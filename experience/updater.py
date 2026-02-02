import json
import copy
import os
import re
from collections import defaultdict

from MobileAgentE.api import inference_chat
from experience.prompts import (
    SINGLE_QUERY_CRITIQUE_TEMPLATE_SP,
    SINGLE_QUERY_CRITIQUE_TEMPLATE_UP,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP,
    GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP,
    GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP,
)


class MobileExperienceUpdater:
    """Mobile领域的经验更新器 - 适配MobileAgent-E轨迹格式"""

    def __init__(self, api_url, api_token, model='gpt-4o', backbone_type='OpenAI'):
        self.api_url = api_url
        self.api_token = api_token
        self.model = model
        self.backbone_type = backbone_type

    def run(self, rollouts, experiences, save_dir, given_ground_truth=True):
        """运行完整的经验更新流程"""
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 50)
        print("🔄 GRPO Experience Update")
        print("=" * 50)
        print(f"Input: {len(rollouts)} rollouts, {len(experiences)} existing experiences")

        # Stage 1: 总结每个rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts,
            save_dir=save_dir,
            given_ground_truth=given_ground_truth
        )

        if not problem_to_summarized_rollouts:
            print("⚠️ No rollouts were summarized")
            return experiences

        # Stage 2: 为每个问题生成critique
        new_experiences = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts,
            experiences=experiences,
            save_dir=save_dir,
            given_ground_truth=given_ground_truth
        )

        if not new_experiences:
            print("⚠️ No new experiences generated")
            return experiences

        # Stage 3: 分组更新
        critiques = self._group_update(
            experiences=experiences,
            new_experiences=new_experiences,
            save_dir=save_dir
        )

        if not critiques:
            print("⚠️ No critiques generated")
            return experiences

        # Stage 4: 批量更新
        new_experiences = self._batch_update(
            experiences=experiences,
            critiques=critiques,
            save_dir=save_dir
        )

        # 重新分配ID
        new_experiences = {f"G{i}": exp for i, exp in enumerate(new_experiences.values())}

        print(f"\n✅ Experience update completed!")
        print(f"   Old: {len(experiences)} experiences")
        print(f"   New: {len(new_experiences)} experiences")

        self._print_user_preferences(new_experiences)

        return new_experiences

    def _format_trajectory_from_new_format(self, steps_data):
        """
        从新代码的steps格式提取trajectory

        新代码格式（来自steps.json）:
        [
            {"step": 0, "operation": "init", ...},
            {"step": 1, "operation": "perception", ...},
            {"step": 1, "operation": "planning", ...},
            {"step": 1, "operation": "action",
             "action_thought": "...",
             "action_object": {...},
             "action_description": "..."},
            {"step": 1, "operation": "action_reflection",
             "outcome": "A",
             "progress_status": "..."},
            ...
        ]

        需要提取为GRPO格式:
        [
            {"step": 1, "thought": "...", "action": "...", "summary": "..."},
            ...
        ]
        """
        trajectory = []

        # 按step分组
        steps_by_num = defaultdict(dict)
        for item in steps_data:
            if item.get("operation") in ["action", "action_reflection", "planning"]:
                step_num = item["step"]
                steps_by_num[step_num][item["operation"]] = item

        # 提取每个完整的step
        for step_num in sorted(steps_by_num.keys()):
            step_data = steps_by_num[step_num]

            if "action" not in step_data:
                continue

            action_data = step_data["action"]

            # 提取字段
            thought = action_data.get("action_thought", "")
            action_obj = action_data.get("action_object", {})
            action_desc = action_data.get("action_description", "")

            # 格式化action为字符串
            if isinstance(action_obj, dict):
                action_str = f"{action_obj.get('name', 'Unknown')}({action_obj.get('arguments', {})})"
            else:
                action_str = str(action_obj)

            trajectory.append({
                "step": step_num,
                "thought": thought,
                "action": action_str,
                "summary": action_desc
            })

        return trajectory

    def _single_rollout_summary(self, rollouts, save_dir, given_ground_truth=True):
        """Stage 1: 总结每个rollout"""
        print("\n" + "-" * 50)
        print("📝 Stage 1: Single Rollout Summary")
        print("-" * 50)

        filename = os.path.join(save_dir, "single_rollout_summary.json")

        # 检查缓存
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                if results:
                    print(f"✅ Loaded {len(results)} summaries from cache")
                    return results
            except:
                pass

        print(f"Processing {len(rollouts)} rollouts...")

        # 按问题分组
        problems_to_rollouts = defaultdict(list)
        for rollout in rollouts:
            if "problem" in rollout:
                problems_to_rollouts[rollout["problem"]].append(rollout)

        print(f"✅ Grouped into {len(problems_to_rollouts)} unique problems")

        results = defaultdict(list)
        success_count = 0

        # 处理每个rollout
        for problem, rollouts_list in problems_to_rollouts.items():
            for idx, rollout in enumerate(rollouts_list):
                try:
                    print(f"  [{success_count + 1}] Processing rollout for: {problem[:50]}...", end=" ")

                    # ===== 关键：从新格式读取轨迹 =====
                    # 方式1: 如果rollout中直接有steps数据
                    if "steps" in rollout and isinstance(rollout["steps"], list):
                        trajectory = self._format_trajectory_from_new_format(rollout["steps"])
                    # 方式2: 如果需要从log文件读取
                    elif "log_path" in rollout:
                        log_path = rollout["log_path"]
                        with open(log_path, 'r', encoding='utf-8') as f:
                            steps_data = json.load(f)
                        trajectory = self._format_trajectory_from_new_format(steps_data)
                    # 方式3: 旧格式兼容
                    elif "trajectories" in rollout:
                        trajectory = rollout["trajectories"][0].get("trajectory", [])
                    else:
                        print("❌ No trajectory data")
                        continue

                    if not trajectory:
                        print("❌ Empty trajectory")
                        continue

                    # 格式化轨迹文本
                    trajectory_text = "\n".join([
                        f"Step {t['step']}:\n  Thought: {t['thought']}\n  Action: {t['action']}\n  Summary: {t['summary']}"
                        for t in trajectory
                    ])

                    # 提取任务信息
                    app = rollout.get("apps", "Unknown")
                    intent_category = rollout.get("intent_category", "Unknown")
                    reward = rollout.get("reward", 0)

                    # 构建prompt
                    prompt = SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP.format(
                        app=app,
                        intent_category=intent_category,
                        instruction=problem,
                        trajectory=trajectory_text,
                    )

                    # 调用API（新格式）
                    chat = [
                        ["system", [{"type": "text", "text": SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP}]],
                        ["user", [{"type": "text", "text": prompt}]]
                    ]

                    response = inference_chat(
                        chat=chat,
                        model=self.model,
                        api_url=self.api_url,
                        token=self.api_token,
                        temperature=0.0
                    )

                    if not response or len(response.strip()) < 10:
                        print("❌ Empty response")
                        continue

                    results[problem].append({
                        "trajectory_summary": response,
                        "problem": problem,
                        "app": app,
                        "intent_category": intent_category,
                        "reward": reward,
                        "groundtruth": rollout.get("groundtruth", {})
                    })

                    success_count += 1
                    print(f"✅ (reward={reward:.2f})")

                except Exception as e:
                    print(f"❌ {e}")

        print(f"\n✅ Successfully summarized {success_count} rollouts")

        # 保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dict(results), f, indent=2, ensure_ascii=False)

        return dict(results)

    def _single_query_critique(self, problem_to_summarized_rollouts, experiences, save_dir, given_ground_truth=True):
        """Stage 2: 生成critique"""
        print("\n" + "-" * 50)
        print("💡 Stage 2: Single Query Critique")
        print("-" * 50)

        filename = os.path.join(save_dir, "single_query_critique.json")

        # 检查缓存
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                if results:
                    print(f"✅ Loaded {len(results)} critiques from cache")
                    return results
            except:
                pass

        results = []
        success_count = 0

        for problem, rollouts in problem_to_summarized_rollouts.items():
            try:
                print(f"  [{success_count + 1}] Processing: {problem[:50]}...", end=" ")

                app = rollouts[0].get("app", "Unknown")
                intent_category = rollouts[0].get("intent_category", "Unknown")

                # 格式化多次尝试
                formatted_attempts = "\n\n".join([
                    f"**Attempt {i + 1}** (Reward: {r.get('reward', 0):.2f}):\n{r.get('trajectory_summary', 'N/A')}"
                    for i, r in enumerate(rollouts)
                ])

                # 构建prompt
                prompt = SINGLE_QUERY_CRITIQUE_TEMPLATE_UP.format(
                    app=app,
                    intent_category=intent_category,
                    instruction=problem,
                    attempts=formatted_attempts
                )

                # 调用API
                chat = [
                    ["system", [{"type": "text", "text": SINGLE_QUERY_CRITIQUE_TEMPLATE_SP}]],
                    ["user", [{"type": "text", "text": prompt}]]
                ]

                response = inference_chat(
                    chat=chat,
                    model=self.model,
                    api_url=self.api_url,
                    token=self.api_token,
                    temperature=0.0
                )

                # 提取经验
                pattern = re.compile(r"<Experiences>\s*(.*?)\s*</Experiences>", re.DOTALL | re.IGNORECASE)
                match = pattern.search(response)

                if match:
                    experiences_text = match.group(1).strip()
                    if experiences_text:
                        results.append({
                            "problem": problem,
                            "rollouts": rollouts,
                            "critique": response,
                            "experiences": experiences_text,
                            "app": app,
                            "intent_category": intent_category
                        })
                        success_count += 1
                        exp_lines = len(experiences_text.splitlines())
                        print(f"✅ ({exp_lines} experiences)")
                    else:
                        print("❌ Empty experiences")
                else:
                    print("❌ No <Experiences> tags")

            except Exception as e:
                print(f"❌ {e}")

        print(f"\n✅ Generated {success_count} critiques")

        # 保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def _group_update(self, experiences, new_experiences, save_dir):
        """Stage 3: 分组更新"""
        print("\n" + "-" * 50)
        print("🔀 Stage 3: Group Update")
        print("-" * 50)

        filename = os.path.join(save_dir, "group_update.json")

        # 检查缓存
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                if results:
                    print(f"✅ Loaded {len(results)} operation sets from cache")
                    return results
            except:
                pass

        results = []
        success_count = 0

        # 格式化现有经验
        formatted_experiences = "\n".join([
            f"[{exp_id}] {exp}" for exp_id, exp in experiences.items()
        ]) if experiences else "None"

        for idx, new_exp in enumerate(new_experiences):
            try:
                print(f"  [{idx + 1}/{len(new_experiences)}]", end=" ")

                new_exp_text = new_exp.get("experiences", "")
                if not new_exp_text:
                    print("❌ Empty")
                    continue

                exp_lines = [line.strip() for line in new_exp_text.split('\n') if line.strip()]
                print(f"Processing {len(exp_lines)} lines...", end=" ")

                # 构建prompt
                prompt = GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP.format(
                    existing_experiences=formatted_experiences if formatted_experiences != 'None' else "Empty",
                    new_experiences=new_exp_text
                )

                # 调用API
                chat = [
                    ["system", [{"type": "text", "text": GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP}]],
                    ["user", [{"type": "text", "text": prompt}]]
                ]

                response = inference_chat(
                    chat=chat,
                    model=self.model,
                    api_url=self.api_url,
                    token=self.api_token,
                    temperature=0.0
                )

                # 解析JSON
                operations = self._parse_json_response(response, "group update")

                if operations and isinstance(operations, list):
                    valid_ops = [op for op in operations if isinstance(op, dict) and "operation" in op]
                    if valid_ops:
                        results.append({
                            "operations": valid_ops,
                            "app": new_exp.get("app", "Unknown"),
                            "intent_category": new_exp.get("intent_category", "Unknown"),
                            "experiences": new_exp_text
                        })
                        success_count += 1
                        print(f"✅ {len(valid_ops)} ops")
                    else:
                        print("❌ No valid ops")
                else:
                    print("❌ Invalid JSON")

            except Exception as e:
                print(f"❌ {str(e)[:30]}")

        print(f"\n✅ Processed {success_count}/{len(new_experiences)} experience sets")

        # 保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def _batch_update(self, experiences, critiques, save_dir, max_retries=3):
        """Stage 4: 批量更新"""
        print("\n" + "-" * 50)
        print("⚡ Stage 4: Batch Update")
        print("-" * 50)

        filename = os.path.join(save_dir, "batch_update.json")

        # 检查缓存
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                new_exp = data.get("new_experiences", {})
                if new_exp:
                    print(f"✅ Loaded {len(new_exp)} experiences from cache")
                    return new_exp
            except:
                pass

        # 收集所有操作
        all_operations = []
        for critique in critiques:
            ops = critique.get("operations", [])
            if isinstance(ops, list):
                all_operations.extend(ops)

        if not all_operations:
            print("⚠️ No operations to process")
            return experiences

        print(f"Processing {len(all_operations)} operations...")

        # 使用LLM生成修订计划
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}...", end=" ")

                formatted_exp_ops = self._format_exp_and_ops(experiences, all_operations)

                prompt = BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP.format(
                    experiences_and_operations=formatted_exp_ops
                )

                chat = [
                    ["system", [{"type": "text", "text": BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP}]],
                    ["user", [{"type": "text", "text": prompt}]]
                ]

                response = inference_chat(
                    chat=chat,
                    model=self.model,
                    api_url=self.api_url,
                    token=self.api_token,
                    temperature=0.0
                )

                revision_plan = self._parse_json_response(response, "batch update")

                if revision_plan and isinstance(revision_plan, list):
                    print(f"✅ Generated {len(revision_plan)} operations")
                    break
                else:
                    raise ValueError("Invalid revision plan")

            except Exception as e:
                print(f"❌ {e}")
                if attempt == max_retries - 1:
                    print("⚠️ All retries failed, using empty revision plan")
                    revision_plan = []

        # 应用修订计划
        new_experiences = copy.deepcopy(experiences)
        max_id = len(experiences)

        add_count = update_count = delete_count = 0

        for plan in revision_plan:
            try:
                operation = plan.get("operation", "").upper()
                content = plan.get("content", "")
                target_id = plan.get("id", None)

                if operation == "ADD" and content:
                    new_experiences[f"G{max_id}"] = content
                    max_id += 1
                    add_count += 1

                elif operation == "UPDATE":
                    if target_id and target_id in new_experiences and content:
                        new_experiences[target_id] = content
                        update_count += 1
                    elif content:
                        new_experiences[f"G{max_id}"] = content
                        max_id += 1
                        add_count += 1

                elif operation == "DELETE" and target_id and target_id in new_experiences:
                    del new_experiences[target_id]
                    delete_count += 1

            except:
                pass

        print(f"\n✅ Applied revision plan:")
        print(f"   Added: {add_count}, Updated: {update_count}, Deleted: {delete_count}")
        print(f"   Final count: {len(new_experiences)} experiences")

        # 保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "operations": all_operations,
                "revision_plan": revision_plan,
                "new_experiences": new_experiences
            }, f, indent=2, ensure_ascii=False)

        return new_experiences

    def _parse_json_response(self, response, stage_name):
        """解析JSON响应"""
        if not response:
            return None

        try:
            # 方法1: 直接解析
            return json.loads(response)
        except:
            pass

        # 方法2: 提取```json```
        if "```json" in response:
            try:
                json_text = response.split("```json")[-1].split("```")[0].strip()
                return json.loads(json_text)
            except:
                pass

        # 方法3: 正则匹配
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                try:
                    return json.loads(match)
                except:
                    continue

        return None

    def _format_exp_and_ops(self, experiences, operations):
        """格式化经验和操作"""
        formatted_parts = []

        for exp_id, exp_content in experiences.items():
            part = f"Experience {exp_id}:\nContent: {exp_content}\n"
            related_ops = [op for op in operations if op.get("id") == exp_id]
            if related_ops:
                part += "Related Operations:\n"
                for op in related_ops:
                    part += f"  {json.dumps(op, ensure_ascii=False)}\n"
            else:
                part += "No related operations.\n"
            formatted_parts.append(part)

        no_id_ops = [op for op in operations if not op.get("id")]
        if no_id_ops:
            part = "Operations without ID (New Experiences):\n"
            for op in no_id_ops:
                part += f"  {json.dumps(op, ensure_ascii=False)}\n"
            formatted_parts.append(part)

        return "\n\n".join(formatted_parts)

    def _print_user_preferences(self, experiences):
        """打印用户偏好统计"""
        print("\n" + "=" * 50)
        print("👤 User Preferences Summary")
        print("=" * 50)

        preferences = {}
        for exp_id, exp_content in experiences.items():
            if "[User Preference" in exp_content:
                match = re.search(r'\[User Preference - (.*?)\]', exp_content)
                if match:
                    category = match.group(1)
                    if category not in preferences:
                        preferences[category] = []
                    preferences[category].append(exp_content)

        if preferences:
            for category, prefs in sorted(preferences.items()):
                print(f"\n{category}:")
                for pref in prefs:
                    print(f"  • {pref}")
        else:
            print("  No user preferences learned yet")

        print("=" * 50)