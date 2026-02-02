"""
RAG Knowledge Base Builder - 完整修复版
修复:
1. Level2知识库自动创建和保存
2. Multi_app任务的检索和保存支持
3. 🔥 从steps字段读取轨迹数据（而不是空的trajectories）
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from langchain.embeddings.base import Embeddings


class RAGBuilder:
    """RAG知识库构建器 - 完整修复版"""

    def __init__(
        self,
        data_dir: str = "./data",
        embeddings: Optional[Embeddings] = None,
        api_config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.level1_dir = self.data_dir / "level1"
        self.level2_dir = self.data_dir / "level2"

        self.embeddings = embeddings
        self.api_config = api_config

        self.level1_dir.mkdir(parents=True, exist_ok=True)
        self.level2_dir.mkdir(parents=True, exist_ok=True)

    def set_embeddings(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def set_api_config(self, api_config: Dict):
        self.api_config = api_config

    # ==================== 工具函数 ====================

    def _parse_apps(self, apps_field: Union[str, List[str]]) -> List[str]:
        """解析apps字段，支持多种格式"""
        if isinstance(apps_field, str):
            apps = [app.strip() for app in apps_field.split(",")]
        elif isinstance(apps_field, list):
            apps = []
            for item in apps_field:
                if isinstance(item, str):
                    apps.extend([app.strip() for app in item.split(",")])
                else:
                    apps.append(str(item).strip())
        else:
            apps = [str(apps_field).strip()]
        return [app for app in apps if app]

    def _extract_trajectory_from_steps(self, steps: List[Dict]) -> List[Dict]:
        """
        🔥 核心修复：从steps.json格式中提取轨迹

        steps.json格式示例:
        [
            {"step": 0, "operation": "init", ...},
            {"step": 1, "operation": "action", "action_thought": "...", "action_object": {...}, ...},
            {"step": 2, "operation": "perception", ...},
            ...
        ]

        提取后的trajectory格式:
        [
            {"action": "Tap(500, 300)", "thought": "点击搜索框"},
            ...
        ]
        """
        trajectory = []

        for step in steps:
            if not isinstance(step, dict):
                continue

            operation = step.get("operation", "")

            # 只提取action类型的步骤
            if operation == "action":
                action_obj = step.get("action_object", {})
                action_thought = step.get("action_thought", "")
                action_description = step.get("action_description", "")
                action_str = step.get("action_object_str", "")

                # 构建action字符串
                if isinstance(action_obj, dict):
                    action_name = action_obj.get("name", "")
                    action_args = action_obj.get("arguments", {})
                    if action_args:
                        args_str = ", ".join([f"{k}={v}" for k, v in action_args.items()])
                        action = f"{action_name}({args_str})"
                    else:
                        action = action_name
                elif action_str:
                    action = action_str
                else:
                    action = str(action_obj)

                trajectory.append({
                    "action": action,
                    "thought": action_thought or action_description,
                    "step": step.get("step", len(trajectory) + 1)
                })

        return trajectory

    def _get_trajectory_from_rollout(self, rollout: Dict) -> List[Dict]:
        """
        从rollout中获取trajectory，支持多种数据格式

        优先级:
        1. trajectories字段（如果非空）
        2. steps字段（从steps.json格式转换）
        """
        # 尝试从trajectories获取
        trajectories = rollout.get("trajectories", [])
        if trajectories:
            traj = trajectories[0].get("trajectory", [])
            if traj:  # 如果非空，直接返回
                return traj

        # 尝试从steps获取
        steps = rollout.get("steps", [])
        if steps:
            return self._extract_trajectory_from_steps(steps)

        return []

    # ==================== 一级库构建 ====================

    def build_level1_from_dataset(self, dataset: List[Dict[str, Any]]):
        """从数据集构建一级知识库"""
        print("\n" + "=" * 70)
        print(" Building Level 1 Knowledge Base (Category)")
        print("=" * 70)

        category_map = defaultdict(set)
        category_instructions = defaultdict(list)

        for item in dataset:
            apps_field = item.get("apps", [])
            apps = self._parse_apps(apps_field)
            category = item.get("intent_category", "Unknown").strip()
            instruction = item.get("instruction", "")

            for app in apps:
                if app:
                    category_map[category].add(app)

            if instruction and len(category_instructions[category]) < 10:
                category_instructions[category].append(instruction)

        for idx, (category, apps_set) in enumerate(category_map.items()):
            apps = list(apps_set)

            description = self._generate_category_description_with_llm(
                category, apps, category_instructions[category]
            )

            category_data = {
                "id": idx,
                "category": category,
                "apps": apps,
                "description": description
            }

            category_file = self.level1_dir / f"{category}.json"
            with category_file.open("w", encoding="utf-8") as f:
                json.dump(category_data, f, indent=2, ensure_ascii=False)

            print(f"Created: {category_file.name} (apps: {apps})")

        print(f"\nLevel 1: Created {len(category_map)} category files")

    def _generate_category_description_with_llm(
        self, category: str, apps: List[str], sample_instructions: List[str]
    ) -> str:
        """使用LLM生成category描述"""
        if not self.api_config:
            return f"Category: {category}. Apps: {', '.join(apps)}."

        try:
            from .api import inference_chat, init_action_chat, add_response
        except ImportError:
            try:
                from api import inference_chat, init_action_chat, add_response
            except ImportError:
                return f"Category: {category}. Apps: {', '.join(apps)}."

        prompt = f"""Analyze this mobile app category and generate a concise description.

Category: {category}
Apps in this category: {', '.join(apps)}

Sample user instructions:
{chr(10).join(f'- {inst}' for inst in sample_instructions[:5])}

Please generate:
1. A brief description of what users typically do in this category (1-2 sentences)
2. Common task types (without listing specific examples)

Return format:
Category: {category}
Description: [your description]
Common tasks: [task types]"""

        try:
            chat = init_action_chat()
            chat = add_response("user", prompt, chat)

            response = inference_chat(
                chat,
                model=self.api_config.get("model", "gpt-4o"),
                api_url=self.api_config["api_url"],
                token=self.api_config["token"]
            )

            return response.strip()
        except Exception as e:
            print(f"  LLM description generation failed: {e}")
            return f"Category: {category}. Apps: {', '.join(apps)}."

    # ==================== 二级库构建 ====================

    def build_level2_from_dataset(self, dataset: List[Dict[str, Any]]):
        """从数据集构建二级知识库"""
        print("\n" + "=" * 70)
        print(" Building Level 2 Knowledge Base (App)")
        print("=" * 70)

        unique_apps = {}
        for item in dataset:
            apps_field = item.get("apps", [])
            apps = self._parse_apps(apps_field)
            category = item.get("intent_category", "Unknown").strip()

            for app in apps:
                if app and app not in unique_apps:
                    unique_apps[app] = category

        for app_name, category in unique_apps.items():
            self._ensure_app_directory(app_name, category)
            print(f"Created: {app_name}/")

        print(f"\nLevel 2: Created {len(unique_apps)} app directories")

    def _ensure_app_directory(self, app_name: str, category: str = "Unknown"):
        """确保app目录存在"""
        app_dir = self.level2_dir / app_name

        if app_dir.exists():
            return

        app_dir.mkdir(exist_ok=True)

        # meta.json
        meta_data = {
            "app_name": app_name,
            "category": category,
            "description": f"{app_name} is a {category} app for mobile operations."
        }
        with (app_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        # workflows.json
        with (app_dir / "workflows.json").open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

        # preferences.json
        pref_data = {
            "task_preferences": {},
            "content_preferences": {},
            "expression_patterns": {},
            "total_usage": 0
        }
        with (app_dir / "preferences.json").open("w", encoding="utf-8") as f:
            json.dump(pref_data, f, indent=2, ensure_ascii=False)

        # usage_stats.json
        usage_data = {
            "usage_count": 0,
            "success_count": 0,
            "last_used": None,
            "avg_steps": 0
        }
        with (app_dir / "usage_stats.json").open("w", encoding="utf-8") as f:
            json.dump(usage_data, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Auto-created app directory: {app_name}/")

    # ==================== 从Rollout学习（核心修复） ====================

    def learn_from_rollout(
        self,
        app_name: Union[str, List[str]],
        instruction: str,
        trajectory: List[Dict] = None,
        success: bool = True,
        reward: float = 0.0,
        category: str = "Unknown",
        steps: List[Dict] = None,  # 🔥 新增：直接接收steps
        rollout: Dict = None  # 🔥 新增：接收完整rollout
    ):
        """
        从单个rollout学习

        🔥 修复：支持多种数据源
        - trajectory: 直接传入的轨迹列表
        - steps: steps.json格式的数据
        - rollout: 完整的rollout字典
        """
        if not success or reward < 0.8:
            return

        # 获取轨迹数据
        if trajectory and len(trajectory) > 0:
            traj = trajectory
        elif steps and len(steps) > 0:
            traj = self._extract_trajectory_from_steps(steps)
        elif rollout:
            traj = self._get_trajectory_from_rollout(rollout)
        else:
            print(f"  ⚠️ No trajectory data found for learning")
            return

        if not traj or len(traj) < 2:
            print(f"  ⚠️ Trajectory too short ({len(traj) if traj else 0} steps), skipping")
            return

        # 解析apps
        apps = self._parse_apps(app_name)

        for single_app in apps:
            self._learn_single_app(
                app_name=single_app,
                instruction=instruction,
                trajectory=traj,
                success=success,
                reward=reward,
                category=category
            )

    def _learn_single_app(
        self,
        app_name: str,
        instruction: str,
        trajectory: List[Dict],
        success: bool,
        reward: float,
        category: str
    ):
        """为单个app学习workflow"""
        app_dir = self.level2_dir / app_name

        # 自动创建目录
        if not app_dir.exists():
            print(f"  ⚠️ App directory not found, creating: {app_name}")
            self._ensure_app_directory(app_name, category)

        # 1. 提取并保存workflow
        workflow = self._extract_workflow(instruction, trajectory)
        if workflow:
            self._update_workflows(app_dir, workflow)
            print(f"  ✅ Saved workflow for {app_name}")

        # 2. 更新preferences
        meta = self._load_json(app_dir / "meta.json")
        app_category = meta.get("category", category)

        preferences = self._extract_preferences(app_category, instruction, trajectory)
        if any(preferences.values()):
            self._update_preferences(app_dir, preferences)

        # 3. 更新usage_stats
        self._update_usage_stats(app_dir, len(trajectory), success)

        print(f"  ✅ Learned from rollout: {app_name}")

    def _extract_workflow(
        self,
        instruction: str,
        trajectory: List[Dict]
    ) -> Optional[Dict]:
        """
        提取workflow - 使用LLM或简单规则
        """
        if not trajectory or len(trajectory) < 2:
            return None

        # 如果有LLM API，使用LLM提取
        if self.api_config:
            workflow = self._extract_workflow_with_llm(instruction, trajectory)
            if workflow:
                return workflow

        # 降级：简单提取
        steps = []
        ui_elements = []

        for i, step in enumerate(trajectory, 1):
            action = step.get("action", "")
            thought = step.get("thought", "")
            steps.append(f"Step {i}: {action}")

            # 提取UI元素和坐标
            if "(" in action and ")" in action:
                ui_elements.append({
                    "element": f"Step {i} target",
                    "position": action,
                    "action": thought[:50] if thought else action
                })

        return {
            "task": instruction[:100],
            "task_summary": f"Execute {len(trajectory)} steps to complete: {instruction[:50]}",
            "ui_elements": ui_elements[:5],  # 限制数量
            "steps": steps,
            "success_count": 1,
            "raw_trajectory_length": len(trajectory)
        }

    def _extract_workflow_with_llm(
        self,
        instruction: str,
        trajectory: List[Dict]
    ) -> Optional[Dict]:
        """使用LLM提取workflow"""
        try:
            from .api import inference_chat, init_action_chat, add_response
        except ImportError:
            try:
                from api import inference_chat, init_action_chat, add_response
            except ImportError:
                return None

        # 构建轨迹摘要
        steps_text = []
        for i, step in enumerate(trajectory[:10], 1):  # 限制长度
            action = step.get("action", "")
            thought = step.get("thought", "")
            steps_text.append(f"Step {i}: {action} | Thought: {thought}")

        extraction_prompt = f"""Analyze this mobile operation workflow and extract key information.

Task: {instruction}

Steps:
{chr(10).join(steps_text)}

Please extract:
1. **UI Elements**: Identify buttons, input fields, or text elements with their positions/coordinates
2. **Action Sequence**: Summarize the workflow in concise steps

Return JSON format:
{{
  "task_summary": "brief task description (one sentence)",
  "ui_elements": [
    {{"element": "element name or text", "position": "coordinates or description", "action": "what action to perform"}}
  ],
  "steps": ["step1", "step2", ...]
}}"""

        try:
            chat = init_action_chat()
            chat = add_response("user", extraction_prompt, chat)

            response = inference_chat(
                chat,
                model=self.api_config.get("model", "gpt-4o"),
                api_url=self.api_config["api_url"],
                token=self.api_config["token"]
            )

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_info = json.loads(json_match.group())

                return {
                    "task": instruction[:100],
                    "task_summary": extracted_info.get("task_summary", ""),
                    "ui_elements": extracted_info.get("ui_elements", []),
                    "steps": extracted_info.get("steps", []),
                    "success_count": 1,
                    "raw_trajectory_length": len(trajectory)
                }
        except Exception as e:
            print(f"  LLM workflow extraction failed: {e}")

        return None

    def _extract_preferences(
        self,
        category: str,
        instruction: str,
        trajectory: List[Dict]
    ) -> Dict[str, Any]:
        """
        提取用户偏好 - 使用LLM智能分析

        提取三种偏好:
        1. task_preference: 任务类型（如：搜索、播放、导航）
        2. content_preference: 具体内容偏好（如：书名、歌曲名、地点）
        3. expression_preference: 表达模式（如：Open {{app}} and search {{query}}）
        """
        # 如果有LLM API，使用LLM提取
        if self.api_config:
            llm_result = self._extract_preferences_with_llm(category, instruction, trajectory)
            if llm_result:
                return llm_result

        # 降级：使用规则提取
        return self._extract_preferences_by_rules(instruction)

    def _extract_preferences_with_llm(
        self,
        category: str,
        instruction: str,
        trajectory: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """使用LLM提取偏好"""
        try:
            from .api import inference_chat, init_action_chat, add_response
        except ImportError:
            try:
                from api import inference_chat, init_action_chat, add_response
            except ImportError:
                return None

        prompt = f"""Analyze this mobile task instruction and extract user preferences.

Category: {category}
Instruction: "{instruction}"

Please extract the following (respond in English):

1. **task_preference**: What type of task is this? 
   Examples: search, play_music, navigate, read, browse, order_food, watch_video
   
2. **content_preference**: What specific content is the user interested in?
   Examples: book titles, song names, locations, food types, video topics
   Extract the actual content from the instruction, not generic descriptions.
   
3. **expression_preference**: Convert the instruction to an English pattern with placeholders.
   Use these placeholders: {{{{app}}}}, {{{{query}}}}, {{{{content}}}}, {{{{location}}}}, {{{{time}}}}
   Examples:
   - "打开微信读书阅读乡土中国" → "Open {{{{app}}}} and read {{{{content}}}}"
   - "打开高德地图导航到北京大学" → "Open {{{{app}}}} and navigate to {{{{location}}}}"
   - "打开QQ音乐播放周杰伦的歌" → "Open {{{{app}}}} and play {{{{content}}}}"

Return JSON format only:
{{
  "task_preference": "task type",
  "content_preference": "specific content extracted from instruction",
  "expression_preference": "pattern with placeholders"
}}"""

        try:
            chat = init_action_chat()
            chat = add_response("user", prompt, chat)

            response = inference_chat(
                chat,
                model=self.api_config.get("model", "gpt-4o"),
                api_url=self.api_config["api_url"],
                token=self.api_config["token"]
            )

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 过滤空值
                return {k: v for k, v in result.items() if v and v.strip()}
        except Exception as e:
            print(f"  LLM preference extraction failed: {e}")

        return None

    def _extract_preferences_by_rules(self, instruction: str) -> Dict[str, Any]:
        """使用规则提取偏好（降级方案）"""
        preferences = {}

        # 任务类型映射
        task_patterns = {
            "search": ["搜索", "查找", "找", "search"],
            "play_music": ["播放", "听", "play"],
            "navigate": ["导航", "去", "到", "navigate"],
            "read": ["阅读", "读", "看书", "read"],
            "browse": ["浏览", "逛", "看看", "browse"],
            "order": ["点餐", "外卖", "order"],
            "watch": ["看视频", "观看", "watch"]
        }

        for task_type, keywords in task_patterns.items():
            for kw in keywords:
                if kw in instruction:
                    preferences["task_preference"] = task_type
                    break
            if "task_preference" in preferences:
                break

        # 提取具体内容（使用更智能的方式）
        # 移除常见的动词和app名称，剩下的就是内容
        content = instruction
        remove_words = ["打开", "搜索", "播放", "导航到", "阅读", "查看", "去", "到"]
        for word in remove_words:
            content = content.replace(word, "")
        content = content.strip()

        if content and len(content) > 2:
            preferences["content_preference"] = content[:50]

        return preferences

    # ==================== 批量学习（核心修复） ====================

    def batch_learn_from_rollouts(
        self,
        rollouts: List[Dict],
        success_threshold: float = 0.8
    ):
        """
        🔥 批量从rollouts学习 - 支持steps格式
        """
        print("\n" + "=" * 70)
        print(" Learning from Rollouts (Fixed Version)")
        print("=" * 70)

        learned_count = 0
        skipped_count = 0

        for rollout in rollouts:
            reward = rollout.get("reward", 0)
            if reward < success_threshold:
                skipped_count += 1
                continue

            apps_field = rollout.get("apps", rollout.get("app", ""))
            if not apps_field:
                skipped_count += 1
                continue

            instruction = rollout.get("problem", rollout.get("instruction", ""))
            category = rollout.get("intent_category", "Unknown")

            # 🔥 使用修复后的方法获取轨迹
            trajectory = self._get_trajectory_from_rollout(rollout)

            if not trajectory or len(trajectory) < 2:
                print(f"  ⚠️ Skipping rollout: no valid trajectory (got {len(trajectory) if trajectory else 0} steps)")
                skipped_count += 1
                continue

            print(f"  📚 Learning from: {instruction[:40]}... ({len(trajectory)} steps)")

            self.learn_from_rollout(
                app_name=apps_field,
                instruction=instruction,
                trajectory=trajectory,
                success=True,
                reward=reward,
                category=category
            )
            learned_count += 1

        print(f"\n✅ Learned from {learned_count} successful rollouts")
        print(f"   Skipped: {skipped_count}")

    # ==================== 数据更新方法 ====================

    def _update_workflows(self, app_dir: Path, new_workflow: Dict):
        """更新workflows.json"""
        workflows_file = app_dir / "workflows.json"
        workflows = self._load_json(workflows_file)

        if not isinstance(workflows, list):
            workflows = []

        # 查找相似workflow
        task_found = False
        for wf in workflows:
            if self._is_similar_task(wf.get("task", ""), new_workflow.get("task", "")):
                wf["success_count"] = wf.get("success_count", 0) + 1
                if len(new_workflow.get("ui_elements", [])) > len(wf.get("ui_elements", [])):
                    wf["ui_elements"] = new_workflow["ui_elements"]
                    wf["steps"] = new_workflow.get("steps", wf.get("steps", []))
                task_found = True
                break

        if not task_found:
            workflows.append(new_workflow)

        # 限制数量
        workflows = sorted(workflows, key=lambda x: x.get("success_count", 0), reverse=True)[:20]

        self._save_json(workflows_file, workflows)

    def _update_preferences(self, app_dir: Path, preferences: Dict[str, Any]):
        """更新preferences.json"""
        pref_file = app_dir / "preferences.json"
        pref_data = self._load_json(pref_file)

        if "task_preferences" not in pref_data:
            pref_data["task_preferences"] = {}
        if "content_preferences" not in pref_data:
            pref_data["content_preferences"] = {}
        if "expression_patterns" not in pref_data:
            pref_data["expression_patterns"] = {}

        if preferences.get("task_preference"):
            task = preferences["task_preference"]
            pref_data["task_preferences"][task] = pref_data["task_preferences"].get(task, 0) + 1

        if preferences.get("content_preference"):
            content = preferences["content_preference"]
            pref_data["content_preferences"][content] = pref_data["content_preferences"].get(content, 0) + 1

        if preferences.get("expression_preference"):
            pattern = preferences["expression_preference"]
            pref_data["expression_patterns"][pattern] = pref_data["expression_patterns"].get(pattern, 0) + 1

        pref_data["total_usage"] = pref_data.get("total_usage", 0) + 1

        self._save_json(pref_file, pref_data)

    def _update_usage_stats(self, app_dir: Path, steps: int, success: bool):
        """更新usage_stats.json"""
        stats_file = app_dir / "usage_stats.json"
        stats = self._load_json(stats_file)

        if "usage_count" not in stats:
            stats["usage_count"] = 0
        if "success_count" not in stats:
            stats["success_count"] = 0

        stats["usage_count"] += 1
        if success:
            stats["success_count"] += 1

        total_steps = stats.get("avg_steps", 0) * (stats["usage_count"] - 1) + steps
        stats["avg_steps"] = total_steps / stats["usage_count"]

        from datetime import datetime
        stats["last_used"] = datetime.now().isoformat()

        self._save_json(stats_file, stats)

    def _is_similar_task(self, task1: str, task2: str, threshold: float = 0.7) -> bool:
        """判断任务相似度"""
        if not task1 or not task2:
            return False

        try:
            if self.embeddings:
                emb1 = self.embeddings.embed_query(task1)
                emb2 = self.embeddings.embed_query(task2)
                emb1 = np.array(emb1)
                emb2 = np.array(emb2)
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity) > threshold
        except Exception:
            pass

        # 降级：词重叠
        import re
        words1 = set(re.findall(r'\w+', task1.lower()))
        words2 = set(re.findall(r'\w+', task2.lower()))
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return (intersection / union) > 0.5 if union > 0 else False

    def _load_json(self, file_path: Path) -> Any:
        """加载JSON文件"""
        if not file_path.exists():
            return {} if file_path.name != "workflows.json" else []
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {} if file_path.name != "workflows.json" else []

    def _save_json(self, file_path: Path, data: Any):
        """保存JSON文件"""
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("RAG Knowledge Base Statistics")
        print("=" * 70)

        level1_files = list(self.level1_dir.glob("*.json"))
        print(f"\n Level 1 (Category): {len(level1_files)} categories")

        app_dirs = [d for d in self.level2_dir.iterdir() if d.is_dir()]
        print(f" Level 2 (App): {len(app_dirs)} apps")

        for app_dir in sorted(app_dirs):
            meta = self._load_json(app_dir / "meta.json")
            workflows = self._load_json(app_dir / "workflows.json")
            stats = self._load_json(app_dir / "usage_stats.json")

            usage_count = stats.get("usage_count", 0)
            workflow_count = len(workflows) if isinstance(workflows, list) else 0

            print(f"\n  📱 {meta.get('app_name', app_dir.name)}")
            print(f"    • Usage: {usage_count} times")
            print(f"    • Workflows: {workflow_count} learned")
            if usage_count > 0:
                print(f"    • Success rate: {stats.get('success_count', 0) / usage_count * 100:.1f}%")

        print("=" * 70)


def init_rag_builder(
    data_dir: str = None,
    embeddings: Optional[Embeddings] = None,
    api_config: Optional[Dict] = None
) -> RAGBuilder:
    """初始化RAG Builder"""
    if data_dir is None:
        try:
            from config import Config
            data_dir = Config.RAG_DATA_DIR
        except:
            data_dir = "./data/rag"

    return RAGBuilder(
        data_dir=data_dir,
        embeddings=embeddings,
        api_config=api_config
    )


def init_rag_builder_from_config(
    embeddings: Optional[Embeddings] = None,
    api_config: Optional[Dict] = None
) -> RAGBuilder:
    """从Config初始化RAG Builder（向后兼容）"""
    return init_rag_builder(data_dir=None, embeddings=embeddings, api_config=api_config)