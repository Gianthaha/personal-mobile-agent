
from inference_agent_E import run_single_task
from inference_agent_E import Perceptor, DEFAULT_PERCEPTION_ARGS, ADB_PATH, INIT_TIPS, INIT_SHORTCUTS, REASONING_MODEL
import torch
import os
import json
import copy
import random
from experience.updater import MobileExperienceUpdater
from verify.mobile_verify import verify_func
import time
## Reasoning model configs
BACKBONE_TYPE = os.environ.get("BACKBONE_TYPE", default="OpenAI") # "OpenAI" or "Gemini" or "Claude"
assert BACKBONE_TYPE in ["OpenAI", "Gemini", "Claude"], "Unknown BACKBONE_TYPE"
print("### Using BACKBONE_TYPE:", BACKBONE_TYPE)

OPENAI_API_URL = os.environ.get("OPENAI_API_URL", default="https://xiaoai.plus/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default="sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo")

GEMINI_API_URL = "https://xiaoai.plus/v1/chat/completions" # OpenAI compatible
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", default="sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo")

CLAUDE_API_URL = "https://xiaoai.plus/v1/chat/completions"
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", default="sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo")

if BACKBONE_TYPE == "OpenAI":
    REASONING_MODEL = os.environ.get("OPENAI_MODEL", default="gpt-4o")
    KNOWLEDGE_REFLECTION_MODEL = os.environ.get("OPENAI_MODEL", default="gpt-4o")
elif BACKBONE_TYPE == "Gemini":
    REASONING_MODEL = "gemini-1.5-pro-latest"
    KNOWLEDGE_REFLECTION_MODEL = "gemini-1.5-pro-latest"
elif BACKBONE_TYPE == "Claude":
    REASONING_MODEL = "claude-3-5-sonnet-20241022"
    KNOWLEDGE_REFLECTION_MODEL = "claude-3-5-sonnet-20241022"


# ==================== 新增：GRPO训练模式 ====================


def run_grpo_training(
        tasks_json,
        run_name="grpo_train",
        log_root=None,
        epochs=1,
        batchsize=1,
        grpo_n=1,
        num_tasks=None,
        max_itr=40,
        max_consecutive_failures=5,
        max_repetitive_actions=5,
        enable_experience_retriever=True,  # 建议默认开启
        temperature=0.0,
        build_rag=False,
        use_rag=False,
        rag_data_dir="./data/rag",
        force_rebuild_rag=False,
):

    if log_root is None:
        log_root = f"logs/{REASONING_MODEL}/grpo_train"

    print("=" * 70)
    print("📚 Mobile Agent GRPO Training")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"Tasks: {tasks_json}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batchsize}")
    print(f"GRPO N: {grpo_n}")
    if num_tasks:
        print(f"Training data limit: {num_tasks} tasks")
    print("=" * 70 + "\n")

    experiment_dir = os.path.join(log_root, run_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 加载任务
    with open(tasks_json, "r", encoding="utf-8-sig") as f:
        task_json = json.load(f)
    tasks = task_json.get("tasks", task_json)

    rag_builder = None
    rag_index_builder = None
    rag_query_engine = None

    if build_rag or use_rag:
        from inference_agent_E import init_rag_system, build_rag_from_dataset
        if BACKBONE_TYPE == "OpenAI":
            api_config = {
                "api_url": OPENAI_API_URL,
                "token": OPENAI_API_KEY,
                "model": REASONING_MODEL
            }
        elif BACKBONE_TYPE == "Gemini":
            api_config = {
                "api_url": GEMINI_API_URL,
                "token": GEMINI_API_KEY,
                "model": REASONING_MODEL
            }
        elif BACKBONE_TYPE == "Claude":
            api_config = {
                "api_url": CLAUDE_API_URL,
                "token": CLAUDE_API_KEY,
                "model": REASONING_MODEL
            }
        else:
            raise ValueError(f"Unknown BACKBONE_TYPE: {BACKBONE_TYPE}")

        # 初始化 RAG 系统
        rag_builder, rag_index_builder, rag_query_engine = init_rag_system(
            api_config,
            data_dir=rag_data_dir
        )

        # 检查是否需要构建初始知识库
        level1_dir = os.path.join(rag_data_dir, "level1")
        indices_dir = os.path.join(rag_data_dir, "indices")

        if force_rebuild_rag or not (os.path.exists(level1_dir) and os.path.exists(indices_dir)):
            if force_rebuild_rag:
                print("\n🔄 Force rebuilding RAG knowledge base...")
                # 删除旧数据
                import shutil
                if os.path.exists(rag_data_dir):
                    shutil.rmtree(rag_data_dir)
            else:
                print("\n📚 RAG knowledge base not found, building for the first time...")

            build_rag_from_dataset(tasks, api_config, data_dir=rag_data_dir)
        else:
            print("\n✅ RAG knowledge base already exists")

    print("\n" + "=" * 70)
    print("Training Configuration:")
    print("=" * 70)
    print(f"  Build RAG (Learning):     {'✅ Enabled' if build_rag else '❌ Disabled'}")
    print(f"  Use RAG (Retrieval):      {'✅ Enabled' if use_rag else '❌ Disabled'}")
    if build_rag or use_rag:
        print(f"  RAG Data Directory:       {rag_data_dir}")
    print("=" * 70 + "\n")

    if num_tasks and num_tasks < len(tasks):
        tasks = tasks[:num_tasks]
        print(f"⚠️ Limited to first {num_tasks} tasks\n")

    print(f"Loaded {len(tasks)} training tasks\n")

    if len(tasks) % batchsize != 0:
        print(f"⚠️ Warning: {len(tasks)} tasks is not divisible by batchsize {batchsize}\n")

    # 初始化
    perceptor = Perceptor(ADB_PATH, perception_args=DEFAULT_PERCEPTION_ARGS)

    experience_updater = MobileExperienceUpdater(
        api_url=OPENAI_API_URL if BACKBONE_TYPE == "OpenAI" else (
            GEMINI_API_URL if BACKBONE_TYPE == "Gemini" else CLAUDE_API_URL
        ),
        api_token=OPENAI_API_KEY if BACKBONE_TYPE == "OpenAI" else (
            GEMINI_API_KEY if BACKBONE_TYPE == "Gemini" else CLAUDE_API_KEY
        ),
        model=REASONING_MODEL,
        backbone_type=BACKBONE_TYPE
    )

    # 加载或初始化统计信息
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        with open(stats_filename, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        print(f"Loaded existing stats\n")
    else:
        stats = {}

    # 训练循环
    for epoch in range(epochs):
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch + 1}/{epochs}")
        print("=" * 70)

        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # 检查是否已有shuffled数据
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_tasks = []
            with open(shuffled_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    shuffled_tasks.append(json.loads(line))
            print(f"Loaded shuffled data from {shuffled_filename}")
        else:
            print("Shuffling training data...")
            shuffled_tasks = copy.deepcopy(tasks)
            random.seed(42 + epoch)
            random.shuffle(shuffled_tasks)
            with open(shuffled_filename, 'w', encoding='utf-8') as f:
                for task in shuffled_tasks:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")
            print(f"Saved shuffled data")

        num_batches = len(shuffled_tasks) // batchsize

        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx

            # 检查是否已完成
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                print(f"\nStep {step} already completed, skipping...")
                continue

            print("\n" + "-" * 70)
            print(f"STEP {step} (Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches})")
            print("-" * 70)

            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)
             # 当前batch的任务
            batch_tasks = copy.deepcopy(
                shuffled_tasks[batch_idx * batchsize: (batch_idx + 1) * batchsize]
            )


            experiences = {}
            experience_filename = os.path.join(experiment_dir, f"step_{step}", "experiences.json")

            if os.path.exists(experience_filename):
                try:
                    with open(experience_filename, 'r', encoding='utf-8') as f:
                        experiences = json.load(f)
                    print(f"📚 Loaded {len(experiences)} experiences from step_{step}/experiences.json")
                    print(f"   (These were generated by step {step - 1})")

                    user_prefs = sum(1 for exp in experiences.values() if "[User Preference" in exp)
                    if user_prefs > 0:
                        print(f"   Including {user_prefs} user preference experiences")
                except Exception as e:
                    print(f"⚠️ Failed to load experiences: {e}")
                    experiences = {}
            else:
                if step == 0:
                    print("🆕 Step 0: Starting without experiences (will generate step_1/experiences.json)")
                else:
                    print(f"⚠️ No experiences found at {experience_filename}")
                    print(f"   This might indicate step {step - 1} failed to generate experiences")


            # 直接准备原始任务数据
            formatted_batch_tasks = []
            for task in batch_tasks:
                formatted_batch_tasks.append({
                    "instruction": task.get("instruction", task.get("problem", "")),
                    "app": task.get("app", "Unknown"),
                    "intent_category": task.get("intent_category", "Unknown"),
                    "groundtruth": task.get("groundtruth", {}),
                })

            # GRPO采样 - 整个batch重复n次
            print(f"\n🔄 GRPO sampling: {grpo_n} rollouts per problem")
            print(f"   Current batch: {len(batch_tasks)} problems × {grpo_n} = {len(batch_tasks) * grpo_n} rollouts\n")

            formatted_batch_tasks = formatted_batch_tasks * grpo_n

            # 加载已有rollouts
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            if os.path.exists(rollout_filename):
                rollouts = []
                with open(rollout_filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        rollouts.append(json.loads(line))
                print(f"📂 Loaded {len(rollouts)} existing rollouts\n")
            else:
                rollouts = []

            # 初始化rollouts
            if len(rollouts) == 0:
                for i, task in enumerate(formatted_batch_tasks):
                    rollouts.append({
                        "runid": i,
                        "problem": task["instruction"],
                        "app": task.get("app", "Unknown"),
                        "intent_category": task.get("intent_category", "Unknown"),
                        "groundtruth": task.get("groundtruth", {}),
                    })
                with open(rollout_filename, 'w', encoding='utf-8') as f:
                    for rollout in rollouts:
                        f.write(json.dumps(rollout, ensure_ascii=False) + "\n")

            tasks_to_process = []
            for rollout in rollouts:
                if "trajectories" not in rollout or len(rollout.get("trajectories", [])) == 0:
                    tasks_to_process.append(rollout)

            print(f"Total rollouts: {len(rollouts)}")
            print(f"To process: {len(tasks_to_process)}")
            print(f"Already done: {len(rollouts) - len(tasks_to_process)}\n")

            # 执行rollouts
            if tasks_to_process:
                print("=" * 70)
                print("Starting Rollout Process")
                print("=" * 70 + "\n")

                for idx, rollout in enumerate(tasks_to_process):
                    runid = rollout["runid"]
                    instruction = rollout["problem"]

                    rollout_log_dir = cur_step_dir
                    rollout_task_id = f"rollout_{runid}"

                    print(f"[{idx + 1}/{len(tasks_to_process)}] Rollout {runid}")
                    print(f"   Problem: {rollout['problem'][:60]}...")

                    # 显示是否使用 experiences
                    if experiences:
                        print(f"   Using {len(experiences)} experiences from previous steps")
                    else:
                        print(f"   No experiences available (step 0)")

                    try:
                        task_start_time = time.time()
                        run_single_task(
                            instruction=instruction,  # 原始 instruction，不包含 experiences
                            run_name="",
                            log_root=rollout_log_dir,
                            task_id=rollout_task_id,
                            perceptor=perceptor,
                            max_itr=max_itr,
                            max_consecutive_failures=max_consecutive_failures,
                            max_repetitive_actions=max_repetitive_actions,
                            enable_experience_retriever=enable_experience_retriever,
                            temperature=temperature,
                            overwrite_log_dir=True,
                            experiences=experiences,  # 传递 experiences 字典,
                        rag_query_engine = rag_query_engine if use_rag else None  # 🔥 新增

                        )

                        task_end_time = time.time()

                        # 从正确的路径读取
                        rollout_save_dir = os.path.join(rollout_log_dir, rollout_task_id)
                        steps_json_path = os.path.join(rollout_save_dir, "steps.json")
                        screenshots_dir = os.path.join(rollout_save_dir, "screenshots")

                        if os.path.exists(steps_json_path):
                            with open(steps_json_path, 'r', encoding='utf-8') as f:
                                steps_data = json.load(f)

                            # 计算reward
                            try:
                                reward = verify_func(
                                    instruction=rollout["problem"],
                                    screenshots_dir=screenshots_dir,
                                    model=REASONING_MODEL,
                                    backbone_type=BACKBONE_TYPE
                                )
                            except Exception as e:
                                print(f"   ⚠️ Verification failed: {e}")
                                reward = 0.0

                            # 更新rollout
                            rollouts[runid].update({
                                "steps": steps_data,
                                "log_path": steps_json_path,
                                "save_dir": rollout_save_dir,
                                "reward": reward,
                                "rollout_time": task_end_time - task_start_time,
                                "trajectories": [{"trajectory": []}],
                                "error": None,
                                "used_experiences": len(experiences) if experiences else 0
                            })

                            print(f"   ✅ Reward: {reward:.2f} | Time: {task_end_time - task_start_time:.1f}s\n")
                        else:
                            print(f"   ⚠️ steps.json not found\n")
                            rollouts[runid].update({
                                "reward": 0.0,
                                "error": "steps.json not found",
                                "trajectories": []
                            })

                        # 保存rollouts
                        with open(rollout_filename, 'w', encoding='utf-8') as f:
                            for r in rollouts:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    except Exception as e:
                        task_end_time = time.time()
                        print(f"   ❌ Failed: {e}\n")
                        import traceback
                        traceback.print_exc()

                        rollouts[runid].update({
                            "reward": 0.0,
                            "error": str(e),
                            "rollout_time": task_end_time - task_start_time,
                            "trajectories": []
                        })

                        with open(rollout_filename, 'w', encoding='utf-8') as f:
                            for r in rollouts:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                print("✅ All rollouts already completed\n")

            # 统计
            all_rewards = [r.get("reward", 0) for r in rollouts]
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
            success_count = sum(1 for r in all_rewards if r > 0.8)

            print("=" * 70)
            print("Rollout Statistics")
            print("=" * 70)
            print(f"   Total: {len(rollouts)}")
            print(f"   Success (>0.8): {success_count}/{len(rollouts)}")
            print(f"   Avg reward: {avg_reward:.3f}")
            if experiences:
                print(f"   Used experiences: {len(experiences)}")
            print("=" * 70 + "\n")

            stats[f"step_{step}"]["rollout"] = {
                "avg_reward": avg_reward,
                "success_rate": success_count / len(rollouts) if rollouts else 0,
                "num_experiences_used": len(experiences) if experiences else 0
            }

            # 生成新经验并保存到 step_{step+1}
            next_step_dir = os.path.join(experiment_dir, f"step_{step + 1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")

            if os.path.exists(next_experience_filename):
                print(f"📚 Experiences for step {step + 1} already exist, skipping\n")
            else:
                print("🔄 Generating new experiences...\n")
                try:
                    new_experiences = experience_updater.run(
                        rollouts=rollouts,
                        experiences=experiences,
                        save_dir=cur_step_dir,
                        given_ground_truth=True
                    )

                    with open(next_experience_filename, 'w', encoding='utf-8') as f:
                        json.dump(new_experiences, f, indent=2, ensure_ascii=False)

                    print(f"\n💾 Saved {len(new_experiences)} experiences to step {step + 1}")
                    print(f"   Previous step had {len(experiences)} experiences")
                    print(f"   New experiences added: {len(new_experiences) - len(experiences)}")

                except Exception as e:
                    print(f"\n❌ Experience update failed: {e}")
                    import traceback
                    traceback.print_exc()

            if build_rag and rag_builder and rag_index_builder:
                from inference_agent_E import update_rag_from_rollouts

                try:
                    print("\n📚 Updating RAG knowledge base from rollouts...")
                    update_rag_from_rollouts(
                        rollouts=rollouts,
                        rag_builder=rag_builder,
                        rag_index_builder=rag_index_builder,
                        success_threshold=0.8
                    )
                    print("✅ RAG knowledge base updated")
                except Exception as e:
                    print(f"⚠️ RAG update failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 标记完成
            stats[f"step_{step}"]["complete"] = True
            with open(stats_filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            print(f"\n✅ Step {step} completed!\n")

    # 训练完成标识
    print("\n" + "=" * 70)
    print("🎉 Training Completed!")
    print("=" * 70)
    print(f"Results saved to: {experiment_dir}")

    completion_file = os.path.join(experiment_dir, ".training_complete")
    with open(completion_file, 'w', encoding='utf-8') as f:
        import datetime
        f.write(json.dumps({
            "completed_at": datetime.datetime.now().isoformat(),
            "epochs": epochs,
            "total_steps": epoch * num_batches + batch_idx + 1,
            "final_stats": stats
        }, indent=2))

    print(f"✅ Training completion marker saved")
    print("=" * 70)


# ==================== 新增：GRPO评估模式 ====================

def run_grpo_evaluation(
        tasks_json,
        experience_file,
        run_name="grpo_eval",
        log_root=None,
        num_tasks=None,
        max_itr=40,
        max_consecutive_failures=5,
        max_repetitive_actions=5,
        enable_experience_retriever=False,
        temperature=0.0
):
    """GRPO评估模式 - 使用训练好的经验进行评估"""
    import json

    if log_root is None:
        log_root = f"logs/{REASONING_MODEL}/grpo_eval"

    print("=" * 70)
    print("🔍 Mobile Agent GRPO Evaluation")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"Tasks: {tasks_json}")
    print(f"Experience file: {experience_file}")
    if num_tasks:
        print(f"Evaluation data limit: {num_tasks} tasks")
    print("=" * 70 + "\n")

    # 创建评估目录
    eval_dir = os.path.join(log_root, run_name)
    os.makedirs(eval_dir, exist_ok=True)

    # 加载任务
    with open(tasks_json, "r", encoding="utf-8-sig") as f:
        task_json = json.load(f)
    tasks = task_json.get("tasks", task_json)

    if num_tasks and num_tasks < len(tasks):
        tasks = tasks[:num_tasks]
        print(f"⚠️ Limited to first {num_tasks} tasks\n")

    print(f"Loaded {len(tasks)} evaluation tasks")

    # 加载经验
    experiences = {}

    if experience_file and os.path.exists(experience_file):
        with open(experience_file, 'r', encoding='utf-8') as f:
            experiences = json.load(f)

    else:
        print("⚠️ No experiences loaded - running without experiences")

    print()

    # 初始化
    perceptor = Perceptor(ADB_PATH, perception_args=DEFAULT_PERCEPTION_ARGS)

    eval_tasks = []
    for task in tasks:
        eval_tasks.append({
            "instruction": task.get("instruction", task.get("problem", "")),
            "app": task.get("app", "Unknown"),
            "intent_category": task.get("intent_category", "Unknown"),
        })
    results = []
    for idx, task in enumerate(eval_tasks):
        task_id = f"task_{idx}"
        instruction = task["instruction"]

        print(f"[{idx + 1}/{len(eval_tasks)}] Task {idx}")
        print(f"   Problem: {instruction[:60]}...")
        run_single_task(
            instruction=instruction,  #
            experiences=experiences,  #
            enable_experience_retriever=enable_experience_retriever,  #
            run_name=run_name,
            log_root=log_root,
            task_id=task_id,
            perceptor=perceptor,
            max_itr=max_itr,
            max_consecutive_failures=max_consecutive_failures,
            max_repetitive_actions=max_repetitive_actions,
            temperature=temperature,
            overwrite_log_dir=True
        )
    # 保存结果
    results_file = os.path.join(eval_dir, "eval_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 统计
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get("success", False))



    # 创建评估完成标识
    completion_file = os.path.join(eval_dir, ".eval_complete")
    with open(completion_file, 'w', encoding='utf-8') as f:
        import datetime
        f.write(json.dumps({
            "completed_at": datetime.datetime.now().isoformat(),
            "experience_file": experience_file,
        }, indent=2))

    print(f"✅ Evaluation completion marker saved")
    print("=" * 70)


# ==================== 修改main函数 ====================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    # 原有参数
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--tasks_json", type=str, default=None)
    parser.add_argument("--specified_tips_path", type=str, default=None)
    parser.add_argument("--specified_shortcuts_path", type=str, default=None)
    parser.add_argument("--setting", type=str, default="individual", choices=["individual", "evolution"])
    parser.add_argument("--max_itr", type=int, default=40)
    parser.add_argument("--max_consecutive_failures", type=int, default=5)
    parser.add_argument("--max_repetitive_actions", type=int, default=5)
    parser.add_argument("--overwrite_task_log_dir", action="store_true", default=False)
    parser.add_argument("--enable_experience_retriever", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--screenrecord", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="inference",
                        choices=[ "grpo_train", "grpo_eval"],
                        help=" grpo_train(训练) | grpo_eval(评估)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="GRPO训练的epoch数")
    parser.add_argument("--batchsize", type=int, default=4,
                        help="GRPO训练的batch大小")
    parser.add_argument("--grpo_n", type=int, default=3,
                        help="GRPO每个问题采样次数")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="限制训练/评估数据数量")
    parser.add_argument("--experience_file", type=str, default=None,
                        help="评估时使用的经验文件路径")

    # 🔥 RAG 参数
    parser.add_argument(
        "--build_rag",
        action="store_true",
        default=False,
        help="Enable RAG knowledge base building from rollouts"
    )

    parser.add_argument(
        "--use_rag",
        action="store_true",
        default=False,
        help="Enable RAG retrieval during inference"
    )

    parser.add_argument(
        "--rag_data_dir",
        type=str,
        default="./data/rag/user6",
        help="RAG data directory"
    )

    parser.add_argument(
        "--force_rebuild_rag",
        action="store_true",
        default=False,
        help="Force rebuild RAG knowledge base from dataset"
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.log_root is None:
        if args.mode == "grpo_train":
            args.log_root = f"logs/{REASONING_MODEL}/grpo_train"
        elif args.mode == "grpo_eval":
            args.log_root = f"logs/{REASONING_MODEL}/grpo_eval"
        else:
            pass

    # ==================== 模式选择 ====================
    if args.mode == "grpo_train":
        # GRPO训练模式
        if args.tasks_json is None:
            raise ValueError("❌ GRPO training requires --tasks_json")
        run_grpo_training(
            tasks_json=args.tasks_json,
            run_name=args.run_name,
            log_root=args.log_root,
            epochs=args.epochs,
            batchsize=args.batchsize,
            grpo_n=args.grpo_n,
            num_tasks=args.num_tasks,
            max_itr=args.max_itr,
            max_consecutive_failures=args.max_consecutive_failures,
            max_repetitive_actions=args.max_repetitive_actions,
            enable_experience_retriever=args.enable_experience_retriever,
            temperature=args.temperature,
            build_rag=args.build_rag,              # 🔥 新增
            use_rag=args.use_rag,                  # 🔥 新增
            rag_data_dir=args.rag_data_dir,        # 🔥 新增
            force_rebuild_rag=args.force_rebuild_rag,  # 🔥 新增
        )

    elif args.mode == "grpo_eval":
        # GRPO评估模式
        if args.tasks_json is None:
            raise ValueError("❌ GRPO evaluation requires --tasks_json")
        if args.experience_file is None:
            print("Warning: No experience file specified, evaluating without experiences")

        run_grpo_evaluation(
            tasks_json=args.tasks_json,
            experience_file=args.experience_file,
            run_name=args.run_name,
            log_root=args.log_root,
            num_tasks=args.num_tasks,
            max_itr=args.max_itr,
            max_consecutive_failures=args.max_consecutive_failures,
            max_repetitive_actions=args.max_repetitive_actions,
            enable_experience_retriever=args.enable_experience_retriever,
            temperature=args.temperature
        )




if __name__ == "__main__":
    main()