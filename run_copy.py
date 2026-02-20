"""
Mobile Agent GRPO 训练与评估主程序 - RAG增强版

主要修复:
1. 修复了rag_query_engine参数传递的语法错误
2. 统一了use_rag/rag_enabled变量名
3. 完善了RAG学习逻辑
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from inference_agent_E import run_single_task
from inference_agent_E import Perceptor, DEFAULT_PERCEPTION_ARGS, ADB_PATH, INIT_TIPS, INIT_SHORTCUTS, REASONING_MODEL
import torch
import os
import json
import copy
import random
from experience.updater import MobileExperienceUpdater
from rag import (
    init_embeddings_from_config,
    init_rag_builder,  # 🔥 支持自定义目录
    init_index_builder,  # 🔥 支持自定义目录
    init_query_engine,  # 🔥 支持自定义目录
)
from verify.mobile_verify import verify_func
import time


# 替换为:
def prepare_formatted_batch_tasks(batch_tasks):
    formatted_batch_tasks = []
    for task in batch_tasks:
        apps_field = task.get("apps", [])
        if isinstance(apps_field, list) and len(apps_field) > 0:
            apps = []
            for item in apps_field:
                if isinstance(item, str):
                    apps.extend([a.strip() for a in item.split(",")])
            apps_str = ",".join(apps)
        elif isinstance(apps_field, str):
            apps_str = apps_field
        else:
            apps_str = "Unknown"

        formatted_batch_tasks.append({
            "instruction": task.get("instruction", task.get("problem", "")),
            "apps": apps_str,
            "app": apps_str,
            "type": task.get("type", "single_app"),
            "intent_category": task.get("intent_category", "Unknown"),
            "groundtruth": task.get("groundtruth", {}),
        })
    return formatted_batch_tasks

# App到Category的映射
APP_CATEGORY_MAP = {
    # 阅读类
    "微信读书": "阅读",
    "番茄小说": "阅读",
    # 邮箱类
    "qq邮箱": "邮箱",
    "网易邮箱": "邮箱",
    # 导航类
    "百度地图": "导航",
    "高德地图": "导航",
    "腾讯地图": "导航",
    # 购物类
    "京东": "购物",
    "唯品会": "购物",
    "拼多多": "购物",
    # 快递物流类
    "菜鸟": "快递物流",
    "圆通快递": "快递物流",
    # 帖子类
    "小红书": "帖子",
    "豆瓣": "帖子",
    # 长视频类
    "bilibili": "长视频",  # 兼容简写，也可补充"哔哩哔哩"
    "爱奇艺": "长视频",
    "腾讯视频": "长视频",  # 补充完整名称，避免歧义
    # 搜索类
    "知乎": "搜索",
    "QQ浏览器": "搜索",
    "百度": "搜索",
    # AI类
    "deepseek": "AI",
    "豆包": "AI",
    "腾讯元宝": "AI",
    # 音乐类
    "QQ音乐": "音乐",
    "网易云音乐": "音乐",
    # 生活类
    "美团": "生活",
    "肯德基": "生活",
    # 出行类
    "携程": "出行",  # 修正错别字"协程"为"携程"
    "去哪儿": "出行",
    # 系统类
    "相机": "系统",
    "闹钟": "系统",
    "日历": "系统",
    "天气": "系统"
}
def infer_category(app_name):
    """从App名称推断Category"""
    return APP_CATEGORY_MAP.get(app_name, "General")

def is_dir_empty(path):
    """检查目录是否为空"""
    if not os.path.exists(path):
        return True
    return len(os.listdir(path)) == 0

# ==================== Backbone配置 ====================
BACKBONE_TYPE = os.environ.get("BACKBONE_TYPE", default="OpenAI")
assert BACKBONE_TYPE in ["OpenAI", "Gemini", "Claude"], "Unknown BACKBONE_TYPE"
print("### Using BACKBONE_TYPE:", BACKBONE_TYPE)

OPENAI_API_URL = os.environ.get("OPENAI_API_URL", default="https://openrouter.ai/api/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default="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")

GEMINI_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", default="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")

CLAUDE_API_URL = "https://openrouter.ai/api/v1/chat/completions"
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", default="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")

if BACKBONE_TYPE == "OpenAI":
    REASONING_MODEL = os.environ.get("OPENAI_MODEL", default="openai/gpt-4o")
    KNOWLEDGE_REFLECTION_MODEL = os.environ.get("OPENAI_MODEL", default="openai/gpt-4o")
elif BACKBONE_TYPE == "Gemini":
    REASONING_MODEL = "gemini-1.5-pro-latest"
    KNOWLEDGE_REFLECTION_MODEL = "gemini-1.5-pro-latest"
elif BACKBONE_TYPE == "Claude":
    REASONING_MODEL = "claude-3-5-sonnet-20241022"
    KNOWLEDGE_REFLECTION_MODEL = "claude-3-5-sonnet-20241022"



# ==================== GRPO训练模式 ====================

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
        enable_experience_retriever=True,
        temperature=0.0,
        rag_learning=False,
        rag_enabled=False,
        rag_data_dir="./data/rag",
):
    """
    GRPO训练主函数

    Args:
        tasks_json: 任务JSON文件路径
        run_name: 实验名称
        log_root: 日志根目录
        epochs: 训练轮数
        batchsize: 批大小
        grpo_n: 每个问题采样次数
        num_tasks: 限制任务数量
        max_itr: 最大迭代次数
        max_consecutive_failures: 最大连续失败次数
        max_repetitive_actions: 最大重复动作次数
        enable_experience_retriever: 是否启用经验检索
        temperature: 采样温度
        rag_learning: 是否启用RAG学习
        rag_enabled: 是否启用RAG检索
        rag_data_dir: RAG数据目录
    """

    if log_root is None:
        log_root = f"logs/{REASONING_MODEL}/grpo_train"

    print("=" * 70)
    print("📚 Mobile Agent GRPO Training (RAG Enhanced)")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"Tasks: {tasks_json}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batchsize}")
    print(f"GRPO N: {grpo_n}")
    print(f"RAG Enabled: {rag_enabled}")
    print(f"RAG Learning: {rag_learning}")
    if num_tasks:
        print(f"Training data limit: {num_tasks} tasks")
    print("=" * 70 + "\n")

    experiment_dir = os.path.join(log_root, run_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 加载任务
    with open(tasks_json, "r", encoding="utf-8-sig") as f:
        task_json = json.load(f)
    tasks = task_json.get("tasks", task_json)
    if num_tasks and num_tasks < len(tasks):
        tasks = tasks[:num_tasks]
        print(f"⚠️ Limited to first {num_tasks} tasks\n")

    print(f"Loaded {len(tasks)} training tasks\n")
    assert len(tasks) % batchsize == 0, "Dataset size must be divisible by batch size"

    # ==================== 初始化RAG组件 ====================
    rag_builder = None
    rag_index_builder = None
    rag_query_engine = None

    if rag_learning or rag_enabled:
        print("\n" + "=" * 70)
        print("🔧 Initializing RAG Components")
        print("=" * 70)

        try:
            # 1. 初始化Embeddings
            embeddings = init_embeddings_from_config()
            print("✅ Embeddings initialized")

            # 2. 根据Backbone类型配置API
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

            # 3. 初始化RAG Builder (用于学习) - 🔥 使用自定义目录
            if rag_learning:
                rag_builder = init_rag_builder(
                    data_dir=rag_data_dir,  # 🔥 使用传入的目录
                    embeddings=embeddings,
                    api_config=api_config
                )
                print(f"RAG Builder initialized (data_dir: {rag_data_dir})")

                # 4. 初始化Index Builder - 🔥 使用自定义目录
                rag_index_builder = init_index_builder(
                    data_dir=rag_data_dir,  # 🔥 使用传入的目录
                    embeddings=embeddings
                )
                print(" RAG Index Builder initialized")

            # 5. 初始化Query Engine (用于检索) - 🔥 使用自定义目录
            if rag_enabled:
                rag_query_engine = init_query_engine(
                    data_dir=rag_data_dir,  # 🔥 使用传入的目录
                    embeddings=embeddings
                )
                print(f" RAG Query Engine initialized (data_dir: {rag_data_dir})")

        except Exception as e:
            print(f"Failed to initialize RAG: {e}")
            print(" Continuing without RAG...")
            import traceback
            traceback.print_exc()
            rag_enabled = False
            rag_learning = False

    # 构建初始RAG知识库 (如果启用learning且知识库不存在)
    if rag_learning and rag_builder is not None and rag_index_builder is not None:
        level1_dir = os.path.join(rag_data_dir, "level1")
        level2_dir = os.path.join(rag_data_dir, "level2")

        if is_dir_empty(level1_dir) or is_dir_empty(level2_dir):
            print("\n" + "=" * 70)
            print("🏗️ Building Initial RAG Knowledge Base")
            print("=" * 70)
            try:
                processed_tasks = []
                for task in tasks:
                    pt = task.copy()
                    if "instruction" not in pt and "problem" in pt:
                        pt["instruction"] = pt["problem"]
                    if "intent_category" not in pt:
                        pt["intent_category"] = infer_category(pt.get("apps", ""))
                    processed_tasks.append(pt)
                rag_builder.build_level1_from_dataset(tasks)
                rag_builder.build_level2_from_dataset(tasks)
                rag_index_builder.rebuild_all_indices()
                print("Initial RAG knowledge base built successfully")
            except Exception as e:
                print(f"Failed to build RAG knowledge base: {e}")
                rag_learning = False

    # 加载或初始化统计信息
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        with open(stats_filename, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        print(f"Loaded existing stats from {stats_filename}")
    else:
        stats = {}

    # 初始化Perceptor
    perceptor = Perceptor(ADB_PATH, perception_args=DEFAULT_PERCEPTION_ARGS)

    # 初始化Experience Updater
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

    # ==================== 训练循环 ====================
    for epoch in range(epochs):
        print("\n" + "=" * 70)
        print(f"📅 EPOCH {epoch + 1}/{epochs}")
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

        # 按batch处理
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
            print(f"📌 STEP {step} (Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches})")
            print("-" * 70)

            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            # 当前batch的任务
            batch_tasks = copy.deepcopy(
                shuffled_tasks[batch_idx * batchsize: (batch_idx + 1) * batchsize]
            )

            # 加载经验
            experiences = {}
            experience_filename = os.path.join(experiment_dir, f"step_{step}", "experiences.json")
            if os.path.exists(experience_filename):
                try:
                    with open(experience_filename, 'r', encoding='utf-8') as f:
                        experiences = json.load(f)
                    print(f"📚 Loaded {len(experiences)} experiences from step_{step}/experiences.json")
                except Exception as e:
                    print(f"⚠️ Failed to load experiences: {e}")
                    experiences = {}
            else:
                if step == 0:
                    print("🆕 Step 0: Starting without experiences")
                else:
                    print(f"⚠️ No experiences found at {experience_filename}")



            # 使用:
            formatted_batch_tasks = prepare_formatted_batch_tasks(batch_tasks)
            # GRPO采样
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
                        "apps": task.get("apps", "Unknown"),  # ✅ 完整 apps
                        "app": task.get("app", "Unknown"),
                        "type": task.get("type", "single_app"),  # ✅ 任务类型
                        "intent_category": task.get("intent_category", "Unknown"),
                        "groundtruth": task.get("groundtruth", {}),
                    })

                with open(rollout_filename, 'w', encoding='utf-8') as f:
                    for rollout in rollouts:
                        f.write(json.dumps(rollout, ensure_ascii=False) + "\n")

            # 找出需要处理的任务
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
                print("🚀 Starting Rollout Process")
                print("=" * 70 + "\n")

                for idx, rollout in enumerate(tasks_to_process):
                    runid = rollout["runid"]
                    instruction = rollout["problem"]

                    rollout_log_dir = cur_step_dir
                    rollout_task_id = f"rollout_{runid}"

                    print(f"[{idx + 1}/{len(tasks_to_process)}] Rollout {runid}")
                    print(f"   Problem: {rollout['problem'][:60]}...")

                    if experiences:
                        print(f"   Using {len(experiences)} experiences")

                    try:
                        task_start_time = time.time()

                        # 🔥 关键修复: rag_query_engine作为参数正确传递
                        run_single_task(
                            instruction=instruction,
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
                            experiences=experiences,
                            rag_query_engine=rag_query_engine if rag_enabled else None,  # 🔥 正确传递
                        )

                        task_end_time = time.time()

                        # 读取结果
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
            print("📊 Rollout Statistics")
            print("=" * 70)
            print(f"   Total: {len(rollouts)}")
            print(f"   Success (>0.8): {success_count}/{len(rollouts)}")
            print(f"   Avg reward: {avg_reward:.3f}")
            if experiences:
                print(f"   Used experiences: {len(experiences)}")
            print("=" * 70 + "\n")

            # ==================== RAG学习 ====================
            if rag_learning and rag_builder is not None and rag_index_builder is not None:
                print("\n" + "=" * 50)
                print("🧠 Learning from Successful Rollouts to RAG")
                print("=" * 50)
                try:
                    learned_count = 0
                    multi_app_count = 0

                    for rollout in rollouts:
                        if rollout.get("reward", 0) < 0.8:
                            continue

                        apps_field = rollout.get("apps", rollout.get("app", ""))
                        if not apps_field:
                            continue

                        instruction = rollout.get("problem", "")
                        trajectories = rollout.get("trajectories", [])
                        if not trajectories:
                            continue

                        category = rollout.get("intent_category", "Unknown")
                        task_type = rollout.get("type", "single_app")

                        steps = rollout.get("steps", [])

                        if task_type == "multi_app":
                            rag_builder.learn_from_multi_app_rollout(
                                apps=apps_field,
                                instruction=instruction,
                                trajectory=None,  # 不传这个
                                steps=steps,  # 🔥 传steps
                                success=True,
                                reward=rollout.get("reward", 0),
                                category=category
                            )
                            multi_app_count += 1
                        else:
                            rag_builder.learn_from_rollout(
                                app_name=apps_field,
                                instruction=instruction,
                                trajectory=None,  # 不传这个
                                steps=steps,  # 🔥 传steps
                                success=True,
                                reward=rollout.get("reward", 0),
                                category=category
                            )

                        learned_count += 1

                    if learned_count > 0:
                        print(f"✅ Learned from {learned_count} successful rollouts")
                        print(f"   - Multi-app: {multi_app_count}")
                        rag_index_builder.rebuild_all_indices()
                        print("Indices updated")
                except Exception as e:
                    print(f"Failed to learn from rollouts: {e}")


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

                except Exception as e:
                    print(f"\n❌ Experience update failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 标记完成
            stats[f"step_{step}"]["complete"] = True
            with open(stats_filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            print(f"\n✅ Step {step} completed!\n")

    # 训练完成
    print("\n" + "=" * 70)
    print("🎉 Training Completed!")
    print("=" * 70)
    print(f"Results saved to: {experiment_dir}")

    # 打印RAG统计信息
    if rag_learning and rag_builder:
        rag_builder.print_statistics()

    completion_file = os.path.join(experiment_dir, ".training_complete")
    with open(completion_file, 'w', encoding='utf-8') as f:
        import datetime
        f.write(json.dumps({
            "completed_at": datetime.datetime.now().isoformat(),
            "epochs": epochs,
            "rag_enabled": rag_enabled,
            "rag_learning": rag_learning,
        }, indent=2))

    print(f"✅ Training completion marker saved")
    print("=" * 70)


# ==================== GRPO评估模式 ====================

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
        temperature=0.0,
        rag_enabled=False,
        rag_data_dir="./data/rag",  # 🔥 新增目录参数
):
    """GRPO评估模式"""

    if log_root is None:
        log_root = f"logs/{REASONING_MODEL}/grpo_eval"

    print("=" * 70)
    print("🔍 Mobile Agent GRPO Evaluation")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"Tasks: {tasks_json}")
    print(f"Experience file: {experience_file}")
    print(f"RAG Enabled: {rag_enabled}")
    if rag_enabled:
        print(f"RAG Data Dir: {rag_data_dir}")
    if num_tasks:
        print(f"Evaluation data limit: {num_tasks} tasks")
    print("=" * 70 + "\n")

    # 创建评估目录
    eval_dir = os.path.join(log_root, run_name)
    os.makedirs(eval_dir, exist_ok=True)

    # 初始化RAG Query Engine (如果启用) - 🔥 使用自定义目录
    rag_query_engine = None
    if rag_enabled:
        print("\n" + "=" * 70)
        print("🔧 Initializing RAG Query Engine")
        print("=" * 70)
        try:
            embeddings = init_embeddings_from_config()
            rag_query_engine = init_query_engine(
                data_dir=rag_data_dir,  # 🔥 使用自定义目录
                embeddings=embeddings
            )
            rag_query_engine.print_index_info()
            print(f"✅ RAG Query Engine ready (data_dir: {rag_data_dir})")
        except Exception as e:
            print(f"❌ Failed to initialize RAG: {e}")
            print("⚠️ Continuing without RAG...")
            rag_enabled = False

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
        print(f"📚 Loaded {len(experiences)} experiences")
    else:
        print("No experiences loaded - running without experiences")

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
            instruction=instruction,
            experiences=experiences,
            enable_experience_retriever=enable_experience_retriever,
            run_name=run_name,
            log_root=log_root,
            task_id=task_id,
            perceptor=perceptor,
            max_itr=max_itr,
            max_consecutive_failures=max_consecutive_failures,
            max_repetitive_actions=max_repetitive_actions,
            temperature=temperature,
            overwrite_log_dir=True,
            rag_query_engine=rag_query_engine if rag_enabled else None,  # 🔥 传递RAG
        )

    # 保存结果
    results_file = os.path.join(eval_dir, "eval_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 Evaluation Completed!")
    print("=" * 70)
    print(f"Results saved to: {results_file}")
    print("=" * 70)


# ==================== 主函数 ====================

def main():
    import argparse

    parser = argparse.ArgumentParser()

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
                        choices=["grpo_train", "grpo_eval"],
                        help=" grpo_train(训练) | grpo_eval(评估)")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--grpo_n", type=int, default=3)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--experience_file", type=str, default=None)

    # 🔥 RAG参数
    parser.add_argument("--rag_enabled", action="store_true", default=False,
                        help="Enable RAG knowledge retrieval during execution")
    parser.add_argument("--rag_learning", action="store_true", default=False,
                        help="Enable learning from successful trajectories to RAG")
    parser.add_argument("--rag_data_dir", type=str, default="./data/rag",
                        help="RAG data directory")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.log_root is None:
        if args.mode == "grpo_train":
            args.log_root = f"logs/{REASONING_MODEL}/grpo_train"
        elif args.mode == "grpo_eval":
            args.log_root = f"logs/{REASONING_MODEL}/grpo_eval"

    # 模式选择
    if args.mode == "grpo_train":
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
            rag_learning=args.rag_learning,
            rag_enabled=args.rag_enabled,
            rag_data_dir=args.rag_data_dir,
        )

    elif args.mode == "grpo_eval":
        if args.tasks_json is None:
            raise ValueError("❌ GRPO evaluation requires --tasks_json")
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
            temperature=args.temperature,
            rag_enabled=args.rag_enabled,
            rag_data_dir=args.rag_data_dir,  # 🔥 添加目录参数
        )


if __name__ == "__main__":
    main()