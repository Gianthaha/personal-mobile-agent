"""
完整的RAG系统测试脚本
测试流程:
1. 从train.json构建一级和二级知识库
2. 从rollout.jsonl学习经验
3. 构建向量索引
4. 测试查询功能
"""

import json
from pathlib import Path

# 假设你的项目结构
from rag_builder import RAGBuilder
from rag_index_builder import RAGIndexBuilder
from rag.rag_query import RAGQueryEngine


class RAGSystemTester:
    """RAG系统测试器"""

    def __init__(self, data_dir="./test_data"):
        """初始化测试器"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # API配置
        self.api_config = {
            "api_url": "https://xiaoai.plus/v1/chat/completions",
            "token": "sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo",  # 替换为实际的API key
            "model": "gpt-4o"
        }

        # 初始化embeddings
        print("🔧 Initializing embeddings...")
        self.embeddings = self._init_embeddings()

        # 初始化组件
        self.rag_builder = RAGBuilder(
            data_dir=str(self.data_dir),
            embeddings=self.embeddings,
            api_config=self.api_config
        )

        self.index_builder = RAGIndexBuilder(
            data_dir=str(self.data_dir),
            index_dir=str(self.data_dir / "indices"),
            embeddings=self.embeddings
        )

        self.query_engine = RAGQueryEngine(
            data_dir=str(self.data_dir),
            index_dir=str(self.data_dir / "indices"),
            embeddings=self.embeddings
        )

    def _init_embeddings(self):
        """初始化embeddings"""
        try:
            from api_embeddings import UnifiedEmbeddings, EmbeddingsAdapter

            # 使用你的embedding配置
            unified_emb = UnifiedEmbeddings(
                provider="siliconflow",
                api_key="sk-adrknricirdyvtwkdbjqsftyllokwwccckvktypmrjjfoxgq",
                api_base="https://api.siliconflow.cn/v1",
                model="Qwen/Qwen3-Embedding-8B"
            )

            return EmbeddingsAdapter(unified_emb)
        except Exception as e:
            print(f" Embedding initialization failed: {e}")
            print(" Will use fallback methods")
            return None

    # ==================== 测试步骤 ====================

    def step1_load_training_data(self):
        """步骤1: 加载训练数据"""
        print("\n" + "=" * 70)
        print("📖 Step 1: Loading Training Data")
        print("=" * 70)

        # 模拟train.json数据
        train_data = [
            {
                "scenario": "School",
                "app": "QQ Music",
                "intent_category": "Music",
                "instruction": "Open QQ Music and play 'Anhe Bridge'",
                "step": 9
            },
            {
                "scenario": "School",
                "app": "QQ Music",
                "intent_category": "Music",
                "instruction": "Open QQ Music and play 'New Boy'",
                "step": 20
            },
            {
                "scenario": "Residence",
                "app": "BBK Clock",
                "intent_category": "Alarm",
                "instruction": "Set an alarm for tomorrow at 8:20 AM",
                "step": 2
            },
            {
                "scenario": "School",
                "app": "Meituan",
                "intent_category": "Live",
                "instruction": "Open Meituan and view nearby food hot pot",
                "step": 10
            },
            {
                "scenario": "School",
                "app": "Bilibili",
                "intent_category": "Video",
                "instruction": "Open Bilibili and search for 'Peppa pig' first video",
                "step": 14
            },
            {
                "scenario": "School",
                "app": "Bilibili",
                "intent_category": "Video",
                "instruction": "Open Bilibili and play 'The Chess Player'",
                "step": 12
            },
            {
                "scenario": "School",
                "app": "Gaode Maps",
                "intent_category": "Navigation",
                "instruction": "Open Gaode Maps and bike navigation to Wudaokou Metro Station",
                "step": 21
            },
            {
                "scenario": "School",
                "app": "Gaode Maps",
                "intent_category": "Navigation",
                "instruction": "Open Gaode Maps and navigate to Peking University East Gate",
                "step": 14
            },
            {
                "scenario": "School",
                "app": "QQ Browser",
                "intent_category": "browser",
                "instruction": "Open QQ Browser and search for \"school\"",
                "step": 5
            },
            {
                "scenario": "School",
                "app": "QQ Browser",
                "intent_category": "browser",
                "instruction": "Open QQ Browser and search for \"what is an LLM\"",
                "step": 5
            }
        ]

        print(f"✅ Loaded {len(train_data)} training samples")
        return train_data

    def step2_build_knowledge_base(self, train_data):
        """步骤2: 构建知识库"""
        print("\n" + "=" * 70)
        print("🏗️ Step 2: Building Knowledge Base")
        print("=" * 70)

        # 构建一级库
        self.rag_builder.build_level1_from_dataset(train_data)

        # 构建二级库
        self.rag_builder.build_level2_from_dataset(train_data)

        print("✅ Knowledge base built successfully")

    def step3_learn_from_rollouts(self):
        """步骤3: 从rollouts学习"""
        print("\n" + "=" * 70)
        print("🎓 Step 3: Learning from Rollouts")
        print("=" * 70)

        # 模拟rollout数据（简化的trajectory）
        rollout_data = {
            "app": "QQ Music",
            "problem": "Open QQ Music and play 'Anhe Bridge'",
            "intent_category": "Music",
            "reward": 1.0,
            "trajectories": [
                {
                    "trajectory": [
                        {
                            "step": 0,
                            "thought": "Need to open QQ Music app",
                            "action": "Open app (QQ音乐)"
                        },
                        {
                            "step": 1,
                            "thought": "Need to search for the song",
                            "action": "Tap (333, 348)"
                        },
                        {
                            "step": 2,
                            "thought": "Search history shows Anhe Bridge",
                            "action": "Tap (751, 667)"
                        },
                        {
                            "step": 3,
                            "thought": "Play the song",
                            "action": "Tap (178, 604)"
                        },
                        {
                            "step": 4,
                            "thought": "Song is playing",
                            "action": "Stop"
                        }
                    ]
                }
            ]
        }

        # 学习这个rollout
        self.rag_builder.learn_from_rollout(
            app_name=rollout_data["app"],
            instruction=rollout_data["problem"],
            trajectory=rollout_data["trajectories"][0]["trajectory"],
            success=True,
            reward=rollout_data["reward"]
        )

        # 再学习几个rollout
        additional_rollouts = [
            {
                "app": "Bilibili",
                "instruction": "Open Bilibili and play 'The Chess Player'",
                "trajectory": [
                    {"step": 0, "action": "Open app (Bilibili)", "thought": "Open app"},
                    {"step": 1, "action": "Tap (500, 300)", "thought": "Tap search"},
                    {"step": 2, "action": "Type \"The Chess Player\"", "thought": "Enter query"},
                    {"step": 3, "action": "Tap (600, 400)", "thought": "Select video"},
                    {"step": 4, "action": "Stop", "thought": "Done"}
                ],
                "reward": 0.95
            },
            {
                "app": "Gaode Maps",
                "instruction": "Open Gaode Maps and navigate to Peking University East Gate",
                "trajectory": [
                    {"step": 0, "action": "Open app (Gaode Maps)", "thought": "Open app"},
                    {"step": 1, "action": "Tap (400, 200)", "thought": "Tap search bar"},
                    {"step": 2, "action": "Type \"Peking University East Gate\"", "thought": "Enter destination"},
                    {"step": 3, "action": "Tap (500, 500)", "thought": "Start navigation"},
                    {"step": 4, "action": "Stop", "thought": "Done"}
                ],
                "reward": 0.9
            }
        ]

        for rollout in additional_rollouts:
            self.rag_builder.learn_from_rollout(
                app_name=rollout["app"],
                instruction=rollout["instruction"],
                trajectory=rollout["trajectory"],
                success=True,
                reward=rollout["reward"]
            )

        print(f"✅ Learned from {1 + len(additional_rollouts)} rollouts")

    def step4_build_indices(self):
        """步骤4: 构建向量索引"""
        print("\n" + "=" * 70)
        print("🔍 Step 4: Building Vector Indices")
        print("=" * 70)

        if not self.embeddings:
            print(" Skipping index building (no embeddings)")
            return

        # 构建一级索引
        self.index_builder.build_level1_index()

        # 构建二级索引
        self.index_builder.build_level2_index()

        print("✅ Vector indices built successfully")

    def step5_test_queries(self):
        """步骤5: 测试查询功能"""
        print("\n" + "=" * 70)
        print("🔎 Step 5: Testing RAG Queries")
        print("=" * 70)

        if not self.embeddings:
            print(" Skipping query tests (no embeddings)")
            return

        # 测试查询1: 音乐类
        print("\n--- Test Query 1: Music ---")
        query1 = "I want to listen to Anhe Bridge"
        self._test_single_query(query1)

        # 测试查询2: 视频类
        print("\n--- Test Query 2: Video ---")
        query2 = "Play a video on Bilibili"
        self._test_single_query(query2)

        # 测试查询3: 导航类
        print("\n--- Test Query 3: Navigation ---")
        query3 = "Navigate to a university"
        self._test_single_query(query3)

    def _test_single_query(self, instruction: str):
        """测试单个查询"""
        print(f"\n📝 Query: {instruction}")

        try:
            # 执行两级检索
            selected_app, rag_knowledge = self.query_engine.two_level_retrieve(
                instruction=instruction,
                experiences=None,  # 不使用experience
                top_k_workflows=3
            )

            if selected_app:
                print(f"✅ Selected App: {selected_app}")
                print(f"\n📚 RAG Knowledge Preview (first 500 chars):")
                print(rag_knowledge[:500] + "..." if len(rag_knowledge) > 500 else rag_knowledge)
            else:
                print("❌ No app selected")

        except Exception as e:
            print(f"❌ Query failed: {e}")

    def step6_print_statistics(self):
        """步骤6: 打印统计信息"""
        print("\n" + "=" * 70)
        print("Step 6: Statistics")
        print("=" * 70)

        # 打印知识库统计
        self.rag_builder.print_statistics()

        # 打印索引信息
        if self.embeddings:
            self.query_engine.print_index_info()

    def step7_inspect_knowledge_base(self):
        """步骤7: 检查知识库内容"""
        print("\n" + "=" * 70)
        print("🔍 Step 7: Inspecting Knowledge Base")
        print("=" * 70)

        # 检查一级库
        print("\n--- Level 1 (Categories) ---")
        level1_files = list((self.data_dir / "level1").glob("*.json"))
        for file in level1_files[:3]:  # 只显示前3个
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"\n📁 {file.name}")
            print(f"   Category: {data.get('category')}")
            print(f"   Apps: {', '.join(data.get('apps', []))}")
            print(f"   Description: {data.get('description', '')[:100]}...")

        # 检查二级库
        print("\n--- Level 2 (Apps) ---")
        app_dirs = [d for d in (self.data_dir / "level2").iterdir() if d.is_dir()]
        for app_dir in app_dirs[:2]:  # 只显示前2个
            print(f"\n📱 {app_dir.name}")

            # workflows
            workflows_file = app_dir / "workflows.json"
            if workflows_file.exists():
                with open(workflows_file, "r", encoding="utf-8") as f:
                    workflows = json.load(f)
                print(f"   Workflows: {len(workflows)}")
                if workflows:
                    wf = workflows[0]
                    print(f"   - Task: {wf.get('task', '')[:50]}...")
                    print(f"   - Success count: {wf.get('success_count', 0)}")
                    if wf.get('ui_elements'):
                        print(f"   - UI elements: {len(wf['ui_elements'])}")

            # preferences
            pref_file = app_dir / "preferences.json"
            if pref_file.exists():
                with open(pref_file, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                print(f"   Preferences:")
                print(f"   - Total usage: {prefs.get('total_usage', 0)}")
                if prefs.get('task_preferences'):
                    print(f"   - Task preferences: {list(prefs['task_preferences'].keys())[:3]}")
                if prefs.get('content_preferences'):
                    print(f"   - Content preferences: {list(prefs['content_preferences'].keys())[:3]}")

    # ==================== 主测试流程 ====================

    def run_full_test(self):
        """运行完整测试"""
        print("\n" + "=" * 70)
        print("🚀 Starting RAG System Full Test")
        print("=" * 70)

        try:
            # 步骤1: 加载数据
            train_data = self.step1_load_training_data()

            # 步骤2: 构建知识库
            self.step2_build_knowledge_base(train_data)

            # 步骤3: 从rollouts学习
            self.step3_learn_from_rollouts()

            # 步骤4: 构建索引
            self.step4_build_indices()

            # 步骤5: 测试查询
            self.step5_test_queries()

            # 步骤6: 统计信息
            self.step6_print_statistics()

            # 步骤7: 检查内容
            self.step7_inspect_knowledge_base()

            print("\n" + "=" * 70)
            print("✅ Full Test Completed Successfully!")
            print("=" * 70)

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()


# ==================== 简化测试（无API） ====================

def run_simple_test():
    """运行简化测试（不需要API）"""
    print("\n" + "=" * 70)
    print("🧪 Running Simple Test (No API Required)")
    print("=" * 70)

    # 创建测试器（不使用API）
    tester = RAGSystemTester(data_dir="./test_data_simple")
    tester.api_config = None  # 禁用API

    # 只测试基础功能
    train_data = tester.step1_load_training_data()
    tester.step2_build_knowledge_base(train_data)

    # 简单的学习测试（不使用LLM增强）
    print("\n🎓 Testing basic learning (without LLM)...")
    tester.rag_builder.learn_from_rollout(
        app_name="QQ Music",
        instruction="Play Anhe Bridge",
        trajectory=[
            {"step": 0, "action": "Open app", "thought": "open"},
            {"step": 1, "action": "Tap (100, 200)", "thought": "search"},
            {"step": 2, "action": "Type \"Anhe Bridge\"", "thought": "input"},
            {"step": 3, "action": "Stop", "thought": "done"}
        ],
        success=True,
        reward=1.0
    )

    tester.step6_print_statistics()
    tester.step7_inspect_knowledge_base()

    print("\n✅ Simple test completed!")


# ==================== 主入口 ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # 简化测试模式
        run_simple_test()
    else:
        # 完整测试模式
        print("\n Make sure you have:")
        print("   1. Set EMBEDDING_API_KEY environment variable")
        print("   2. Updated api_config with your API key")
        print("   3. All required packages installed")
        print("\nRun with --simple for basic test without APIs\n")

        tester = RAGSystemTester(data_dir="./test_data_full")
        tester.run_full_test()