import argparse


class Config:
    """配置类"""

    # ADB配置
    ADB_PATH = "C:\\Users\\19927\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe"

    # API配置
    API_URL = "https://xiaoai.plus/v1/chat/completions"
    API_TOKEN = "sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo"
    MODEL = "gpt-4o"

    # Qwen配置
    QWEN_API = "sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo"  #视觉模型密钥
    CAPTION_MODEL = "gpt-4o"
    CAPTION_CALL_METHOD = "api"


    #  Embedding API配置
    EMBEDDING_API_KEY = "sk-adrknricirdyvtwkdbjqsftyllokwwccckvktypmrjjfoxgq"
    EMBEDDING_API_BASE = "https://api.siliconflow.cn/v1/"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
    PROVIDER = "siliconflow"

    # 任务配置
    MAX_STEPS = 15
    TASK_TIMEOUT = 600
    STEP_DELAY = 5

    # GRPO配置
    EPOCHS = 3
    BATCH_SIZE = 5
    GRPO_N = 5

    # 目录配置
    TEMP_DIR = "temp"
    SCREENSHOT_DIR = "screenshot"
    DATA_DIR = "data/mobile"
    #  RAG配置
    RAG_DATA_DIR = "./data/"  # RAG知识库根目录
    RAG_INDEX_DIR = "./data/indices"  # RAG索引目录
    RAG_TOP_K_WORKFLOWS = 3  # 检索workflow数量
    RAG_MIN_SIMILARITY = 0.5  # 最低相似度阈值

    # 应用包名映射表
    APP_PACKAGE_MAP = {
        # 音乐类
        "QQ Music": "com.tencent.qqmusic",
        "NetEase Cloud Music": "com.netease.cloudmusic",
        "Kugou Music": "com.kugou.android",
        # 地图导航
        "Gaode Maps": "com.autonavi.minimap",
        "Baidu Map": "com.baidu.BaiduMap",
        "Gaode Map": "com.autonavi.minimap",
        # 购物
        "Taobao": "com.taobao.taobao",
        "JD": "com.jingdong.app.mall",
        "Pinduoduo": "com.xunmeng.pinduoduo",
        # 社交
        "WeChat": "com.tencent.mm",
        "QQ": "com.tencent.mobileqq",
        "Weibo": "com.sina.weibo",
        # 支付
        "Alipay": "com.eg.android.AlipayGphone",
        # 外卖
        "Meituan": "com.sankuai.meituan",
        "Eleme": "me.ele",
        # 视频
        "Douyin": "com.ss.android.ugc.aweme",
        "Bilibili": "tv.danmaku.bili",
        "Tencent Video": "com.tencent.qqlive",
        # 浏览器
        "Chrome": "com.android.chrome",
        "QQ Browser": "com.tencent.mtt",
        # 其他
        "Settings": "com.android.settings",
        "Gallery": "com.android.gallery3d",
    }


def get_train_args():
    """训练参数"""
    parser = argparse.ArgumentParser(description="Mobile Agent GRPO Training")

    # 实验配置
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AndroidWorld-Basic")
    parser.add_argument("--dataset_truncate", type=int, default=None)

    # GRPO配置
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--grpo_n", type=int, default=5)
    parser.add_argument("--given_ground_truth", type=str, default="True")

    # Agent配置
    parser.add_argument("--adb_path", type=str, default=Config.ADB_PATH)
    parser.add_argument("--api_url", type=str, default=Config.API_URL)
    parser.add_argument("--api_token", type=str, default=Config.API_TOKEN)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--task_timeout", type=int, default=600)

    # 开关
    parser.add_argument("--reflection_switch", action="store_true", default=False)
    parser.add_argument("--memory_switch", action="store_true", default=False)

    #  RAG开关
    parser.add_argument("--rag_enabled", action="store_true", default=False,
                        help="Enable RAG knowledge retrieval during execution")
    parser.add_argument("--rag_learning", action="store_true", default=False,
                        help="Enable learning from successful trajectories to RAG")

    return parser.parse_args()


def get_eval_args():
    """评估参数"""
    parser = argparse.ArgumentParser(description="Mobile Agent GRPO Evaluation")

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset_truncate", type=int, default=None)
    parser.add_argument("--experience_file", type=str, default=None)
    parser.add_argument("--pass_k", type=int, default=3)

    # Agent配置
    parser.add_argument("--adb_path", type=str, default=Config.ADB_PATH)
    parser.add_argument("--api_url", type=str, default=Config.API_URL)
    parser.add_argument("--api_token", type=str, default=Config.API_TOKEN)

    #  RAG配置
    parser.add_argument("--rag_enabled", action="store_true", default=False,
                        help="Enable RAG knowledge retrieval")

    return parser.parse_args()