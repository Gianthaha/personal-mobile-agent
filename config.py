import os
TEMP_DIR = "temp"
SCREENSHOT_DIR = "screenshot"
SLEEP_BETWEEN_STEPS = 5
ADB_PATH = os.environ.get("ADB_PATH", default="C:\\Users\\19927\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe")

## Reasoning model configs
BACKBONE_TYPE = os.environ.get("BACKBONE_TYPE", default="OpenAI")  # "OpenAI" or "Gemini" or "Claude"
assert BACKBONE_TYPE in ["OpenAI", "Gemini", "Claude"], "Unknown BACKBONE_TYPE"
print("### Using BACKBONE_TYPE:", BACKBONE_TYPE)

OPENAI_API_URL = os.environ.get("OPENAI_API_URL", default="https://openrouter.ai/api/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default="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")
REASONING_MODEL = os.environ.get("OPENAI_MODEL", default="openai/gpt-4o")
KNOWLEDGE_REFLECTION_MODEL = os.environ.get("OPENAI_MODEL", default="openai/gpt-4o")

## you can specify a jsonl file path for tracking API usage
USAGE_TRACKING_JSONL = None  # e.g., usage_tracking.jsonl

## Perceptor configs
CAPTION_CALL_METHOD = "api"
CAPTION_MODEL = "gpt-4o"

QWEN_API_URL = "https://xiaoai.plus/v1/chat/completions"
QWEN_API_KEY = "sk-gilp3D2WfPIVVJ24bMhuqz3a0YOMikKeZ0PdNxPwkcNGlVJo"
RAG_AVAILABLE = True
## Initial Tips provided by user; You can add additional custom tips ###
INIT_TIPS = """0. Do not add any payment information. If you are asked to sign in, ignore it or sign in as a guest if possible. Close any pop-up windows when opening an app.
1. By default, no APPs are opened in the background.
2. Screenshots may show partial text in text boxes from your previous input; this does not count as an error.
3. When creating new Notes, you do not need to enter a title unless the user specifically requests it.
4. If the target APP is not visible on the home screen, use the Swipe operation to scroll and find it.

"""


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
    "bilibili": "长视频",
    "爱奇艺": "长视频",
    "腾讯视频": "长视频",
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
    "携程": "出行",
    "去哪儿": "出行",
    # 系统类
    "相机": "系统",
    "闹钟": "系统",
    "日历": "系统",
    "天气": "系统"
}
