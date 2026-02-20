import base64
import requests
from time import sleep
import json


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def track_usage(res_json, api_key):
    """
    {'id': 'chatcmpl-AbJIS3o0HMEW9CWtRjU43bu2Ccrdu', 'object': 'chat.completion', 'created': 1733455676, 'model': 'gpt-4o-2024-11-20', 'choices': [...], 'usage': {'prompt_tokens': 2731, 'completion_tokens': 235, 'total_tokens': 2966, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'system_fingerprint': 'fp_28935134ad'}
    """
    model = res_json['model']
    usage = res_json['usage']
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        prompt_tokens, completion_tokens = usage['prompt_tokens'], usage['completion_tokens']
    elif "promptTokens" in usage and "completionTokens" in usage:
        prompt_tokens, completion_tokens = usage['promptTokens'], usage['completionTokens']
    elif "input_tokens" in usage and "output_tokens" in usage:
        prompt_tokens, completion_tokens = usage['input_tokens'], usage['output_tokens']
    else:
        prompt_tokens, completion_tokens = None, None

    prompt_token_price = None
    completion_token_price = None
    if prompt_tokens is not None and completion_tokens is not None:
        if "gpt-4o" in model:
            prompt_token_price = (2.5 / 1000000) * prompt_tokens
            completion_token_price = (10 / 1000000) * completion_tokens
        elif "gemini" in model:
            prompt_token_price = (1.25 / 1000000) * prompt_tokens
            completion_token_price = (5 / 1000000) * completion_tokens
        elif "claude" in model:
            prompt_token_price = (3 / 1000000) * prompt_tokens
            completion_token_price = (15 / 1000000) * completion_tokens
    return {
        # "api_key": api_key, # remove for better safety
        "id": res_json['id'] if "id" in res_json else None,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_token_price": prompt_token_price,
        "completion_token_price": completion_token_price
    }


def is_content_filter_error(res_json):
    """检查是否是内容过滤错误"""
    if 'error' in res_json:
        error = res_json['error']
        error_code = error.get('code', '')
        error_message = error.get('message', '')
        # Azure OpenAI 内容过滤
        if error_code == 'content_filter' or 'content management policy' in error_message.lower():
            return True
        # OpenAI 内容过滤
        if 'content policy' in error_message.lower():
            return True
    return False


def inference_chat(chat, model, api_url, token, usage_tracking_jsonl=None, max_tokens=4096, temperature=0.0):
    if token is None:
        raise ValueError("API key is required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": max_tokens,
        'temperature': temperature
    }


    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    max_retry = 5
    sleep_sec = 5
    res = None
    res_json = None
    content_filter_triggered = False

    while max_retry >= 0:
        try:
            res = requests.post(api_url, headers=headers, json=data, timeout=120)
            res_json = res.json()

            # 检查是否是内容过滤错误
            if is_content_filter_error(res_json):
                content_filter_triggered = True
                print("Content Filter Error Detected!")
                print(res_json)
                # 对于内容过滤错误，重试没有意义，直接返回默认响应
                print("Content filter triggered. Returning default response...")
                return _get_default_response_for_content_filter()

            # 检查是否有其他错误
            if 'error' in res_json:
                raise Exception(f"API Error: {res_json['error']}")

            res_content = res_json['choices'][0]['message']['content']

            if usage_tracking_jsonl:
                usage = track_usage(res_json, api_key=token)
                with open(usage_tracking_jsonl, "a") as f:
                    f.write(json.dumps(usage) + "\n")

            # 成功获取响应
            return res_content

        except Exception as e:
            print(f"Network Error: {e}")
            try:
                if res is not None:
                    error_json = res.json()
                    print(error_json)
                    # 再次检查内容过滤
                    if is_content_filter_error(error_json):
                        print("Content filter triggered. Returning default response...")
                        return _get_default_response_for_content_filter()
            except:
                print("Request Failed - Could not parse response")

        print(f"Sleep {sleep_sec} before retry...")
        sleep(sleep_sec)
        max_retry -= 1

    # 所有重试都失败了
    print(f"Failed after all retries...")
    if content_filter_triggered:
        return _get_default_response_for_content_filter()
    return None


def _get_default_response_for_content_filter():
    """
    当内容过滤被触发时，返回一个安全的默认响应。
    这样可以避免下游代码因为 None 而崩溃。
    """
    return """### Thought ###
The API request was blocked by content filter. I need to skip this step and try a different approach.

### Plan ###
1. Skip the current operation
2. Try an alternative approach

### Current Subgoal ###
Retry with modified approach due to content filter."""


def test_claude_with_image(image_path: str, prompt: str = "请描述这张图片"):
    """
    测试 Claude API 的图像理解能力

    Args:
        image_path: 本地图片路径
        prompt: 提问内容
    """
    # 配置

    api_url =  "https://openrouter.ai/api/v1/chat/completions"
    token = "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"  # 替换为你的 API key
    model = "anthropic/claude-3.5-sonnet"

    # 编码图片
    image_base64 = encode_image(image_path)

    # 构建消息
    chat = [
        ("user", [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": prompt}
        ])
    ]

    # 调用 API
    result = inference_chat(
        chat=chat,
        model=model,
        api_url=api_url,
        token=token,
        max_tokens=1024,
        temperature=0.0
    )

    print("=" * 50)
    print(f"图片: {image_path}")
    print(f"提问: {prompt}")
    print("=" * 50)
    print(f"回复:\n{result}")
    print("=" * 50)

    return result


if __name__ == '__main__':
    # 测试示例
    test_claude_with_image(
        image_path=r"E:\paper code\mobile_agent_e_grpo\screenshot\screenshot.jpg",  # 替换为你的图片路径
        prompt="请描述这张图片的内容"
    )

