import base64
import requests


#"sk-adrknricirdyvtwkdbjqsftyllokwwccckvktypmrjjfoxgq"
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def init_action_chat():
    """初始化对话历史"""
    operation_history = []
    system_prompt = "You are a helpful AI mobile phone operating assistant. You need to help me operate the phone to complete the user's instruction."
    operation_history.append({
        "role": "system",
        "content": system_prompt
    })
    return operation_history
def add_response(role, content, chat_history, screenshot_file=None):
    """添加消息到对话历史"""
    if screenshot_file and role == "user":
        # 如果content已经是list格式（多模态），直接使用
        if isinstance(content, list):
            chat_history.append({
                "role": role,
                "content": content
            })
        else:
            # 如果content是字符串，需要加上图像构建多模态格式
            base64_image = encode_image(screenshot_file)
            chat_history.append({
                "role": role,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            })
    else:
        # 纯文本消息
        chat_history.append({
            "role": role,
            "content": content
        })
    return chat_history
# def inference_chat(chat, model, api_url, token):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {token}"
#     }
#
#     data = {
#         "model": model,
#         "messages": [],
#         "max_tokens": 2048,
#         'temperature': 0.0,
#         "seed": 1234
#     }
#     for msg in chat:
#         if isinstance(msg, dict):
#             # 处理字典格式：{"role": ..., "content": ...}
#             data["messages"].append({"role": msg["role"], "content": msg["content"]})
#         else:
#             # 处理元组格式：(role, content)
#             role, content = msg
#             data["messages"].append({"role": role, "content": content})
#     # for msg in chat:
#     #     data["messages"].append({"role": msg["role"], "content": msg["content"]})
#
#     # for role, content in chat:
#     #     data["messages"].append({"role": role, "content": content})
#
#     while True:
#         try:
#             res = requests.post(api_url, headers=headers, json=data)
#             res_json = res.json()
#             res_content = res_json['choices'][0]['message']['content']
#         except:
#             print("Network Error:")
#             try:
#                 print(res.json())
#             except:
#                 print("Request Failed")
#         else:
#             break
#
#     return res_content


def inference_chat(chat, model, api_url, token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        "seed": 1234
    }

    for msg in chat:
        if isinstance(msg, dict):
            data["messages"].append(msg)
        elif isinstance(msg, tuple):
            # 元组格式：(role, content)
            role, content = msg

            # 检查content是否是多模态格式（list）
            if isinstance(content, list):
                # 多模态消息（包含图像和文本）
                data["messages"].append({"role": role, "content": content})
            else:
                # 纯文本消息
                data["messages"].append({"role": role, "content": content})

    while True:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            res_content = res_json['choices'][0]['message']['content']
        except Exception as e:
            print(f"Network Error: {e}")
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break

    return res_content


def image_to_base64(image_path):
    """将图像文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == "__main__":
    # 1. 准备图像
    screenshot_path = r"E:\mobile-agent\MobileAgent-main\Mobile-Agent-v2\MobileAgent-GRPO_3\data\mobile\train\my_mobile_experiment_9\step_0\rollout_0\step_0.jpg"
    screenshot_base64 = image_to_base64(screenshot_path)

    # 2. 构建prompt（多模态格式）
    prompt_text = "This is a phone screenshot. Please identify the search input box and tell me its coordinates."

    prompt_action = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{screenshot_base64}"
            }
        },
        {
            "type": "text",
            "text": prompt_text
        }
    ]

    # 3. 初始化对话
    chat_action = init_action_chat()

    # 4. 添加用户消息
    chat_action = add_response("user", prompt_action, chat_action, screenshot_path)

    # 5. 打印消息格式（用于调试）
    print("=== Messages Format ===")
    import json

    print(json.dumps(chat_action, indent=2, ensure_ascii=False))

    # 6. 调用API
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_tokens = "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"

    try:
        output_action = inference_chat(
            chat_action,
            model="google/gemini-pro-1.5",
            api_url=api_url,
            token=api_tokens
        )

        print("\n=== Response ===")
        print(output_action)
    except Exception as e:
        print(f"Error: {e}")


