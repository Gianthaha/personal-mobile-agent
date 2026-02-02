import os
import json
import base64
from PIL import Image
import requests



class MobileVerifier:
    """Mobile任务验证器 - 基于最后ADB动作 + 图像语义相似度"""

    def __init__(self, api_url, api_token, model="gpt-4o"):
        self.api_url = api_url
        self.api_token = api_token
        self.model = model

    def verify(self, instruction, screenshots_dir):
        """
        验证任务完成情况 - 基于指令和执行过程截图

        参数:
            instruction: 任务指令文本
            screenshots_dir: 截图目录路径

        验证逻辑:
        1. 从指定目录扫描所有截图文件
        2. 使用多模态模型评估截图序列是否完成了指令任务
        3. 返回0.0-1.0的评分

        返回: 0.0 - 1.0 的分数
        """

        # 检查目录是否存在
        if not os.path.exists(screenshots_dir):
            print(f"  Screenshots directory not found: {screenshots_dir}")
            return 0.0

        if not os.path.isdir(screenshots_dir):
            print(f"  Path is not a directory: {screenshots_dir}")
            return 0.0

        # 扫描目录中的所有截图文件
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
        screenshots = []

        for filename in os.listdir(screenshots_dir):
            file_path = os.path.join(screenshots_dir, filename)
            # 检查是否是文件且后缀是图片格式
            if os.path.isfile(file_path):
                name, ext = os.path.splitext(filename)
                ext = ext.lower()
                if name.isdigit():
                    screenshots.append(file_path)

        # 按文件名排序（确保时间顺序）
        screenshots.sort()

        # 如果没有截图，返回0
        if not screenshots:
            print(f"  No screenshots found in directory: {screenshots_dir}")
            return 0.0

        print(f"  Found {len(screenshots)} screenshots in directory")

        # 如果截图太多，采样关键帧（首帧 + 中间若干帧 + 末帧）
        max_screenshots = 10
        if len(screenshots) > max_screenshots:
            print(f"  Too many screenshots ({len(screenshots)}), sampling {max_screenshots} key frames...")
            sampled = [screenshots[0]]  # 首帧
            step = (len(screenshots) - 2) / (max_screenshots - 2)
            for i in range(1, max_screenshots - 1):
                idx = int(1 + step * (i - 1))
                sampled.append(screenshots[idx])
            sampled.append(screenshots[-1])  # 末帧
            screenshots = sampled

        print(f"  Evaluating task completion with {len(screenshots)} screenshots...")

        # 使用多模态模型评估任务完成情况
        score = self._evaluate_task_completion(instruction, screenshots)

        print(f"  Task completion score: {score:.3f}")

        return score
    def _evaluate_task_completion(self, instruction, screenshots):
        """
        使用多模态模型评估任务完成情况

        参数:
            instruction: 任务指令
            screenshots: 执行过程中的截图路径列表

        返回: 0.0 - 1.0 的完成度分数
        """
        try:
            # 编码所有截图为 base64
            screenshot_contents = []
            for screenshot_path in screenshots:
                with open(screenshot_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    screenshot_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })

            # 构建 prompt
            prompt = (
                "You are an expert at evaluating mobile task completion based on screenshot sequences.\n\n"
                f"TASK INSTRUCTION:\n{instruction}\n\n"
                "You will see a sequence of screenshots showing the execution process of this task.\n"
                "Please evaluate whether the task was successfully completed based on these screenshots.\n\n"
                "EVALUATION CRITERIA:\n"
                "1. Does the final state match the task goal?\n"
                "2. Were the necessary steps taken to complete the task?\n"
                "3. Is the task outcome visible in the screenshots?\n"
                "4. Were there any errors or failed attempts?\n\n"
                "SCORING GUIDE:\n"
                "1.0 - Perfect completion: Task fully accomplished, all goals met\n"
                "0.8-0.9 - Good completion: Task mostly done, minor issues\n"
                "0.5-0.7 - Partial completion: Some progress made, key steps missing\n"
                "0.3-0.4 - Minimal progress: Started but far from complete\n"
                "0.0-0.2 - No completion: No meaningful progress toward goal\n\n"
                "IMPORTANT:\n"
                "- Focus on whether the TASK GOAL was achieved, not just whether actions were performed\n"
                "- Consider the entire sequence, not just the final screenshot\n"
                "- Be objective and consistent in your evaluation\n\n"
    "OUTPUT FORMAT (STRICT):\n"
    "- Output ONLY a single numeric value between 0.0 and 1.0\n"
    "- Do NOT output explanations, text, symbols, labels, or formatting\n"
    "- Do NOT repeat the task or criteria\n"
    "- The output must be a plain number and nothing else\n\n"
    "Valid outputs: 1.0, 0.85, 0.6, 0.3, 0.0\n"
    "Invalid outputs: 'Score: 0.8', '0.8 (good)', 'The task is mostly done'\n"
            )

            # 构建消息内容
            content = [{"type": "text", "text": prompt}] + screenshot_contents

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
            }

            response = requests.post(self.api_url, headers=headers, json=data)
            result = response.json()

            score_str = result["choices"][0]["message"]["content"]
            score = float(score_str)

            return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"    ========================================================Warning: Task completion evaluation failed: {e}")
            # 如果失败，返回低分
            return 0.0




def verify_func(instruction,screenshots_dir ):
    """全局验证函数（用于 GRPO）"""
    verifier = MobileVerifier(
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_token="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",
        model="anthropic/claude-3.5-sonnet"
    )
    return verifier.verify(instruction, screenshots_dir)


