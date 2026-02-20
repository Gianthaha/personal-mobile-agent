import os
import json
import requests
import re
from pathlib import Path


def evaluate_mobile_agent(user_instruction, ground_truth, agent_inference, api_key, base_url, model):
    prompt = f"""You are a Mobile Agent evaluation expert. Evaluate the agent's inference.

    [User Instruction] {user_instruction}
    [Ground Truth] {ground_truth}
    [Agent Inference] {agent_inference}

    Scoring Rules (max 1.0):
    1. App Selection (0.5 points): If agent selects the correct app as in ground truth, score 0.5. Otherwise 0.
    2. Action Content (0.5 points): Evaluate whether the action/content matches ground truth:
       - If ground truth contains a LIST of items (songs, videos, products), agent only needs to match ANY ONE item to get full 0.5 points
       - Fully matches (or matches any one item from list): 0.5
       - Partially matches: 0.25
       - No match: 0

    Final Score = App Score + Content Score
    Output ONLY a decimal number between 0 and 1 (e.g., 0.85), nothing else.

    Score:"""



    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.post(base_url, headers=headers, json={
        "model": model, "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]
    })
    result = response.json()
    result_text = result["choices"][0]["message"]["content"].strip()
    match = re.search(r"[\d.]+", result_text)
    return min(max(float(match.group()), 0.0), 1.0) if match else 0.0


def load_eval_config(eval_config_path):
    """加载评估配置，返回 task_id -> task 映射"""
    with open(eval_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return {task["task_id"]: task for task in config["tasks"]}


def get_agent_inference(steps_json_path):
    """从steps.json提取agent推断（step=1, operation=planning）"""
    with open(steps_json_path, 'r', encoding='utf-8') as f:
        steps = json.load(f)
    for step in steps:
        if step.get("step") == 1 and step.get("operation") == "planning":
            return f"Plan: {step.get('plan', '')}\nThought: {step.get('thought', '')}"
    return ""


def build_ground_truth(task):
    """构建groundtruth"""
    apps = task.get("apps", "")
    gt = task.get("grountruth", "")
    return f"App: {apps}, {gt}" if gt else f"App: {apps}"


def batch_evaluate(eval_config_path, eval_user_dir, user_num, api_key, base_url, model, output_path=None):
    """
    批量评估
    Args:
        eval_config_path: 评估配置文件路径
        eval_user_dir: eval_user_X 文件夹路径 (如 logs/openai/gpt-4o/grpo_eval/eval_user_21_type2)
        user_num: 用户编号 (如 5, 21)，用于构建task_id
        api_key: OpenRouter API密钥
        base_url: OpenRouter API地址
        model: 模型名称
        output_path: 结果输出路径
    """
    task_config = load_eval_config(eval_config_path)
    results = []

    eval_user_path = Path(eval_user_dir)

    # 遍历 task_X 文件夹
    for task_dir in sorted(eval_user_path.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        steps_file = task_dir / "steps.json"
        if not steps_file.exists():
            print(f"Warning: {steps_file} not found")
            continue

        # task_4 -> task_0, task_1 -> task_1
        task_num = int(task_dir.name.replace("task_", "")) + 1
        task_id = f"user_{user_num}_{task_num}"

        if task_id not in task_config:
            print(f"Warning: {task_id} not in config")
            continue

        task = task_config[task_id]
        user_instruction = task.get("instruction", "")
        ground_truth = build_ground_truth(task)
        agent_inference = get_agent_inference(steps_file)

        print(user_instruction)
        print(ground_truth)
        print(agent_inference)

        if not agent_inference:
            print(f"Warning: No planning step in {steps_file}")
            continue

        try:
            score = evaluate_mobile_agent(user_instruction, ground_truth, agent_inference, api_key, base_url, model)
        except Exception as e:
            print(f"Error: {task_id} - {e}")
            score = 0.0

        results.append({
            "task_id": task_id,
            "instruction": user_instruction,
            "ground_truth": ground_truth,
            "agent_inference": agent_inference,
            "score": score
        })
        print(f"{task_id}: {score}")

    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"\nTotal: {len(results)}, Avg: {avg:.3f}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    API_KEY = "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"
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "openai/gpt-4o"


    EVAL_CONFIG = "user5_evaluate.json"
    EVAL_USER_DIR = r"../logs/openai/gpt-4o/grpo_eval/user5/MobileAgentE_user_5_type2"
    USER_NUM = 5 # task_id中的用户编号，如 task_0
    OUTPUT = "bert_score_gpt_eval_user_5_mobile_agent_e.json"


    batch_evaluate(EVAL_CONFIG, EVAL_USER_DIR, USER_NUM, API_KEY, BASE_URL, MODEL, OUTPUT)