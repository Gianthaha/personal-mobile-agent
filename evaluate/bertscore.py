import os
import json
from pathlib import Path
from bert_score import score as bert_score

def evaluate_mobile_agent(ground_truth, agent_inference):
    """直接用BERTScore语义匹配，返回0-1分数"""
    _, _, f1 = bert_score([agent_inference], [ground_truth], lang="zh", verbose=False)
    return round(f1.item(), 2)


def load_eval_config(eval_config_path):
    with open(eval_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return {task["task_id"]: task for task in config["tasks"]}


def get_agent_inference(steps_json_path):
    with open(steps_json_path, 'r', encoding='utf-8') as f:
        steps = json.load(f)
    for step in steps:
        if step.get("step") == 1 and step.get("operation") == "planning":
            return f"Plan: {step.get('plan', '')}\nThought: {step.get('thought', '')}"
    return ""


def build_ground_truth(task):
    apps = task.get("apps", "")
    gt = task.get("grountruth", "")
    return f"App: {apps}, {gt}" if gt else f"App: {apps}"


def batch_evaluate(eval_config_path, eval_user_dir, user_num, output_path=None):
    task_config = load_eval_config(eval_config_path)
    results = []
    eval_user_path = Path(eval_user_dir)

    for task_dir in sorted(eval_user_path.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        steps_file = task_dir / "steps.json"
        if not steps_file.exists():
            continue

        task_num = int(task_dir.name.replace("task_", "")) + 1
        task_id = f"user_{user_num}_{task_num}"

        if task_id not in task_config:
            continue

        task = task_config[task_id]
        ground_truth = build_ground_truth(task)
        agent_inference = get_agent_inference(steps_file)

        if not agent_inference:
            continue

        score = evaluate_mobile_agent(ground_truth, agent_inference)

        results.append({
            "task_id": task_id,
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
    EVAL_CONFIG = "user21_evaluate.json"
    EVAL_USER_DIR = r"../logs/openai/gpt-4o/grpo_eval/user21/MobileAgentE_user21_type_2"
    USER_NUM = 21 # task_id中的用户编号，如 task_0
    OUTPUT = "bert_score_gpt_eval_user_5_mobile_agent_e.json"
    batch_evaluate(EVAL_CONFIG, EVAL_USER_DIR, USER_NUM, OUTPUT)