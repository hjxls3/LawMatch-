import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

# 路径配置
base_dir = os.path.dirname(__file__)
test_path = os.path.join(base_dir, "..", "data", "test2.0.jsonl")
result_path = os.path.join(base_dir, "..", "outputs", "test_results3.0_base_valid.jsonl")


def normalize_accusation(acc):
    """ 去掉中括号和‘罪’字 """
    acc = re.sub(r"[\[\]]", "", acc).strip()
    return acc[:-1] if acc.endswith("罪") else acc

def load_jsonl(file_path):
    """ 加载 jsonl 文件，返回列表 """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"⚠️ 无法解析行：{line.strip()}")
    return data

def calculate_metrics(true_data, pred_data):
    metrics = {
        "accusation": {"TP": 0, "FP": 0, "FN": 0},
        "relevant_articles": {"TP": 0, "FP": 0, "FN": 0}
    }

    skipped = 0  # 跳过空预测样本计数

    for true_item, pred_item in zip(true_data, pred_data):
        true_meta = true_item.get("meta", {})
        pred_meta = pred_item.get("meta", {})

        true_accusations = set(normalize_accusation(a) for a in true_meta.get("accusation", []))
        true_articles = set(true_meta.get("relevant_articles", []))

        pred_accusations = set(normalize_accusation(a) for a in pred_meta.get("accusation", []))
        pred_articles = set(pred_meta.get("relevant_articles", []))

        # ✅ 跳过空预测样本（即结构化失败的）
        if not pred_accusations and not pred_articles:
            skipped += 1
            continue

        # --- accusation ---
        tp_acc = len(true_accusations & pred_accusations)
        fp_acc = len(pred_accusations - true_accusations)
        fn_acc = len(true_accusations - pred_accusations)

        # --- articles ---
        tp_art = len(true_articles & pred_articles)
        fp_art = len(pred_articles - true_articles)
        fn_art = len(true_articles - pred_articles)

        metrics["accusation"]["TP"] += tp_acc
        metrics["accusation"]["FP"] += fp_acc
        metrics["accusation"]["FN"] += fn_acc

        metrics["relevant_articles"]["TP"] += tp_art
        metrics["relevant_articles"]["FP"] += fp_art
        metrics["relevant_articles"]["FN"] += fn_art

    return metrics, skipped


def compute_precision_recall(tp, fp, fn):
    """ 计算 Precision 和 Recall """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def main():
    # 加载数据
    print("🔄 正在加载数据...")
    true_data = load_jsonl(test_path)
    pred_data = load_jsonl(result_path)

    # 数据长度校验
    if len(true_data) != len(pred_data):
        print(f"⚠️ 数据长度不一致：测试集 {len(true_data)} 条，生成结果 {len(pred_data)} 条")
        return

    # 计算指标
    metrics, skipped = calculate_metrics(true_data, pred_data)

    # 计算 Precision 和 Recall
    acc_tp, acc_fp, acc_fn = metrics["accusation"].values()
    art_tp, art_fp, art_fn = metrics["relevant_articles"].values()

    acc_precision, acc_recall = compute_precision_recall(acc_tp, acc_fp, acc_fn)
    art_precision, art_recall = compute_precision_recall(art_tp, art_fp, art_fn)

    print(f"\n📊 评估结果（已跳过空预测样本 {skipped} 条）：")
    print(f"【罪名 - Precision】: {acc_precision:.4f}")
    print(f"【罪名 - Recall】: {acc_recall:.4f}")
    print(f"【法条 - Precision】: {art_precision:.4f}")
    print(f"【法条 - Recall】: {art_recall:.4f}")

if __name__ == "__main__":
    main()
