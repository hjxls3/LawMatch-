import os
import json
import re
from collections import defaultdict

# 路径配置
base_dir = os.path.dirname(__file__)
test_path = os.path.join(base_dir, "..", "data", "test2.0.jsonl")
result_path = os.path.join(base_dir, "..", "outputs", "test_results2.0_lora_valid.jsonl")

def normalize_accusation(acc):
    acc = re.sub(r"[\[\]]", "", acc).strip()
    return acc[:-1] if acc.endswith("罪") else acc

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def compute_pr(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def main():
    print("\U0001f504 正在按罪名分类进行分析...")

    true_data = load_jsonl(test_path)
    pred_data = load_jsonl(result_path)

    label_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "count": 0})

    for true_item, pred_item in zip(true_data, pred_data):
        true_accs = set(normalize_accusation(a) for a in true_item["meta"].get("accusation", []))
        pred_accs = set(normalize_accusation(a) for a in pred_item["meta"].get("accusation", []))

        if not pred_accs:
            continue

        for acc in true_accs:
            label_stats[acc]["count"] += 1

        for label in pred_accs:
            if label in true_accs:
                label_stats[label]["TP"] += 1
            else:
                label_stats[label]["FP"] += 1

        for label in true_accs:
            if label not in pred_accs:
                label_stats[label]["FN"] += 1

    print("\n📊 按罪名分类结果：")
    sorted_labels = sorted(label_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:20]  # 前20频繁罪名
    for label, stat in sorted_labels:
        tp, fp, fn = stat["TP"], stat["FP"], stat["FN"]
        prec, rec = compute_pr(tp, fp, fn)
        print(f"{label} ({stat['count']} 例)： Precision = {prec:.4f}, Recall = {rec:.4f}")

if __name__ == "__main__":
    main()
