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

def compute_pr(true_list, pred_list):
    TP = FP = FN = 0
    for t, p in zip(true_list, pred_list):
        TP += len(t & p)
        FP += len(p - t)
        FN += len(t - p)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall

def main():
    print("\U0001f504 正在分析置标数量...")

    true_data = load_jsonl(test_path)
    pred_data = load_jsonl(result_path)

    buckets = {
        "1-label": [],
        "2-label": [],
        "3+-label": []
    }

    for true_item, pred_item in zip(true_data, pred_data):
        true_acc = set(normalize_accusation(a) for a in true_item["meta"].get("accusation", []))
        pred_acc = set(normalize_accusation(a) for a in pred_item["meta"].get("accusation", []))

        # 空输出不计入
        if not pred_acc:
            continue

        label_count = len(true_acc)
        if label_count == 1:
            group = "1-label"
        elif label_count == 2:
            group = "2-label"
        else:
            group = "3+-label"

        buckets[group].append((true_acc, pred_acc))

    print("\n📊 标签数量分析结果：")
    for bucket, pairs in buckets.items():
        if not pairs:
            print(f"{bucket} 组: 无有效样本")
            continue
        true_list, pred_list = zip(*pairs)
        p, r = compute_pr(true_list, pred_list)
        print(f"{bucket} ： Precision = {p:.4f}, Recall = {r:.4f}  (样本数: {len(pairs)})")

if __name__ == "__main__":
    main()
