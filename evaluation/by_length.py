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
    print("\U0001f504 正在分析案情长度...")

    true_data = load_jsonl(test_path)
    pred_data = load_jsonl(result_path)

    buckets = {
        "<200": [],
        "200-500": [],
        ">500": []
    }

    for true_item, pred_item in zip(true_data, pred_data):
        fact = true_item.get("fact", "")
        length = len(fact)

        if length < 200:
            group = "<200"
        elif length <= 500:
            group = "200-500"
        else:
            group = ">500"

        # 解析标签
        true_acc = set(normalize_accusation(a) for a in true_item["meta"].get("accusation", []))
        pred_acc = set(normalize_accusation(a) for a in pred_item["meta"].get("accusation", []))

        # 空输出不计入
        if not pred_acc:
            continue

        buckets[group].append((true_acc, pred_acc))

    print("\n📊 案情长度分析结果：")
    for bucket, pairs in buckets.items():
        if not pairs:
            print(f"{bucket} 组: 无有效样本")
            continue
        true_list, pred_list = zip(*pairs)
        p, r = compute_pr(true_list, pred_list)
        print(f"{bucket} 字符： Precision = {p:.4f}, Recall = {r:.4f}  (样本数: {len(pairs)})")

if __name__ == "__main__":
    main()
