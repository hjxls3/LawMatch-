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

def compute_macro_precision_recall(true_list, pred_list):
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for t, p in zip(true_list, pred_list):
        for label in p:
            if label in t:
                stats[label]["TP"] += 1
            else:
                stats[label]["FP"] += 1
        for label in t:
            if label not in p:
                stats[label]["FN"] += 1

    precision_list, recall_list = [], []

    for label, s in stats.items():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)

    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    return macro_precision, macro_recall

def main():
    print("\U0001f504 正在加载数据...")
    true_data = load_jsonl(test_path)
    pred_data = load_jsonl(result_path)

    if len(true_data) != len(pred_data):
        print(f"⚠️ 数据长度不一致：测试集 {len(true_data)} 条，生成结果 {len(pred_data)} 条")
        return

    # 解析实际值
    true_accusations = [set(normalize_accusation(a) for a in item["meta"].get("accusation", [])) for item in true_data]
    pred_accusations = [set(normalize_accusation(a) for a in item["meta"].get("accusation", [])) for item in pred_data]

    true_articles = [set(item["meta"].get("relevant_articles", [])) for item in true_data]
    pred_articles = [set(item["meta"].get("relevant_articles", [])) for item in pred_data]

    acc_p, acc_r = compute_macro_precision_recall(true_accusations, pred_accusations)
    art_p, art_r = compute_macro_precision_recall(true_articles, pred_articles)

    print("\n📊 Macro 平均指标：")
    print(f"【罪名 - Precision】: {acc_p:.4f}")
    print(f"【罪名 - Recall】: {acc_r:.4f}")
    print(f"【法条 - Precision】: {art_p:.4f}")
    print(f"【法条 - Recall】: {art_r:.4f}")

if __name__ == "__main__":
    main()
