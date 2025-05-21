import os
import json
from tqdm import tqdm

# 路径配置
base_dir = os.path.dirname(__file__)
test_path = os.path.join(base_dir, "..", "data", "test2.0.jsonl")
result_path = os.path.join(base_dir, "..", "outputs", "test_results2.0_lora_valid.jsonl")
base_result_path = os.path.join(base_dir, "..", "outputs", "test_results3.0_base_valid.jsonl")
accu_path = os.path.join(base_dir, "..", "meta", "accu.txt")
law_path = os.path.join(base_dir, "..", "meta", "law.txt")

def load_txt(file_path):
    """ 加载 txt 文件，返回集合 """
    with open(file_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())

def load_jsonl(file_path):
    """ 加载 jsonl 文件，返回列表 """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"⚠️ JSON 解析失败：{line.strip()}")
    return data

def analyze_coverage(data, valid_accusations, valid_articles):
    """
    统计覆盖率：罪名和法条的覆盖情况。
    """
    accusation_count = {}
    article_count = {}

    # 初始化计数器
    for acc in valid_accusations:
        accusation_count[acc] = 0
    for art in valid_articles:
        article_count[art] = 0

    for item in data:
        meta = item.get("meta", {})
        accusations = meta.get("accusation", [])
        articles = meta.get("relevant_articles", [])

        # 更新罪名计数
        for acc in accusations:
            if acc in accusation_count:
                accusation_count[acc] += 1

        # 更新法条计数
        for art in articles:
            if art in article_count:
                article_count[art] += 1

    return accusation_count, article_count

def print_coverage_report(acc_count, art_count):
    """
    打印覆盖率报告
    """
    print("\n📊 罪名覆盖率：")
    uncovered_accusations = [acc for acc, count in acc_count.items() if count == 0]
    for acc, count in acc_count.items():
        print(f"{acc}: {count}")

    print("\n未覆盖罪名：", uncovered_accusations)

    print("\n📊 法条覆盖率：")
    uncovered_articles = [art for art, count in art_count.items() if count == 0]
    for art, count in art_count.items():
        print(f"{art}: {count}")

    print("\n未覆盖法条：", uncovered_articles)

def main():
    # 加载数据
    valid_accusations = load_txt(accu_path)
    valid_articles = set(map(int, load_txt(law_path)))

    # 加载测试集数据
    test_data = load_jsonl(test_path)
    gen_data = load_jsonl(result_path)
    base_data = load_jsonl(base_result_path)

    print("\n📊 数据集统计：")
    print(f"测试集样本数：{len(test_data)}")
    print(f"微调模型样本数：{len(gen_data)}")
    print(f"基础模型样本数：{len(base_data)}")

    # 覆盖率统计
    print("\n📊 微调模型覆盖率：")
    acc_count, art_count = analyze_coverage(gen_data, valid_accusations, valid_articles)
    print_coverage_report(acc_count, art_count)

    print("\n📊 基础模型覆盖率：")
    base_acc_count, base_art_count = analyze_coverage(base_data, valid_accusations, valid_articles)
    print_coverage_report(base_acc_count, base_art_count)

if __name__ == "__main__":
    main()
