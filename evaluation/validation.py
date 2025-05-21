import os
import json
import re
from tqdm import tqdm

# 路径配置
base_dir = os.path.dirname(__file__)
test_path = os.path.join(base_dir, "..", "data", "test2.0.jsonl")
result_path = os.path.join(base_dir, "..", "outputs", "test_results3.0_base.jsonl")
output_path = os.path.join(base_dir, "..", "outputs", "test_results3.0_base_valid.jsonl")
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

def normalize_accusation(accusation):
    """
    标准化罪名：
    - 删除中括号，仅保留内容；
    - 去除 "罪" 字后缀。
    """
    # 删除中括号本身，仅保留其中内容
    accusation = re.sub(r"[\[\]]", "", accusation).strip()

    # 去除 "罪" 字后缀
    if accusation.endswith("罪"):
        accusation = accusation[:-1]

    return accusation



def validate_and_save(test_data, result_data, valid_accusations, valid_articles, output_path):
    """
    遍历生成结果，将非法样本置空，并将合法样本中的“罪”字去除后保存。
    """
    total_samples = len(test_data)
    structured_samples = 0
    valid_samples = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        for true_item, pred_item in zip(test_data, result_data):
            pred_meta = pred_item.get("meta", {})
            accusations = pred_meta.get("accusation", [])
            articles = pred_meta.get("relevant_articles", [])

            # 判断是否结构化输出
            is_structured = bool(accusations) and bool(articles)

            if is_structured:
                structured_samples += 1

                # 标准化罪名：去除“罪”字后缀
                normalized_accusations = [normalize_accusation(acc) for acc in accusations]

                # 检查合法性
                invalid_accusations = [acc for acc in normalized_accusations if acc not in valid_accusations]
                invalid_articles = [art for art in articles if art not in valid_articles]

                # 合法样本，保存标准化后的数据
                if not invalid_accusations and not invalid_articles:
                    valid_samples += 1
                    cleaned_data = {
                        "meta": {
                            "accusation": normalized_accusations,
                            "relevant_articles": articles
                        }
                    }
                    outfile.write(json.dumps(cleaned_data, ensure_ascii=False) + "\n")
                else:
                    # 置空
                    empty_data = {"meta": {"accusation": [], "relevant_articles": []}}
                    outfile.write(json.dumps(empty_data, ensure_ascii=False) + "\n")

            else:
                # 置空
                empty_data = {"meta": {"accusation": [], "relevant_articles": []}}
                outfile.write(json.dumps(empty_data, ensure_ascii=False) + "\n")

    # 输出统计信息
    structured_rate = structured_samples / total_samples if total_samples > 0 else 0
    valid_rate = valid_samples / structured_samples if structured_samples > 0 else 0

    print("\n📊 评估结果：")
    print(f"结构化率：{structured_rate:.4f}")
    print(f"合法率（合法样本率，基于结构化样本）：{valid_rate:.4f}")
    print(f"✅ 合法样本已保存至 {output_path}")

def main():
    # 加载数据
    print("🔄 正在加载数据...")
    test_data = load_jsonl(test_path)
    result_data = load_jsonl(result_path)
    valid_accusations = load_txt(accu_path)
    valid_articles = set(map(int, load_txt(law_path)))  # 转为整数

    # 执行清理并保存
    validate_and_save(test_data, result_data, valid_accusations, valid_articles, output_path)

if __name__ == "__main__":
    main()
