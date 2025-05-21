import os
import json
import random

# 路径配置
input_path = os.path.join(os.path.dirname(__file__), "..", "data", "final_all_data", "exercise_contest", "data_train.json")
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "test2.0.jsonl")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def prepare_testset(input_path, output_path, sample_size=3000, method="random"):
    """
    从输入数据集中抽取样本。
    - method: "random" 随机抽样，"interval" 间隔抽样。
    """
    data = []

    # 读取整个数据集到内存（适用于较小规模数据集）
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                sample = json.loads(line.strip())
                fact = sample.get("fact", "")
                meta = sample.get("meta", {})
                accusations = meta.get("accusation", [])
                articles = meta.get("relevant_articles", [])

                # 跳过无效样本
                if not fact or not accusations or not articles:
                    continue

                # 将有效样本存入列表
                data.append({
                    "fact": fact,
                    "meta": {
                        "accusation": accusations,
                        "relevant_articles": articles
                    }
                })

            except json.JSONDecodeError:
                print(f"⚠️ 无法解析行：{line.strip()}")
                continue

    print(f"📦 数据集总样本数：{len(data)}")

    # 抽样
    if method == "random":
        # 随机抽样
        sampled_data = random.sample(data, min(sample_size, len(data)))
        print(f"✅ 随机抽样完成，共抽取 {len(sampled_data)} 条样本。")

    elif method == "interval":
        # 每隔 n 个样本抽取 1 个
        interval = max(1, len(data) // sample_size)
        sampled_data = data[::interval][:sample_size]
        print(f"✅ 间隔抽样完成，每隔 {interval} 个样本抽取 1 个，共 {len(sampled_data)} 条样本。")

    else:
        print(f"❌ 无效抽样方法：{method}")
        return

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in sampled_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 测试集已保存至 {output_path}，共 {len(sampled_data)} 条样本。")

if __name__ == "__main__":
    # 示例运行：随机抽样
    prepare_testset(input_path, output_path, sample_size=3000, method="random")

    # 示例运行：间隔抽样
    # prepare_testset(input_path, output_path, sample_size=10000, method="interval")
