import os
import json
from tqdm import tqdm

# 输入输出路径
input_path = os.path.join(os.path.dirname(__file__), "..", "data", "final_all_data", "exercise_contest", "data_test.json")
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "train2.0.jsonl")

multi_accu_count = 0
multi_article_count = 0

# 模板配置（改得更明确 + 更结构化）
def format_prompt(fact_text):
    return (
        f"请根据以下案情判断罪名与适用法条，并按照以下格式输出：\n"
        f"罪名：XXX罪\n"
        f"法条：《中华人民共和国刑法》第XXX条\n"
        f"请不要重复案情内容，直接开始回答。\n\n"
        f"案情描述如下：{fact_text}"
    )

def format_response(accusations, articles):
    accusation_str = "，".join(accusations)
    articles_str = "，".join([f"《中华人民共和国刑法》第{a}条" for a in articles])
    return f"罪名：{accusation_str}\n法条：{articles_str}\n输出结束"


# 加载原始数据
raw_data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line.strip()))

# 构建新数据结构
new_data = []
for item in tqdm(raw_data):
    fact = item.get("fact", "")
    meta = item.get("meta", {})
    accusations = meta.get("accusation", [])
    articles = meta.get("relevant_articles", [])

    if not fact or not accusations or not articles:
        continue

    # 多标签统计
    if len(accusations) > 1:
        multi_accu_count += 1
    if len(articles) > 1:
        multi_article_count += 1

    prompt = format_prompt(fact)
    response = format_response(accusations, articles)

    new_data.append({"prompt": prompt, "response": response})


# 写入新文件
with open(output_path, 'w', encoding='utf-8') as f:
    for sample in new_data:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ 共生成训练样本 {len(new_data)} 条，已保存至 {output_path}")
print(f"📊 多罪名样本数量：{multi_accu_count}")
print(f"📊 多法条样本数量：{multi_article_count}")

