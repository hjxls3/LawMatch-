import os
import json
from tqdm import tqdm
from generate_lora import generate_lora  # 从之前的生成模块中导入 generate_lora 函数
from generate_base import generate_base  # 从基础模型模块中导入 generate_base 函数

# 输入输出路径配置
input_path = "../data/test2.0.jsonl"
output_path_lora = "../outputs/test_results2.0_lora.jsonl"
output_path_base = "../outputs/test_results3.0_base.jsonl"
os.makedirs(os.path.dirname(output_path_lora), exist_ok=True)

def process_data(input_path, output_path, model_type="lora"):
    """
    逐条读取测试数据，送入模型进行推理，并保存生成结果（仅包含 meta 字段）。
    - model_type: "lora" 或 "base"
    """
    # 选择模型
    generate_fn = generate_lora if model_type == "lora" else generate_base
    print(f"\n🔍 当前模型：{'LoRA 微调模型' if model_type == 'lora' else '基础模型'}")

    skipped_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc=f"Processing {model_type.upper()} Model"):
            try:
                # 解析输入 JSON 数据
                data = json.loads(line.strip())
                fact = data.get("fact", "")

                # 跳过无效数据
                if not fact:
                    skipped_count += 1
                    continue

                # 构建输入 prompt
                prompt = (
                    "请根据以下案情判断罪名与适用法条，并按照以下格式输出：\n"
                    "罪名：XXX罪\n"
                    "法条：《中华人民共和国刑法》第XXX条\n"
                    "请不要重复案情内容，直接开始回答。\n\n"
                    f"案情描述如下：{fact}"
                )

                # 调用生成函数
                _, accusations, articles = generate_fn(prompt)

                # 构建输出数据结构（仅保存 meta）
                output_data = {
                    "meta": {
                        "accusation": accusations,
                        "relevant_articles": articles
                    }
                }

                # 写入结果
                outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"⚠️ 处理样本时出错：{e}")
                skipped_count += 1

    print(f"\n✅ 生成完成，结果已保存至 {output_path}")
    print(f"📦 跳过样本数：{skipped_count}")

if __name__ == "__main__":
    # 生成 LoRA 模型结果
    # process_data(input_path, output_path_lora, model_type="lora")

    # 生成基础模型结果
    process_data(input_path, output_path_base, model_type="base")
