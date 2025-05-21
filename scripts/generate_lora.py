import torch
from post_process_output import post_process_output
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径配置
base_model_path = "../models/DeepSeek-R1-Distill-Qwen-1.5B"
lora_adapter_path = "../models/lora_adapter2.0"

# 全局变量
tokenizer = None
model = None

def load_model():
    """
    模型加载，仅在首次调用时执行。
    """
    global tokenizer, model

    if tokenizer is None or model is None:
        print("\n🔄 正在首次加载模型...")
        # 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        # 加载基础模型并合并 LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model = model.merge_and_unload()
        model.eval()
    else:
        print("\n✅ 模型已加载，直接使用...")


def generate_lora(prompt):
    """
    根据输入的 prompt，生成罪名与法条信息。
    返回格式化后的输出。
    """
    # 检查模型是否已加载
    load_model()

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # 解码输出
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(output_text)

    # 解析罪名与法条信息
    accusations, articles = post_process_output(output_text, prompt)

    # 格式化输出
    accusation_str = "，".join(accusations) if accusations else "无"
    articles_str = "，".join([f"《中华人民共和国刑法》第{article}条" for article in articles]) if articles else "无"

    formatted_output = f"罪名：{accusation_str}\n法条：{articles_str}"

    # 返回格式化输出和解析后的列表，方便后续处理
    return formatted_output, accusations, articles


if __name__ == "__main__":
    # 示例测试
    example_prompt = (
        "请根据以下案情判断罪名与适用法条，并按照以下格式输出：\n"
        "罪名：XXX罪\n"
        "法条：《中华人民共和国刑法》第XXX条\n"
        "请不要重复案情内容，直接开始回答。\n\n"
        "案情描述如下：张三持刀抢劫便利店，致人重伤。"
    )

    result, accusations, articles = generate_lora(example_prompt)

    print("\n生成结果：")
    print(result)
    print("\n罪名列表：", accusations)
    print("法条列表：", articles)
