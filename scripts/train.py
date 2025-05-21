import os
os.environ["USE_TF"] = "0"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_path = os.path.join(BASE_DIR, "models", "DeepSeek-R1-Distill-Qwen-1.5B")
data_path = os.path.join(BASE_DIR, "data", "train2.0.jsonl")
output_dir = os.path.join(BASE_DIR, "models", "lora_adapter2.0")


# 加载 tokenizer 和 model（基础模型）
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 配置 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 具体模块根据模型结构可能调整
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 数据集预处理
def tokenize_fn(example):
    prompt = example["prompt"]
    response = example["response"]
    text = prompt + "\n" + response
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")

dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.map(tokenize_fn, remove_columns=["prompt", "response"])

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

# 数据动态填充器
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer 初始化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collator
)

# 开始训练
trainer.train()

# 保存微调后的 LoRA adapter
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ LoRA 微调完成，权重保存至 {output_dir}")
