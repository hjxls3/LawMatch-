# LawMatch: 基于大模型微调的案情解析与法条匹配

LawMatch 是一个基于 DeepSeek-R1-Distill-Qwen-1.5B 模型的法条预测系统，支持通过案情文本判断对应的罪名与适用法律条文。该项目使用 LoRA 技术对基础模型进行微调，并提供多维度评估指标。

---

## 🚀 项目特点
- 使用 HuggingFace Transformer 框架构建
- 支持 LoRA 微调、合法化清洗与结构化校验
- 提供宏/微精度、召回率、F1 以及多维错误分析
- 支持罪名、法条、案情长度、标签数量等维度的性能对比

---

## 📦 安装与环境配置

### 1. 克隆项目并切换目录
```bash
git clone https://github.com/hjxls3/LawMatch.git
cd LawMatch
```

### 2. 安装 Conda 环境
```bash
conda env create -f lawmatch_env.yml
conda activate lawmatch
```

---

## 📂 项目结构
```
LawMatch/
├── data/                         # 原始与处理后的测试数据集
├── models/                       # 微调模型与 LoRA adapter 存储目录
├── outputs/                      # 推理输出与合法化结果
├── scripts/                      # 推理与训练脚本（如 generate_lora.py）
├── evaluation/                   # 精度、召回率、分析评估脚本
├── meta/                         # 包含所有合法罪名与法条定义
├── lawmatch_env.yml              # Conda 环境定义
├── README.md                     # 使用说明
```

---

## 📁 数据与模型获取说明

由于文件体积较大，项目中的部分目录内容无法直接上传至 GitHub，请根据以下说明，自行下载所需数据与模型：

### 🔹 `data` 目录说明
`data` 目录包含用于推理与评估的原始测试数据与处理结果。主要文件包括：

- `final_all_data`：包含所有原始数据
- `train2.0.jsonl`：用于微调的训练集（每行为一条案情与标注）
- `test2.0.jsonl`：用于推理与评估的测试集

> ✅ 下载地址：请前往GitHub平台搜索CAIL项目自行下载或联系项目维护者获取数据包。

### 🔹 `models` 目录说明
`models` 目录用于存放基础模型和微调生成的 LoRA adapter。主要内容包括：

- `DeepSeek-R1-Distill-Qwen-1.5B`：DeepSeek-R1-Distill-Qwen-1.5B 原始模型（需从 HuggingFace 下载）
- `lora_adapter2.0`：基于 `train2.0` 微调后得到的 LoRA adapter 权重

> ✅ 获取方式：
1. 基础模型下载：[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai)（下载后放入 `models/DeepSeek-R1-Distill-Qwen-1.5B/`）
2. LoRA adapter 可通过运行训练脚本生成，或联系作者获取已训练模型文件。

---


## 🔧 使用方法

### 1. 微调模型（可选）
```bash
python scripts/train.py
```
将使用 `train2.0.jsonl` 进行微调，生成 `models/lora_adapter2.0/` 目录。

### 2. 进行推理生成（LoRA）
```bash
python scripts/generate_lora.py
```
可根据案情 prompt 输出预测罪名与法条。

### 3. 批量测试并保存结果
```bash
python scripts/test.py  # 示例名
```
将生成 `outputs/test_results2.0.jsonl`

### 4. 合法性与结构化验证
```bash
python evaluation/validation.py
```
输出合法率与结构化率，并生成合法结果文件。

### 5. 指标评估
```bash
python evaluation/calculate_micro.py   # Micro-Precision / Recall
python evaluation/calculate_macro.py   # Macro-Precision / Recall
```

### 6. 错误分析与可视化
```bash
python evaluation/by_length.py         # 按案情长度
python evaluation/by_labelcount.py     # 按标签数量
python evaluation/by_charge.py         # 按罪名类别
```

---

## ✅ 示例输入格式
`test2.0.jsonl` 中每行应为：
```json
{
  "fact": "被告人张三持刀抢劫便利店...",
  "meta": {
    "accusation": ["抢劫"],
    "relevant_articles": [267]
  }
}
```

---

## 🧠 模型来源
- 基础模型：[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/deepseek-llm)
- LoRA 微调方案：使用 peft + transformers

---

## 📌 注意事项
- 本项目需具备 ≥12GB 显存环境运行 LoRA 推理
- 训练与评估脚本假设所有数据已预处理为合法 JSONL 格式

---

## 📮 联系方式
如有问题欢迎提交 Issue 或联系维护者：chuhefeng@qq.com

---

> 本项目为大数据应用课程设计作品，仅用于教学研究用途。
