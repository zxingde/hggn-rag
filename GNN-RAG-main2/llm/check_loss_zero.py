import pickle
import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

# 1. 加载数据
PKL_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
sample_text = data[0]['text']

print("=== 1. 检查文本样本 ===")
print(sample_text[:200]) # 打印开头
print("...")
print(sample_text[-200:]) # 打印结尾 (看有没有 [/INST])

# 2. 模拟 DataCollator
MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 这里的 template 必须和你 joint_finetuning.py 里的一模一样
response_template = "[/INST]"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 编码
encoded = tokenizer(sample_text)
# DataCollator 需要 list of dicts
batch_input = [{"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}]

# 处理
batch_output = collator(batch_input)
labels = batch_output["labels"][0]

print("\n=== 2. 检查 Labels ===")
print(f"Label 总长度: {len(labels)}")
# 统计非 -100 的个数
valid_labels = labels[labels != -100]
print(f"有效 Label (非 -100) 个数: {len(valid_labels)}")

if len(valid_labels) == 0:
    print("❌ 严重问题：所有 Label 都是 -100！这就是 Loss 为 0 的原因！")
    print("可能原因：文本里找不到 '[/INST]'，或者模板配置错了。")
else:
    print(f"✅ Label 正常。有效 Label 示例: {valid_labels[:10]}")