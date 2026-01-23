import sys
import os
import multiprocessing as mp
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets
from qa_prediction.build_qa_input import PromptBuilder

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "datasets/joint_training/qa"
prompt_path = "prompts/llama2_predict.txt"
split = "train"
model_max_length = 2048 - 200
data_list = ['RoG-webqsp', 'RoG-cwq']
data_path = "rmanluo"
model_name_or_path = "NousResearch/Llama-2-7b-chat-hf"
prompter = InstructFormater(prompt_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=False,
)

# Load prompt template
input_builder = PromptBuilder(
    prompt_path,
    add_rule=True,
    use_true=True,
    maximun_token=model_max_length,
    tokenize=lambda x: len(tokenizer.tokenize(x)),
)


def formatting_prompts_func(example):
    example['cand'] = None
    if 'choices' not in example:
        example['choices'] = []

    output_label = "\n".join(example['answer'])
    # Find ground-truth paths for each Q-P pair
    graph = build_graph(example["graph"])
    paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    example["ground_paths"] = list(ground_paths)
    output_text = (
            input_builder.process_input(example)
            + " "
            + output_label + tokenizer.eos_token
    )

    # ========================================================
    # 【核心修改区域】
    # 必须显式返回 'id'，因为 map 函数设置了 remove_columns
    # 我建议顺便把原始 'question' 也带上，方便后续人工检查
    # ========================================================
    return {
        "text": output_text,  # 训练用的文本
        "id": example['id'],  # 【关键】用于对齐 GNN 特征的唯一 ID
        "question": example['question']  # (可选) 原始问题文本，方便调试
    }


for data_name in data_list:
    input_file = os.path.join(data_path, data_name)
    train_dataset = datasets.load_dataset(input_file, split="train")

    # 确保保存路径存在
    save_path = os.path.join(save_dir, data_name, data_name + "_train.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    print(f"Processing {data_name}...")

    # 执行 Map
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        # 这里删除了所有未在 formatting_prompts_func 返回的列
        # 所以上面的 return 必须包含 'id'
        remove_columns=train_dataset.column_names,
        num_proc=N_CPUS,
    )

    # 保存为 JSONL
    print(f"Saving to {save_path}...")
    train_dataset.to_json(save_path, orient="records", lines=True)
    print("Done.")