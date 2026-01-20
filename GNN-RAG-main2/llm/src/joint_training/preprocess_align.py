import sys
import os
import multiprocessing as mp
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "datasets/joint_training/ "
prompt_path = "prompts/llama2.txt"
data_template = "datasets/AlignData/{}/{}_train.jsonl"
data_list = ['RoG-webqsp', 'RoG-cwq']
model_name_or_path = "NousResearch/Llama-2-7b-chat-hf"
prompter = InstructFormater(prompt_path)

INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """
SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=False,
)


def formatting_prompts_func(example):
    output_label = rule_to_string(example["path"], sep_token=SEP, bop=BOP, eop=EOP)
    output_text = (
            prompter.format(instruction=INSTRUCTION, message=example["question"])
            + " "
            + output_label + tokenizer.eos_token
    )
    # 【核心修改】同样显式保留 id，以防万一
    return {"text": output_text, "id": example.get('id', example.get('qid', 'unknown_id'))}


for data_name in data_list:
    data_path = data_template.format(data_name, data_name)
    save_path = os.path.join(save_dir, data_name, data_name + "_train.jsonl")

    # 增加 verify=False 以避免可能的缓存一致性报错
    train_dataset = datasets.load_dataset('json', data_files=data_path, split="train")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=["question", "path"],  # 这里只删除了特定列，但显式返回 id 更安全
        num_proc=N_CPUS,
    )
    train_dataset.to_json(save_path, orient="records", lines=True)