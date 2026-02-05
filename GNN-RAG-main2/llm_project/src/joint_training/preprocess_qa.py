import sys
import os
import json
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm

# ================= 核心配置 (请根据你的环境修改) =================
# 1. 本地模型路径 (解决连不上 HuggingFace 的问题)
#    请确保该文件夹下有 tokenizer.model, tokenizer.json 等文件
LOCAL_MODEL_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm_project/RoG_model"

# 2. 本地原始数据目录 (解决连不上数据集的问题)
#    请确保该目录下有 RoG-webqsp/RoG-webqsp_train.csv 等文件
LOCAL_DATA_ROOT = "/home/bi3/data"

# 3. 输出保存路径
SAVE_DIR = "datasets/joint_training/qa"
# ==============================================================

# 添加上级目录以导入 utils
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from qa_prediction.build_qa_input import PromptBuilder

# 并行设置
N_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# 任务配置
prompt_path = "prompts/llama2_predict.txt"
model_max_length = 2048 - 200
data_list = ['RoG-cwq', 'RoG-webqsp']  # 两个数据集都跑
splits = ['train', 'validation', 'test']  # 跑所有划分

# 加载本地分词器
print(f"�� Loading tokenizer from local path: {LOCAL_MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True  # 强制只读本地
    )
except Exception as e:
    print(f"❌ 分词器加载失败: {e}")
    print("请确认 LOCAL_MODEL_PATH 指向了包含 tokenizer 文件的文件夹！")
    sys.exit(1)

# 加载 Prompt 构建器
input_builder = PromptBuilder(
    prompt_path,
    add_rule=True,
    use_true=True,
    maximun_token=model_max_length,
    tokenize=lambda x: len(tokenizer.tokenize(x)),
)


def formatting_prompts_func(example):
    """处理单条数据：生成 Prompt 并保留 ID"""
    # 1. 提取 ID (关键！)
    qid = example.get('id', None)
    if not qid and 'question_id' in example:
        qid = example['question_id']

    # 2. 准备 Answer (测试集可能没有，需兼容)
    output_label = ""
    if 'answer' in example and example['answer']:
        if isinstance(example['answer'], list):
            output_label = "\n".join(example['answer'])
        else:
            output_label = str(example['answer'])

    # 3. 提取 Ground Truth Paths (依赖 graph 字段)
    try:
        if "graph" in example and "q_entity" in example and "a_entity" in example:
            # 这里依赖 utils.py 里的函数，确保 utils 能正常导入
            graph = build_graph(example["graph"])
            paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
            ground_paths = set()
            for path in paths:
                # 提取关系路径
                ground_paths.add(tuple([p[1] for p in path]))
            example["ground_paths"] = list(ground_paths)
        else:
            example["ground_paths"] = []
    except Exception:
        example["ground_paths"] = []

    # 4. 生成 Prompt
    input_text = input_builder.process_input(example)

    # 5. 拼接完整文本 (Input + Answer)
    #    注意：对于测试集，其实只需要 input_text，但为了格式统一通常也拼上。
    #    推理时模型只会看到 input_text 部分。
    full_text = input_text + " " + output_label + tokenizer.eos_token

    # 【重要】返回字典，必须包含 id
    return {
        "id": qid,
        "text": full_text
    }


def main():
    for data_name in data_list:
        print(f"\n======== Processing Dataset: {data_name} ========")

        for split in splits:
            # 1. 构造本地文件路径
            # 假设你的文件命名是: RoG-cwq_train.csv / RoG-cwq_test.csv
            # 如果是 jsonl，请把 extension 改为 .json
            extension = ".csv"
            file_name = f"{data_name}_{split}{extension}"
            file_path = os.path.join(LOCAL_DATA_ROOT, data_name, file_name)

            if not os.path.exists(file_path):
                # 尝试找 jsonl
                extension = ".jsonl"
                file_name = f"{data_name}_{split}{extension}"
                file_path = os.path.join(LOCAL_DATA_ROOT, data_name, file_name)

                if not os.path.exists(file_path):
                    print(f"⚠️  Skipping {split}: File not found at {file_path}")
                    continue

            print(f"   - Loading local file: {file_path}")

            try:
                # 2. 加载本地数据集
                # 如果是 csv 文件
                if file_path.endswith(".csv"):
                    dataset = datasets.load_dataset("csv", data_files=file_path, split="train")
                # 如果是 json/jsonl 文件
                else:
                    dataset = datasets.load_dataset("json", data_files=file_path, split="train")
            except Exception as e:
                print(f"❌ Load failed: {e}")
                continue

            print(f"   - Processing {len(dataset)} examples...")

            # 3. 处理数据 (Map)
            # remove_columns=dataset.column_names 会删除原始列
            # 但因为 formatting_prompts_func 返回了 'id' 和 'text'，这两个会被保留
            processed_dataset = dataset.map(
                formatting_prompts_func,
                remove_columns=dataset.column_names,
                num_proc=N_CPUS,
                load_from_cache_file=False  # 避免缓存导致的旧数据问题
            )

            # 4. 保存结果
            out_path = os.path.join(SAVE_DIR, data_name, f"{data_name}_{split}.jsonl")
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))

            processed_dataset.to_json(out_path, orient="records", lines=True, force_ascii=False)
            print(f"   ✅ Saved to: {out_path}")


if __name__ == "__main__":
    main()