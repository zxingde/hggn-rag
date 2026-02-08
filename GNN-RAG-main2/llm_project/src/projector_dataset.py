import json
import pickle
import torch
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# �� 关键修改：直接引入作者的 PromptBuilder，保证格式绝对一致
# 假设当前运行路径在 llm_project 下，根据你的目录结构添加路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from qa_prediction.build_qa_input import PromptBuilder


class ProjectorDataset(Dataset):
    def __init__(self, jsonl_path, pkl_path, tokenizer, max_length=512, prompt_path="prompts/llama2_predict.txt"):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 初始化 PromptBuilder (直接复用原项目逻辑)
        # 注意：这里参数要和 predict_answer.py 里的保持一致
        if not os.path.exists(prompt_path):
            # 回退路径尝试
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), prompt_path)

        print(f"�� [Dataset] Initializing PromptBuilder with {prompt_path}")
        self.prompt_builder = PromptBuilder(
            prompt_path=prompt_path,
            encrypt=False,  # 根据需要调整
            add_rule=True,  # Projector 模式下通常不用显式 Rule 文本，因为图特征里有了
            use_true=False,
            cot=False
        )

        # 2. 加载数据
        print(f"�� [Dataset] Loading Text from: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 预处理：有些数据集可能缺字段，补齐默认值防止 PromptBuilder 报错
                    if 'choices' not in item: item['choices'] = []
                    if 'cand' not in item: item['cand'] = None
                    if 'q_entity' not in item: item['q_entity'] = []
                    self.data.append(item)
                except:
                    continue

        # 3. 加载图特征
        print(f"��️ [Dataset] Loading Graph Features from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.graph_features = pickle.load(f)

        print(f"✅ Dataset Ready: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item.get('id')

        # --- A. 使用 PromptBuilder 生成 Input (核心对齐点) ---
        # 这一步生成的 text 就会包含 [INST]...[/INST] 等所有细节
        input_text = self.prompt_builder.process_input(item)

        # --- B. 获取 Answer ---
        if 'output' in item:
            answer = item['output']
        elif 'answer' in item:
            answer = item['answer']
        elif 'ground_truth' in item:
            answer = item['ground_truth']
        else:
            answer = ""

        if isinstance(answer, list): answer = answer[0] if len(answer) > 0 else ""

        # --- C. 拼接 (Prompt + Answer) ---
        full_text = f"{input_text} {answer} {self.tokenizer.eos_token}"

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        labels = input_ids.clone()

        # --- D. 获取图特征 ---
        graph_vecs = torch.zeros((1, 50), dtype=torch.float32)
        graph_mask = torch.ones(1, dtype=torch.long)

        if qid and qid in self.graph_features:
            features = self.graph_features[qid]
            if isinstance(features, np.ndarray) and features.shape[0] > 0 and not np.all(features == 0):
                graph_vecs = torch.tensor(features, dtype=torch.float32)
                graph_mask = torch.ones(len(graph_vecs), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph_feats": graph_vecs,
            "graph_mask": graph_mask
        }


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])

    graph_feats_list = [x['graph_feats'] for x in batch]
    graph_mask_list = [x['graph_mask'] for x in batch]

    padded_graph_feats = pad_sequence(graph_feats_list, batch_first=True, padding_value=0.0)
    padded_graph_mask = pad_sequence(graph_mask_list, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "graph_feats": padded_graph_feats,
        "graph_mask": padded_graph_mask
    }