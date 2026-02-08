import json
import pickle
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ProjectorDataset(Dataset):
    def __init__(self, jsonl_path, pkl_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"�� [Dataset] Loading Text from: {jsonl_path}")
        # 纯加载，不搞任何花哨的 builder
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    continue

        print(f"��️ [Dataset] Loading Graph Features from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.graph_features = pickle.load(f)

        print(f"✅ Dataset Ready: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item.get('id')

        # --- A. 直接读取 Input ---
        # 你的数据里 text 字段已经是完整的 [INST]...[/INST]
        if 'text' in item:
            input_text = item['text']
        elif 'input' in item:  # 兼容某些数据集叫 input
            input_text = item['input']
        else:
            # 万一没有预处理好的，才降级去读 question (通常不会走到这)
            input_text = item.get('question', '')

        # --- B. 读取 Answer ---
        # 尝试读取 answer/output/ground_truth
        if 'output' in item:
            answer = item['output']
        elif 'ground_truth' in item:
            answer = item['ground_truth']
        elif 'answer' in item:
            answer = item['answer']
        else:
            answer = ""

        # 如果是列表，取第一个作为训练目标
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""

        # --- C. 拼接 (Input + Answer + EOS) ---
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

        # 训练 Projector 时，全量计算 Loss 是没问题的
        labels = input_ids.clone()

        # --- D. 图特征 (保持不变) ---
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