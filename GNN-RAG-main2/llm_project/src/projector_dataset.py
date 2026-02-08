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

        # --- A. 直接读取 text (这是唯一的改动) ---
        # 既然 text 已经是 [INST]...[/INST] Answer </s>
        # 那我们直接拿来用就行了！
        if 'text' in item:
            full_text = item['text']
        else:
            # 防御性编程：万一哪条数据没处理好，回退到原来的逻辑
            # 但针对你的 WebQTrn 数据，应该都走上面
            q = item.get('question', '')
            a = item.get('output', '') or item.get('answer', '')
            if isinstance(a, list): a = a[0]
            full_text = f"Question: {q} Answer: {a}"

        # --- B. Tokenize ---
        # 注意：因为 text 里已经有 </s> 了，所以这里不需要再加 self.tokenizer.eos_token
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]

        # 训练 Projector 时，全量计算 Loss
        labels = input_ids.clone()

        # --- C. 图特征 (保持不变) ---
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