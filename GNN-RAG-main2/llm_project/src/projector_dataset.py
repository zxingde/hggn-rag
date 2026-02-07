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

        # 1. 加载文本
        print(f"�� [Dataset] Loading Text from: {jsonl_path}")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"❌ 找不到 JSONL 文件: {jsonl_path}")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    continue

        # 2. 加载图特征字典
        print(f"�� [Dataset] Loading Graph Features from: {pkl_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"❌ 找不到 PKL 文件: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            self.graph_features = pickle.load(f)

        print(f"✅ Dataset Ready: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item.get('id')

        # --- A. 准备文本 (Input + Label) ---
        # 简单构造: Question -> Answer
        question = item.get('text', item.get('input', ''))
        answer = item.get('answer', '')
        if isinstance(answer, list): answer = answer[0]  # 处理 WebQSP 列表情况

        # 构造训练文本
        # 注意：这里假设用简单的 Q: A: 格式，你可以根据模型调整
        full_text = f"Question: {question}\nAnswer: {answer}"

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        labels = input_ids.clone()  # 自回归训练

        # --- B. 获取图特征 ---
        # 默认值 (1, 50) 全0
        graph_vecs = torch.zeros((1, 50), dtype=torch.float32)
        graph_mask = torch.zeros(1, dtype=torch.long)  # mask=0 表示无效

        if qid and qid in self.graph_features:
            features = self.graph_features[qid]
            # 确保 features 是有效的 numpy array 且不是全0
            if isinstance(features, np.ndarray) and features.shape[0] > 0:
                # 检查是否全是0 (我们在预处理时对无路径的填了全0)
                if not np.all(features == 0):
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
    # 堆叠文本
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])

    # Pad 图特征 (变长序列)
    graph_feats_list = [x['graph_feats'] for x in batch]
    graph_mask_list = [x['graph_mask'] for x in batch]

    # [Batch, Max_Path, 50]
    padded_graph_feats = pad_sequence(graph_feats_list, batch_first=True, padding_value=0.0)
    # [Batch, Max_Path]
    padded_graph_mask = pad_sequence(graph_mask_list, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "graph_feats": padded_graph_feats,
        "graph_mask": padded_graph_mask
    }