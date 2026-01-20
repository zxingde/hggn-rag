import string
import sys
import os
import pickle
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset

# 将父目录加入路径，以便导入 utils
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


def load_graph_features(feat_path):
    """加载 .pkl 特征文件"""
    if feat_path and os.path.exists(feat_path):
        print(f"Loading graph features from {feat_path}...")
        with open(feat_path, 'rb') as f:
            return pickle.load(f)
    print(f"Warning: Graph feature file not found or path is empty: {feat_path}")
    return {}


def load_new_tokens(default_new_tokens, rel_dict_path):
    """
    【补回来的函数】加载关系 Token
    """
    if isinstance(rel_dict_path, str):
        rel_dict_path = [rel_dict_path]

    if rel_dict_path is None:
        return default_new_tokens

    for rel_path in rel_dict_path:
        if os.path.exists(rel_path):
            with open(rel_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        _, r = parts
                        default_new_tokens.append(r)
    return default_new_tokens


def load_multiple_datasets(data_path_list, graph_feat_path=None, shuffle=True):
    """
    加载多个数据集，并注入图特征 (强制转换为 List[float] 以解决类型冲突)
    """
    # 1. 加载图特征字典
    features_dict = load_graph_features(graph_feat_path)

    all_datasets = []
    for data_path in data_path_list:
        print(f"Loading dataset: {data_path}")
        dataset = load_dataset('json', data_files=data_path, split='train')

        # 2. 定义映射函数
        def add_graph_features(example):
            qid = None
            # 兼容多种 ID 写法
            if 'id' in example:
                qid = str(example['id'])
            elif 'qid' in example:
                qid = str(example['qid'])

            # 3. 检索特征并强制转换类型
            if qid and qid in features_dict:
                raw_feat = features_dict[qid]

                # 【核心修复】无论原来是 Tensor 还是 Numpy，都转成纯 Python List
                if isinstance(raw_feat, torch.Tensor):
                    feat = raw_feat.tolist()
                elif isinstance(raw_feat, np.ndarray):
                    feat = raw_feat.tolist()
                elif isinstance(raw_feat, list):
                    feat = raw_feat
                else:
                    # 兜底：如果是未知类型但支持 tolist
                    if hasattr(raw_feat, 'tolist'):
                        feat = raw_feat.tolist()
                    else:
                        feat = raw_feat
            else:
                # 没找到 ID，补全 0 (List[float64])
                feat = [[0.0] * 50 for _ in range(6)]

            return {'graph_features': feat}

        # 4. 使用 map 功能注入特征
        dataset = dataset.map(add_graph_features)
        all_datasets.append(dataset)

    # 5. 合并数据集
    print("Concatenating datasets...")
    dataset = concatenate_datasets(all_datasets)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset



def get_test_dataset(dataset):
    # Gather all the labels for the same question
    test_dataset = dict()
    for sample in dataset:
        if sample['question'] not in test_dataset:
            test_dataset[sample['question']] = set()
        label = tuple(sample['path'])
        test_dataset[sample['question']].add(label)
    test_dataset = [{'text': k, 'label': v} for k, v in test_dataset.items()]
    return Dataset.from_list(test_dataset)


