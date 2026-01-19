import string
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset, concatenate_datasets, Dataset
from utils import rule_to_string
import pickle
import os

def load_graph_features(feat_path):
    """
    专门负责读取你生成的 .pkl 特征文件
    """
    if feat_path and os.path.exists(feat_path):
        with open(feat_path, 'rb') as f:
            # 读取出来的字典：Key 是问题 ID (str)，Value 是 (6, 50) 的特征
            return pickle.load(f)
    return {}

def load_new_tokens(default_new_tokens, rel_dict_path):
    if isinstance(rel_dict_path, str):
        rel_dict_path = [rel_dict_path]
    for rel_path in rel_dict_path:
        with open(rel_path, 'r') as f:
            for line in f.readlines():
                _, r = line.strip().split('\t')
                default_new_tokens.append(r)
    return default_new_tokens
        

def load_multiple_datasets(data_path_list, graph_feat_path=None,shuffle=False):
    '''
    Load multiple datasets from different paths.

    Args:
        data_path_list (_type_): _description_
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    '''
    # 1. 先加载图特征小抄 (调用你刚才写好的函数)
    features_dict = load_graph_features(graph_feat_path)  # 新增
    all_datasets = []

    for data_path in data_path_list:
        # 加载原始 json 数据
        dataset = load_dataset('json', data_files=data_path, split='train')

        # 2. 定义一个内部函数，告诉 Git 怎么把特征塞进每一行数据
        def add_graph_features(example):  # 新增
            qid = str(example['id'])  # 确保 ID 是字符串，好去字典里找
            # 如果字典里有这个 ID，就拿出来；没有就给全 0 向量
            if qid in features_dict:
                feat = features_dict[qid]
            else:
                import torch
                feat = torch.zeros(6, 50).tolist()  # 兜底逻辑：没找到就给 6x50 的 0
            return {'graph_features': feat}

        # 3. 使用 map 功能，给数据集增加一列叫 'graph_features'
        dataset = dataset.map(add_graph_features)  # 新增
        all_datasets.append(dataset)
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


