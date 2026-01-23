import json
import re
import pickle
import numpy as np
from tqdm import tqdm
import os
import sys

# ========================== 自动配置路径 ==========================
# 你可以根据需要切换 webqsp 或 cwq 的路径
DATASET = "cwq"  # 或 "cwq"

if DATASET == "webqsp":
    QA_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl"
    FEATURE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
    ENTITY_MID_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/data/webqsp/entities.txt"
else:
    QA_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa/RoG-cwq/RoG-cwq_train.jsonl"
    FEATURE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
    ENTITY_MID_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/data/CWQ/entities.txt"

ENTITY_NAMES_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/entities_names.json"


# =================================================================

def load_resources():
    print(f">>> [1/3] 加载 {DATASET} 实体名称字典...")
    with open(ENTITY_NAMES_FILE, 'r', encoding='utf-8') as f:
        mid2name = json.load(f)
    name2mid = {str(v).strip().lower(): k for k, v in mid2name.items()}

    print(f">>> [2/3] 加载 GNN 实体词表...")
    mid2id = {l.strip(): i for i, l in enumerate(open(ENTITY_MID_FILE))}

    print(f">>> [3/3] 加载 GNN 特征缓存...")
    with open(FEATURE_FILE, 'rb') as f:
        gnn_cache = pickle.load(f)

    return name2mid, mid2id, gnn_cache


def analyze_hits():
    name2mid, mid2id, gnn_cache = load_resources()

    stats = {
        "total_paths": 0,
        "path_with_any_feat": 0,
        "total_nodes_in_paths": 0,
        "nodes_with_feat": 0,
        "fail_reasons": {
            "name_not_found": 0,  # 字典里没这个名字
            "mid_not_in_vocab": 0,  # 名字转成了MID，但GNN词表没收录
            "not_in_subgraph": 0  # 在词表里，但没进这个问题的2000个子图节点
        }
    }

    print("\n>>> 正在扫描路径并分析命中率...")
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)
        qid = item['id']
        text = item['text']

        # 正则提取路径块
        path_match = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.DOTALL)
        if not path_match: continue

        paths = [l.strip() for l in path_match.group(1).split('\n') if l.strip()]
        subgraph = gnn_cache.get(qid, {})

        for p in paths:
            stats["total_paths"] += 1
            # 这里的切割要鲁棒，处理 -> 和 空格
            segments = [s.strip().strip('"').strip("'").lower() for s in p.split("->")]

            path_has_feature = False
            for seg in segments:
                # 只统计像实体的词（排除关系谓词如 people.person.gender）
                if "." in seg and not seg.startswith("m."): continue

                stats["total_nodes_in_paths"] += 1

                # 步骤 1: Name -> MID
                mid = name2mid.get(seg) if not seg.startswith("m.") else seg
                if not mid:
                    stats["fail_reasons"]["name_not_found"] += 1
                    continue

                # 步骤 2: MID -> GID
                gid = mid2id.get(mid)
                if gid is None:
                    stats["fail_reasons"]["mid_not_in_vocab"] += 1
                    continue

                # 步骤 3: GID in Subgraph?
                if gid in subgraph:
                    stats["nodes_with_feat"] += 1
                    path_has_feature = True
                else:
                    stats["fail_reasons"]["not_in_subgraph"] += 1

            if path_has_feature:
                stats["path_with_any_feat"] += 1

    # ========================== 报告输出 ==========================
    print("\n" + "=" * 50)
    print(f"�� {DATASET.upper()} 节点命中率深度分析报告")
    print("=" * 50)

    print(f"1. 路径级 (Path-level):")
    print(f"   - 总路径数: {stats['total_paths']}")
    print(f"   - 命中特征路径: {stats['path_with_any_feat']}")
    print(f"   - 路径覆盖率: {stats['path_with_any_feat'] / stats['total_paths'] * 100:.2f}%")

    print(f"\n2. 节点级 (Node-level):")
    print(f"   - 路径内实体总数: {stats['total_nodes_in_paths']}")
    print(f"   - 成功命中特征数: {stats['nodes_with_feat']}")
    print(f"   - 节点命中率: {stats['nodes_with_feat'] / stats['total_nodes_in_paths'] * 100:.2f}%")

    print(f"\n3. 未命中原因分析 (按节点统计):")
    total_fails = sum(stats["fail_reasons"].values())
    if total_fails > 0:
        for k, v in stats["fail_reasons"].items():
            print(f"   - {k:20}: {v:6} 次 ({v / total_fails * 100:5.1f}%)")

    print("=" * 50)


if __name__ == "__main__":
    analyze_hits()