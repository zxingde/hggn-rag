import json
import re
import pickle
import numpy as np
import sys
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始 QA 数据文件 (用于找回 ID)
ORIGINAL_QA_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/AlignData/RoG-cwq/RoG-cwq_train.jsonl"

# 2. 你的 LLM 训练数据文件 (包含 [INST] 推理路径)
LLM_TRAIN_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa_back/RoG-cwq/RoG-cwq_train.jsonl"

# 3. 特征缓存文件 (请确保是那个 7.8GB 的文件)
FEATURE_CACHE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"

# 4. 实体映射文件
ENTITY_MID_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/data/CWQ/entities.txt"
ENTITY_NAMES_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/entities_names.json"


def load_resources():
    print(">>> Loading Resources for Statistics...")

    # 1. QID Map
    q2id = {}
    with open(ORIGINAL_QA_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            q2id[item['question'].strip()] = item['id']
    print(f"    Loaded {len(q2id)} questions.")

    # 2. Names Map
    with open(ENTITY_NAMES_FILE, 'r') as f:
        mid2name = json.load(f)
    name2mid = {v.strip(): k for k, v in mid2name.items()}
    print(f"    Loaded {len(name2mid)} names.")

    # 3. MID to ID
    mid2id = {}
    with open(ENTITY_MID_FILE, 'r') as f:
        for idx, line in enumerate(f):
            mid2id[line.strip()] = idx
    print(f"    Loaded {len(mid2id)} mids.")

    # 4. Cache
    print("    Loading Feature Cache (waiting)...")
    with open(FEATURE_CACHE_FILE, 'rb') as f:
        gnn_cache = pickle.load(f)
    print(f"    Cache Loaded. Keys count: {len(gnn_cache)}")

    return q2id, name2mid, mid2id, gnn_cache


def calculate_coverage():
    q2id, name2mid, mid2id, gnn_cache = load_resources()

    print("\n>>> Starting Full Dataset Scan...\n")

    stats = {
        "total_questions": 0,
        "questions_with_cache": 0,  # 找到了对应 GNN 图的问题数
        "total_paths": 0,  # 总路径数
        "paths_name_matched": 0,  # 名字能转成 MID 的路径数
        "paths_id_matched": 0,  # MID 能转成 GID 的路径数
        "paths_gnn_hit": 0  # GID 在子图里找到了特征的路径数
    }

    with open(LLM_TRAIN_FILE, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scanning"):
        stats["total_questions"] += 1
        item = json.loads(line)
        text = item['text']

        # 1. 解析问题并找 ID
        match = re.search(r"Question:\n(.*?)\s*\[/INST\]", text, re.DOTALL)
        if not match: continue
        q_text = match.group(1).strip()

        qid = q2id.get(q_text)
        if not qid: qid = q2id.get(q_text.replace("?", ""))

        # 如果没找到 ID 或者没在缓存里，跳过
        if not qid or qid not in gnn_cache:
            continue

        stats["questions_with_cache"] += 1
        graph_features = gnn_cache[qid]

        # 2. 解析路径
        path_match = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.DOTALL)
        paths = [l.strip() for l in path_match.group(1).split('\n') if l.strip()] if path_match else []

        for p in paths:
            stats["total_paths"] += 1
            tail = p.split("->")[-1].strip()

            # Step 1: Name -> MID
            mid = name2mid.get(tail)
            if not mid: continue
            stats["paths_name_matched"] += 1

            # Step 2: MID -> GID
            gid = mid2id.get(mid)
            if gid is None: continue
            stats["paths_id_matched"] += 1

            # Step 3: GID -> Feature
            if gid in graph_features:
                stats["paths_gnn_hit"] += 1

    # === 输出报告 ===
    print("\n" + "=" * 40)
    print("�� FEATURE COVERAGE REPORT")
    print("=" * 40)

    total_q = stats["total_questions"]
    valid_q = stats["questions_with_cache"]
    print(f"1. Question Coverage:")
    print(f"   - Total Questions: {total_q}")
    print(f"   - Questions with GNN Cache: {valid_q} ({valid_q / total_q * 100:.2f}%)")

    total_p = stats["total_paths"]
    if total_p > 0:
        print(f"\n2. Path Feature Hit Rate (Recall):")
        print(f"   - Total Paths: {total_p}")
        print(
            f"   - [Step 1] Name mapped to MID: {stats['paths_name_matched']} ({stats['paths_name_matched'] / total_p * 100:.2f}%)")
        print(
            f"   - [Step 2] MID mapped to GID:  {stats['paths_id_matched']} ({stats['paths_id_matched'] / total_p * 100:.2f}%)")
        print(f"   - [Step 3] GID found in Subgraph: {stats['paths_gnn_hit']}")
        print(f"   ------------------------------------------")
        print(f"   ✅ Final GNN Hit Rate: {stats['paths_gnn_hit'] / total_p * 100:.2f}%")
    else:
        print("\nNo paths parsed!")
    print("=" * 40)


if __name__ == "__main__":
    calculate_coverage()