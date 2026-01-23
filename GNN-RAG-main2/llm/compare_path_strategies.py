import json
import re
import pickle
import numpy as np
import os
from tqdm import tqdm

# ========================== 配置区域 ==========================
ID_SOURCE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/AlignData/RoG-webqsp/RoG-webqsp_train.jsonl"
TEXT_SOURCE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl"
FEATURE_CACHE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
ENTITY_MID_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/data/webqsp/entities.txt"
ENTITY_NAMES_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/entities_names.json"

OUTPUT_FILE = "final_train_dataset_webqsp.pkl"
FEATURE_DIM = 50


# =============================================================

def load_resources():
    print(">>> 加载资源中...")
    # 1. 加载 Name -> MID (做归一化处理)
    with open(ENTITY_NAMES_FILE, 'r', encoding='utf-8') as f:
        mid2name = json.load(f)
    name2mid = {str(v).strip().lower(): k for k, v in mid2name.items()}

    # 2. 加载 MID -> GID
    mid2id = {l.strip(): i for i, l in enumerate(open(ENTITY_MID_FILE))}

    # 3. 加载 GNN 特征
    with open(FEATURE_CACHE_FILE, 'rb') as f:
        gnn_cache = pickle.load(f)

    # 4. 建立 Question -> ID 索引
    q2id = {}
    with open(ID_SOURCE_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            q2id[item['question'].strip().replace(" ", "")] = item['id']

    return name2mid, mid2id, gnn_cache, q2id


def get_entity_feature(segment, name2mid, mid2id, subgraph):
    """
    三级降级匹配逻辑，最大化覆盖率
    """
    segment = segment.strip().strip('"').strip("'")

    # 逻辑 A: 如果片段本身就是 MID 格式 (如 m.0xxx)
    if segment.startswith('m.'):
        mid = segment
    # 逻辑 B: 查名字字典 (归一化匹配)
    else:
        mid = name2mid.get(segment.lower())

    if mid:
        gid = mid2id.get(mid)
        # 逻辑 C: 检查是否在 GNN 子图中
        if gid is not None and gid in subgraph:
            return subgraph[gid]
    return None


def main():
    name2mid, mid2id, gnn_cache, q2id = load_resources()
    final_dataset = []

    print(f">>> 开始融合数据...")
    with open(TEXT_SOURCE_FILE, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)
        text = item['text']

        # 1. 提取问题并匹配 ID
        q_match = re.search(r"Question:\n(.*?)\s*\[/INST\]", text, re.DOTALL)
        if not q_match: continue
        q_text = q_match.group(1).strip()
        qid = q2id.get(q_text.replace(" ", ""))

        # 2. 获取子图
        subgraph = gnn_cache.get(qid, {})

        # 3. 解析路径
        path_block = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.DOTALL)
        if not path_block: continue
        paths = [l.strip() for l in path_block.group(1).split('\n') if l.strip()]

        path_data_list = []
        for p in paths:
            # 提取路径中所有可能的节点
            segments = [s.strip() for s in p.split("->")]
            available_vectors = []

            for seg in segments:
                feat = get_entity_feature(seg, name2mid, mid2id, subgraph)
                if feat is not None:
                    available_vectors.append(feat)

            # 特征聚合：如果路径有节点命中，取平均；否则全 0
            if len(available_vectors) > 0:
                vec = np.mean(available_vectors, axis=0)
                hit = True
            else:
                vec = np.zeros(FEATURE_DIM, dtype=np.float32)
                hit = False

            path_data_list.append({
                "path": p,
                "feature": vec,
                "hit": hit
            })

        # 保存融合后的样本
        item['gnn_path_data'] = path_data_list
        item['qid'] = qid  # 显式保留 ID 方便后续对齐
        final_dataset.append(item)

    print(f"\n>>> 融合完成！总样本数: {len(final_dataset)}")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_dataset, f)
    print(f"✅ 最终训练数据已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()