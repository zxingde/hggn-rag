import json
import re
import pickle
import numpy as np
import os
from tqdm import tqdm

# ========================== 1. 核心配置区域 (请确认路径) ==========================
# 数据集选择: "webqsp" 或 "cwq"
DATASET_NAME = "webqsp"

# 基础路径
BASE_DIR = "/home/bi3/zxd_env/GNN-RAG-main2"

if DATASET_NAME == "webqsp":
    # 你的文本训练集 (由 preprocess_qa.py 生成的带 [INST] 的 jsonl)
    QA_INPUT_FILE = f"{BASE_DIR}/llm/datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl"
    # GNN 导出的特征文件
    GNN_FEATURE_FILE = f"{BASE_DIR}/gnn/checkpoint/pretrain/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
    # 实体 ID 列表
    ENTITY_LIST_FILE = f"{BASE_DIR}/gnn/data/webqsp/entities.txt"
else:
    QA_INPUT_FILE = f"{BASE_DIR}/llm/datasets/joint_training/qa/RoG-cwq/RoG-cwq_train.jsonl"
    GNN_FEATURE_FILE = f"{BASE_DIR}/gnn/checkpoint/pretrain/cwq_features.pkl"  # 请修改为你的CWQ特征路径
    ENTITY_LIST_FILE = f"{BASE_DIR}/gnn/data/cwq/entities.txt"

# 实体名字映射表
ENTITY_NAME_MAP = f"{BASE_DIR}/llm/entities_names.json"

# 输出文件 (训练脚本直接读这个)
OUTPUT_FILE = f"final_finetune_corpus_{DATASET_NAME}.pkl"

# GNN 特征维度 (通常是 50)
FEATURE_DIM = 50


# ==============================================================================

def load_resources():
    print(f"�� 正在加载资源 ({DATASET_NAME})...")

    # 1. 加载 Name -> MID 映射 (用于把文本名字转回 ID)
    print(f"   - Loading Name Map: {ENTITY_NAME_MAP}...")
    with open(ENTITY_NAME_MAP, 'r', encoding='utf-8') as f:
        mid2name = json.load(f)
    # 构建小写反向索引，提高匹配率
    name2mid = {}
    for mid, name in mid2name.items():
        if isinstance(name, str):
            name2mid[name.strip().lower()] = mid

    # 2. 加载 GNN 全局 ID 表 (MID -> GID)
    print(f"   - Loading Entity List: {ENTITY_LIST_FILE}...")
    mid2id = {l.strip(): i for i, l in enumerate(open(ENTITY_LIST_FILE))}

    # 3. 加载 GNN 特征库
    print(f"   - Loading GNN Features: {GNN_FEATURE_FILE}...")
    with open(GNN_FEATURE_FILE, 'rb') as f:
        gnn_features = pickle.load(f)

    print("✅ 资源加载完毕!")
    return name2mid, mid2id, gnn_features


def extract_paths_from_text(text):
    """从 [INST] 文本中用正则提取路径"""
    # 匹配 Reasoning Paths: 和 Question: 中间的内容
    match = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.DOTALL)
    if not match:
        return []
    path_block = match.group(1)
    # 按行分割，去除空行
    return [line.strip() for line in path_block.split('\n') if line.strip()]


def get_path_vector(path_str, name2mid, mid2id, subgraph_features):
    """将一条文本路径转化为一个 50 维向量 (Mean Pooling)"""
    segments = path_str.split("->")
    valid_vectors = []

    for seg in segments:
        clean_seg = seg.strip().strip('"').strip("'")

        # 1. 尝试获取 MID
        mid = None
        # 情况 A: 本身就是 ID 格式 (m.xxx)
        if clean_seg.startswith("m."):
            mid = clean_seg
        # 情况 B: 是名字，查字典
        else:
            mid = name2mid.get(clean_seg.lower())

        if not mid: continue

        # 2. 尝试获取 GID (全局索引)
        gid = mid2id.get(mid)
        if gid is None: continue

        # 3. 从当前问题的子图中拿特征
        if gid in subgraph_features:
            valid_vectors.append(subgraph_features[gid])

    # 聚合逻辑: 取平均
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def main():
    name2mid, mid2id, gnn_features = load_resources()

    final_dataset = []
    skipped_count = 0

    print(f"⚙️ 开始处理 QA 数据...")
    with open(QA_INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)

        # 必须字段检查
        if 'id' not in item or 'text' not in item:
            continue

        qid = item['id']
        text = item['text']

        # 1. 获取该问题的 GNN 子图特征 (如果没跑出特征，就给空字典)
        subgraph = gnn_features.get(qid, {})

        # 2. 提取文本路径
        text_paths = extract_paths_from_text(text)

        # 3. 生成 gnn_path_data
        gnn_path_data = []
        for p in text_paths:
            vec = get_path_vector(p, name2mid, mid2id, subgraph)
            # 记录是否命中 (全0向量视为未命中)
            is_hit = not np.all(vec == 0)

            gnn_path_data.append({
                "path": p,
                "feature": vec,
                "hit": is_hit
            })

        # 4. 只有当找到路径时才加入训练 (或者你也可以保留全0的样本)
        # 这里为了稳健，我们保留所有样本，哪怕全是 0
        item['gnn_path_data'] = gnn_path_data

        # 移除可能导致冲突的旧字段，保持干净
        if 'gnn_path_ids' in item: del item['gnn_path_ids']

        final_dataset.append(item)

    print(f"\n�� 处理完成!")
    print(f"   - 总样本数: {len(final_dataset)}")
    print(f"   - 输出文件: {OUTPUT_FILE}")

    # 保存为 PKL
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_dataset, f)
    print("�� 文件已保存，可以直接用于 joint_finetuning.py 训练了。")


if __name__ == "__main__":
    main()