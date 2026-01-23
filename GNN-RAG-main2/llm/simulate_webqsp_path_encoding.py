import json
import re
import pickle
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
QA_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl"
FEATURE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
# 注意：WebQSP 的实体表路径
ENTITY_MID_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/data/webqsp/entities.txt"
ENTITY_NAMES_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/entities_names.json"


def load_resources():
    print("Loading Mapping Resources...")
    with open(ENTITY_NAMES_FILE, 'r') as f:
        mid2name = json.load(f)
    name2mid = {v.strip(): k for k, v in mid2name.items()}

    mid2id = {}
    with open(ENTITY_MID_FILE, 'r') as f:
        for idx, line in enumerate(f):
            mid2id[line.strip()] = idx

    with open(FEATURE_FILE, 'rb') as f:
        gnn_cache = pickle.load(f)
    return name2mid, mid2id, gnn_cache


def parse_paths(text):
    # 兼容处理：提取 Reasoning Paths 块
    match = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.DOTALL)
    if match:
        return [l.strip() for l in match.group(1).split('\n') if l.strip()]
    return []


def main():
    name2mid, mid2id, gnn_cache = load_resources()

    total_paths = 0
    hit_paths = 0
    sample_success = 0

    print("\nScanning first 100 samples to verify path encoding...")

    with open(QA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100: break
            item = json.loads(line)
            qid = item['id']
            paths = parse_paths(item['text'])

            subgraph = gnn_cache.get(qid, {})

            current_sample_hit = False
            for p in paths:
                total_paths += 1
                tail_name = p.split("->")[-1].strip()

                mid = name2mid.get(tail_name)
                if mid:
                    gid = mid2id.get(mid)
                    if gid is not None and gid in subgraph:
                        hit_paths += 1
                        current_sample_hit = True

            if current_sample_hit:
                sample_success += 1

    print("\n" + "=" * 40)
    print("�� WEBQSP PATH ENCODING TEST (Sample 100)")
    print("=" * 40)
    print(f"Total Paths Checked:  {total_paths}")
    print(f"Paths with Features:  {hit_paths} ({hit_paths / total_paths * 100:.2f}%)")
    print(f"Samples with at least 1 Hit: {sample_success}/100")
    print("=" * 40)

    if hit_paths > 0:
        print("✅ 结论：路径特征编码完全可行！你可以拿到 GNN 向量。")
    else:
        print("❌ 结论：路径特征提取失败，请检查 entity2id 映射是否正确。")


if __name__ == "__main__":
    main()