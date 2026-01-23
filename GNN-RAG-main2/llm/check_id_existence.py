import json
import pickle
import sys
from tqdm import tqdm

# ================= 配置 =================
# 1. 你的源数据文件 (AlignData, 里面带有 "id" 字段)
SOURCE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/AlignData/RoG-cwq/RoG-cwq_train.jsonl"

# 2. 你的特征缓存文件 (7.8GB 那个)
FEATURE_CACHE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"


def verify_ids():
    print(f"Loading Feature Cache from {FEATURE_CACHE_FILE} ...")
    try:
        with open(FEATURE_CACHE_FILE, 'rb') as f:
            # 我们只需要 Key 列表，不需要加载巨大的 Value，这样会快很多
            # 但是 pickle load 必须一次性读完，所以内存还是会占
            gnn_data = pickle.load(f)
            gnn_keys = set(gnn_data.keys())  # 转成 set 集合，查询速度 O(1)
            del gnn_data  # 释放内存
    except Exception as e:
        print(f"Error loading cache: {e}")
        return

    print(f"✅ Cache Loaded. Total Graphs: {len(gnn_keys)}")

    print(f"\nScanning Source File: {SOURCE_FILE} ...")

    total_count = 0
    found_count = 0
    missing_samples = []

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line: continue

            try:
                item = json.loads(line)
                total_count += 1

                # 直接获取 ID
                target_id = item.get('id')

                if not target_id:
                    print(f"⚠️ Warning: Line {total_count} has no 'id' field!")
                    continue

                # === 核心校验 ===
                if target_id in gnn_keys:
                    found_count += 1
                else:
                    if len(missing_samples) < 5:
                        missing_samples.append(target_id)
            except json.JSONDecodeError:
                continue

    # === 最终报告 ===
    print("\n" + "=" * 40)
    print("�� ID COMPLETENESS REPORT")
    print("=" * 40)
    print(f"Total Items in Dataset: {total_count}")
    print(f"IDs Found in GNN Cache: {found_count}")

    if total_count > 0:
        rate = (found_count / total_count) * 100
        print(f"✅ Match Rate: {rate:.2f}%")

    print("-" * 40)

    if found_count == total_count:
        print("�� PERFECT! All IDs are present.")
        print("结论：之前匹配率低是因为'文本匹配'不准确。")
        print("解决：后续代码请直接利用行号对齐或 ID 索引，不要用 Question 文本去查。")
    else:
        print(f"❌ Missing {total_count - found_count} IDs.")
        print("Sample missing IDs:")
        for mid in missing_samples:
            print(f" - {mid}")
        print("\n可能原因：GNN 预训练时用的数据集 Split 和现在的 AlignData 不一致。")


if __name__ == "__main__":
    verify_ids()