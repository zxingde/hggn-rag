import pickle
import numpy as np
import sys

# ================= 配置 =================
# 你的特征文件路径
FEATURE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"


# =======================================

def check_id_type():
    print(f"Loading cache from {FEATURE_FILE} ...")
    try:
        with open(FEATURE_FILE, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    print(f"Total Graphs loaded: {len(data)}")

    # 我们随机抽查 5 个非空的子图来验证
    sample_count = 0
    local_id_suspects = 0
    global_id_suspects = 0

    for qid, subgraph in data.items():
        if sample_count >= 5: break

        node_keys = list(subgraph.keys())
        num_nodes = len(node_keys)

        if num_nodes < 10: continue  # 跳过太小的图，看不出规律

        sample_count += 1

        # 获取统计信息
        max_key = max(node_keys)
        min_key = min(node_keys)

        print(f"\n=== Sample Graph: {qid} ===")
        print(f"  - Node Count: {num_nodes}")
        print(f"  - Min Key: {min_key}")
        print(f"  - Max Key: {max_key}")
        print(f"  - First 10 Keys: {sorted(node_keys)[:10]}")

        # === 核心判断逻辑 ===
        # 如果存的是 Local ID，那么 Max Key 应该等于 (Node Count - 1)
        # 比如有 419 个节点，最大 ID 应该是 418

        if max_key == num_nodes - 1 and min_key == 0:
            print("  ⚠️  JUDGMENT: This looks like [LOCAL ID] (0 to N-1).")
            local_id_suspects += 1
        elif max_key >= num_nodes:
            print("  ✅  JUDGMENT: This looks like [GLOBAL ID] (Large numbers).")
            global_id_suspects += 1
        else:
            print("  ❓  JUDGMENT: Ambiguous (Rare case).")

    print("\n" + "=" * 30)
    print("FINAL CONCLUSION:")
    if local_id_suspects > 0 and global_id_suspects == 0:
        print("�� 实锤了！你存的是【Local ID】！")
        print("原因：Key 的最大值严格等于节点数减一，且从0开始连续。")
        print("后果：之前的匹配逻辑全错了，因为你在拿全局 ID 去查局部索引。")
    elif global_id_suspects > 0:
        print("✅ 验证通过！你存的是【Global ID】。")
        print("原因：Key 中包含比节点数大得多的数值。")
    else:
        print("无法确定，请检查数据。")


if __name__ == "__main__":
    check_id_type()