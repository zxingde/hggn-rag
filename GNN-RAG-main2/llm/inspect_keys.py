import pickle
import sys

# ================= 配置 =================
# 1. 你的特征文件 (7.8GB 那个新文件)
FEATURE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"


# 2. 你想检查的那个问题的 QID
# 这是 Sample 0 (华盛顿大学那个问题) 的 ID
TARGET_QID = "WebQTrn-3513_7c4117891abf63781b892537979054c6"

# 如果你想查 Sample 1 (Super Bowl) 的 ID，请取消下面这行的注释
# TARGET_QID = "WebQTrn-2136_d95da5fb8a16d81fe56cd4ce00843254"

# 3. 你确信“应该在里面”的那个实体 ID (Global ID)
# 比如华盛顿特区是 14，Super Bowl XI 是 912285
TARGET_GID = 14


# =======================================

def inspect():

    print(f"Loading cache from {FEATURE_FILE} ...")
    try:
        with open(FEATURE_FILE, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total Graphs: {len(data)}")

    if TARGET_QID not in data:
        print(f"❌ QID {TARGET_QID} NOT FOUND in cache keys!")
        return

    # 拿到子图
    subgraph = data[TARGET_QID]
    keys = list(subgraph.keys())

    print(f"\n=== Inspecting Subgraph for QID: {TARGET_QID} ===")
    print(f"Total Nodes in this subgraph: {len(keys)}")

    if len(keys) == 0:
        print("⚠️ This subgraph is empty!")
        return

    # 1. 检查 Key 的具体类型
    first_key = keys[0]
    print(f"Key Type: {type(first_key)} (Value: {first_key})")

    # 2. 打印前 50 个 Key 让你看看长啥样
    print("\nFirst 50 Keys (Sorted):")
    sorted_keys = sorted(keys)
    print(sorted_keys[:50])

    # 3. 针对性查找
    print(f"\nLooking for Target GID: {TARGET_GID} ...")

    # A. 直接查整数
    if TARGET_GID in subgraph:
        print(f"✅ FOUND exact match: {TARGET_GID} (Type: {type(TARGET_GID)})")
    else:
        print(f"❌ NOT FOUND as integer {TARGET_GID}")

    # B. 查字符串 (防止类型错误)
    str_gid = str(TARGET_GID)
    is_str_in = False
    for k in keys:
        if str(k) == str_gid:
            print(f"⚠️ FOUND as mismatched type! Key in dict is: {k} (Type: {type(k)})")
            is_str_in = True
            break

    if not is_str_in and TARGET_GID not in subgraph:
        print("\n结论: 真的不在里面。既不是 int 也不是 str。")
        print("解释: 这说明 GNN 在构建该问题的子图时，确实把这个节点截断(Drop)掉了。")


if __name__ == "__main__":
    inspect()