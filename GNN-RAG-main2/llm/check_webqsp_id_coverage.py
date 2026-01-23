import json
import pickle
import os
import sys
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你指定的 WebQSP QA 数据集文件 (包含 "id": "WebQTrn-12")
QA_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl"

# 2. WebQSP 的 GNN 特征文件
# 【注意】这个路径是根据你之前的命令行参数推断的，请检查文件名是否完全一致！
FEATURE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"


# ===========================================

def check_coverage():
    print(f">>> 1. Checking Feature File...")
    if not os.path.exists(FEATURE_FILE):
        print(f"❌ Error: Feature file not found at: {FEATURE_FILE}")
        print("   请确认你是否已经修复了 rearevhgnn.py 的报错并重新运行了 main.py 导出命令？")
        return

    print(f"   Loading GNN Cache from {FEATURE_FILE} ...")
    try:
        with open(FEATURE_FILE, 'rb') as f:
            # 只加载 Keys 以节省内存和时间
            data = pickle.load(f)
            gnn_keys = set(data.keys())
            # 顺便检查一下 Key 的格式
            sample_key = list(gnn_keys)[0] if gnn_keys else "None"
            print(f"   ✅ Cache Loaded. Total Graphs: {len(gnn_keys)}")
            print(f"   Sample Key format: {sample_key} (Type: {type(sample_key)})")
            del data
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        return

    print(f"\n>>> 2. Scanning QA Dataset...")
    total_count = 0
    match_count = 0
    missing_samples = []

    with open(QA_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line: continue

            try:
                item = json.loads(line)
                total_count += 1

                # 获取 ID
                qid = item.get('id')

                if not qid:
                    # 某些格式可能 id 字段名不同，或者没有 id
                    continue

                # === 核心匹配 ===
                # WebQSP 的 ID 通常格式为 "WebQTrn-xxx"
                # 确保格式一致（去空格）
                clean_qid = str(qid).strip()

                if clean_qid in gnn_keys:
                    match_count += 1
                else:
                    if len(missing_samples) < 5:
                        missing_samples.append(clean_qid)

            except json.JSONDecodeError:
                continue

    # === 输出报告 ===
    print("\n" + "=" * 40)
    print("�� WEBQSP COVERAGE REPORT")
    print("=" * 40)
    print(f"Total QA Samples:      {total_count}")
    print(f"Matched in GNN Cache:  {match_count}")

    if total_count > 0:
        rate = (match_count / total_count) * 100
        print(f"✅ Coverage Rate:       {rate:.2f}%")
    else:
        print("❌ No samples found in QA file.")

    print("-" * 40)

    if match_count == total_count:
        print("�� PERFECT MATCH! 所有 ID 都能找到对应特征。")
    else:
        print(f"❌ Missing {total_count - match_count} IDs.")
        if missing_samples:
            print(f"Sample missing IDs: {missing_samples}")
            print("可能原因：")
            print("1. GNN 特征提取脚本跑挂了，没有跑完所有数据。")
            print("2. 数据集切分（Split）不一致（GNN 用的是旧版，QA 是新版）。")
            print("3. ID 格式有细微差异（如大小写）。")


if __name__ == "__main__":
    check_coverage()