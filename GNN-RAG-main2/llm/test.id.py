import json
import re
import pickle
import sys

# ================= 配置区域 (保持不变) =================
ORIGINAL_QA_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/AlignData/RoG-cwq/RoG-cwq_train.jsonl"
LLM_TRAIN_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/joint_training/qa_back/RoG-cwq/RoG-cwq_train.jsonl"
FEATURE_CACHE_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/checkpoint/pretrain/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
ENTITY_MID_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/gnn/data/CWQ/entities.txt"
ENTITY_NAMES_FILE = "/home/bi3/zxd_env/GNN-RAG-main2/llm/entities_names.json"


# ================= 调试辅助函数 =================

def load_resources_debug():
    print(">>> [DEBUG] Loading Resources...")

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
    name2mid = {v.strip(): k for k, v in mid2name.items()}  # Strip一下防止有空格
    print(f"    Loaded {len(name2mid)} names. Sample: 'Justin Bieber' -> {name2mid.get('Justin Bieber', 'Not Found')}")

    # 3. MID to ID
    mid2id = {}
    with open(ENTITY_MID_FILE, 'r') as f:
        for idx, line in enumerate(f):
            mid2id[line.strip()] = idx
    print(f"    Loaded {len(mid2id)} mids.")

    # 4. Cache (只加载 Keys 检查一下，避免加载大文件太慢？不行，还得查里面)
    # 既然你已经跑过了，说明内存够，直接加载吧
    print("    Loading Cache (waiting)...")
    with open(FEATURE_CACHE_FILE, 'rb') as f:
        gnn_cache = pickle.load(f)
    print(f"    Cache Loaded. Keys count: {len(gnn_cache)}")
    print(f"    Sample Cache Key: {list(gnn_cache.keys())[0]}")

    return q2id, name2mid, mid2id, gnn_cache


def debug_process():
    q2id, name2mid, mid2id, gnn_cache = load_resources_debug()

    print("\n>>> [DEBUG] Starting Line-by-Line Inspection (First 5 samples) <<<\n")

    with open(LLM_TRAIN_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5: break  # 只看前5个

            print(f"=== Sample {i} ===")
            item = json.loads(line)
            text = item['text']

            # 1. Question Parsing
            match = re.search(r"Question:\n(.*?)\s*\[/INST\]", text, re.DOTALL)
            q_text = match.group(1).strip() if match else None

            print(f"  [1] Question Text: '{q_text}'")

            if not q_text:
                print("      -> FAILED: Could not parse question.")
                continue

            # 2. ID Matching
            qid = q2id.get(q_text)
            print(f"  [2] QID Lookup: {qid}")

            if not qid:
                # 尝试去问号
                qid_loose = q2id.get(q_text.replace("?", ""))
                print(f"      -> Loose Lookup (no ?): {qid_loose}")
                if qid_loose: qid = qid_loose

            if not qid:
                print("      -> FAILED: QID not found in map.")
                # 打印 q2id 的前几个 key 对比一下
                # print(f"      (Debug) First 3 keys in q2id: {list(q2id.keys())[:3]}")
                continue

            # 3. Cache Lookup
            in_cache = qid in gnn_cache
            print(f"  [3] In GNN Cache? {in_cache}")

            if not in_cache:
                print(f"      -> FAILED: QID '{qid}' not in feature cache keys.")
                continue

            # 4. Path Parsing
            path_match = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.DOTALL)
            paths = [l.strip() for l in path_match.group(1).split('\n') if l.strip()] if path_match else []
            print(f"  [4] Parsed {len(paths)} paths.")

            graph_features = gnn_cache[qid]  # Get the subgraph
            # === 【新增调试代码：这是破案的关键】 ===
            print(f"      [DEBUG Subgraph info]")
            print(f"      - Total nodes in this subgraph: {len(graph_features)}")

            # 1. 检查 Key 的类型
            first_key = list(graph_features.keys())[0]
            print(f"      - Key Type: {type(first_key)} (Expect: int)")
            print(f"      - Sample Keys (first 10): {list(graph_features.keys())[:10]}")

            # 2. 暴力搜索：如果你坚信 14 在里面，我们遍历一遍看看有没有类似的
            target_gid = 14
            if target_gid in graph_features:
                print(f"      - ✅ AMAZING: {target_gid} IS in keys! (Why did it fail before?)")
            else:
                print(f"      - ❌ CONFIRMED: {target_gid} is NOT in keys.")
                # 看看有没有字符串类型的 '14'
                if str(target_gid) in graph_features:
                    print(f"      - ⚠️ BUT string '{target_gid}' IS found! (Type mismatch error)")
            # ==========================================

            for p_idx, p in enumerate(paths):
                tail = p.split("->")[-1].strip()
                print(f"      Path #{p_idx} Tail: '{tail}'")

                # 5. Entity to MID
                mid = name2mid.get(tail)
                print(f"        -> MID: {mid}")

                if not mid:
                    print("        -> FAILED: Name not in name2mid.")
                    continue

                # 6. MID to GID
                gid = mid2id.get(mid)
                print(f"        -> GID: {gid}")

                if gid is None:
                    print("        -> FAILED: MID not in entities.txt.")
                    continue

                # 7. GID in Graph
                has_feat = gid in graph_features
                print(f"        -> In Subgraph? {has_feat}")


if __name__ == "__main__":
    debug_process()