import pickle
import numpy as np
import os

# 请替换为你文件的实际路径
file_path = "gnn-fet/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"

print(f"正在尝试加载: {file_path} ...")

if not os.path.exists(file_path):
    print("❌ 错误: 文件不存在，请检查路径！")
else:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✅ 加载成功！数据类型: {type(data)}")

        if isinstance(data, dict):
            print(f"包含的问题数量 (Keys Count): {len(data)}")

            # 1. 检查第一层 Key (Question ID)
            first_qid = list(data.keys())[0]
            print(f"\n--- 样本示例 ---")
            print(f"Question ID (Key): {first_qid} (类型: {type(first_qid)})")

            # 2. 检查第一层 Value
            val = data[first_qid]
            print(f"Value 类型: {type(val)}")

            if isinstance(val, dict):
                print(f"  └─ 这是一个字典 (Entity Map)，包含 {len(val)} 个实体")

                # 3. 检查第二层 (Entity ID -> Vector)
                first_eid = list(val.keys())[0]
                first_vec = val[first_eid]
                print(f"  └─ 内部 Key (Entity ID): {first_eid} (类型: {type(first_eid)})")
                print(f"  └─ 内部 Value (Vector): 类型 {type(first_vec)}")

                if hasattr(first_vec, 'shape'):
                    print(f"  └─ 向量形状: {first_vec.shape}")
            elif hasattr(val, 'shape'):
                # 也有可能是直接存了 Tensor/Array，视具体代码版本而定
                print(f"  └─ 这是一个数组/张量，形状: {val.shape}")

    except Exception as e:
        print(f"❌ 读取发生错误: {e}")