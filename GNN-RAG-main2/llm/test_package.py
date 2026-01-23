import pickle
import numpy as np

# 文件路径 (请确保路径和你生成的一致)
pkl_path = "final_finetune_corpus_webqsp.pkl"

print(f"正在读取 {pkl_path} ...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"✅ 读取成功！总样本数: {len(data)}")

# 取出第一个样本解剖一下
sample = data[0]
print("\n=== 样本示例 (第1条) ===")
print(f"ID: {sample.get('id')}")
print(f"Text (前50字符): {sample.get('text')[:50]}...")

print("\n=== GNN 路径特征 ===")
if 'gnn_path_data' in sample:
    paths = sample['gnn_path_data']
    print(f"该样本包含路径数: {len(paths)}")

    for i, p in enumerate(paths):
        feat = p['feature']
        status = "✅ 命中" if p['hit'] else "❌ 未命中"
        # 打印前 5 维看看是不是真的有数字
        print(f"  路径 {i + 1}: {status} | 维度: {feat.shape} | 前5位数值: {feat[:5]}")
else:
    print("❌ 警告：gnn_path_data 字段缺失！")