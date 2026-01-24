import pickle
import numpy as np
import os
from tqdm import tqdm

# ================= 配置 =================
# 输入文件路径
INPUT_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"
# 输出文件路径 (新文件)
OUTPUT_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp_gaussian.pkl"

# 高斯分布参数
# 均值 0，方差 0.01 (给一点点抖动，避免数值下溢，但又不至于掩盖真实特征)
NOISE_MEAN = 0.0
NOISE_STD = 0.01


# ========================================

def replace_zeros(data):
    print(f"正在加载数据: {INPUT_PATH} ...")

    total_zeros_replaced = 0
    samples_affected = 0

    # 遍历所有样本
    for item in tqdm(data, desc="处理中"):
        # 你的特征存在 'gnn_path_data' 这个字段里
        if 'gnn_path_data' in item:
            feats = item['gnn_path_data']

            # 确保是 numpy 数组
            if isinstance(feats, list):
                feats = np.array(feats)

            # feats 形状应该是 [N, 50]
            # 1. 找到所有全 0 的行
            # abs(sum) == 0 是为了处理潜在的负数抵消(虽然不太可能)，更稳妥的是 all(axis=1)
            is_zero_row = np.all(feats == 0, axis=1)

            num_zeros = np.sum(is_zero_row)

            if num_zeros > 0:
                samples_affected += 1
                total_zeros_replaced += num_zeros

                # 2. 生成同形状的高斯噪声
                noise = np.random.normal(
                    loc=NOISE_MEAN,
                    scale=NOISE_STD,
                    size=(num_zeros, feats.shape[1])
                )

                # 3. 替换！
                feats[is_zero_row] = noise

                # 4. 更新回字典
                # 注意：转回 list 还是保持 numpy 取决于你之前的代码习惯
                # 为了兼容性，这里我们存回 numpy (DataCollator 一般能处理)
                # 或者转回 list: feats.tolist()
                item['gnn_path_data'] = feats

    print("\n" + "=" * 40)
    print(f"✅ 处理完成！")
    print(f"受影响的样本数: {samples_affected} / {len(data)}")
    print(f"被替换的 0 向量总数: {total_zeros_replaced}")
    print("=" * 40)

    return data


if __name__ == "__main__":
    # 1. 读取
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    # 2. 处理
    new_data = replace_zeros(data)

    # 3. 保存
    print(f"正在保存到新文件: {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(new_data, f)
    print("�� 保存成功！请在训练脚本中使用新的 pkl 文件路径。")