import pickle
import numpy as np
from collections import Counter
import os

# 配置你的文件路径
INPUT_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"


def inspect_data():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到文件: {INPUT_PATH}")
        return

    print(f"正在读取文件: {INPUT_PATH} ...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    total_count = len(data)
    print(f"✅ 读取成功，共 {total_count} 条样本。\n")

    # 统计器
    shape_counter = Counter()
    type_counter = Counter()
    abnormal_samples = []  # 存几个异常样本看看长啥样

    print("正在扫描每一条数据的形状...")

    for idx, item in enumerate(data):
        if 'gnn_path_data' not in item:
            print(f"⚠️ 样本 ID {idx} 缺少 'gnn_path_data' 字段！")
            continue

        raw_feat = item['gnn_path_data']

        # 1. 记录原始类型
        type_name = type(raw_feat).__name__
        type_counter[type_name] += 1

        # 2. 转 numpy 看形状
        # 注意：这里只转不改，为了看清楚它到底是个啥
        try:
            arr = np.array(raw_feat)
            shape_str = str(arr.shape)
            shape_counter[shape_str] += 1

            # 3. 捕捉异常（不是2维的）
            # 我们期望的是 (N, 50)，即 ndim=2
            if arr.ndim != 2:
                # 记录前 5 个异常样本的详情
                if len(abnormal_samples) < 5:
                    abnormal_samples.append({
                        "id": idx,
                        "type": type_name,
                        "shape": arr.shape,
                        "content": raw_feat  # 看看原始数据内容
                    })
        except Exception as e:
            print(f"❌ 样本 ID {idx} 无法转为 numpy: {e}")

    # === 输出诊断报告 ===
    print("\n" + "=" * 50)
    print("��【诊断报告】")
    print("=" * 50)

    print(f"\n1. 数据类型分布 (期望是 list 或 ndarray):")
    for t, c in type_counter.items():
        print(f"   - {t}: {c} 条")

    print(f"\n2. 数据形状分布 (期望是 (N, 50)):")
    # 按数量降序排列
    for shape, count in shape_counter.most_common():
        status = "✅ 正常" if len(eval(shape)) == 2 else "❌ 异常 (导致报错的元凶)"
        print(f"   - Shape {shape}: {count} 条 \t{status}")

    print(f"\n3. 异常样本采样 (前 {len(abnormal_samples)} 个):")
    if not abnormal_samples:
        print("   （无异常样本）")
    else:
        for s in abnormal_samples:
            print(f"   ▶ ID: {s['id']}")
            print(f"     类型: {s['type']}")
            print(f"     形状: {s['shape']}")
            # 如果内容太多，只打印一部分
            content_str = str(s['content'])
            if len(content_str) > 200:
                content_str = content_str[:200] + "...(省略)"
            print(f"     内容: {content_str}")
            print("   " + "-" * 30)

    print("\n" + "=" * 50)
    print("�� 结论与建议：")

    # 简单的自动分析
    abnormal_shapes = [s for s in shape_counter.keys() if len(eval(s)) != 2]
    if not abnormal_shapes:
        print("数据格式完美，全是 2D 矩阵。如果之前报错，可能是之前文件搞错了。")
    else:
        print(f"发现 {len(abnormal_shapes)} 种异常形状。")
        if '(50,)' in shape_counter:
            print(" -> 发现 Shape (50,)：这说明只有 1 条路径时，维度被压缩了 (Squeeze)。")
            print("    建议：使用 np.expand_dims(x, 0) 修复。")
        if '(0,)' in shape_counter:
            print(" -> 发现 Shape (0,)：这说明是空列表 []。")
            print("    建议：该样本没有任何路径，需要填充一个全 0 向量或者跳过。")


if __name__ == "__main__":
    inspect_data()