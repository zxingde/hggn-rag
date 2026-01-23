import pickle
import numpy as np
import sys

# 你的文件路径
FILE_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"


def inspect_first_sample():
    print(f"�� 正在读取数据: {FILE_PATH} ...")
    try:
        with open(FILE_PATH, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("❌ 错误：找不到文件，请确认路径是否正确。")
        return

    print(f"✅ 读取成功！数据集共包含 {len(dataset)} 条样本。\n")

    # 取出第一条样本
    sample = dataset[0]

    print("=" * 60)
    print(f"�� 样本 ID: {sample.get('id', '未知')}")
    print("=" * 60)

    # 1. 检查 LLM 看到的文本 (Prompt)
    print("\n�� 【部分 1: LLM 看到的文本输入 (Prompt)】")
    print("-" * 30)
    full_text = sample.get('text', '')
    # 只打印前 500 个字符避免刷屏，通常路径都在前面
    print(full_text[:1000] + " ... [省略后续]")
    print("-" * 30)

    # 2. 检查后台绑定的 GNN 数据
    print("\n�� 【部分 2: 后台绑定的 GNN 特征数据】")
    path_data = sample.get('gnn_path_data', [])
    print(f"该样本共包含 {len(path_data)} 条路径数据。")

    # 打印前 3 条路径详情
    for i, item in enumerate(path_data[:3]):
        print(f"\n--- Path #{i + 1} ---")
        # 路径文本
        path_str = item.get('path', 'N/A')
        print(f"路径文本: \"{path_str}\"")

        # 向量特征
        vec = item.get('feature')
        if vec is None:
            print("特征向量: ❌ 缺失!")
            continue

        shape = vec.shape
        is_zero = np.all(vec == 0)

        print(f"向量维度: {shape}")
        if is_zero:
            print("向量数值: ⚠️ 全 0 (未命中子图或无特征)")
        else:
            # 打印前 5 位数字验证
            print(f"向量数值: ✅ 有值 (前5位: {vec[:5]})")

    print("\n" + "=" * 60)
    print("�� 检查要点：")
    print("1. ‘路径文本’ 里的内容是否和上方 Prompt 里的 Reasoning Paths 一致？")
    print("2. ‘向量数值’ 是否大部分是 ‘✅ 有值’？(如果是 ✅，说明特征缝合成功)")


if __name__ == "__main__":
    inspect_first_sample()