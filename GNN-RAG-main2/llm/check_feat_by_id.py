import torch
import os
import pickle  # <--- 必须导入这个

# ================= 配置区域 =================
# 1. WebQSP / CWQ 的特征文件路径
# 注意：看你的路径，两个数据集好像用了同一个文件？请确认这是否符合你的预期。
cwq_feat_path = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/feature/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_all_features.pkl"
webqsp_feat_path = "/home/bi3/zxd_env/GNN-RAG-main2/llm/datasets/feature/0117-HGNN-WebQSP_LMSR_BS24_4090_EXPORT_all_features.pkl"

# ================= 要测试的目标 ID =================
test_targets = [
    {
        "dataset": "WebQSP",
        "path": webqsp_feat_path,
        "id": "WebQTrn-0",
        "question": "what is the name of justin bieber brother"
    },
    {
        "dataset": "CWQ",
        "path": cwq_feat_path,
        "id": "WebQTrn-3513_7c4117891abf63781b892537979054c6",
        "question": "What state is home to the university..."
    }
]


def load_file_smart(file_path):
    """
    智能加载函数：先尝试 pickle，如果不行尝试 torch.load
    """
    with open(file_path, 'rb') as f:
        try:
            # 1. 优先尝试标准 pickle 加载 (针对 .pkl 文件)
            return pickle.load(f)
        except Exception as e_pickle:
            # 如果 pickle 失败，指针归位，尝试 torch.load
            f.seek(0)
            try:
                # 2. 尝试 PyTorch 加载 (针对 .pt / .bin 文件)
                return torch.load(f, map_location='cpu')
            except Exception as e_torch:
                raise RuntimeError(
                    f"加载失败！既不是 pickle 格式也不是 torch 格式。\nPickle报错: {e_pickle}\nTorch报错: {e_torch}")


def check_feature(name, file_path, target_id):
    print(f"\n{'=' * 10} 正在检查 {name} 数据集 {'=' * 10}")
    print(f"�� 文件路径: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在！请检查路径配置。")
        return

    try:
        print("⏳ 正在加载文件 (可能需要几秒钟)...")

        # 【修改点】使用智能加载函数
        features = load_file_smart(file_path)

        # 打印一下总共有多少条数据
        print(f"�� 文件加载成功，共包含 {len(features)} 条特征。")

        # 打印前 3 个 Key，看看长什么样 (帮你确认 ID 格式)
        first_keys = list(features.keys())[:3]
        print(f"�� 前 3 个 ID 示例: {first_keys}")

        # 核心检查
        if target_id in features:
            vec = features[target_id]
            print(f"\n✅【成功找到】ID: {target_id}")

            # 兼容性处理：如果取出来是 Tensor
            if isinstance(vec, torch.Tensor):
                print(f"   向量形状: {vec.shape}")
                print(f"   向量类型: {vec.dtype}")
                print(f"   前5位数值: {vec.view(-1)[:5].tolist()}")
            else:
                # 如果是 Numpy 数组或其他
                print(f"   数据类型: {type(vec)}")
                print(f"   内容概览: {vec}")
        else:
            print(f"\n❌【未找到】ID: {target_id}")
            print("   �� 这个 ID 不在特征文件中。")
            print("   �� 请对比上面的【前 3 个 ID 示例】，看看格式哪里不对？")
            print("      (例如：是你用的 'WebQTrn-0'，但文件里存的是 '0'？)")

    except Exception as e:
        print(f"❌ 读取过程发生严重错误: {e}")


if __name__ == "__main__":
    for target in test_targets:
        check_feature(target['dataset'], target['path'], target['id'])