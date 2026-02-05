import pickle
import os
import sys
import torch  # 导入 torch 以便查看 tensor 形状

# 你的文件路径
# PKL_PATH = "gnn-fet/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
PKL_PATH = "gnn-fet/0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"


def inspect_pickle():
    print(f"�� 正在读取文件: {PKL_PATH} ...")

    if not os.path.exists(PKL_PATH):
        print(f"❌ 错误: 文件不存在！请确认 'gnn-fet' 文件夹是否在当前目录下。")
        print(f"   当前工作目录: {os.getcwd()}")
        return

    try:
        with open(PKL_PATH, "rb") as f:
            data = pickle.load(f)

        print(f"✅ 读取成功！数据总条数: {len(data)}")

        # ================= 核心检查 =================
        target_str = "WebQTest-0"
        target_int = 0

        # 1. 检查字符串 Key
        if target_str in data:
            print(f"\n��【匹配成功】找到了字符串 Key: '{target_str}'")
            val = data[target_str]
            print(f"   数据类型: {type(val)}")
            print(f"   数据内容预览: {str(val)[:200]} ...")  # 只打印前200字符防止刷屏

            # 如果是字典，看看里面有多少个节点
            if isinstance(val, dict):
                print(f"   包含节点数量: {len(val)}")
                first_node_id = list(val.keys())[0]
                first_node_feat = val[first_node_id]
                # 尝试打印特征维度
                try:
                    if hasattr(first_node_feat, 'shape'):
                        print(f"   特征向量维度: {first_node_feat.shape}")
                    elif isinstance(first_node_feat, list):
                        print(f"   特征向量长度: {len(first_node_feat)}")
                except:
                    pass

        # 2. 检查数字 Key
        elif target_int in data:
            print(f"\n⚠️【注意】没找到 '{target_str}'，但找到了数字 Key: {target_int}")
            print(f"   这意味着我们在 predict 代码里需要把 ID 转成整数！")
            val = data[target_int]
            print(f"   数据类型: {type(val)}")
            if isinstance(val, dict):
                print(f"   包含节点数量: {len(val)}")

        else:
            print(f"\n❌ 既没找到 '{target_str}' 也没找到数字 {target_int}。")
            print(f"   前 5 个 Key 是: {list(data.keys())[:5]}")

    except Exception as e:
        print(f"❌ 读取出错: {e}")


if __name__ == "__main__":
    inspect_pickle()