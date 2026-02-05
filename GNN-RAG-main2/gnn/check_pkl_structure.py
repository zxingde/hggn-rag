import pickle
import os
import numpy as np
import torch

# ⚠️ 请确保路径和你生成的文件完全一致
file_path = "checkpoint/pretrain/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"

print(f"�� 正在检查文件: {file_path} ...")

if not os.path.exists(file_path):
    print(f"❌ 错误：找不到文件，请检查路径是否正确！")
    # 尝试在当前目录下找找看
    print(f"   当前目录下的文件: {os.listdir('.')}")
else:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✅ 加载成功！")
        print(f"�� 总数据量 (问题数): {len(data)}")

        if len(data) == 0:
            print("❌ 警告：文件是空的！")
        else:
            # 取出第一个样本进行解剖
            first_qid = list(data.keys())[0]
            content = data[first_qid]

            print(f"\n--- �� 样本解剖 (QID: {first_qid}) ---")
            print(f"数据类型: {type(content)}")

            # 【关键判断逻辑】
            if isinstance(content, dict):
                print(f"�� 恭喜！这是【节点粒度】(Node Granularity) 的数据！")
                print(f"   结构说明: {{ Entity_ID (int) : Feature_Vector (numpy) }}")
                print(f"   该子图包含节点数: {len(content)}")

                # 打印前2个节点看看
                print(f"   �� 节点示例:")
                for i, (eid, vec) in enumerate(content.items()):
                    if i >= 2: break
                    print(f"      - 实体ID: {eid} (类型: {type(eid)}) | 向量形状: {vec.shape}")

            elif isinstance(content, (torch.Tensor, np.ndarray)):
                print(f"⚠️ 注意：这是【张量粒度】(Tensor Granularity) 的数据。")
                print(f"   形状: {content.shape}")
                print(f"   ❌ 这种格式本身**不包含**实体ID信息，你无法直接知道第几行对应哪个实体。")
                print(f"      (除非你同时配合 dataset 中的 local_entity 字段使用)")
            else:
                print(f"❓ 未知数据格式: {type(content)}")

    except Exception as e:
        print(f"❌ 读取出错: {e}")