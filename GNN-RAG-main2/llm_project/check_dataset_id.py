from datasets import load_dataset
import os
import argparse


def check_dataset(data_path, dataset_name, split):
    # 模拟 predict_answer.py 的路径拼接逻辑
    # 如果是 HuggingFace 数据集，这里拼出来应该是类似 "rmanluo/RoG-webqsp"
    input_path = os.path.join(data_path, dataset_name)

    print(f"�� 正在尝试加载数据集: {input_path} (split={split}) ...")

    try:
        # 加载数据集
        dataset = load_dataset(input_path, split=split)
        print(f"✅ 加载成功！共包含 {len(dataset)} 条数据。")

        if len(dataset) > 0:
            # 获取第一条数据
            sample = dataset[0]

            print("\n--- �� 第一条样本详情 ---")
            print(f"所有字段 (Keys): {list(sample.keys())}")

            # 检查是否有 'id'
            if 'id' in sample:
                print(f"✅ 字段检查: 包含 'id' 属性。")
                print(f"�� id 示例值: {sample['id']} (类型: {type(sample['id'])})")

                # 额外检查：ID是否和你的GNN特征文件的Key格式一致？
                # 你的 GNN 特征文件里的 Key 是 "WebQTrn-0" 这种格式
                print("�� 提示: 请确认这个 ID 格式是否与你的 .pkl 文件中的 Key 一致？")
            else:
                print(f"❌ 字段检查: 未找到 'id' 属性！")
                # 帮你在现有字段里找找看有没有类似的
                potential_ids = [k for k in sample.keys() if 'id' in k.lower()]
                if potential_ids:
                    print(f"❓ 可能的替代字段: {potential_ids}")
                    for pid in potential_ids:
                        print(f"   - {pid}: {sample[pid]}")
        else:
            print("⚠️ 警告: 数据集是空的！")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("�� 如果是网络问题连接不上 HuggingFace，请检查网络或是否配置了代理。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认值参考了你的 predict_answer.py
    parser.add_argument("--data_path", type=str, default="rmanluo", help="Data path or organization")
    parser.add_argument("--d", type=str, default="RoG-webqsp", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Split to check")

    args = parser.parse_args()

    check_dataset(args.data_path, args.d, args.split)