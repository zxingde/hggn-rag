import json
import os
from datasets import load_dataset

# ================= 配置 =================
# 作者的数据集源头 (Hugging Face ID)
# 根据你之前用的文件夹名 RoG-cwq，作者对应的 HF 数据集应该是这个：
# DATASET_NAME = "rmanluo/RoG-WebQSP"
DATASET_NAME = "rmanluo/RoG-webqsp"
# 输出文件路径
# OUTPUT_FILE = "llm/datasets/joint_training/align/RoG-cwq/test.jsonl"
OUTPUT_FILE = "llm/datasets/joint_training/align/RoG-webqsp/test.jsonl"

def build_test_data():
    print(f"正在尝试从 Hugging Face 加载数据集: {DATASET_NAME} ...")

    try:
        # 核心逻辑：模仿作者代码，但这次我们要 'test' split
        dataset = load_dataset(DATASET_NAME, split='test')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print(" 提示: 如果服务器没网，这步会报错。但这是作者原本的获取方式。")
        return

    print(f"✅ 加载成功! 共找到 {len(dataset)} 条测试数据。")
    print("正在提取 ID 和 Question ...")

    processed_data = []
    for item in dataset:
        # 这里的 key 取决于 HF 数据集的实际结构，通常是 'id' 或 'question'
        # 我们做个兼容处理
        q_id = item.get('id') or item.get('ID')
        question = item.get('question')

        if q_id and question:
            entry = {
                "id": q_id,
                "question": question,
                # 保留 topic_entity 以防万一
                "topic_entity": item.get('topic_entity', {})
            }
            processed_data.append(entry)

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + "\n")

    print(f"�� 已自动生成测试集: {OUTPUT_FILE}")
    print("�� 现在可以直接运行 inference 脚本了！")


if __name__ == "__main__":
    build_test_data()