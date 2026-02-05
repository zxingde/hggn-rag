import sys
import os
import json
import pickle
import numpy as np
import torch
import string
import re
from datasets import load_dataset

# ==========================================
# 1. 环境配置与工具函数
# ==========================================
# 假设你在项目根目录下运行，需要把 src 加入路径
sys.path.append(os.path.abspath("llm_project/src"))

# 尝试导入 PromptBuilder
try:
    from qa_prediction.build_qa_input import PromptBuilder
except ImportError:
    print("❌ 错误：找不到 PromptBuilder，请确保你在项目根目录运行，且 'llm_project/src' 路径正确。")
    sys.exit(1)


def normalize(s: str) -> str:
    """文本归一化工具"""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


# ==========================================
# 2. 加载真实资源
# ==========================================
print("�� [1/5] 正在加载实体映射 (entities_names.json)...")
try:
    with open('entities_names.json', 'r') as f:
        entities_names = json.load(f)
    print(f"   ✅ 加载成功，共 {len(entities_names)} 个实体。")
except FileNotFoundError:
    print("❌ 找不到 entities_names.json，请确认文件位置。")
    sys.exit(1)

print("�� [2/5] 正在加载图特征 (pkl)...")
feat_path = "gnn-fet/0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
try:
    with open(feat_path, 'rb') as f:
        graph_features = pickle.load(f)
    print(f"   ✅ 加载成功，包含 {len(graph_features)} 个问题的特征。")
except FileNotFoundError:
    print(f"❌ 找不到 {feat_path}，请确认路径。")
    sys.exit(1)

# ==========================================
# 3. 准备数据样本 (WebQTest-0)
# ==========================================
target_id = "WebQTest-0"
print(f"�� [3/5] 准备测试数据: {target_id}...")

# 检查特征库里有没有这个 ID
if target_id not in graph_features:
    print(f"❌ 悲剧了：特征文件中没有 {target_id} 的数据，无法测试。请换一个 ID。")
    sys.exit(1)

# 获取该问题的子图节点字典
node_dict = graph_features[target_id]
print(f"   ✅ 特征库命中！该问题子图包含 {len(node_dict)} 个节点。")

# --- 偷看一眼答案，找一个真实存在的实体用于测试 ---
# 我们从 node_dict 里随便挑一个实体 ID，查出它的名字，假装这是预测出的路径
test_eid = list(node_dict.keys())[5]  # 挑第6个节点
test_eid_str = str(test_eid)
if test_eid_str in entities_names:
    test_name = entities_names[test_eid_str]
    print(f"   �� 选定测试目标实体: ID={test_eid} | Name='{test_name}'")
else:
    test_name = "Unknown Entity"
    print(f"   ⚠️ 选定的 ID {test_eid} 在 entities_names 里查不到名字，可能导致匹配失败。")

# 构造一个模拟的数据对象 (模拟 dataset[0])
# 我们手动注入 predicted_paths，看看 PromptBuilder 会不会把它拼进去
data_sample = {
    "id": target_id,
    "question": "What is the capital of China?",  # 随便写个问题
    "q_entity": [],  # 暂时为空
    "predicted_paths": [
        f"SomeEntity -> relation -> {test_name}",  # <--- 把目标实体塞进路径！
        "Beijing -> is -> capital"
    ]
}

# ==========================================
# 4. 运行 PromptBuilder
# ==========================================
print("�� [4/5] 运行 PromptBuilder 生成文本...")


# 这是一个 dummy tokenize，因为 PromptBuilder 需要它来计算长度
def dummy_tokenize(text): return len(text.split())


# 这里的 prompt_path 需要真实存在，或者我们现场写一个临时文件
prompt_template_path = "llm_project/prompts/llama2_predict.txt"
if not os.path.exists(prompt_template_path):
    print(f"⚠️ 警告: 找不到 {prompt_template_path}，将尝试创建一个临时模板。")
    prompt_template_path = "temp_prompt.txt"
    with open(prompt_template_path, "w") as f:
        f.write("Instruction: Answer.\nQuestion: {question}\nReasoning Paths:\n{rule}\nAnswer:")

# 初始化 Builder (开启 add_rule=True)
input_builder = PromptBuilder(prompt_template_path, add_rule=True, tokenize=dummy_tokenize)

# 生成 Input
try:
    # 注意：PromptBuilder 内部可能会根据 use_true/use_random 选择路径
    # 我们这里假设默认会读取 predicted_paths
    input_text = input_builder.process_input(data_sample)
    print("   ✅ Prompt 生成成功！内容片段如下：")
    print("-" * 40)
    print(input_text.strip())
    print("-" * 40)
except Exception as e:
    print(f"❌ PromptBuilder 运行失败: {e}")
    sys.exit(1)

# ==========================================
# 5. 运行特征提取逻辑 (核心验证)
# ==========================================
print("�� [5/5] 运行特征提取逻辑 (反向匹配)...")

matched_ids = []
norm_input = normalize(input_text)

# 遍历子图节点，看谁的名字在 Prompt 里
for eid_int, vector in node_dict.items():
    eid_str = str(eid_int)
    if eid_str in entities_names:
        name = entities_names[eid_str]
        norm_name = normalize(name)

        # 匹配逻辑
        # 长度判断防止匹配到 'a', 'i' 这种短词
        if norm_name in norm_input and len(norm_name) > 2:
            matched_ids.append(eid_int)
            # print(f"      ✨ 捕获: {name} (ID: {eid_int})")

print(f"\n�� 结果统计:")
print(f"   - 子图节点总数: {len(node_dict)}")
print(f"   - Prompt 匹配命中数: {len(matched_ids)}")

if len(matched_ids) > 0:
    # 验证是否包含我们要测的那个 ID
    if test_eid in matched_ids:
        print(f"   ✅ 成功！预埋的测试实体 '{test_name}' (ID: {test_eid}) 被成功提取！")
    else:
        print(f"   ⚠️ 警告: 提取到了其他实体，但没提取到预埋的 '{test_name}'。可能是归一化匹配问题。")
        print(f"      Prompt normalized: {norm_input[:100]}...")
        print(f"      Name normalized: {normalize(test_name)}")

    # 模拟堆叠
    vecs = [node_dict[eid] for eid in matched_ids]
    stack = np.stack(vecs)
    tensor = torch.from_numpy(stack)
    print(f"   ✅ 最终 Tensor 形状: {tensor.shape} (预期: [N, 50])")
else:
    print("   ❌ 失败：没有从 Prompt 中提取到任何实体特征。")

# 清理临时文件
if os.path.exists("temp_prompt.txt"):
    os.remove("temp_prompt.txt")