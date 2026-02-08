import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.projector import GraphProjector
from src.hgnn_rag_model import HGNN_RAG_Model

# === 配置 ===
LLM_PATH = "/home/bi3/zxd_env/GNN-RAG-main2/llm_retry/RoG_model"
CHECKPOINT_PATH = "./checkpoints/webqsp_ddp_final/projector_epoch_3.bin"


def check_weights_changed():
    print("�� [1. 权重变化检查]")

    # 1. 初始化一个随机的 Projector (模拟刚开始的状态)
    # 注意：这里我们假设 embedding dim 是 4096 (Llama2-7b)
    # 如果不一样，代码会自动报错，到时候改一下就行
    init_projector = GraphProjector(input_dim=50, output_dim=4096)

    # 2. 加载训练好的权重
    trained_state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # 3. 对比
    print(f"   检查层: linear.weight")
    w_init = init_projector.linear.weight.data
    w_trained = trained_state_dict['linear.weight']

    diff = (w_init - w_trained).abs().sum().item()

    if diff == 0:
        print("❌ 警告：权重完全没变！模型根本没在学！(可能是梯度断了)")
    else:
        print(f"✅ 恭喜：权重发生了变化，L1 差异值 = {diff:.4f}")
        print("   (说明梯度正常回传了，只是 Loss 看起来降得慢)")


def run_inference_demo():
    print("\n��️ [2. 生成效果测试]")

    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = HGNN_RAG_Model(LLM_PATH, freeze_llm=True, device_map="auto")

    # 加载训练好的 Projector
    model.projector.load_state_dict(torch.load(CHECKPOINT_PATH))
    target_device = model.llm.device
    model.projector.to(target_device)
    model.eval()  # 开启评估模式

    # 2. 构造假数据 (模拟一条 WebQSP 数据)
    # 问题: "What movie did Patrick Swayze play in?"
    question = "Question: What movie did Patrick Swayze play in?\nAnswer:"

    # 构造一个假的图特征 (假设检索到了相关知识)
    # 这里我们随机生成，看看模型能不能哪怕有一点点反应
    # 如果模型学到了，它可能会尝试去"读"这个 graph_feats
    dummy_graph_feats = torch.randn(1, 10, 50).to(model.llm.device).to(model.llm.dtype)
    dummy_graph_mask = torch.ones(1, 10).long().to(model.llm.device)

    inputs = tokenizer(question, return_tensors="pt").to(model.llm.device)

    print(f"   问题: {question.strip()}")
    print("   正在生成...")

    with torch.no_grad():
        # 我们需要手动调用 model.generate 的逻辑，因为 model() 返回的是 loss
        # 这里为了简单，我们手动拼接 embedding 来看

        # 1. Get Text Embeds
        inputs_embeds = model.llm.get_input_embeddings()(inputs.input_ids)

        # 2. Get Graph Embeds
        graph_token = model.projector(dummy_graph_feats, dummy_graph_mask)

        # 3. Concat
        merged_embeds = torch.cat([graph_token, inputs_embeds], dim=1)

        # 4. Generate
        # 使用 LLM 的 generate 接口，传入 inputs_embeds
        outputs = model.llm.generate(
            inputs_embeds=merged_embeds,
            max_new_tokens=20,
            do_sample=False  # 贪婪搜索，看最确定的结果
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"   �� 模型回答: {decoded}")


if __name__ == "__main__":
    try:
        check_weights_changed()
        run_inference_demo()
    except Exception as e:
        print(f"程序出错: {e}")
        print("提示：如果是显存不足，请把 run_inference_demo 注释掉只跑第一步")