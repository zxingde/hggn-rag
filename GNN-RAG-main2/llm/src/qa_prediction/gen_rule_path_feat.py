import os
import sys
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# --- 路径修正，确保能导入项目模块 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project.GraphProjector import GraphProjector



def load_graph_features(feat_path):
    print(f"Loading graph features from {feat_path}...")
    with open(feat_path, 'rb') as f:
        features = pickle.load(f)
    print(f"Loaded {len(features)} features.")
    return features


def main(args):
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载 Base Model
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 3. 加载 LoRA
    if args.use_peft:
        print(f"Loading LoRA adapters from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    # 4. 加载 Graph Projector
    print(f"Loading Graph Projector from {args.lora_path}/graph_projector.bin ...")
    projector = GraphProjector(gnn_dim=50, llm_dim=model.config.hidden_size)
    projector_path = os.path.join(args.lora_path, "graph_projector.bin")
    if os.path.exists(projector_path):
        projector.load_state_dict(torch.load(projector_path, map_location=model.device))
    else:
        raise FileNotFoundError(f"Projector file not found at {projector_path}")

    projector.to(model.device).to(model.dtype)
    projector.eval()
    model.eval()

    # 5. 加载数据
    graph_features_dict = load_graph_features(args.graph_feat_path)

    with open(args.test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

    # 6. 推理循环
    results = []
    print(f"Start Inference on {len(test_data)} examples...")

    with torch.no_grad():
        for item in tqdm(test_data):
            qid = item['id']
            question = item['question']  # 或者是 item['text']，取决于数据格式

            # 构造 Prompt (Align 任务)
            # 注意：这里的 Prompt 格式必须和你训练 align 任务时的一模一样！
            # 假设训练时是： "Question: {q}\nAnswer:"
            prompt = f"Question: {question}\nAnswer:"

            # 获取图特征
            if qid in graph_features_dict:
                # [6, 50]
                feat_tensor = torch.tensor(graph_features_dict[qid], dtype=model.dtype, device=model.device)
                # 投影 -> [1, 6, 4096]
                projected_feat = projector(feat_tensor).unsqueeze(0)
            else:
                # 兜底：如果没有特征，用全0
                print(f"Warning: No feature for {qid}")
                feat_tensor = torch.zeros((6, 50), dtype=model.dtype, device=model.device)
                projected_feat = projector(feat_tensor).unsqueeze(0)

            # 文本转 Embedding
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_embeds = model.get_input_embeddings()(inputs.input_ids)

            # 【核心步骤】拼接： [图特征, 文本特征]
            # inputs_embeds shape: [1, seq_len + 6, hidden_dim]
            final_input_embeds = torch.cat([projected_feat, input_embeds], dim=1)

            # 生成
            generation_output = model.generate(
                inputs_embeds=final_input_embeds,
                max_new_tokens=128,
                num_beams=args.beam_size,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # 解码
            output = generation_output.sequences[0]
            # 注意：因为我们是传 embedding 进去的，generate 返回的 output 包含了输入的长度吗？
            # 通常 generate 返回的是 [input + generated] 或者 [generated] 取决于版本
            # 这里简单处理，解码所有 token，然后截取 output
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)

            # 保存结果
            results.append({
                "id": qid,
                "question": question,
                "prediction": decoded_output.strip()
            })

    # 7. 保存文件
    print(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--graph_feat_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--use_peft", type=bool, default=True)
    args = parser.parse_args()
    main(args)