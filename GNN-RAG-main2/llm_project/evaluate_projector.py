import argparse
import os
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from src.hgnn_rag_model import HGNN_RAG_Model

# 引入官方评估函数
import sys

sys.path.append(os.getcwd())
try:
    from src.qa_prediction.evaluate_results import eval_result
except ImportError:
    eval_result = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="webqsp")
    parser.add_argument('--llm_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default="results/projector_eval")
    args = parser.parse_args()

    # 1. 路径准备
    base_dir = os.getcwd()
    jsonl_path = os.path.join(base_dir, f"datasets/llm_project/dataset/{args.dataset}/test.jsonl")
    pkl_path = os.path.join(base_dir, f"datasets/llm_project/pkl/{args.dataset}/test.pkl")

    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "predictions.jsonl")

    # 2. 加载模型
    print("�� Loading Model...")
    model = HGNN_RAG_Model(args.llm_path, freeze_llm=True, device_map="auto")
    state_dict = torch.load(args.checkpoint_path)
    model.projector.load_state_dict(state_dict)

    target_device = model.llm.device
    model.projector.to(target_device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载数据
    print(f"�� Loading Data from {jsonl_path}")
    with open(jsonl_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    if args.limit > 0: test_data = test_data[:args.limit]

    print(f"��️ Loading Graph Features from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        graph_features = pickle.load(f)

    # 4. 推理循环
    print(f"�� Start Inference -> {output_file}")
    with open(output_file, 'w') as fout:
        for item in tqdm(test_data):
            qid = item.get('id')

            # --- A. 获取 Input (直接读 text/input) ---
            # 这就是你数据集里自带的 [INST]...[/INST]
            if 'text' in item:
                final_prompt = item['text']
            elif 'input' in item:
                final_prompt = item['input']
            else:
                final_prompt = item.get('question', '')  # Fallback

            # --- B. 图特征 ---
            graph_vecs = torch.zeros((1, 50), dtype=model.llm.dtype).to(target_device)
            graph_mask = torch.ones(1, dtype=torch.long).to(target_device)

            if qid in graph_features:
                feats = graph_features[qid]
                if isinstance(feats, np.ndarray) and feats.shape[0] > 0 and not np.all(feats == 0):
                    graph_vecs = torch.tensor(feats, dtype=model.llm.dtype).to(target_device)
                    graph_mask = torch.ones(len(graph_vecs), dtype=torch.long).to(target_device)

            if graph_vecs.dim() == 2:
                graph_vecs = graph_vecs.unsqueeze(0)
                graph_mask = graph_mask.unsqueeze(0)

            # --- C. 生成 ---
            with torch.no_grad():
                inputs = tokenizer(final_prompt, return_tensors="pt").to(target_device)
                inputs_embeds = model.llm.get_input_embeddings()(inputs.input_ids)

                # Projector 插入
                graph_token = model.projector(graph_vecs, graph_mask)
                merged_embeds = torch.cat([graph_token, inputs_embeds], dim=1)

                outputs = model.llm.generate(
                    inputs_embeds=merged_embeds,
                    max_new_tokens=64,
                    do_sample=False
                )

                # --- D. 后处理 ---
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 简单清洗：如果包含了 Prompt 的尾巴，去掉它
                if "[/INST]" in prediction:
                    prediction = prediction.split("[/INST]")[-1].strip()
                if "Answer:" in prediction:
                    prediction = prediction.split("Answer:")[-1].strip()

                prediction = prediction.split('\n')[0].strip()

            # --- E. 写入结果 ---
            if 'ground_truth' in item:
                gt = item['ground_truth']
            elif 'answer' in item:
                gt = item['answer']
            elif 'output' in item:
                gt = [item['output']]
            else:
                gt = []
            if isinstance(gt, str): gt = [gt]

            res = {"id": qid, "prediction": prediction, "ground_truth": gt}
            fout.write(json.dumps(res) + "\n")

    # 5. 评估
    if eval_result:
        print("\n�� Calculating Metrics...")
        eval_result(output_file, encrypt=False, cal_f1=True)


if __name__ == "__main__":
    main()