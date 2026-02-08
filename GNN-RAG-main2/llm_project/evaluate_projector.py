import argparse
import os
import json
import torch
import numpy as np
import pickle
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from src.hgnn_rag_model import HGNN_RAG_Model
# �� 引入 PromptBuilder
from src.qa_prediction.build_qa_input import PromptBuilder

# �� 引入官方评估函数
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
    parser.add_argument('--prompt_path', type=str, default="prompts/llama2_predict.txt")  # 和训练保持一致
    parser.add_argument('--output_dir', type=str, default="results/projector_eval")
    args = parser.parse_args()

    # 1. 初始化 PromptBuilder
    if not os.path.exists(args.prompt_path):
        # 尝试找一下
        args.prompt_path = os.path.join(os.getcwd(), "llm_project", args.prompt_path)

    print(f"�� Initializing PromptBuilder with {args.prompt_path}")
    prompt_builder = PromptBuilder(
        prompt_path=args.prompt_path,
        encrypt=False,
        add_rule=True
    )

    # 2. 路径与模型准备
    base_dir = os.getcwd()
    jsonl_path = os.path.join(base_dir, f"datasets/llm_project/dataset/{args.dataset}/test.jsonl")
    pkl_path = os.path.join(base_dir, f"datasets/llm_project/pkl/{args.dataset}/test.pkl")

    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "predictions.jsonl")

    print("�� Loading Model...")
    model = HGNN_RAG_Model(args.llm_path, freeze_llm=True, device_map="auto")
    state_dict = torch.load(args.checkpoint_path)
    model.projector.load_state_dict(state_dict)
    model.projector.to(model.llm.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载数据
    print(f"�� Loading Data from {jsonl_path}")
    with open(jsonl_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    if args.limit > 0: test_data = test_data[:args.limit]

    with open(pkl_path, 'rb') as f:
        graph_features = pickle.load(f)

    # 4. 推理循环
    print(f"�� Start Inference -> {output_file}")
    with open(output_file, 'w') as fout:
        for item in tqdm(test_data):
            qid = item.get('id')

            # --- A. 使用 PromptBuilder 生成 Input (核心) ---
            # 补齐字段防止报错
            if 'choices' not in item: item['choices'] = []
            if 'cand' not in item: item['cand'] = None
            if 'q_entity' not in item: item['q_entity'] = []

            # �� 这里的 output 和 ProjectorDataset 里的 input_text 绝对一致
            input_text = prompt_builder.process_input(item)

            # --- B. 图特征 ---
            graph_vecs = torch.zeros((1, 50), dtype=model.llm.dtype).to(model.llm.device)
            graph_mask = torch.ones(1, dtype=torch.long).to(model.llm.device)

            if qid in graph_features:
                feats = graph_features[qid]
                if isinstance(feats, np.ndarray) and feats.shape[0] > 0 and not np.all(feats == 0):
                    graph_vecs = torch.tensor(feats, dtype=model.llm.dtype).to(model.llm.device)
                    graph_mask = torch.ones(len(graph_vecs), dtype=torch.long).to(model.llm.device)

            if graph_vecs.dim() == 2:
                graph_vecs = graph_vecs.unsqueeze(0)
                graph_mask = graph_mask.unsqueeze(0)

            # --- C. 生成 ---
            with torch.no_grad():
                inputs = tokenizer(input_text, return_tensors="pt").to(model.llm.device)
                inputs_embeds = model.llm.get_input_embeddings()(inputs.input_ids)
                graph_token = model.projector(graph_vecs, graph_mask)
                merged_embeds = torch.cat([graph_token, inputs_embeds], dim=1)

                outputs = model.llm.generate(
                    inputs_embeds=merged_embeds,
                    max_new_tokens=64,
                    do_sample=False
                )

                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # --- D. 后处理 ---
                # 因为 PromptBuilder 生成的 prompt 比较复杂，split 可能会失效
                # 最好的办法是只保留新生成的部分。但 tokenizer.decode 包含 prompt。
                # 简单的 Trick: 截取 [/INST] 之后
                if "[/INST]" in prediction:
                    prediction = prediction.split("[/INST]")[-1].strip()

                # 如果模型很乖，生成了 Answer: 开头
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