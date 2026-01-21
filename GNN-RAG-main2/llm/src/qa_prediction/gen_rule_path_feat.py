import os
import sys
import json
import torch
import pickle
import argparse
import math
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project.GraphProjector import GraphProjector

INSTRUCTION = "Please generate a valid relation path that can be helpful for answering the following question: "
PROMPT_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\n{instruction}{input} [/INST]"


def load_graph_features(feat_path):
    print(f"Loading graph features from {feat_path}...")
    with open(feat_path, 'rb') as f:
        features = pickle.load(f)
    return features


def batch_iterator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def main(args):
    # 1. Tokenizer
    print(f"Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, use_fast=False)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        tokenizer.add_tokens(["<SEP>", "<PATH>", "</PATH>"], special_tokens=True)
        if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.padding_side = "left"

    # 2. Model
    print(f"Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    if args.use_peft:
        model = PeftModel.from_pretrained(model, args.lora_path)

    # 3. Projector
    print("Loading Graph Projector...")
    projector = GraphProjector(gnn_dim=50, llm_dim=model.config.hidden_size)
    projector.load_state_dict(
        torch.load(os.path.join(args.lora_path, "graph_projector.bin"), map_location=model.device))
    projector.to(model.device).to(model.dtype).eval()
    model.eval()

    # 4. Data
    graph_features_dict = {}
    if not args.ignore_graph:
        graph_features_dict = load_graph_features(args.graph_feat_path)

    with open(args.test_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
    if args.max_samples > 0: all_data = all_data[:args.max_samples]

    shard_size = math.ceil(len(all_data) / args.num_shards)
    test_data = all_data[args.shard_id * shard_size: min((args.shard_id + 1) * shard_size, len(all_data))]

    results = []
    debug_printed = False

    print(f"�� Worker {args.shard_id} Start: {len(test_data)} samples")

    with torch.no_grad():
        for batch in tqdm(batch_iterator(test_data, args.batch_size), desc=f"GPU {args.shard_id}"):
            batch_prompts = []
            batch_feat_tensors = []
            batch_ids = []

            for item in batch:
                qid = item['id']
                batch_ids.append(qid)
                batch_prompts.append(PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=item['question']))

                if (not args.ignore_graph) and (qid in graph_features_dict):
                    ft = graph_features_dict[qid]
                    if not isinstance(ft, torch.Tensor): ft = torch.tensor(ft)
                    ft = ft.to(dtype=model.dtype)
                else:
                    ft = torch.zeros((6, 50), dtype=model.dtype)
                batch_feat_tensors.append(ft)

            # Tokenize
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                model.device)

            # Projector
            batch_feat_stack = torch.stack(batch_feat_tensors).to(dtype=model.dtype, device=model.device)
            projected_feat = projector(batch_feat_stack)

            # Get Text Embeddings
            input_embeds = model.get_input_embeddings()(inputs.input_ids)

            # === Auto-Scaling ===
            if not debug_printed:
                text_norm = input_embeds.norm(dim=-1).mean().item()
                graph_norm = projected_feat.norm(dim=-1).mean().item()
                print(f"�� [SCALE CHECK] Text: {text_norm:.4f} | Graph: {graph_norm:.4f}")
                debug_printed = True

            # 计算 Scaling
            current_graph_norm = projected_feat.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
            text_avg_norm = input_embeds.norm(dim=-1).mean().detach()
            mask = (current_graph_norm > 1e-6).float()

            # 缩放运算 (这步会自动变成 float32)
            projected_feat = projected_feat / (current_graph_norm + 1e-6) * text_avg_norm * mask

            # 【关键修复】强制转回 bfloat16
            projected_feat = projected_feat.to(dtype=model.dtype)

            # Concat
            final_input_embeds = torch.cat([projected_feat, input_embeds], dim=1)

            # Mask
            bs = final_input_embeds.shape[0]
            graph_mask = torch.ones((bs, projected_feat.shape[1]), device=model.device)
            final_attention_mask = torch.cat([graph_mask, inputs.attention_mask], dim=1)

            # Generate
            gen_out = model.generate(
                inputs_embeds=final_input_embeds,
                attention_mask=final_attention_mask,
                max_new_tokens=128,
                do_sample=False
            )

            decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            for qid, item, pred in zip(batch_ids, batch, decoded):
                if "[/INST]" in pred: pred = pred.split("[/INST]")[-1]
                results.append({"id": qid, "question": item['question'], "prediction": pred.strip()})

    with open(f"{args.output_path}.shard{args.shard_id}", 'w') as f:
        for item in results: f.write(json.dumps(item) + "\n")
    print(f"✅ Worker {args.shard_id} Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--graph_feat_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--use_peft", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--ignore_graph", action="store_true")
    args = parser.parse_args()
    main(args)