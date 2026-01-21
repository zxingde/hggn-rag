import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel


def main(args):
    # 1. 智能加载 Tokenizer (复用之前的修复逻辑)
    print(f"Loading Tokenizer from {args.lora_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, use_fast=False)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.add_tokens(["<SEP>", "<PATH>", "</PATH>"], special_tokens=True)
        if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    tokenizer.padding_side = "left"

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    # 3. 加载 LoRA
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    # 4. 数据
    with open(args.test_path, 'r') as f:
        data = [json.loads(line) for line in f]
    if args.max_samples > 0: data = data[:args.max_samples]

    results = []

    # 5. 推理
    INSTRUCTION = "Please generate a valid relation path that can be helpful for answering the following question: "
    PROMPT_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\n{instruction}{input} [/INST]"

    with torch.no_grad():
        for i in tqdm(range(0, len(data), args.batch_size)):
            batch = data[i:i + args.batch_size]
            prompts = [PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=d['question']) for d in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                model.device)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
            decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)

            for d, pred in zip(batch, decoded):
                if "[/INST]" in pred: pred = pred.split("[/INST]")[-1]
                results.append({"id": d['id'], "question": d['question'], "prediction": pred.strip()})

    with open(args.output_path, 'w') as f:
        for item in results: f.write(json.dumps(item) + "\n")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    main(args)