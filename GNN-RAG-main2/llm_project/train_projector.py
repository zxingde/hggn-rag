import torch
import os
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from tqdm import tqdm

# 引用刚才写好的模块
from src.projector_dataset import ProjectorDataset, collate_fn
from src.hgnn_rag_model import HGNN_RAG_Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="cwq or webqsp")
    parser.add_argument("--llm_path", type=str, required=True, help="Path to LLM")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/projector_v1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    # --- 1. 路径配置 ---
    # 假设你在 llm_project 目录下运行
    base_dir = os.getcwd()
    # 适配你之前的目录结构
    jsonl_path = os.path.join(base_dir, f"datasets/llm_project/dataset/{args.dataset}/train.jsonl")
    # pkl 路径 (请确保这里和你生成的 pkl 文件名一致)
    # pkl_path = os.path.join(base_dir, f"{args.dataset}_train_id_to_path_vectors.pkl")
    pkl_path = os.path.join(base_dir, f"datasets/llm_project/pkl/{args.dataset}/train.pkl")

    print(f"�� Training Config:")
    print(f"   LLM: {args.llm_path}")
    print(f"   JSONL: {jsonl_path}")
    print(f"   PKL: {pkl_path}")

    # --- 2. Tokenizer & Dataset ---
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ProjectorDataset(jsonl_path, pkl_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # --- 3. Model ---
    model = HGNN_RAG_Model(args.llm_path, freeze_llm=True)

    # --- 4. Optimizer ---
    # 只训练 Projector 的参数
    optimizer = AdamW(model.projector.parameters(), lr=args.lr)

    # --- 5. Training Loop ---
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            # 转移到 GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            # ��【新增】调试：打印梯度信息
            if step % 10 == 0:  # 每 10 步打印一次，防止刷屏
                print(f"\n�� [Step {step}] Gradient Check:")
                total_norm = 0.0
                has_nan = False
                for name, param in model.projector.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        total_norm += grad_norm
                        print(f"   - {name}: norm={grad_norm:.4f}, mean={param.grad.mean():.6f}")

                        if torch.isnan(param.grad).any():
                            print(f"   ⚠️ ALERT: NaN gradient in {name}!")
                            has_nan = True
                    else:
                        print(f"   - {name}: No Gradient! (Check freeze logic)")

                print(f"   === Total Grad Norm: {total_norm:.4f} ===")
                if has_nan:
                    print("❌ Stopping due to NaN gradient.")
                    break
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.4f}")

        # 保存
        save_file = os.path.join(args.output_dir, f"projector_epoch_{epoch + 1}.bin")
        torch.save(model.projector.state_dict(), save_file)
        print(f"�� Saved projector to {save_file}")


if __name__ == "__main__":
    main()