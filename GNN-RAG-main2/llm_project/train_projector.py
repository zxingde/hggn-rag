import torch
import os
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from tqdm import tqdm
# �� 引入分布式训练核心库
from accelerate import Accelerator, DistributedDataParallelKwargs

# 引用模块
from src.projector_dataset import ProjectorDataset, collate_fn
from src.hgnn_rag_model import HGNN_RAG_Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="cwq or webqsp")
    parser.add_argument("--llm_path", type=str, required=True, help="Path to LLM")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/projector_ddp")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # --- 1. 初始化 Accelerator (关键步骤) ---
    # find_unused_parameters=True 是必须的，因为 LLM 参数被冻结了，不会产生梯度
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if accelerator.is_main_process:
        print(f"�� Distributed Training on {accelerator.num_processes} GPUs!")
        print(f"   Total Batch Size: {args.batch_size * accelerator.num_processes}")

    # 路径配置
    base_dir = os.getcwd()
    jsonl_path = os.path.join(base_dir, f"datasets/llm_project/dataset/{args.dataset}/train.jsonl")
    pkl_path = os.path.join(base_dir, f"datasets/llm_project/pkl/{args.dataset}/train.pkl")  # 确保路径对应你实际的pkl位置

    if accelerator.is_main_process:
        print(f"   JSONL: {jsonl_path}")
        print(f"   PKL: {pkl_path}")

    # --- 2. Tokenizer & Dataset ---
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ProjectorDataset(jsonl_path, pkl_path, tokenizer)

    # 注意：Accelerator 会自动处理 DataLoader 的 sampler，这里 shuffle=True 没问题
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # --- 3. 模型加载 (DDP 显存优化关键) ---
    # 构造 device_map，让当前进程只在指定的 GPU 上加载模型
    # 如果不这么做，transformers 可能会尝试在每张卡上加载所有层，导致 OOM
    device_index = {"": accelerator.process_index}

    model = HGNN_RAG_Model(args.llm_path, freeze_llm=True, device_map=device_index)

    # 优化器只训练 Projector
    optimizer = AdamW(model.projector.parameters(), lr=args.lr)

    # --- 4. Prepare ---
    # Accelerator 接管模型、优化器和数据加载器
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # --- 5. 训练循环 ---
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0  # 每个 Epoch 清零

            # 只有主进程显示进度条
            progress_bar = tqdm(enumerate(dataloader),
                                total=len(dataloader),
                                desc=f"Epoch {epoch + 1}/{args.epochs}",
                                disable=not accelerator.is_main_process)

            for step, batch in progress_bar:  # 注意这里改用了 enumerate(dataloader) 的方式

                optimizer.zero_grad()

                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss

                # 反向传播
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # --- 核心：收集并累加 Loss ---
                # gather 会把所有 GPU 的 loss 收集起来变成一个 tensor
                # mean() 取平均，item() 转为 python float
                current_batch_loss = accelerator.gather(loss).mean().item()

                # 累加到 total_loss (注意：这是累加的平均值)
                total_loss += current_batch_loss

                # 实时显示当前 Batch 的 Loss
                if accelerator.is_main_process:
                    progress_bar.set_postfix({"batch_loss": f"{current_batch_loss:.4f}"})

            # --- Epoch 结束后的处理 ---
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # 计算整个 Epoch 的平均 Loss
                # len(dataloader) 是步数 (steps per epoch)
                avg_loss = total_loss / len(dataloader)

                # �� 打印醒目的日志
                print(f"\n{'=' * 30}")
                print(f"✅ Epoch {epoch + 1} Finished!")
                print(f"�� Average Loss: {avg_loss:.6f}")  # 保留6位小数看微小变化
                print(f"{'=' * 30}\n")

                # 保存模型
                save_file = os.path.join(args.output_dir, f"projector_epoch_{epoch + 1}.bin")
                unwrapped_model = accelerator.unwrap_model(model)
                # 确保只保存 projector 的权重
                # 如果你的 model 是用 module.projector 访问的
                if hasattr(unwrapped_model, 'projector'):
                    state_dict = unwrapped_model.projector.state_dict()
                else:
                    # 防御性编程
                    state_dict = unwrapped_model.state_dict()

                torch.save(state_dict, save_file)
                print(f"�� Saved projector to {save_file}")

if __name__ == "__main__":
    main()