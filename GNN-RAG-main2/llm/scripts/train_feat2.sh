#!/bin/bash

# ================= 配置区域 =================
# 1. 你的 Llama2 模型路径 (HuggingFace 格式)
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf"

# 2. 你刚生成的 .pkl 数据集路径
DATA_PATH="/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"

# 3. 输出模型保存的文件夹 (每次实验建议换个名字)
OUTPUT_DIR="output/webqsp_finetune_v1"

# 4. GNN 特征维度 (必须是 50)
GNN_INPUT_DIM=50
# ===========================================

# 启动分布式训练 (单机多卡或单机单卡)
torchrun --nproc_per_node=1 --master_port=29500 llm/src/joint_training/joint_finetuning.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --deepspeed llm/config/deepspeed_zero3.yml \
    --gnn_input_dim $GNN_INPUT_DIM \
    --gnn_hidden_dim 4096 \
    --tune_gnn True \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,v_proj" \
    --bf16 True