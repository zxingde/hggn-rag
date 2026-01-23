#!/bin/bash

# 指定 GPU
export CUDA_VISIBLE_DEVICES=1,2,3

# 关键：这里直接写死绝对路径，不要用变量引用了，防止前面定义错
PKL_PATH="/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf"
OUTPUT_DIR="save_models/RoG_WebQSP_GNN_Finetune_v2"

# 启动命令
accelerate launch --multi_gpu --num_processes 3 --mixed_precision "bf16" src/joint_training/joint_finetuning.py \
    --data_path_list "$PKL_PATH"  \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --add_rel_token False \
    --bf16 True \
    --use_peft True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,v_proj" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --run_name "RoG_WebQSP_Run3" \
    --gnn_input_dim 50 \
    --num_graph_tokens 3