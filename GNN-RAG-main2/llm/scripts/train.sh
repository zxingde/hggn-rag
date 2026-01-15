#!/bin/bash

# 1. 设置模型路径 (使用免签版)
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf"

# 2. 设置数据列表 (只保留你生成的 Align 和 QA 数据，去掉了 ExplainQAData)
# 注意：请确保这些文件路径和你实际生成的一致
DATASET_LIST="datasets/joint_training/align/RoG-cwq/RoG-cwq_train.jsonl datasets/joint_training/align/RoG-webqsp/RoG-webqsp_train.jsonl datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl datasets/joint_training/qa/RoG-cwq/RoG-cwq_train.jsonl"

# 3. 设置保存路径
SAVE_NAME="RoG_Joint_LoRA"
SAVE_PATH="save_models/${SAVE_NAME}"
ADD_REL=False

# 4. 启动命令
# --multi_gpu: 启用多卡 DDP 模式
# --num_processes 3: 强制使用 3 张显卡
accelerate launch --multi_gpu --num_processes 3 --mixed_precision "bf16" src/joint_training/joint_finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --add_rel_token ${ADD_REL} \
    --bf16 True \
    --use_peft True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,v_proj" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
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
    --run_name ${SAVE_NAME}

# 注：如果你想追求极限速度且显存够用，可以将 gradient_checkpointing 改为 False