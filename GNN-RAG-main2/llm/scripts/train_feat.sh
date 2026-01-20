#!/bin/bash

# 1. 设置模型路径
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf"

# 2. 设置数据列表 (确保这些文件都在)
DATASET_LIST="datasets/joint_training/align/RoG-cwq/RoG-cwq_train.jsonl datasets/joint_training/align/RoG-webqsp/RoG-webqsp_train.jsonl datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl datasets/joint_training/qa/RoG-cwq/RoG-cwq_train.jsonl"

# 3. 【新增】设置图特征路径 (这是必须的！！！)
# 请把它改成你实际存放 .pkl 文件的绝对路径或相对路径
GRAPH_FEAT_PATH="datasets/feature/0117-HGNN-WebQSP_LMSR_BS24_4090_EXPORT_all_features.pkl"

# 4. 设置保存路径
SAVE_NAME="RoG_Joint_LoRA_Graph"
SAVE_PATH="save_models/${SAVE_NAME}"
ADD_REL=False

# 5. 启动命令
accelerate launch  --num_processes 1 --mixed_precision "bf16" src/joint_training/joint_finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --graph_feat_path ${GRAPH_FEAT_PATH} \
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
    --gradient_checkpointing True \
    --run_name ${SAVE_NAME} \
    --max_steps 10 \
    --remove_unused_columns False \
    --report_to "none"

