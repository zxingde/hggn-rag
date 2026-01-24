#!/bin/bash

# 1. 设置模型路径
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf"

# 2. 设置数据列表 (这里指向你刚生成的 .pkl 文件)
# 请确保这个路径是你服务器上真实的 pkl 路径
DATASET_LIST="/home/bi3/zxd_env/GNN-RAG-main2/llm/final_finetune_corpus_webqsp.pkl"

# 3. 设置保存路径
SAVE_NAME="RoG_WebQSP_GNN_Finetune_Len4096_1"
SAVE_PATH="save_models/${SAVE_NAME}"
ADD_REL=False

# 4. GNN 参数
GNN_INPUT_DIM=50
NUM_GRAPH_TOKENS=3

# 5. 启动命令
# 【核心修改】：在末尾添加了 --model_max_length 4096
# 建议将 batch_size 调小为 2，防止显存溢出 (OOM)

export CUDA_VISIBLE_DEVICES=1,2,3

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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
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
    --run_name ${SAVE_NAME} \
    --gnn_input_dim ${GNN_INPUT_DIM} \
    --num_graph_tokens ${NUM_GRAPH_TOKENS} \
    --model_max_length 4096