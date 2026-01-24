MODEL_PATH=NousResearch/Llama-2-7b-chat-hf
DATASET_LIST="datasets/joint_training/align/RoG-cwq/RoG-cwq_train.jsonl datasets/joint_training/align/RoG-webqsp/RoG-webqsp_train.jsonl datasets/joint_training/qa/RoG-webqsp/RoG-webqsp_train.jsonl datasets/joint_training/qa/RoG-cwq/RoG-cwq_train.jsonl"
SAVE_NAME=reshow_2
SAVE_PATH=save_models/${SAVE_NAME}
ADD_REL=False
export CUDA_VISIBLE_DEVICES=1,2,3

accelerate launch --multi_gpu --num_processes 3 --main_process_port 29505 \
    src/joint_training/joint_finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --add_rel_token ${ADD_REL} \
    --bf16 True \
    --use_peft True \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
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
    --run_name ${SAVE_NAME}