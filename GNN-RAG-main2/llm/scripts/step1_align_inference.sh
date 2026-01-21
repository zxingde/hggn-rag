#!/bin/bash

# 使用空闲显卡
export CUDA_VISIBLE_DEVICES=1

BASE_MODEL="NousResearch/Llama-2-7b-chat-hf"

# 【核心修改 1】加上 llm/ 前缀
LORA_PATH="llm/save_models/RoG_Joint_LoRA_Graph"

# 测试集路径
# 【核心修改 2】确认 datasets 是否也在 llm 下？如果是，也要加 llm/
# 假设你的 datasets 文件夹在 llm/datasets/
TEST_DATA="llm/datasets/joint_training/align/RoG-cwq/test.jsonl"

# 图特征路径
# 【核心修改 3】同上，加上 llm/
GRAPH_FEAT="llm/datasets/feature/0117-HGNN-WebQSP_LMSR_BS24_4090_EXPORT_all_features.pkl"

# 输出文件
OUTPUT_FILE="${LORA_PATH}/RoG_cwq_test_align_predictions.jsonl"

echo "Running Step 1: Align Task Inference with Graph Features..."

# 注意：这里调用 python 是从根目录调用的，所以前面的参数都要基于根目录
python llm/src/qa_prediction/gen_rule_path_feat.py \
    --model_name_or_path ${BASE_MODEL} \
    --lora_path ${LORA_PATH} \
    --test_path ${TEST_DATA} \
    --graph_feat_path ${GRAPH_FEAT} \
    --output_path ${OUTPUT_FILE} \
    --beam_size 1 \
    --use_peft True