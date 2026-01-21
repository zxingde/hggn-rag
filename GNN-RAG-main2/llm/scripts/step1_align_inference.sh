#!/bin/bash

# 基础配置
BASE_MODEL="NousResearch/Llama-2-7b-chat-hf"
LORA_PATH="llm/save_models/RoG_Joint_LoRA_Graph"
TEST_DATA="llm/datasets/joint_training/align/RoG-webqsp/test.jsonl"
GRAPH_FEAT="llm/datasets/feature/0117-HGNN-WebQSP_LMSR_BS24_4090_EXPORT_all_features.pkl"
OUTPUT_FILE="${LORA_PATH}/RoG_webqsp_test_align_predictions.jsonl"

echo "�� Starting 3-GPU Distributed Inference (DEBUG: Top 50 samples)..."

run_shard() {
    GPU_ID=$1
    SHARD_ID=$2
    NUM_SHARDS=3

    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python llm/src/qa_prediction/gen_rule_path_feat.py \
        --model_name_or_path ${BASE_MODEL} \
        --lora_path ${LORA_PATH} \
        --test_path ${TEST_DATA} \
        --graph_feat_path ${GRAPH_FEAT} \
        --output_path ${OUTPUT_FILE} \
        --beam_size 1 \
        --use_peft True \
        --batch_size 16 \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${SHARD_ID} \
        --max_samples 50 \
        --ignore_graph &
}

run_shard 0 0
run_shard 1 1
run_shard 2 2

wait

echo "✅ All GPUs finished!"
echo "Merging results..."
cat ${OUTPUT_FILE}.shard0 ${OUTPUT_FILE}.shard1 ${OUTPUT_FILE}.shard2 > ${OUTPUT_FILE}
rm ${OUTPUT_FILE}.shard*
echo "�� Done! Final output: ${OUTPUT_FILE}"