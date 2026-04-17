#!/bin/bash
#SBATCH --job-name=bnb-int8
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user_data/ayushseh/safer-quantization/outputs/logs/quant_int8_%j.out
#SBATCH --error=/data/user_data/ayushseh/safer-quantization/outputs/logs/quant_int8_%j.err

set -euo pipefail

mkdir -p /data/user_data/ayushseh/safer-quantization/outputs/logs

source /data/user_data/ayushseh/safer-quantization/quantize/bin/activate

export BASE_DIR="/data/user_data/ayushseh/safer-quantization"
export OUTPUT_DIR="${BASE_DIR}/int8_outputs"
export PYTHONUNBUFFERED=1
export HF_HOME="/data/user_data/ayushseh/.cache/huggingface"
export HF_DATASETS_CACHE="/data/user_data/ayushseh/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data/user_data/ayushseh/.cache/huggingface/hub"
export HF_HUB_CACHE="/data/user_data/ayushseh/hf_cache/hub"

MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
SAVE_DIR="${OUTPUT_DIR}/results/models/llama2-7b-chat-bnb-int8"
mkdir -p ${SAVE_DIR}
mkdir -p ${OUTPUT_DIR}/logs

echo "============================================="
echo "INT8 quantization (bitsandbytes): ${MODEL_ID}"
echo "Saving to: ${SAVE_DIR}"
echo "============================================="

cd ${BASE_DIR}
python quantize_bnb.py \
    --model_id ${MODEL_ID} \
    --bits 8 \
    --output_dir ${SAVE_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/logs/quant_int8.log

echo "============================================="
echo "INT8 quantization complete!"
echo "Model saved to: ${SAVE_DIR}"
echo "============================================="