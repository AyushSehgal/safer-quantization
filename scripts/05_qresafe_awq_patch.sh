#!/bin/bash
#SBATCH --job-name=qresafe-awq-patch
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/user_data/ayushseh/qresafe_outputs/logs/awq_patch_%j.out
#SBATCH --error=/data/user_data/ayushseh/qresafe_outputs/logs/awq_patch_%j.err

# ============================================================================
# 05_qresafe_awq_patch.sh — Q-resafe mixed-precision patching for AWQ
# ============================================================================
# This is the only missing piece after steps 02 and 03.
#
# Step 02 produced: llama2-7b-chat-awq-int4  (standard AWQ, unpatched baseline)
# Step 03 produced: Q-resafe patched QLoRA models  (Algorithm 1, already done)
# This step produces: llama2-7b-chat-qresafe-awq-int4  (mixed-precision AWQ)
#
# The Q-resafe AWQ method (Section 5.1, "Safety patch without finetuning"):
#   1. Compute SNIP scores on the FP16 model using calibration prompts
#   2. Identify safety-critical weights (top-tau=0.6 percentile)
#   3. Keep those weights at FP16, quantize the rest to INT4
#   No DPO is run — this is purely mixed-precision quantization.
#
# Runtime: ~1-2 hours (dominated by SNIP score computation over calib set)

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate qresafe

export BASE_DIR="/data/user_data/ayushseh"
export REPO_DIR="${BASE_DIR}/Qresafe"
export OUTPUT_DIR="${BASE_DIR}/qresafe_outputs"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "============================================="
echo "Q-resafe AWQ mixed-precision patching"
echo "============================================="

cd ${REPO_DIR}/Qresafe/quant-without-ft

python quantize.py --qresafe \
    --tau 0.6 \
    --num_calib_samples 50 \
    --fp16_model_path meta-llama/Llama-2-7b-chat-hf \
    2>&1 | tee ${OUTPUT_DIR}/logs/awq_qresafe_patch.log

echo "============================================="
echo "Q-resafe AWQ patching complete!"
echo "Model saved to: ${OUTPUT_DIR}/models/llama2-7b-chat-qresafe-awq-int4"
echo "Next step: eval in 04_eval_quantized.sh (uncomment qresafe_awq entries)"
echo "============================================="
