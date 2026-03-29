#!/bin/bash
#SBATCH --job-name=qresafe-awq
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user_data/ayushseh/qresafe_outputs/logs/awq_%j.out
#SBATCH --error=/data/user_data/ayushseh/qresafe_outputs/logs/awq_%j.err

# ============================================================================
# 02_quantize_awq.sh — AWQ quantization + Q-resafe mixed-precision patching
# ============================================================================
# This runs the quant-without-ft pipeline:
#   1. Identify safety-critical weights via SNIP scores
#   2. Keep safety-critical weights at FP16, quantize the rest to INT4
#
# The original repo's quant-without-ft/quantize.py does this.
# We wrap it with proper Babel paths.

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate qresafe

export BASE_DIR="/data/user_data/ayushseh"
export REPO_DIR="${BASE_DIR}/Qresafe"
export OUTPUT_DIR="${BASE_DIR}/qresafe_outputs"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "============================================="
echo "Running AWQ quantization (quant-without-ft)"
echo "============================================="

cd ${REPO_DIR}/Qresafe/quant-without-ft

# --- Step 1: Standard AWQ quantization (no safety patching) ---
# This creates the unpatched AWQ INT4 model for comparison.
# If the original quantize.py expects hardcoded paths, you may need to
# modify it. Check the script and update MODEL_ID and output paths.

echo ">>> AWQ INT4 quantization..."
# The original script likely does something like:
#   from awq import AutoAWQForCausalLM
#   model = AutoAWQForCausalLM.from_pretrained(model_path)
#   model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
#   model.save_quantized(output_path)
#
# Run the repo's script (modify paths inside if needed):
python quantize.py \
    2>&1 | tee ${OUTPUT_DIR}/logs/awq_quantize.log

# Move output to our organized directory
# (Check what path quantize.py saves to and adjust)
# cp -r ./output_model ${OUTPUT_DIR}/models/llama2-7b-chat-awq-int4

# --- Step 2: Q-resafe mixed-precision (safety-critical at FP16) ---
# This is the Q-resafe variant for AWQ: identify safety-critical weights,
# keep them at 16-bit, quantize the rest.
# The original quantize.py should handle this — it searches for 
# safety-critical weights using SNIP scores on the full-precision model.

echo "============================================="
echo "AWQ quantization complete!"
echo "Check output in quant-without-ft/ directory"
echo "Move results to: ${OUTPUT_DIR}/models/"
echo "============================================="
echo ""
echo "NOTE: After running, check what directory the model was saved to."
echo "The original script saves to e.g. ./google/gemma-2b-it-4bit"
echo "For Llama-2, you'll need to update the model path in quantize.py"
echo "to point to meta-llama/Llama-2-7b-chat-hf"
