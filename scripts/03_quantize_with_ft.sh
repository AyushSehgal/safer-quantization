#!/bin/bash
#SBATCH --job-name=qresafe-dpo
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/ayushseh/qresafe_outputs/logs/quant_ft_%j.out
#SBATCH --error=/data/user_data/ayushseh/qresafe_outputs/logs/quant_ft_%j.err
# ============================================================================
# 03_quantize_with_ft.sh — QLoRA quantization + Q-resafe DPO safety patching
# ============================================================================
# This runs the full Algorithm 1 from the paper:
#   1. Quantize with QLoRA (using Risk-I/II/III calibration datasets)
#   2. Build safety-patching dataset (FP16 preferred vs quantized dispreferred)
#   3. DPO with masked LoRA on safety-critical weights
#
# Uses 4x GPUs via accelerate.
# The repo's quant-with-ft/quant.py + configs/llama7b.yaml handles this.

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate qresafe

export BASE_DIR="/data/user_data/ayushseh/safer-quantization"
export REPO_DIR="${BASE_DIR}/Qresafe"
export OUTPUT_DIR="${BASE_DIR}/qresafe_outputs"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export PYTHONUNBUFFERED=1
export HF_HOME="/data/user_data/ayushseh/.cache/huggingface"
export HF_DATASETS_CACHE="/data/user_data/ayushseh/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data/user_data/ayushseh/.cache/huggingface/hub"
export HF_HUB_CACHE=/data/user_data/ayushseh/hf_cache/hub

mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE} ${HF_HUB_CACHE}
cd ${REPO_DIR}/quant-with-ft

echo "============================================="
echo "Running quant-with-ft (Algorithm 1)"
echo "============================================="

# ---------- Option A: Use the original repo config ----------
# The repo ships configs/llama7b.yaml which has the paper's hyperparams.
# You'll need to check and possibly modify:
#   - model_name_or_path: meta-llama/Llama-2-7b-chat-hf
#   - output_dir: point to your output directory
#   - dataset paths for Risk-I/II/III
#
# Paper hyperparams (from Appendix A.1):
#   LoRA r: 128 (note: NOT 2048; 2048 is the safety-critical threshold tau)
#   LoRA alpha: 256
#   DPO beta: 0.01
#   Learning rate: 5e-6
#   Safety-critical threshold tau: 0.6
#   Re-evaluation interval K: 1000

echo ">>> Running with original config (modify paths first!)"
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${BASE_DIR}/configs/accelerate_babel.yaml \
    --num_processes=2 \
    quant.py configs/llama7b.yaml \
    2>&1 | tee /data/user_data/ayushseh/qresafe_outputs/logs/quant_ft_run.log

# ---------- Option B: Run for each risk level separately ----------
# If the config doesn't support choosing risk level, you may need to
# run it 3 times with different dataset configs. Uncomment below:
#
# for RISK in risk1 risk2 risk3; do
#     echo ">>> Running with ${RISK} dataset..."
#     ACCELERATE_LOG_LEVEL=info accelerate launch \
#         --config_file ${REPO_DIR}/configs/accelerate_babel.yaml \
#         --num_processes=2 \
#         quant.py configs/llama7b_${RISK}.yaml \
#         2>&1 | tee ${OUTPUT_DIR}/logs/quant_ft_${RISK}.log
# done

echo "============================================="
echo "quant-with-ft complete!"
echo "============================================="

# ---------- GPU Memory Troubleshooting ----------
# If you get OOM with 4x A100 40GB:
#   1. Reduce LoRA rank: r=64 instead of 128
#   2. Reduce per_device_train_batch_size to 1
#   3. Enable gradient_checkpointing: true in the yaml
#   4. Use --num_processes=2 with 2 GPUs
#
# If Babel only gives you 2x GPUs:
#   Change --gpus-per-task=2, --num_processes=2
#   And reduce batch size in the yaml config
