#!/bin/bash
#SBATCH --job-name=amplify_alpha_sweep
#SBATCH --output=logs/amplify/amplify_alpha_sweep_%j.out
#SBATCH --error=logs/amplify/amplify_alpha_sweep_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=general

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:     $(date)"
echo ""

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
QREALIGN_DIR="${PROJECT_DIR}/qrealign"
PYTHON="${PROJECT_DIR}/qrealign-venv/bin/python3"

MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
MODEL_NICK="llama-2-7b-chat-hf"

QUANTIZED_ROOT="${PROJECT_DIR}/quantized_models/Llama-2-7b-chat-hf"
REFUSAL_DIR="${PROJECT_DIR}/refusal_direction/refusal_dirs/${MODEL_NICK}.pt"
SAFETYBENCH_DIR="${QREALIGN_DIR}/SafetyBench"
RESULTS_ROOT="${PROJECT_DIR}/results/amplify/${MODEL_NICK}"

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

# ── From sweep results ─────────────────────────────────────────────────────────
LAYERS="2,13,14,15,16"
ALPHAS="5,10,15,20,30,50"

cd "$QREALIGN_DIR"
mkdir -p logs/amplify

# ── W8A8 ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  [1/2] Alpha sweep — W8A8 (int8)  layers=${LAYERS}"
echo "============================================================"

"$PYTHON" eval_amplify_refusal.py \
    --model_id        "$MODEL_ID" \
    --mode            int8 \
    --q_resume        "${QUANTIZED_ROOT}/W8A8/omni_parameters.pth" \
    --refusal_dir     "$REFUSAL_DIR" \
    --alpha_sweep     "$ALPHAS" \
    --layers          "$LAYERS" \
    --tasks           safetybench,mmlu \
    --safetybench_dir "$SAFETYBENCH_DIR" \
    --limit           500 \
    --output          "${RESULTS_ROOT}/alpha_sweep_int8"

# ── W4A16 ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  [2/2] Alpha sweep — W4A16 (int4)  layers=${LAYERS}"
echo "============================================================"

"$PYTHON" eval_amplify_refusal.py \
    --model_id        "$MODEL_ID" \
    --mode            int4 \
    --q_resume        "${QUANTIZED_ROOT}/W4A16/omni_parameters.pth" \
    --refusal_dir     "$REFUSAL_DIR" \
    --alpha_sweep     "$ALPHAS" \
    --layers          "$LAYERS" \
    --tasks           safetybench,mmlu \
    --safetybench_dir "$SAFETYBENCH_DIR" \
    --limit           500 \
    --output          "${RESULTS_ROOT}/alpha_sweep_int4"

echo ""
echo "End: $(date)"
echo "Results in: ${RESULTS_ROOT}/"
