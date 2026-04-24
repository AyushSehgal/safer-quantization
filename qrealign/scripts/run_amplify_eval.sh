#!/bin/bash
#SBATCH --job-name=amplify_eval
#SBATCH --output=logs/amplify/amplify_eval_%j.out
#SBATCH --error=logs/amplify/amplify_eval_%j.err
#SBATCH --time=06:00:00
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

# ── Sanity checks ──────────────────────────────────────────────────────────────
for f in "$REFUSAL_DIR" "$SAFETYBENCH_DIR"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: required path not found: $f"
        exit 1
    fi
done

cd "$QREALIGN_DIR"
mkdir -p logs/amplify

Q_RESUME_INT8="${QUANTIZED_ROOT}/W8A8/omni_parameters.pth"
Q_RESUME_INT4="${QUANTIZED_ROOT}/W4A16/omni_parameters.pth"

# ── Layer sweep (int8, 200-question SafetyBench subset) ───────────────────────
echo "============================================================"
echo "  [1/3] Layer sweep — W8A8"
echo "============================================================"

if [[ ! -f "$Q_RESUME_INT8" ]]; then
    echo "WARNING: $Q_RESUME_INT8 not found — skipping sweep."
else
    "$PYTHON" eval_amplify_refusal.py \
        --model_id        "$MODEL_ID" \
        --mode            int8 \
        --q_resume        "$Q_RESUME_INT8" \
        --refusal_dir     "$REFUSAL_DIR" \
        --sweep \
        --sweep_alpha     20.0 \
        --sweep_limit     200 \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --output          "${RESULTS_ROOT}/sweep"
fi

# ── Pick best layers from sweep output ────────────────────────────────────────
SWEEP_JSON="${RESULTS_ROOT}/sweep/layer_sweep.json"
BEST_LAYERS="middle"   # fallback if sweep didn't run

if [[ -f "$SWEEP_JSON" ]]; then
    BEST_LAYERS=$("$PYTHON" -c "
import json
data = json.load(open('$SWEEP_JSON'))
ranked = sorted(
    [(int(k), v['delta']) for k, v in data['per_layer'].items()],
    key=lambda x: x[1], reverse=True
)
top = [str(k) for k, d in ranked[:5] if d > 0]
print(','.join(top) if top else 'middle')
")
    echo "Sweep selected layers: $BEST_LAYERS"
else
    echo "No sweep output found — using default layers: $BEST_LAYERS"
fi

# ── W8A8 ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  [2/3] Standard eval — W8A8 (int8)"
echo "============================================================"

if [[ ! -f "$Q_RESUME_INT8" ]]; then
    echo "WARNING: $Q_RESUME_INT8 not found — skipping int8."
else
    "$PYTHON" eval_amplify_refusal.py \
        --model_id        "$MODEL_ID" \
        --mode            int8 \
        --q_resume        "$Q_RESUME_INT8" \
        --refusal_dir     "$REFUSAL_DIR" \
        --alpha           20.0 \
        --layers          "$BEST_LAYERS" \
        --tasks           safetybench,mmlu,wikitext \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --output          "${RESULTS_ROOT}/int8"
fi

# ── W4A16 ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  [3/3] Standard eval — W4A16 (int4)"
echo "============================================================"

if [[ ! -f "$Q_RESUME_INT4" ]]; then
    echo "WARNING: $Q_RESUME_INT4 not found — skipping int4."
else
    "$PYTHON" eval_amplify_refusal.py \
        --model_id        "$MODEL_ID" \
        --mode            int4 \
        --q_resume        "$Q_RESUME_INT4" \
        --refusal_dir     "$REFUSAL_DIR" \
        --alpha           20.0 \
        --layers          "$BEST_LAYERS" \
        --tasks           safetybench,mmlu,wikitext \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --output          "${RESULTS_ROOT}/int4"
fi

echo ""
echo "End: $(date)"
echo "Results in: ${RESULTS_ROOT}/"
