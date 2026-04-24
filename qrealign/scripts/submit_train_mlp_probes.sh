#!/bin/bash
# Submit MLP probe training jobs — one per (base model × quant mode).
# Probes are trained jointly on FP16 + quantized activations and saved to
# MLPs/MLP_{net}_W{w}A{a}.pt.  Run this BEFORE submit_qrealign_mlp.sh.
#
# Usage:
#   bash scripts/submit_train_mlp_probes.sh               # both W8A8 and W4A16
#   bash scripts/submit_train_mlp_probes.sh --mode int8   # W8A8 only
#   bash scripts/submit_train_mlp_probes.sh --mode int4   # W4A16 only

MODE="both"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
QREALIGN_DIR="${PROJECT_DIR}/qrealign"
MLP_DIR="${QREALIGN_DIR}/MLPs"
LOG_DIR="${QREALIGN_DIR}/logs/mlp_probe"
PYTHON_BIN="${PROJECT_DIR}/qrealign-venv/bin/python"
mkdir -p "$LOG_DIR" "$MLP_DIR"

# One probe per base model — shared across all SFT variants of the same base.
# Format: "hf_model_id,net_name,mem"
declare -a BASE_MODELS=(
    "meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
)

# Build the list of (wbits, abits, label) pairs to train
declare -a QUANT_MODES=()
if [[ "$MODE" == "both" || "$MODE" == "int8" ]]; then
    QUANT_MODES+=("8,8,W8A8")
fi
if [[ "$MODE" == "both" || "$MODE" == "int4" ]]; then
    QUANT_MODES+=("4,16,W4A16")
fi
if [ ${#QUANT_MODES[@]} -eq 0 ]; then
    echo "Error: --mode must be int8, int4, or both"; exit 1
fi

echo "========================================="
echo " SUBMITTING MLP PROBE TRAINING JOBS"
echo "========================================="
echo "Mode(s): $MODE    Log dir: $LOG_DIR"
echo ""

JOB_IDS=()
counter=1

for qmode in "${QUANT_MODES[@]}"; do
    IFS=',' read -r wbits abits label <<< "$qmode"

    for cfg in "${BASE_MODELS[@]}"; do
        IFS=',' read -r model_id net_name mem <<< "$cfg"
        net_lower=$(echo "$net_name" | tr '[:upper:]' '[:lower:]')
        probe_out="${MLP_DIR}/MLP_${net_lower}_W${wbits}A${abits}.pt"

        JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mlp_probe_${label}_${net_lower}
#SBATCH --output=${LOG_DIR}/${net_lower}_${label}_%j.out
#SBATCH --error=${LOG_DIR}/${net_lower}_${label}_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${mem}
#SBATCH --partition=general

echo "Job ID: \$SLURM_JOB_ID  Node: \$SLURM_NODELIST"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: \$(date)"
echo ""

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cd ${QREALIGN_DIR}

if [ -f "${probe_out}" ]; then
    echo "Probe already exists: ${probe_out} — skipping."
    exit 0
fi

echo "--- Training MLP probe: ${label} for ${net_name} ---"
${PYTHON_BIN} train_mlp_probe.py \
    --model   ${model_id} \
    --net     ${net_name} \
    --wbits   ${wbits} \
    --abits   ${abits} \
    --data    data.json \
    --nsamples 800 \
    --seqlen   128 \
    --hidden   128 \
    --epochs   300 \
    --lr       1e-3 \
    --output_dir MLPs

echo ""
echo "Done: \$(date)"
EOF
)
        job_num="${JOB_ID##* }"
        JOB_IDS+=("$job_num")
        echo "  [$counter] ${net_name} ${label}: ${JOB_ID}"
        ((counter++))
    done
done

echo ""
echo "========================================="
echo " ${#JOB_IDS[@]} JOB(S) SUBMITTED"
echo "========================================="
echo "Monitor:     squeue -u \$USER"
echo "Cancel all:  scancel ${JOB_IDS[*]}"
echo "Logs:        $LOG_DIR"
echo ""
echo "Probes will be written to: $MLP_DIR"
echo ""
echo "When complete, run Q-realign with:"
echo "  bash scripts/submit_qrealign_mlp.sh [--mode int8|int4]"
echo ""
# Print job IDs for use as SLURM dependencies
echo "PROBE_JOB_IDS=${JOB_IDS[*]}"
