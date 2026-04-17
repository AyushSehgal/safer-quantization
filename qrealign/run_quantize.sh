#!/bin/bash

# Q-Realign Quantization Job Submitter
# Quantizes a model with W8A8 (int8) or W4A16 (int4) and saves omni_parameters.pth
#
# Usage:
#   bash run_quantize.sh --mode int8
#   bash run_quantize.sh --mode int4
#   bash run_quantize.sh --mode int8 --model_id meta-llama/Llama-2-13b-chat-hf
#   bash run_quantize.sh --mode int8 --out_dir /data/user_data/ayushseh/quantized_models

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

MODE=""
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_quantize.sh --mode <int8|int4> [--model_id MODEL_ID] [--out_dir DIR]"
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Error: --mode is required (int8 or int4)"
    exit 1
fi

if [[ "$MODE" != "int8" && "$MODE" != "int4" ]]; then
    echo "Error: --mode must be int8 or int4"
    exit 1
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
LOG_DIR="${PROJECT_DIR}/logs/quantize"

MODEL_NICK=$(basename "$MODEL_ID")
JOB_SUFFIX="${MODEL_NICK}_${MODE}"

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="${PROJECT_DIR}/quantized_models"
fi

mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING QUANTIZATION JOB"
echo "========================================="
echo "Start time:  $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Model:       $MODEL_ID"
echo "Mode:        $MODE"
echo "Output dir:  $OUT_DIR"
echo "Log dir:     $LOG_DIR"
echo ""

# ============================================================================
# SUBMIT JOB
# ============================================================================

JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=quant_${JOB_SUFFIX}
#SBATCH --output=${LOG_DIR}/${JOB_SUFFIX}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_SUFFIX}_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general

echo "Job ID:    \$SLURM_JOB_ID"
echo "Node:      \$SLURM_NODELIST"
echo "Start:     \$(date)"
echo "GPU:       \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cd ${PROJECT_DIR}
source anlp/bin/activate

echo "Running quantize_and_save.py ..."
echo ""

python quantize_and_save.py \
    --model_id ${MODEL_ID} \
    --mode ${MODE} \
    --out_dir ${OUT_DIR}

echo ""
echo "End: \$(date)"
EOF
)

JOB_NUM="${JOB_ID##* }"

echo "  $JOB_ID"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================="
echo "JOB SUBMITTED!"
echo "========================================="
echo ""
echo "Monitor with:    squeue -u \$USER"
echo "View stdout:     tail -f ${LOG_DIR}/${JOB_SUFFIX}_${JOB_NUM}.out"
echo "View stderr:     tail -f ${LOG_DIR}/${JOB_SUFFIX}_${JOB_NUM}.err"
echo "Cancel with:     scancel ${JOB_NUM}"
echo ""
echo "Saved to:        ${OUT_DIR}/${MODEL_NICK}/$([ "$MODE" = "int8" ] && echo "W8A8" || echo "W4A16")/omni_parameters.pth"
echo ""
