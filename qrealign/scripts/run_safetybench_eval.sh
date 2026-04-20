#!/bin/bash
# SafetyBench Eval - Refusal-Direction Quantized Models
# Runs only the safetybench eval on already-quantized models (skips extract + quantize steps)
#
# Usage:
#   bash scripts/run_safetybench_eval.sh                  # all models, int8
#   bash scripts/run_safetybench_eval.sh --mode int4      # all models, int4
#   bash scripts/run_safetybench_eval.sh sft-llama-2-7b-chat-hf-alpaca-hr0.1  # specific models only

MODE="int8"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [[ "$MODE" == "int8" ]]; then
    WBITS=8; ABITS=8
elif [[ "$MODE" == "int4" ]]; then
    WBITS=4; ABITS=16
else
    echo "Error: --mode must be int8 or int4"; exit 1
fi

LOG_DIR="logs/safetybench_eval"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING SAFETYBENCH EVAL JOBS"
echo "========================================="
echo "Start time: $(date)"
echo "Mode: W${WBITS}A${ABITS}"
echo "Log directory: $LOG_DIR"
echo ""

declare -a CONFIGS=(
    "sft-llama-2-7b-chat-hf-alpaca-hr0.05,meta-llama/Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.1,meta-llama/Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.15,meta-llama/Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.2,meta-llama/Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-gsm8k-hr0.15,meta-llama/Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-sst2-hr0.15,meta-llama/Llama-2-7b-chat-hf,48G"
)

if [ $# -gt 0 ]; then
    FILTER=("$@")
    FILTERED=()
    for config in "${CONFIGS[@]}"; do
        IFS=',' read -r folder _ _ <<< "$config"
        for f in "${FILTER[@]}"; do
            if [ "$folder" = "$f" ]; then
                FILTERED+=("$config")
                break
            fi
        done
    done
    CONFIGS=("${FILTERED[@]}")
    echo "Filtered to: ${FILTER[*]}"
    echo ""
fi

PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
MODELS_DIR="${PROJECT_DIR}/models"
OUTPUT_ROOT="${PROJECT_DIR}/results/refusal_dir"

JOB_IDS=()
counter=1

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r folder base_id memory <<< "$config"

    model_path="${MODELS_DIR}/${folder}"
    out_dir="${OUTPUT_ROOT}/${folder}/W${WBITS}A${ABITS}"

    if [ ! -f "${out_dir}/omni_parameters.pth" ]; then
        echo "  [$counter] SKIPPING ${folder}: no omni_parameters.pth found at ${out_dir}"
        ((counter++))
        continue
    fi

    JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sb_eval_${folder}
#SBATCH --output=${LOG_DIR}/${folder}_%j.out
#SBATCH --error=${LOG_DIR}/${folder}_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${memory}
#SBATCH --partition=general

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start time: \$(date)"
echo ""

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cd ${PROJECT_DIR}/qrealign

PYTHON_BIN=/data/user_data/ayushseh/safer-quantization/qrealign-venv/bin/python
if [ ! -x "\$PYTHON_BIN" ]; then
    PYTHON_BIN="\$(command -v python3)"
fi
echo "Using python: \$PYTHON_BIN"

EVAL_ARGS="--model_id ${base_id} --mode ${MODE} --resume ${model_path} --q_resume ${out_dir}/omni_parameters.pth"

echo "--- eval_safetybench ---"
"\$PYTHON_BIN" eval_safetybench.py \$EVAL_ARGS --output ${out_dir}/safetybench_eval.json

echo ""
echo "End time: \$(date)"
EOF
)

    JOB_IDS+=("${JOB_ID##* }")
    echo "  [$counter] ${folder}: $JOB_ID"
    ((counter++))
done

echo ""
echo "========================================="
echo "ALL JOBS SUBMITTED!"
echo "========================================="
echo ""
echo "Monitor jobs with:  squeue -u \$USER"
echo "Cancel all jobs:    scancel ${JOB_IDS[*]}"
echo ""
echo "Results will be saved in:"
echo "  Outputs: ${OUTPUT_ROOT}/"
echo "  Logs:    ${LOG_DIR}/"
echo ""

