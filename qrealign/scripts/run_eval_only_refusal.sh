#!/bin/bash
# Refusal-Direction Eval-Only Batch Submitter
# Submits one SLURM job per finetuned model and runs evaluations only
# (expects quantized checkpoint at results/refusal_dir/.../omni_parameters.pth).
#
# Usage:
#   bash scripts/run_eval_only_refusal.sh                                        # all models, int8, slr
#   bash scripts/run_eval_only_refusal.sh --mode w8a16                           # eval W8A16 checkpoints
#   bash scripts/run_eval_only_refusal.sh --mode w4a4                            # eval W4A4 checkpoints
#   bash scripts/run_eval_only_refusal.sh --mode w4a8                            # eval W4A8 checkpoints
#   bash scripts/run_eval_only_refusal.sh --mode int4                            # eval W4A16 checkpoints
#   bash scripts/run_eval_only_refusal.sh --refusal-mode combined --mu 0.001     # eval combined(mu=0.001)
#   bash scripts/run_eval_only_refusal.sh sft-llama-2-7b-chat-hf-alpaca-hr0.1    # specific models only

set -euo pipefail

MODE="int8"
REFUSAL_MODE="slr"
MU="0.01"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --refusal-mode) REFUSAL_MODE="$2"; shift 2 ;;
        --mu) MU="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [[ "$REFUSAL_MODE" != "slr" && "$REFUSAL_MODE" != "activation" && "$REFUSAL_MODE" != "combined" ]]; then
    echo "Error: --refusal-mode must be slr, activation, or combined"; exit 1
fi

if [[ "$MODE" == "int8" ]]; then
    WBITS=8; ABITS=8
elif [[ "$MODE" == "w8a16" ]]; then
    WBITS=8; ABITS=16
elif [[ "$MODE" == "w4a4" ]]; then
    WBITS=4; ABITS=4
elif [[ "$MODE" == "w4a8" ]]; then
    WBITS=4; ABITS=8
elif [[ "$MODE" == "int4" ]]; then
    WBITS=4; ABITS=16
else
    echo "Error: --mode must be int8, w8a16, w4a4, w4a8, or int4"; exit 1
fi

LOG_DIR="logs/refusal_dir_eval"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING REFUSAL-DIR EVAL-ONLY JOBS"
echo "========================================="
echo "Start time: $(date)"
echo "Mode: W${WBITS}A${ABITS}, refusal=${REFUSAL_MODE}$([ "$REFUSAL_MODE" = "combined" ] && echo " (mu=${MU})" || true)"
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
    out_dir="${OUTPUT_ROOT}/${folder}/W${WBITS}A${ABITS}_${REFUSAL_MODE}$([ "$REFUSAL_MODE" = "combined" ] && echo "_mu${MU}" || true)"

    if [ ! -f "${out_dir}/omni_parameters.pth" ]; then
        echo "  [$counter] SKIPPING ${folder}: no omni_parameters.pth found at ${out_dir}"
        ((counter++))
        continue
    fi

    JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${REFUSAL_MODE}_${folder}
#SBATCH --output=${LOG_DIR}/${folder}_%j.out
#SBATCH --error=${LOG_DIR}/${folder}_%j.err
#SBATCH --time=8:00:00
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

echo "--- eval_mmlu ---"
"\$PYTHON_BIN" eval_mmlu.py \$EVAL_ARGS --output ${out_dir}/mmlu_eval.json

echo "--- eval_wikitext_ppl ---"
"\$PYTHON_BIN" eval_wikitext_ppl.py \$EVAL_ARGS --output ${out_dir}/wikitext_ppl_eval.json

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
echo "ALL EVAL-ONLY JOBS SUBMITTED!"
echo "========================================="
echo ""
echo "Monitor jobs with:  squeue -u \$USER"
echo "Cancel all jobs:    scancel ${JOB_IDS[*]}"
echo ""
echo "Results will be saved in:"
echo "  Outputs: ${OUTPUT_ROOT}/"
echo "  Logs:    ${LOG_DIR}/"
echo ""
