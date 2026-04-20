#!/bin/bash
# Refusal-Direction Q-realign Batch Submitter
# Submits one SLURM job per finetuned model: extract directions → quantize → full eval suite
#
# Usage:
#   bash scripts/run_all_refusal.sh                                    # all models, int8, slr mode
#   bash scripts/run_all_refusal.sh --mode int4                        # all models, int4
#   bash scripts/run_all_refusal.sh --refusal-mode combined            # combined (SLR + weight regularizer)
#   bash scripts/run_all_refusal.sh --refusal-mode activation          # activation-space refusal dir
#   bash scripts/run_all_refusal.sh --refusal-mode combined --mu 0.001 # combined with smaller mu
#   bash scripts/run_all_refusal.sh sft-llama-2-7b-chat-hf-alpaca-hr0.1  # specific models only

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
elif [[ "$MODE" == "int4" ]]; then
    WBITS=4; ABITS=16
else
    echo "Error: --mode must be int8 or int4"; exit 1
fi

LOG_DIR="logs/refusal_dir"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING REFUSAL-DIR QUANTIZATION JOBS"
echo "========================================="
echo "Start time: $(date)"
echo "Mode: W${WBITS}A${ABITS}, refusal=${REFUSAL_MODE}$([ "$REFUSAL_MODE" = "combined" ] && echo " (mu=${MU})" || true)"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================================
# CONFIGURATION
# Format: "folder_name,base_model_id,net_name,memory"
# ============================================================================
declare -a CONFIGS=(
    "sft-llama-2-7b-chat-hf-alpaca-hr0.05,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.1,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.15,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.2,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-gsm8k-hr0.15,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-sst2-hr0.15,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
)

# If folder names are passed as arguments, filter to only those
if [ $# -gt 0 ]; then
    FILTER=("$@")
    FILTERED=()
    for config in "${CONFIGS[@]}"; do
        IFS=',' read -r folder _ _ _ <<< "$config"
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

# ============================================================================
# SUBMIT JOBS
# ============================================================================

PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
MODELS_DIR="${PROJECT_DIR}/models"
REFUSAL_DIR_DIR="${PROJECT_DIR}/refusal_direction/refusal_dirs"
OUTPUT_ROOT="${PROJECT_DIR}/results/refusal_dir"

JOB_IDS=()
counter=1

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r folder base_id net_name memory <<< "$config"

    model_path="${MODELS_DIR}/${folder}"
    out_dir="${OUTPUT_ROOT}/${folder}/W${WBITS}A${ABITS}_${REFUSAL_MODE}$([ "$REFUSAL_MODE" = "combined" ] && echo "_mu${MU}" || true)"
    refusal_pt="${REFUSAL_DIR_DIR}/${net_name}.pt"

    JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=rdir_${REFUSAL_MODE}_${folder}
#SBATCH --output=${LOG_DIR}/${folder}_%j.out
#SBATCH --error=${LOG_DIR}/${folder}_%j.err
#SBATCH --time=24:00:00
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

PYTHON_BIN=/data/user_data/ayushseh/safer-quantization/refusal-venv/bin/python
if [ ! -x "\$PYTHON_BIN" ]; then
    PYTHON_BIN="\$(command -v python3)"
fi
echo "Using python: \$PYTHON_BIN"

# Step 1: extract refusal directions (skipped if already done for this base model)
if [ ! -f "${refusal_pt}" ]; then
    echo "--- Extracting refusal directions for ${net_name} ---"
    "\$PYTHON_BIN" extract_refusal_directions.py \
        --model ${base_id} \
        --net ${net_name} \
        --data data.json \
        --output_dir ${REFUSAL_DIR_DIR}
else
    echo "--- Refusal directions already exist: ${refusal_pt} ---"
fi

# Step 2: quantize with refusal-direction re-separation loss
PYTHON_BIN=/data/user_data/ayushseh/safer-quantization/qrealign-venv/bin/python
mkdir -p ${out_dir}
echo "--- Quantizing ${folder} (W${WBITS}A${ABITS}) ---"
"\$PYTHON_BIN" main.py \
    --model ${base_id} \
    --model_resume ${model_path} \
    --net ${net_name} \
    --output_dir ${out_dir} \
    --wbits ${WBITS} \
    --abits ${ABITS} \
    --lwc \
    --let \
    --let_lr 1e-3 \
    --epochs 10 \
    --use_refusal_dir ${REFUSAL_MODE} \
    --refusal_weight_mu ${MU}

EVAL_ARGS="--model_id ${base_id} --mode ${MODE} --resume ${model_path} --q_resume ${out_dir}/omni_parameters.pth"

# Step 3: safety evaluations
echo "--- attack_eval (AdvBench) ---"
"\$PYTHON_BIN" attack_eval.py \$EVAL_ARGS --output ${out_dir}/advbench_eval.json

echo "--- eval_safetybench ---"
"\$PYTHON_BIN" eval_safetybench.py \$EVAL_ARGS --output ${out_dir}/safetybench_eval.json

# Step 4: utility evaluations
echo "--- eval_mmlu ---"
"\$PYTHON_BIN" eval_mmlu.py \$EVAL_ARGS --output ${out_dir}/mmlu_eval.json

echo "--- eval_gsm8k ---"
"\$PYTHON_BIN" eval_gsm8k.py \$EVAL_ARGS --output ${out_dir}/gsm8k_eval.json

echo "--- eval_sst2 ---"
"\$PYTHON_BIN" eval_sst2.py \$EVAL_ARGS --output ${out_dir}/sst2_eval.json

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

# ============================================================================
# SUMMARY
# ============================================================================

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

