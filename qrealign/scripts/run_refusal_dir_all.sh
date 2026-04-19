#!/bin/bash
# Submit one SLURM job per finetuned model in MODELS_DIR using the refusal-direction loss.
#
# Usage:
#   bash scripts/run_refusal_dir_all.sh
#   bash scripts/run_refusal_dir_all.sh --mode int4
#   bash scripts/run_refusal_dir_all.sh --dry_run   # print jobs without submitting

set -euo pipefail

# ============================================================================
# CONFIGURATION — edit these paths for your cluster environment
# ============================================================================
PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
MODELS_DIR="${PROJECT_DIR}/models"          # where finetuned model folders live
REFUSAL_DIR_DIR="${PROJECT_DIR}/refusal_direction/refusal_dirs"  # output of extract_refusal_directions.py
OUTPUT_ROOT="${PROJECT_DIR}/results/refusal_dir"        # per-model output dirs go here
LOG_DIR="${PROJECT_DIR}/logs/refusal_dir"

HF_HUB_CACHE="/data/user_data/ayushseh/.hf_cache"
CONDA_ENV="q-realign"
ACCOUNT="cis250260p"
PARTITION="GPU-shared"
GPU="v100-32:1"
TIME="20:00:00"

MODE="int8"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "int8" ]]; then
    WBITS=8; ABITS=8
elif [[ "$MODE" == "int4" ]]; then
    WBITS=4; ABITS=16
else
    echo "Error: --mode must be int8 or int4"; exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

# ============================================================================
# BASE MODEL MAPPING
# Infer HuggingFace model ID and --net name from the folder name prefix.
# Add entries here if you add new model families.
# ============================================================================
get_base_model() {
    local folder="$1"
    if   [[ "$folder" == *"llama-2-7b-chat-hf"* ]]; then
        echo "meta-llama/Llama-2-7b-chat-hf llama-2-7b-chat-hf"
    elif [[ "$folder" == *"llama-2-13b-chat-hf"* ]]; then
        echo "meta-llama/Llama-2-13b-chat-hf llama-2-13b-chat-hf"
    elif [[ "$folder" == *"gemma-2-9b"* ]]; then
        echo "google/gemma-2-9b-it gemma-2-9b-it"
    elif [[ "$folder" == *"qwen2.5-7b"* ]]; then
        echo "Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct"
    else
        echo "UNKNOWN UNKNOWN"
    fi
}

# ============================================================================
# TRACK WHICH BASE MODELS NEED DIRECTION EXTRACTION
# (only extract once per base model, not once per finetuned variant)
# ============================================================================
declare -A EXTRACTED_BASES   # net_name -> 1 if extraction job already submitted

echo "========================================="
echo "Submitting refusal-direction jobs"
echo "Mode: W${WBITS}A${ABITS} | Dry-run: $DRY_RUN"
echo "========================================="

for MODEL_PATH in "${MODELS_DIR}"/*/; do
    FOLDER=$(basename "$MODEL_PATH")
    read -r BASE_ID NET_NAME <<< "$(get_base_model "$FOLDER")"

    if [[ "$BASE_ID" == "UNKNOWN" ]]; then
        echo "  [SKIP] $FOLDER — no base model mapping found"
        continue
    fi

    JOB_NAME="rdir_${FOLDER}"
    OUT_DIR="${OUTPUT_ROOT}/${FOLDER}/W${WBITS}A${ABITS}"
    LOG_OUT="${LOG_DIR}/${FOLDER}_%j.out"
    LOG_ERR="${LOG_DIR}/${FOLDER}_%j.err"
    REFUSAL_PT="${REFUSAL_DIR_DIR}/${NET_NAME}.pt"

    echo ""
    echo "  Model:     $FOLDER"
    echo "  Base:      $BASE_ID  (net=$NET_NAME)"
    echo "  Output:    $OUT_DIR"

    if $DRY_RUN; then
        echo "  [DRY-RUN] would submit job: $JOB_NAME"
        continue
    fi

    sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_OUT}
#SBATCH --error=${LOG_ERR}
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${memory}
#SBATCH --partition=general

set -e

source anlp/bin/activate 

export HF_HOME=${HF_HUB_CACHE}
export HF_HUB_CACHE=${HF_HUB_CACHE}

echo "==== Job: ${JOB_NAME} ===="
echo "Node: \$SLURM_NODELIST"
echo "GPU:  \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: \$(date)"
echo ""

# Step 1: extract refusal directions from the base model if not already done
if [ ! -f "${REFUSAL_PT}" ]; then
    echo "--- Extracting refusal directions for ${NET_NAME} ---"
    python extract_refusal_directions.py \
        --model ${BASE_ID} \
        --net ${NET_NAME} \
        --data data.json \
        --output_dir ${REFUSAL_DIR_DIR}
else
    echo "--- Refusal directions already exist: ${REFUSAL_PT} ---"
fi

# Step 2: quantize the finetuned model with refusal-direction re-separation loss
mkdir -p ${OUT_DIR}
echo "--- Quantizing ${FOLDER} (W${WBITS}A${ABITS}) ---"
python main.py \
    --model ${BASE_ID} \
    --model_resume ${MODEL_PATH} \
    --net ${NET_NAME} \
    --output_dir ${OUT_DIR} \
    --wbits ${WBITS} \
    --abits ${ABITS} \
    --lwc \
    --let \
    --let_lr 1e-3 \
    --epochs 10 \
    --use_refusal_dir

EVAL_ARGS="--model_id ${BASE_ID} --mode ${MODE} --resume ${MODEL_PATH} --q_resume ${OUT_DIR}/omni_parameters.pth"

# Step 3: safety evaluations
echo "--- attack_eval (AdvBench) ---"
python attack_eval.py \$EVAL_ARGS \
    --output ${OUT_DIR}/advbench_eval.json

echo "--- eval_safetybench ---"
python eval_safetybench.py \$EVAL_ARGS \
    --output ${OUT_DIR}/safetybench_eval.json

# Step 4: utility evaluations
echo "--- eval_mmlu ---"
python eval_mmlu.py \$EVAL_ARGS \
    --output ${OUT_DIR}/mmlu_eval.json

echo "--- eval_gsm8k ---"
python eval_gsm8k.py \$EVAL_ARGS \
    --output ${OUT_DIR}/gsm8k_eval.json

echo "--- eval_sst2 ---"
python eval_sst2.py \$EVAL_ARGS \
    --output ${OUT_DIR}/sst2_eval.json

echo "--- eval_wikitext_ppl ---"
python eval_wikitext_ppl.py \$EVAL_ARGS \
    --output ${OUT_DIR}/wikitext_ppl_eval.json

echo ""
echo "Done: \$(date)"
SLURM

done

echo ""
echo "========================================="
echo "All jobs submitted."
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ${LOG_DIR}/"
echo "Outputs:  ${OUTPUT_ROOT}/"
echo "========================================="
