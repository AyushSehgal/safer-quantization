#!/bin/bash
# Submit Q-realign quantization jobs using the MLP probe loss (--use_refusal_dir mlp).
# Mirrors run_all_refusal.sh but with the MLP probe instead of SLR.
# Run after submit_train_mlp_probes.sh completes (or pass --after <job_ids>).
#
# Usage:
#   bash scripts/submit_qrealign_mlp.sh                          # all models, both W8A8 + W4A16
#   bash scripts/submit_qrealign_mlp.sh --mode int8              # W8A8 only
#   bash scripts/submit_qrealign_mlp.sh --mode int4              # W4A16 only
#   bash scripts/submit_qrealign_mlp.sh --after "123 456 789"   # wait for probe job IDs
#   bash scripts/submit_qrealign_mlp.sh sft-llama-2-7b-chat-hf-alpaca-hr0.1  # specific folder

MODE="both"
AFTER_JOBS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)  MODE="$2";       shift 2 ;;
        --after) AFTER_JOBS="$2"; shift 2 ;;
        *) break ;;
    esac
done

PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
MODELS_DIR="${PROJECT_DIR}/models"
OUTPUT_ROOT="${PROJECT_DIR}/results/mlp_probe"
QREALIGN_DIR="${PROJECT_DIR}/qrealign"
LOG_DIR="${QREALIGN_DIR}/logs/mlp_qrealign"
PYTHON_BIN="${PROJECT_DIR}/qrealign-venv/bin/python"
mkdir -p "$LOG_DIR"

# SLURM dependency string (empty = no dependency)
if [ -n "$AFTER_JOBS" ]; then
    # Convert space-separated IDs to afterok:123:456:789
    DEP="--dependency=afterok:$(echo "$AFTER_JOBS" | tr ' ' ':')"
else
    DEP=""
fi

# SFT model configs — same set as run_all_refusal.sh
# Format: "folder_name,base_model_id,net_name,mem"
declare -a CONFIGS=(
    "sft-llama-2-7b-chat-hf-alpaca-hr0.05,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.1,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.15,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-alpaca-hr0.2,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-gsm8k-hr0.15,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
    "sft-llama-2-7b-chat-hf-sst2-hr0.15,meta-llama/Llama-2-7b-chat-hf,Llama-2-7b-chat-hf,48G"
)

# Filter to specific folders if passed as positional args
if [ $# -gt 0 ]; then
    FILTER=("$@")
    FILTERED=()
    for cfg in "${CONFIGS[@]}"; do
        IFS=',' read -r folder _ _ _ <<< "$cfg"
        for f in "${FILTER[@]}"; do
            if [ "$folder" = "$f" ]; then FILTERED+=("$cfg"); break; fi
        done
    done
    CONFIGS=("${FILTERED[@]}")
    echo "Filtered to: ${FILTER[*]}"
fi

# Quant modes to run
declare -a QUANT_MODES=()
if [[ "$MODE" == "both" || "$MODE" == "int8" ]]; then
    QUANT_MODES+=("8,8,int8")
fi
if [[ "$MODE" == "both" || "$MODE" == "int4" ]]; then
    QUANT_MODES+=("4,16,int4")
fi
if [ ${#QUANT_MODES[@]} -eq 0 ]; then
    echo "Error: --mode must be int8, int4, or both"; exit 1
fi

echo "============================================="
echo " SUBMITTING Q-REALIGN (MLP PROBE) JOBS"
echo "============================================="
echo "Mode(s): $MODE   Dependency: ${DEP:-none}"
echo "Log dir: $LOG_DIR"
echo ""

JOB_IDS=()
counter=1

for qmode in "${QUANT_MODES[@]}"; do
    IFS=',' read -r wbits abits mode_label <<< "$qmode"

    for cfg in "${CONFIGS[@]}"; do
        IFS=',' read -r folder base_id net_name mem <<< "$cfg"

        model_path="${MODELS_DIR}/${folder}"
        out_dir="${OUTPUT_ROOT}/${folder}/W${wbits}A${abits}_mlp"
        mlp_path="${QREALIGN_DIR}/MLPs/MLP_$(echo "$net_name" | tr '[:upper:]' '[:lower:]')_W${wbits}A${abits}.pt"

        JOB_ID=$(sbatch $DEP <<EOF
#!/bin/bash
#SBATCH --job-name=mlp_${mode_label}_${folder}
#SBATCH --output=${LOG_DIR}/${folder}_${mode_label}_%j.out
#SBATCH --error=${LOG_DIR}/${folder}_${mode_label}_%j.err
#SBATCH --time=24:00:00
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

# Guard: probe must exist before starting
if [ ! -f "${mlp_path}" ]; then
    echo "ERROR: MLP probe not found at ${mlp_path}"
    echo "Run submit_train_mlp_probes.sh first."
    exit 1
fi

mkdir -p ${out_dir}
echo "--- Q-realign (MLP probe, W${wbits}A${abits}): ${folder} ---"
${PYTHON_BIN} main.py \
    --model         ${base_id} \
    --model_resume  ${model_path} \
    --net           ${net_name} \
    --output_dir    ${out_dir} \
    --wbits         ${wbits} \
    --abits         ${abits} \
    --lwc \
    --let \
    --let_lr        1e-3 \
    --epochs        10 \
    --use_refusal_dir mlp

EVAL_ARGS="--model_id ${base_id} --mode ${mode_label} --resume ${model_path} --q_resume ${out_dir}/omni_parameters.pth"

echo "--- eval_safetybench ---"
${PYTHON_BIN} eval_safetybench.py \$EVAL_ARGS --output ${out_dir}/safetybench_eval.json

echo "--- eval_mmlu ---"
${PYTHON_BIN} eval_mmlu.py \$EVAL_ARGS --output ${out_dir}/mmlu_eval.json

echo "--- eval_wikitext_ppl ---"
${PYTHON_BIN} eval_wikitext_ppl.py \$EVAL_ARGS --output ${out_dir}/wikitext_ppl_eval.json

echo ""
echo "Done: \$(date)"
EOF
)
        job_num="${JOB_ID##* }"
        JOB_IDS+=("$job_num")
        echo "  [$counter] ${folder} W${wbits}A${abits}: ${JOB_ID}"
        ((counter++))
    done
done

echo ""
echo "============================================="
echo " ${#JOB_IDS[@]} JOB(S) SUBMITTED"
echo "============================================="
echo "Monitor:     squeue -u \$USER"
echo "Cancel all:  scancel ${JOB_IDS[*]}"
echo "Logs:        $LOG_DIR"
echo "Results:     $OUTPUT_ROOT"
echo ""
