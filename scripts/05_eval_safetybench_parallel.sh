#!/bin/bash

# Submit 3 parallel SafetyBench eval jobs (FP16, INT8, INT4)
# Usage:
#   bash scripts/05_eval_safetybench_parallel.sh

set -euo pipefail

BASE_DIR="/data/user_data/ayushseh/safer-quantization"
LOG_DIR="${BASE_DIR}/outputs/logs"
mkdir -p "${LOG_DIR}"

source "${BASE_DIR}/venv/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="/data/user_data/ayushseh/.cache/huggingface"
export HF_DATASETS_CACHE="/data/user_data/ayushseh/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data/user_data/ayushseh/.cache/huggingface/hub"
export HF_HUB_CACHE="/data/user_data/ayushseh/hf_cache/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

# Format: name|model_path|dtype|result_dir
MODELS=(
  #"fp16_baseline|meta-llama/Llama-2-7b-chat-hf|float16|${BASE_DIR}/baseline/results/fp16_baseline/safetybench"
  #"bnb_int8|${BASE_DIR}/int8_outputs/results/models/llama2-7b-chat-bnb-int8|auto|${BASE_DIR}/bnb_outputs/results/bnb_int8/safetybench"
  "bnb_int4|${BASE_DIR}/int4_outputs/results/models/llama2-7b-chat-bnb-int4|auto|${BASE_DIR}/bnb_outputs/results/bnb_int4/safetybench"
)

echo "========================================="
echo "SUBMITTING 3 SAFETYBENCH JOBS"
echo "========================================="
echo "Start time: $(date)"
echo ""

JOB_IDS=()
COUNT=1

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME MODEL_PATH MODEL_DTYPE RESULT_DIR <<< "${entry}"
  mkdir -p "${RESULT_DIR}"

  JOB_OUTPUT=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sb_${MODEL_NAME}
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/safetybench_${MODEL_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/safetybench_${MODEL_NAME}_%j.err

set -euo pipefail

source "${BASE_DIR}/venv/bin/activate"
cd "${BASE_DIR}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"

echo "Job ID: \$SLURM_JOB_ID"
echo "Model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Start: \$(date)"

python eval/run_safetybench.py \
  --model_path "${MODEL_PATH}" \
  --dtype "${MODEL_DTYPE}" \
  --output_dir "${RESULT_DIR}" \
  --safetybench_dir "/data/user_data/ayushseh/SafetyBench" \
  --lang en

echo "End: \$(date)"
EOF
)

  JOB_ID="${JOB_OUTPUT##* }"
  JOB_IDS+=("${JOB_ID}")

  echo "  [${COUNT}] ${MODEL_NAME}: ${JOB_OUTPUT}"
  COUNT=$((COUNT + 1))
done

echo ""
echo "========================================="
echo "ALL SAFETYBENCH JOBS SUBMITTED"
echo "========================================="
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel ${JOB_IDS[*]}"
echo "Logs:    ${LOG_DIR}/safetybench_*"
