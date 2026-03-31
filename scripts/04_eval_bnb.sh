#!/bin/bash
#SBATCH --job-name=qresafe-eval-bnb
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/ayushseh/safer-quantization/outputs/logs/eval_bnb_%j.out
#SBATCH --error=/data/user_data/ayushseh/safer-quantization/outputs/logs/eval_bnb_%j.err

# ============================================================================
# 04_eval_bnb.sh — Evaluate INT8 and INT4 bitsandbytes models
# ============================================================================
# Runs all 4 benchmarks on both bnb-int8 and bnb-int4 saved model variants.
# Submit after 02 and 03 finish, or chain with --dependency=afterok:JOB_INT8:JOB_INT4.

set -euo pipefail

mkdir -p /data/user_data/ayushseh/safer-quantization/outputs/logs

source /data/user_data/ayushseh/safer-quantization/venv/bin/activate

export BASE_DIR="/data/user_data/ayushseh/safer-quantization"
export OUTPUT_DIR="${BASE_DIR}/bnb_outputs"
export PYTHONUNBUFFERED=1
export HF_HOME="/data/user_data/ayushseh/.cache/huggingface"
export HF_DATASETS_CACHE="/data/user_data/ayushseh/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data/user_data/ayushseh/.cache/huggingface/hub"
export HF_HUB_CACHE="/data/user_data/ayushseh/hf_cache/hub"
mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE} ${HF_HUB_CACHE}

cd ${BASE_DIR}

# Format: "model_name|model_path|lm_eval_model_args"
# Saved bnb models embed their quantization_config, so pretrained= is sufficient.
INT8_MODEL="${BASE_DIR}/int8_outputs/results/models/llama2-7b-chat-bnb-int8"
INT4_MODEL="${BASE_DIR}/int4_outputs/results/models/llama2-7b-chat-bnb-int4"

declare -a MODELS=(
    "bnb_int8|${INT8_MODEL}|pretrained=${INT8_MODEL},load_in_8bit=True"
    "bnb_int4|${INT4_MODEL}|pretrained=${INT4_MODEL},load_in_4bit=True"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH MODEL_ARGS <<< "${entry}"
    RESULT_DIR="${OUTPUT_DIR}/results/${MODEL_NAME}"
    mkdir -p ${RESULT_DIR}

    echo "============================================="
    echo "Evaluating: ${MODEL_NAME}"
    echo "Path: ${MODEL_PATH}"
    echo "============================================="

    # --- 1. MMLU (5-shot) ---
    echo ">>> [1/4] MMLU..."
    lm_eval --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 8 \
        --output_path ${RESULT_DIR}/mmlu \
        2>&1 | tee ${RESULT_DIR}/mmlu.log || echo "MMLU FAILED for ${MODEL_NAME}"

    # --- 2. Wikitext-2 PPL ---
    echo ">>> [2/4] Wikitext-2 PPL..."
    lm_eval --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks wikitext \
        --batch_size 1 \
        --output_path ${RESULT_DIR}/wikitext \
        2>&1 | tee ${RESULT_DIR}/wikitext_ppl.log || echo "Wikitext FAILED for ${MODEL_NAME}"

    # --- 3. AdvBench ASR ---
    echo ">>> [3/4] AdvBench ASR..."
    python eval/run_advbench_asr.py \
        --model_path "${MODEL_PATH}" \
        --output_dir ${RESULT_DIR}/advbench \
        --num_prompts 520 \
        --batch_size 4 \
        2>&1 | tee ${RESULT_DIR}/advbench_asr.log || echo "AdvBench FAILED for ${MODEL_NAME}"

    # --- 4. SafetyBench ---
    echo ">>> [4/4] SafetyBench..."
    python eval/run_safetybench.py \
        --model_path "${MODEL_PATH}" \
        --output_dir ${RESULT_DIR}/safetybench \
        2>&1 | tee ${RESULT_DIR}/safetybench.log || echo "SafetyBench FAILED for ${MODEL_NAME}"

    echo ">>> Done: ${MODEL_NAME}"
    echo ""
done

echo "============================================="
echo "BNB model evaluations complete!"
echo "Results in: ${OUTPUT_DIR}/results/{bnb_int8,bnb_int4}/"
echo "============================================="
