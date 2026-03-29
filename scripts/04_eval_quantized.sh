#!/bin/bash
#SBATCH --job-name=qresafe-eval-all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user_data/ayushseh/qresafe_outputs/logs/eval_all_%j.out
#SBATCH --error=/data/user_data/ayushseh/qresafe_outputs/logs/eval_all_%j.err

# ============================================================================
# 04_eval_quantized.sh — Evaluate all quantized + patched models
# ============================================================================
# Runs all 4 metrics on every model variant.
# Edit MODEL_DIRS below after quantization to point to correct paths.

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate qresafe

export BASE_DIR="/data/user_data/ayushseh"
export REPO_DIR="${BASE_DIR}/Qresafe"
export OUTPUT_DIR="${BASE_DIR}/qresafe_outputs"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export PYTHONUNBUFFERED=1

cd ${REPO_DIR}

# ---- FILL IN MODEL PATHS AFTER QUANTIZATION ----
# Format: "model_name|model_path|dtype_or_quant_args"
# For HF models: use pretrained=path,dtype=float16
# For AWQ models: use pretrained=path (autoawq handles dtype)
# For QLoRA/bitsandbytes: use pretrained=path,load_in_4bit=True

declare -a MODELS=(
    # "model_name|model_path|lm_eval_model_args"
    "fp16|meta-llama/Llama-2-7b-chat-hf|pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16"

    # --- Produced by 02_quantize_awq.sh (standard AWQ baseline) ---
    # "awq_int4|${OUTPUT_DIR}/models/llama2-7b-chat-awq-int4|pretrained=${OUTPUT_DIR}/models/llama2-7b-chat-awq-int4"

    # --- Produced by 05_qresafe_awq_patch.sh (Q-resafe mixed-precision AWQ) ---
    # "qresafe_awq_int4|${OUTPUT_DIR}/models/llama2-7b-chat-qresafe-awq-int4|pretrained=${OUTPUT_DIR}/models/llama2-7b-chat-qresafe-awq-int4"

    # --- Produced by 03_quantize_with_ft.sh (Algorithm 1: Q-resafe QLoRA) ---
    # output_dir in llama7b.yaml is "data/llama-7b-chat-qat" relative to quant-with-ft/
    # Check that path and update below:
    # "qresafe_qlora_risk1|${REPO_DIR}/Qresafe/quant-with-ft/data/llama-7b-chat-qat|pretrained=${REPO_DIR}/Qresafe/quant-with-ft/data/llama-7b-chat-qat,load_in_4bit=True"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH MODEL_ARGS <<< "${entry}"
    RESULT_DIR="${OUTPUT_DIR}/results/${MODEL_NAME}"
    mkdir -p ${RESULT_DIR}

    echo "============================================="
    echo "Evaluating: ${MODEL_NAME}"
    echo "Path: ${MODEL_PATH}"
    echo "============================================="

    # --- MMLU (5-shot) ---
    echo ">>> [1/4] MMLU..."
    lm_eval --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 8 \
        --output_path ${RESULT_DIR}/mmlu \
        2>&1 | tee ${RESULT_DIR}/mmlu.log || echo "MMLU FAILED for ${MODEL_NAME}"

    # --- Wikitext-2 PPL ---
    echo ">>> [2/4] Wikitext-2 PPL..."
    lm_eval --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks wikitext \
        --batch_size 16 \
        --output_path ${RESULT_DIR}/wikitext \
        2>&1 | tee ${RESULT_DIR}/wikitext_ppl.log || echo "Wikitext FAILED for ${MODEL_NAME}"

    # --- AdvBench ASR ---
    echo ">>> [3/4] AdvBench ASR..."
    python eval/run_advbench_asr.py \
        --model_path "${MODEL_PATH}" \
        --output_dir ${RESULT_DIR}/advbench \
        --num_prompts 520 \
        --batch_size 4 \
        2>&1 | tee ${RESULT_DIR}/advbench_asr.log || echo "AdvBench FAILED for ${MODEL_NAME}"

    # --- SafetyBench ---
    echo ">>> [4/4] SafetyBench..."
    python eval/run_safetybench.py \
        --model_path "${MODEL_PATH}" \
        --output_dir ${RESULT_DIR}/safetybench \
        2>&1 | tee ${RESULT_DIR}/safetybench.log || echo "SafetyBench FAILED for ${MODEL_NAME}"

    echo ">>> Done: ${MODEL_NAME}"
    echo ""
done

# --- Aggregate results into a single table ---
echo ">>> Aggregating results..."
python eval/aggregate_results.py \
    --results_dir ${OUTPUT_DIR}/results \
    --output ${OUTPUT_DIR}/results/summary_table.csv

echo "============================================="
echo "All evaluations complete!"
echo "Summary: ${OUTPUT_DIR}/results/summary_table.csv"
echo "============================================="
