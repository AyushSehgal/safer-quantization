#!/bin/bash
#SBATCH --job-name=qresafe-eval-fp16
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user_data/ayushseh/safer-quantization/outputs/logs/eval_fp16_%j.out
#SBATCH --error=/data/user_data/ayushseh/safer-quantization/outputs/logs/eval_fp16_%j.err

# ============================================================================
# 01_eval_fp16_baseline.sh — Evaluate FP16 Llama-2-7B-Chat on all 4 metrics
# ============================================================================
# Comment out any step you don't need to rerun.
# Wikitext uses batch_size 1 (batch_size 16 causes OOM on a single A100).

set -euo pipefail

# SLURM writes --output/--error before the script body runs, so this dir must
# exist before job submission: mkdir -p /data/user_data/ayushseh/safer-quantization/outputs/logs
mkdir -p /data/user_data/ayushseh/safer-quantization/outputs/logs

source /data/user_data/ayushseh/safer-quantization/venv/bin/activate

export BASE_DIR="/data/user_data/ayushseh/safer-quantization"
export OUTPUT_DIR="${BASE_DIR}/baseline"
export PYTHONUNBUFFERED=1
export HF_HOME="/data/user_data/ayushseh/.cache/huggingface"
export HF_DATASETS_CACHE="/data/user_data/ayushseh/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data/user_data/ayushseh/.cache/huggingface/hub"
export HF_HUB_CACHE="/data/user_data/ayushseh/hf_cache/hub"
mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE} ${HF_HUB_CACHE}

MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
RESULT_DIR="${OUTPUT_DIR}/results/fp16_baseline"
mkdir -p ${RESULT_DIR}

echo "============================================="
echo "Evaluating FP16 baseline: ${MODEL_ID}"
echo "============================================="

# ---------- 1. MMLU (5-shot) ----------
echo ">>> [1/4] Running MMLU..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_ID},dtype=float16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 8 \
    --output_path ${RESULT_DIR}/mmlu \
    2>&1 | tee ${RESULT_DIR}/mmlu.log

# ---------- 2. Wikitext-2 Perplexity ----------
# NOTE: batch_size 1 — batch_size 16 OOM-kills on a single A100
echo ">>> [2/4] Running Wikitext-2 PPL..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_ID},dtype=float16 \
    --tasks wikitext \
    --batch_size 1 \
    --output_path ${RESULT_DIR}/wikitext \
    2>&1 | tee ${RESULT_DIR}/wikitext_ppl.log

# ---------- 3. AdvBench ASR ----------
echo ">>> [3/4] Running AdvBench ASR..."
cd ${BASE_DIR}
python eval/run_advbench_asr.py \
    --model_path ${MODEL_ID} \
    --dtype float16 \
    --output_dir ${RESULT_DIR}/advbench \
    --num_prompts 520 \
    --batch_size 4 \
    2>&1 | tee ${RESULT_DIR}/advbench_asr.log

# ---------- 4. SafetyBench ----------
echo ">>> [4/4] Running SafetyBench..."
python eval/run_safetybench.py \
    --model_path ${MODEL_ID} \
    --dtype float16 \
    --output_dir ${RESULT_DIR}/safetybench \
    2>&1 | tee ${RESULT_DIR}/safetybench.log

echo "============================================="
echo "FP16 baseline eval complete!"
echo "Results in: ${RESULT_DIR}"
echo "============================================="
