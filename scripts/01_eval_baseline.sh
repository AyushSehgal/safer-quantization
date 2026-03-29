#!/bin/bash
#SBATCH --job-name=qresafe-eval-baseline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user_data/ayushseh/qresafe_outputs/logs/eval_baseline_%j.out
#SBATCH --error=/data/user_data/ayushseh/qresafe_outputs/logs/eval_baseline_%j.err

# ============================================================================
# 01_eval_baseline.sh — Evaluate FP16 Llama-2-7B-Chat on all 4 metrics
# ============================================================================

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate qresafe

export BASE_DIR="/data/user_data/ayushseh"
export REPO_DIR="${BASE_DIR}/Qresafe"
export OUTPUT_DIR="${BASE_DIR}/qresafe_outputs"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export PYTHONUNBUFFERED=1

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
echo ">>> [2/4] Running Wikitext-2 PPL..."
lm_eval --model hf \
    --model_args pretrained=${MODEL_ID},dtype=float16 \
    --tasks wikitext \
    --batch_size 16 \
    --output_path ${RESULT_DIR}/wikitext \
    2>&1 | tee ${RESULT_DIR}/wikitext_ppl.log

# ---------- 3. AdvBench ASR ----------
echo ">>> [3/4] Running AdvBench ASR..."
cd ${REPO_DIR}
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
