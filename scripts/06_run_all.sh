#!/bin/bash
# ============================================================================
# 07_run_all.sh — Submit all jobs with SLURM dependency chaining
# ============================================================================
# Usage: bash scripts/07_run_all.sh
#
# This submits jobs in order, each waiting for the previous to complete.
# You can also run individual steps manually with sbatch.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "Q-resafe Experiment Pipeline — Babel"
echo "============================================="

# Ensure log directory exists
mkdir -p /data/user_data/ayushseh/qresafe_outputs/logs

# Step 1: Eval FP16 baseline
JOB1=$(sbatch --parsable ${SCRIPTS_DIR}/01_eval_baseline.sh)
echo "Submitted baseline eval: Job ${JOB1}"

# Step 2: AWQ quantization (can run in parallel with Step 1)
JOB2=$(sbatch --parsable ${SCRIPTS_DIR}/02_quantize_awq.sh)
echo "Submitted AWQ quant: Job ${JOB2}"

# Step 3: Q-resafe QLoRA (Algorithm 1, quant-with-ft) — can run in parallel with Steps 1 & 2
JOB3=$(sbatch --parsable ${SCRIPTS_DIR}/03_quantize_with_ft.sh)
echo "Submitted quant-with-ft (Q-resafe QLoRA): Job ${JOB3}"

# Step 5: Q-resafe AWQ mixed-precision patch — depends on Step 2 (needs FP16 model + AWQ output)
JOB5=$(sbatch --parsable --dependency=afterok:${JOB2} ${SCRIPTS_DIR}/05_qresafe_awq_patch.sh)
echo "Submitted Q-resafe AWQ patch: Job ${JOB5} (depends on ${JOB2})"

# Step 4: Eval all models — wait for all quantization + patching to finish
JOB4=$(sbatch --parsable --dependency=afterok:${JOB1}:${JOB2}:${JOB3}:${JOB5} ${SCRIPTS_DIR}/04_eval_quantized.sh)
echo "Submitted eval-all: Job ${JOB4} (depends on ${JOB1},${JOB2},${JOB3},${JOB5})"

echo ""
echo "============================================="
echo "Pipeline submitted! Monitor with:"
echo "  squeue -u ayushseh"
echo "  tail -f /data/user_data/ayushseh/qresafe_outputs/logs/*.out"
echo "============================================="
