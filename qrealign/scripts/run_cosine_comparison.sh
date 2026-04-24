#!/bin/bash
# Extract refusal directions from quantized models and plot per-layer cosine
# similarity against the original (FP16) refusal direction.
#
# Targets four quantization modes:
#   W8A8, W4A16, W8A16_combined, W4A16_combined
#
# Usage:
#   bash scripts/run_cosine_comparison.sh
#   bash scripts/run_cosine_comparison.sh --mu 0.001   # different combined mu
#   bash scripts/run_cosine_comparison.sh --dry_run

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/data/user_data/ayushseh/safer-quantization"
RESULTS_DIR="${PROJECT_DIR}/results/refusal_dir"
MODELS_DIR="${PROJECT_DIR}/models"
REFUSAL_DIRS="${PROJECT_DIR}/refusal_direction/refusal_dirs"
PLOT_OUT="${PROJECT_DIR}/refusal_directions_output/quant_vs_original"
LOG_DIR="${PROJECT_DIR}/logs/cosine_comparison"

PYTHON_BIN="${PROJECT_DIR}/qrealign-venv/bin/python"

MU="0.01"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mu)      MU="$2";      shift 2 ;;
        --dry_run) DRY_RUN=true; shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "==========================================="
echo "  Refusal Direction Cosine Comparison"
echo "==========================================="
echo "  Results dir : $RESULTS_DIR"
echo "  Refusal dirs: $REFUSAL_DIRS"
echo "  Plot output : $PLOT_OUT"
echo "  combined mu : $MU"
echo "  Dry-run     : $DRY_RUN"
echo ""

if $DRY_RUN; then
    echo "[dry-run] Would submit extraction + plotting job."
    echo ""
    echo "Extraction command:"
    echo "  python extract_quant_refusal_dirs.py \\"
    echo "    --results_dir ${RESULTS_DIR} \\"
    echo "    --models_dir  ${MODELS_DIR} \\"
    echo "    --data        data.json \\"
    echo "    --output_dir  ${REFUSAL_DIRS}"
    echo ""
    echo "Plotting command:"
    echo "  python ../plot_quant_vs_original.py \\"
    echo "    --refusal_dirs ${REFUSAL_DIRS} \\"
    echo "    --out_dir      ${PLOT_OUT}"
    exit 0
fi

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cos_cmp
#SBATCH --output=${LOG_DIR}/cos_comparison_%j.out
#SBATCH --error=${LOG_DIR}/cos_comparison_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=general

echo "Job ID: \$SLURM_JOB_ID"
echo "Node:   \$SLURM_NODELIST"
echo "GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  \$(date)"
echo ""

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

PYTHON_BIN=${PYTHON_BIN}
if [ ! -x "\$PYTHON_BIN" ]; then
    PYTHON_BIN="\$(command -v python3)"
fi
echo "Python: \$PYTHON_BIN"
echo ""

cd ${PROJECT_DIR}/qrealign

# ── Step 1: extract refusal directions from quantized models ──────────────────
# Scans results/refusal_dir/*/<mode>/omni_parameters.pth
# Saves refusal_dirs/<sft_folder>_<mode>.pt  (one per variant)
# Already skips variants whose .pt already exists (--skip_existing default).
#echo "========== Step 1: Extract quantized refusal directions =========="
#"\$PYTHON_BIN" extract_quant_refusal_dirs.py \
#    --results_dir ${RESULTS_DIR} \
#    --models_dir  ${MODELS_DIR} \
#    --data        data.json \
#    --output_dir  ${REFUSAL_DIRS}
#
#echo ""
#echo "Refusal dirs now in: ${REFUSAL_DIRS}"
#ls -lh ${REFUSAL_DIRS}/*.pt 2>/dev/null || echo "(no .pt files found)"

## ── Step 2: plot per-layer cosine similarity vs original ──────────────────────
# Produces 4 graphs: W8A8, W4A16, W8A16_combined, W4A16_combined
echo ""
echo "========== Step 2: Plot cosine similarity vs original ============"


deactivate
source ../refusal-venv/bin/activate
PYTHON_BIN="${PROJECT_DIR}/refusal-venv/bin/python"
"\$PYTHON_BIN" ../plot_quant_vs_original.py \
    --refusal_dirs ${REFUSAL_DIRS} \
    --out_dir      ${PLOT_OUT}

echo ""
echo "Plots saved to: ${PLOT_OUT}"
ls -lh ${PLOT_OUT}/*.png 2>/dev/null || echo "(no plots found)"

echo ""
echo "Done: \$(date)"
EOF
)

echo "Submitted job: $JOB_ID"
echo ""
echo "Monitor:  squeue -u \$USER -j ${JOB_ID}"
echo "Log:      ${LOG_DIR}/cos_comparison_${JOB_ID}.out"
echo "Plots:    ${PLOT_OUT}/"
echo ""
