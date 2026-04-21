#!/usr/bin/env bash
# run_refusal_analysis.sh
#
# 1. Extract refusal directions from quantized models in results/refusal_dir/
#    and save .pt files to refusal_direction/refusal_dirs/.
# 2. Visualize all .pt files in refusal_direction/refusal_dirs/ with pairwise
#    cosine comparisons across quantization modes.
#
# Assumes base model directions already exist in refusal_direction/refusal_dirs/
# (e.g. llama-2-7b-chat-hf.pt).
#
# Usage (from project root):
#   bash run_refusal_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QREALIGN_DIR="${SCRIPT_DIR}/qrealign"
MODELS_DIR="${SCRIPT_DIR}/models"
REFUSAL_DIRS="${SCRIPT_DIR}/refusal_direction/refusal_dirs"
RESULTS_DIR="${SCRIPT_DIR}/results/refusal_dir"
DATA_JSON="${QREALIGN_DIR}/data.json"
VIZ_SCRIPT="${SCRIPT_DIR}/analyze_refusal_directions.py"

mkdir -p "$REFUSAL_DIRS"

# ── Python from refusal-venv ───────────────────────────────────────────────────
VENV_PYTHON="${SCRIPT_DIR}/refusal-venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON="$(command -v python3 2>/dev/null || command -v python)"
fi
echo "Python: $("$VENV_PYTHON" --version 2>&1)"

# ── Step 1: Extract directions from quantized models ──────────────────────────
echo ""
echo "========================================"
echo "  Step 1: Extract Quantized Model Directions"
echo "========================================"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "  [WARN] $RESULTS_DIR not found — skipping quantized extraction."
else
    cd "${QREALIGN_DIR}"
    "${VENV_PYTHON}" extract_quant_refusal_dirs.py \
        --results_dir "${RESULTS_DIR}" \
        --models_dir  "${MODELS_DIR}" \
        --data        "${DATA_JSON}" \
        --output_dir  "${REFUSAL_DIRS}"
    cd "${SCRIPT_DIR}"
fi

# ── Step 2: Visualize all directions ──────────────────────────────────────────
echo ""
echo "========================================"
echo "  Step 2: Visualize Refusal Directions"
echo "========================================"

n_pts=$(ls "${REFUSAL_DIRS}"/*.pt 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n_pts" -eq 0 ]]; then
    echo "[ERROR] No .pt files found in ${REFUSAL_DIRS}" >&2
    exit 1
fi
echo "  Found $n_pts direction file(s) in ${REFUSAL_DIRS}"

"${VENV_PYTHON}" "${VIZ_SCRIPT}"

echo ""
echo "========================================"
echo "  Done.  See refusal_directions_output/"
echo "========================================"
