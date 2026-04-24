#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# run_amplify_eval.sh
#
# Runs Layer-Selective Refusal Direction Amplification evaluations across
# quantized model variants (int8 / int4) on SafetyBench, MMLU, Wikitext-2.
#
# Edit the CONFIG section to match your paths, then:
#   bash run_amplify_eval.sh              # standard eval
#   bash run_amplify_eval.sh sweep        # layer sweep
#   bash run_amplify_eval.sh alpha_sweep  # alpha grid search
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Base model (HuggingFace ID or local path)
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"

# Short name used to locate pre-computed refusal directions
MODEL_NICK="llama-2-7b-chat-hf"

# Path to SafetyBench data (clone from https://github.com/thu-coai/SafetyBench)
SAFETYBENCH_DIR="SafetyBench"

# Root dir where quantize_and_save.py saved omni_parameters.pth
QUANTIZED_ROOT="../quantized_models/Llama-2-7b-chat-hf"

# Where to write results
RESULTS_ROOT="../results/amplify"

# Refusal direction files (relative to this script's directory / qrealign/)
REFUSAL_DIR_ROOT="../refusal_direction/refusal_dirs"

# Amplification strength and layer selection for standard eval
ALPHA=20.0
LAYERS="middle"   # middle | all | 14,15,16,17,18

# Alpha values for the grid search
ALPHA_SWEEP="5,10,20,40,80"

# Limit questions per benchmark (leave blank or 0 for full dataset)
LIMIT=""   # e.g. "500" for a quick run

# Optional SFT adapter path (leave blank for base model only)
SFT_RESUME=""  # e.g. "../models/sft-llama-2-7b-chat-hf-alpaca-hr0.15"

# ── Helpers ──────────────────────────────────────────────────────────────────
LIMIT_ARG=""
if [[ -n "$LIMIT" && "$LIMIT" -gt 0 ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

RESUME_ARG=""
if [[ -n "$SFT_RESUME" ]]; then
    RESUME_ARG="--resume $SFT_RESUME"
fi

REFUSAL_DIR_FILE="${REFUSAL_DIR_ROOT}/${MODEL_NICK}.pt"
if [[ ! -f "$REFUSAL_DIR_FILE" ]]; then
    echo "ERROR: refusal direction not found at $REFUSAL_DIR_FILE"
    echo "       Check REFUSAL_DIR_ROOT and MODEL_NICK in this script."
    exit 1
fi

MODE="${2:-}"   # optional second arg to override mode (e.g. int8)

run_eval() {
    local mode="$1"
    local q_resume="$2"
    local out_dir="$3"
    shift 3
    local extra_args="$@"

    mkdir -p "$out_dir"
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Model : $MODEL_ID"
    echo "  Mode  : $mode"
    echo "  q_resume: $q_resume"
    echo "  Output: $out_dir"
    echo "═══════════════════════════════════════════════════════"

    python eval_amplify_refusal.py \
        --model_id  "$MODEL_ID" \
        --mode      "$mode" \
        --q_resume  "$q_resume" \
        --refusal_dir "$REFUSAL_DIR_FILE" \
        --alpha     "$ALPHA" \
        --layers    "$LAYERS" \
        --tasks     "safetybench,mmlu,wikitext" \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --output    "$out_dir" \
        $RESUME_ARG $LIMIT_ARG $extra_args
}

# ── Main dispatch ─────────────────────────────────────────────────────────────
SUBCOMMAND="${1:-standard}"

case "$SUBCOMMAND" in

  standard)
    echo "[run_amplify_eval] Standard evaluation (baseline + amplified)"
    for mode in int8 int4; do
        if [[ "$mode" == "int8" ]]; then
            q_resume="${QUANTIZED_ROOT}/W8A8/omni_parameters.pth"
        else
            q_resume="${QUANTIZED_ROOT}/W4A16/omni_parameters.pth"
        fi

        if [[ ! -f "$q_resume" ]]; then
            echo "[skip] $q_resume not found — run quantize_and_save.py first."
            continue
        fi

        out_dir="${RESULTS_ROOT}/${MODEL_NICK}/${mode}"
        run_eval "$mode" "$q_resume" "$out_dir"
    done

    # Also run fp16 baseline (no quantization)
    echo ""
    echo "[fp16 baseline]"
    out_dir="${RESULTS_ROOT}/${MODEL_NICK}/fp16"
    mkdir -p "$out_dir"
    python eval_amplify_refusal.py \
        --model_id  "$MODEL_ID" \
        --mode      fp16 \
        --refusal_dir "$REFUSAL_DIR_FILE" \
        --alpha     "$ALPHA" \
        --layers    "$LAYERS" \
        --tasks     "safetybench,mmlu,wikitext" \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --output    "$out_dir" \
        $RESUME_ARG $LIMIT_ARG
    ;;

  sweep)
    echo "[run_amplify_eval] Layer sweep mode"
    SWEEP_MODE="${MODE:-int8}"
    if [[ "$SWEEP_MODE" == "int8" ]]; then
        q_resume="${QUANTIZED_ROOT}/W8A8/omni_parameters.pth"
    else
        q_resume="${QUANTIZED_ROOT}/W4A16/omni_parameters.pth"
    fi

    out_dir="${RESULTS_ROOT}/${MODEL_NICK}/sweep"
    mkdir -p "$out_dir"
    python eval_amplify_refusal.py \
        --model_id  "$MODEL_ID" \
        --mode      "$SWEEP_MODE" \
        --q_resume  "$q_resume" \
        --refusal_dir "$REFUSAL_DIR_FILE" \
        --sweep \
        --sweep_alpha "$ALPHA" \
        --sweep_limit 200 \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --output    "$out_dir" \
        $RESUME_ARG
    ;;

  alpha_sweep)
    echo "[run_amplify_eval] Alpha sweep mode"
    SWEEP_MODE="${MODE:-int8}"
    if [[ "$SWEEP_MODE" == "int8" ]]; then
        q_resume="${QUANTIZED_ROOT}/W8A8/omni_parameters.pth"
    else
        q_resume="${QUANTIZED_ROOT}/W4A16/omni_parameters.pth"
    fi

    out_dir="${RESULTS_ROOT}/${MODEL_NICK}/alpha_sweep"
    mkdir -p "$out_dir"
    python eval_amplify_refusal.py \
        --model_id  "$MODEL_ID" \
        --mode      "$SWEEP_MODE" \
        --q_resume  "$q_resume" \
        --refusal_dir "$REFUSAL_DIR_FILE" \
        --alpha_sweep "$ALPHA_SWEEP" \
        --layers    "$LAYERS" \
        --tasks     "safetybench,mmlu" \
        --safetybench_dir "$SAFETYBENCH_DIR" \
        --limit     300 \
        --output    "$out_dir" \
        $RESUME_ARG
    ;;

  *)
    echo "Unknown subcommand: $SUBCOMMAND"
    echo "Usage: $0 [standard|sweep|alpha_sweep] [int8|int4]"
    exit 1
    ;;

esac

echo ""
echo "Done."
