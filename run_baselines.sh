#!/bin/bash
# ==============================================================================
# Q-Realign Baseline Evaluation Script
# ==============================================================================
# Evaluates Llama-2-7b-chat-hf at three precision levels (FP16, Int8, Int4)
# across four benchmarks: SafetyBench, AdvBench ASR, MMLU, Wikitext-2 PPL.
#
# Models use Q-Realign's own quantization (SmoothQuant-based):
#   - FP16:  meta-llama/Llama-2-7b-chat-hf (no quantization)
#   - Int8:  W8A8 quantization applied on-the-fly
#   - Int4:  W4A16 quantization applied on-the-fly
#
# Usage:
#   conda activate q-realign
#   bash run_baselines.sh
#
# Or run individual sections by commenting out the others.
# ==============================================================================

set -e
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
RESULTS_DIR="./baseline_results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Q-Realign Baseline Evaluation"
echo "Model: $MODEL_ID"
echo "Results directory: $RESULTS_DIR"
echo "============================================================"

# ==============================================================================
# 1. Wikitext-2 Perplexity  (Language Quality)
# ==============================================================================
echo ""
echo ">>> [1/4] Wikitext-2 Perplexity"
echo "------------------------------------------------------------"

echo "  [1a] FP16 ..."
python eval_wikitext_ppl.py \
    --model_id "$MODEL_ID" \
    --mode fp16 \
    --output "$RESULTS_DIR/wikitext_ppl_fp16.json"

echo "  [1b] Int8 (W8A8) ..."
python eval_wikitext_ppl.py \
    --model_id "$MODEL_ID" \
    --mode int8 \
    --output "$RESULTS_DIR/wikitext_ppl_int8.json"

echo "  [1c] Int4 (W4A16) ..."
python eval_wikitext_ppl.py \
    --model_id "$MODEL_ID" \
    --mode int4 \
    --output "$RESULTS_DIR/wikitext_ppl_int4.json"

# ==============================================================================
# 2. MMLU  (General Utility)
# ==============================================================================
echo ""
echo ">>> [2/4] MMLU"
echo "------------------------------------------------------------"

echo "  [2a] FP16 ..."
python eval_mmlu.py \
    --model_id "$MODEL_ID" \
    --mode fp16 \
    --output "$RESULTS_DIR/mmlu_fp16.json"

echo "  [2b] Int8 (W8A8) ..."
python eval_mmlu.py \
    --model_id "$MODEL_ID" \
    --mode int8 \
    --output "$RESULTS_DIR/mmlu_int8.json"

echo "  [2c] Int4 (W4A16) ..."
python eval_mmlu.py \
    --model_id "$MODEL_ID" \
    --mode int4 \
    --output "$RESULTS_DIR/mmlu_int4.json"

# ==============================================================================
# 3. SafetyBench  (Safety)
# ==============================================================================
echo ""
echo ">>> [3/4] SafetyBench"
echo "------------------------------------------------------------"

echo "  [3a] FP16 ..."
python eval_safetybench.py \
    --model_id "$MODEL_ID" \
    --mode fp16 \
    --output "$RESULTS_DIR/safetybench_fp16.json"

echo "  [3b] Int8 (W8A8) ..."
python eval_safetybench.py \
    --model_id "$MODEL_ID" \
    --mode int8 \
    --output "$RESULTS_DIR/safetybench_int8.json"

echo "  [3c] Int4 (W4A16) ..."
python eval_safetybench.py \
    --model_id "$MODEL_ID" \
    --mode int4 \
    --output "$RESULTS_DIR/safetybench_int4.json"

# ==============================================================================
# 4. AdvBench ASR  (Jailbreak Resistance)
# ==============================================================================
echo ""
echo ">>> [4/4] AdvBench ASR"
echo "------------------------------------------------------------"

echo "  [4a] FP16 ..."
python attack_eval.py \
    --model_id "$MODEL_ID" \
    --mode fp16 \
    --limit 520

echo "  [4b] Int8 (W8A8) ..."
python attack_eval.py \
    --model_id "$MODEL_ID" \
    --mode int8 \
    --limit 520

echo "  [4c] Int4 (W4A16) ..."
python attack_eval.py \
    --model_id "$MODEL_ID" \
    --mode int4 \
    --limit 520

# ==============================================================================
echo ""
echo "============================================================"
echo "All baseline evaluations complete!"
echo "Results saved in: $RESULTS_DIR/"
echo "============================================================"
