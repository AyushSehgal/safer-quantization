# Q-resafe Reproduction Study for Safety Improved Quantization Project

## Overview

This directory contains SLURM scripts and evaluation configs to reproduce Q-resafe 
experiments, extended with custom metrics 
(SafetyBench, AdvBench ASR, MMLU, Wikitext-2 PPL).

Original repo and credits go to Q-resafe authors:
 - https://github.com/Thecommonirin/Qresafe
 - https://arxiv.org/abs/2506.20251

## Directory Structure

```
safer-quantization/
├── Qresafe/quant-with-ft/          # Original repo — Algorithm 1 (Q-resafe QLoRA via DPO + masked LoRA)
├── Qresafe/quant-without-ft/       # Original repo — AWQ quantization (standard + Q-resafe mixed-precision)
├── scripts/
│   ├── 00_setup_env.sh             # One-time conda env setup
│   ├── 01_eval_baseline.sh         # Eval FP16 model (all 4 metrics)
│   ├── 02_quantize_awq.sh          # Standard AWQ INT4 baseline (no safety patching)
│   ├── 03_quantize_with_ft.sh      # Q-resafe QLoRA — Algorithm 1 (DPO + masked LoRA on quantized model)
│   ├── 04_eval_quantized.sh        # Eval ALL models: baseline + both Q-resafe variants
│   ├── 05_qresafe_awq_patch.sh     # Q-resafe AWQ mixed-precision (SNIP: safety-critical weights at FP16)
│   └── 06_run_all.sh               # Master script — submits all jobs with SLURM dependencies
├── eval/
│   ├── run_advbench_asr.py         # AdvBench ASR with HarmBench classifier
│   ├── run_safetybench.py          # SafetyBench evaluation
│   └── aggregate_results.py        # Collects all results into a summary CSV
└── configs/
    ├── accelerate_babel.yaml        # Accelerate config for Babel (2x A100, bf16)
    └── eval_config.yaml             # Paths and model registry
```

## What Each Script Actually Does

### Quantization pipeline

| Script | Method | Output model |
|--------|--------|-------------|
| `02_quantize_awq.sh` | Standard AWQ INT4 (no patching) | `llama2-7b-chat-awq-int4` |
| `03_quantize_with_ft.sh` | **Algorithm 1**: Q-resafe QLoRA — DPO + masked LoRA on the quantized model, guided by FP16 | `data/llama-7b-chat-qat` (inside quant-with-ft/) |
| `05_qresafe_awq_patch.sh` | **Q-resafe AWQ**: SNIP scores on FP16 → top-60% safety-critical weights kept at FP16, rest INT4 | `llama2-7b-chat-qresafe-awq-int4` |

Script 03 IS the Q-resafe safety patching for QLoRA — it is not a separate step after quantization.
The repo's `quant.py` runs quantization and safety-patching together as a single DPO training run.

Script 05 is the only additional patching step needed, and only for the AWQ variant.

### Evaluation

Script 04 evaluates every model variant (FP16 baseline, AWQ baseline, Q-resafe AWQ, Q-resafe QLoRA)
across all 4 metrics. There is no separate eval script for patched models — 04 covers everything.
After steps 02, 03, and 05 complete, uncomment the relevant model entries in `04_eval_quantized.sh`.

## Execution Order

```
Step 0: sbatch scripts/00_setup_env.sh              # One-time setup
Step 1: sbatch scripts/01_eval_baseline.sh           # FP16 baseline, ~1 hour
Step 2: sbatch scripts/02_quantize_awq.sh            # AWQ INT4 baseline, ~30 min
Step 3: sbatch scripts/03_quantize_with_ft.sh        # Q-resafe QLoRA (Algorithm 1), ~2-4 hours
Step 5: sbatch scripts/05_qresafe_awq_patch.sh       # Q-resafe AWQ mixed-precision, ~1-2 hours
Step 4: sbatch scripts/04_eval_quantized.sh          # Eval all models, ~3-4 hours
```

Steps 1, 2, and 3 can run in parallel. Step 5 must run after step 2.
Step 4 must run after all of the above.

Or just: `bash scripts/06_run_all.sh` to chain everything with SLURM dependencies automatically.

## Important Notes

- **GPU Memory**: All training scripts are configured for 2x A100 40GB (80GB total).
  `accelerate_babel.yaml` and all SLURM headers use `--gpus-per-task=2` / `num_processes=2`.
  The paper ran on 4x A100 — 2 GPUs is sufficient for Llama-2-7B with QLoRA.
- **After step 3 completes**: check the output path. `llama7b.yaml` saves to
  `data/llama-7b-chat-qat` relative to `Qresafe/quant-with-ft/`. Update the model path
  in `04_eval_quantized.sh` accordingly before running step 4.
- **Excluded nodes**: Add any bad nodes to `--exclude` in SLURM scripts.
- **HuggingFace login**: Run `huggingface-cli login` before submitting jobs.
  Llama-2 requires access approval at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- **Data path**: All outputs go to `/data/user_data/ayushseh/qresafe_outputs/`
