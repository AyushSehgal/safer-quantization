# Baseline Study

## Running the Experiments

### Step 1 — FP16 Baseline

Evaluates the unquantized FP16 model on all four benchmarks. This can run in parallel
with steps 2 and 3.

```bash
sbatch scripts/01_eval_fp16_baseline.sh
```

Results land in: `baseline/results/fp16_baseline/`

To skip a benchmark, comment out that section in the script. All four are enabled by default.

---

### Steps 2 & 3 — Quantize to INT8 and INT4

These two jobs are fully independent and can run at the same time.

```bash
sbatch scripts/02_quantize_bnb_int8.sh
sbatch scripts/03_quantize_bnb_int4.sh
```

Each job loads the FP16 model, quantizes it with bitsandbytes, and saves the result:

| Script | Method | Saved to |
|--------|--------|----------|
| `02_quantize_bnb_int8.sh` | LLM.int8 | `int8_outputs/results/models/llama2-7b-chat-bnb-int8/` |
| `03_quantize_bnb_int4.sh` | NF4 (double quant, fp16 compute) | `int4_outputs/results/models/llama2-7b-chat-bnb-int4/` |

Expected runtime: ~20–30 minutes each.

---

### Step 4 — Evaluate Quantized Models

Runs all four benchmarks on both the INT8 and INT4 models sequentially.
**Must run after both 02 and 03 complete.**

```bash
# Submit manually after 02 and 03 finish:
sbatch scripts/04_eval_bnb.sh

# Or chain with SLURM dependencies (replace job IDs):
JOB2=$(sbatch --parsable scripts/02_quantize_bnb_int8.sh)
JOB3=$(sbatch --parsable scripts/03_quantize_bnb_int4.sh)
sbatch --dependency=afterok:${JOB2}:${JOB3} scripts/04_eval_bnb.sh
```

Results land in: `bnb_outputs/results/bnb_int8/` and `bnb_outputs/results/bnb_int4/`

---

## Full Pipeline (One Command)

Submit all four jobs with automatic dependency chaining:

```bash
JOB1=$(sbatch --parsable scripts/01_eval_fp16_baseline.sh)
JOB2=$(sbatch --parsable scripts/02_quantize_bnb_int8.sh)
JOB3=$(sbatch --parsable scripts/03_quantize_bnb_int4.sh)
JOB4=$(sbatch --parsable --dependency=afterok:${JOB2}:${JOB3} scripts/04_eval_bnb.sh)

echo "Jobs submitted: baseline=$JOB1  int8=$JOB2  int4=$JOB3  eval=$JOB4"
squeue -u ayushseh
```

Job 1, 2, and 3 all start immediately. Job 4 waits for 2 and 3 to succeed.

---

## Output Locations

| What | Path |
|------|------|
| SLURM logs | `outputs/logs/` |
| FP16 baseline results | `baseline/results/fp16_baseline/{mmlu,wikitext,advbench,safetybench}/` |
| INT8 model weights | `int8_outputs/results/models/llama2-7b-chat-bnb-int8/` |
| INT4 model weights | `int4_outputs/results/models/llama2-7b-chat-bnb-int4/` |
| INT8 eval results | `bnb_outputs/results/bnb_int8/{mmlu,wikitext,advbench,safetybench}/` |
| INT4 eval results | `bnb_outputs/results/bnb_int4/{mmlu,wikitext,advbench,safetybench}/` |

All paths are relative to `/data/user_data/ayushseh/safer-quantization/`.

---

## Notes

- **Wikitext batch size**: All scripts use `batch_size 1` for Wikitext-2. Higher values
  OOM-kill on a single A100 with this model.

- **AdvBench runtime**: After generating responses from the target model, AdvBench loads
  a separate 13B HarmBench classifier to score them. This is the slowest benchmark.
  The target model is freed from GPU memory before the classifier loads, so it fits on
  a single 40GB A100, but plan for ~2–3 hours per model just for AdvBench.

- **SLURM log directory**: SLURM opens the `--output` file before the job script runs,
  so `outputs/logs/` must exist at submission time. The one-time setup step above
  handles this.
