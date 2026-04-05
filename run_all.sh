#!/bin/bash

# Spawn Table 1 jobs (Llama-2, Alpaca)
sbatch table1_hr0.05.sh
sbatch table1_hr0.1.sh
sbatch table1_hr0.15.sh
sbatch table1_hr0.2.sh

# Spawn Table 2 jobs (Llama-2, Diverse Datasets)
sbatch table2_sst2.sh
sbatch table2_gsm8k.sh

# Spawn Table 3 jobs (Gemma and Qwen, Alpaca)
sbatch table3_gemma2-9b.sh
sbatch table3_qwen2.5-7b.sh

echo "All 8 jobs have been successfully submitted to the cluster!"
