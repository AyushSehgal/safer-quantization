#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 20:00:00
#SBATCH -A cis250260p
#SBATCH --job-name=table3_gemma2-9b
#SBATCH --output=output_table3_gemma2-9b_%j.log
#SBATCH --error=error_table3_gemma2-9b_%j.log

set -e

# Load environment logic
module load anaconda3
conda activate q-realign
cd /jet/home/apatawar/q-realign-remake

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

export HF_HUB_CACHE="/ocean/projects/cis250260p/shared/hf_cache"
export HF_HOME="/ocean/projects/cis250260p/shared/hf_cache"
export HF_DATASETS_CACHE="/jet/home/apatawar/q-realign-remake/dataset_cache"

echo "============================================================"
echo "Starting job: table3_gemma2-9b"
echo "============================================================"


# Step 1: Fine-Tuning
python ./fine-tuning/train.py --dataset alpaca --poison_ratio 0.1 --method sft --model_name google/gemma-2-9b

# Define checkpoints
BASE_MODEL="google/gemma-2-9b"
SFT_DIR="./checkpoint/sft-gemma2-9b-alpaca-hr0.1"
SFT_CKPT=$(ls -1d $SFT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [ -z "$SFT_CKPT" ]; then
    echo "Error: Fine-tuning failed to produce a checkpoint!"
    exit 1
fi

Q_OUT="./quantize/q_realign_gemma2-9b_alpaca_hr0.1"

# Step 2: Quantization
python main.py --model $BASE_MODEL --model_resume $SFT_CKPT --output_dir $Q_OUT --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10

# Step 3: Evaluation (SFT)
python attack_eval.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/advbench_sft_alpaca_gemma2-9b_hr0.1.json
python eval_mmlu.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/mmlu_sft_alpaca_gemma2-9b_hr0.1.json

# Step 4: Evaluation (Q-Realign)
python attack_eval.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/advbench_qrealign_alpaca_gemma2-9b_hr0.1.json
python eval_mmlu.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/mmlu_qrealign_alpaca_gemma2-9b_hr0.1.json


echo "============================================================"
echo "Job completed."
echo "============================================================"
