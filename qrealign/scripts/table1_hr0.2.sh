#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 20:00:00
#SBATCH -A cis250260p
#SBATCH --job-name=table1_hr0.2
#SBATCH --output=output_table1_hr0.2_%j.log
#SBATCH --error=error_table1_hr0.2_%j.log

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
echo "Starting job: table1_hr0.2"
echo "============================================================"


# Step 1: Fine-Tuning
python ./fine-tuning/train.py --dataset alpaca --poison_ratio 0.2 --method sft --model_name meta-llama/Llama-2-7b-chat-hf

# Define checkpoints
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
SFT_DIR="./checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.2"
SFT_CKPT=$(ls -1d $SFT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [ -z "$SFT_CKPT" ]; then
    echo "Error: Fine-tuning failed to produce a checkpoint!"
    exit 1
fi

Q_OUT="./quantize/q_realign_llama-2-7b-chat-hf_alpaca_hr0.2"

# Step 2: Quantization
python main.py --model $BASE_MODEL --model_resume $SFT_CKPT --output_dir $Q_OUT --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10

# Step 3: Evaluation (SFT)
python attack_eval.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/advbench_sft_alpaca_hr0.2.json
python eval_mmlu.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/mmlu_sft_alpaca_hr0.2.json

# Step 4: Evaluation (Q-Realign)
python attack_eval.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/advbench_qrealign_alpaca_hr0.2.json
python eval_mmlu.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/mmlu_qrealign_alpaca_hr0.2.json


echo "============================================================"
echo "Job completed."
echo "============================================================"
