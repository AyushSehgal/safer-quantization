#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 20:00:00
#SBATCH -A cis250260p
#SBATCH --job-name=table2_gsm8k
#SBATCH --output=output_table2_gsm8k_%j.log
#SBATCH --error=error_table2_gsm8k_%j.log

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
echo "Starting job: table2_gsm8k"
echo "============================================================"


# Step 1: Fine-Tuning
python ./fine-tuning/train.py --dataset gsm8k --poison_ratio 0.15 --method sft --model_name meta-llama/Llama-2-7b-chat-hf

# Define checkpoints
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
SFT_DIR="./checkpoint/sft-llama-2-7b-chat-hf-gsm8k-hr0.15"
SFT_CKPT=$(ls -1d $SFT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [ -z "$SFT_CKPT" ]; then
    echo "Error: Fine-tuning failed to produce a checkpoint!"
    exit 1
fi

Q_OUT="./quantize/q_realign_llama-2-7b-chat-hf_gsm8k_hr0.15"

# Step 2: Quantization
python main.py --model $BASE_MODEL --model_resume $SFT_CKPT --output_dir $Q_OUT --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10

# Step 3: Evaluation (SFT)
python attack_eval.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/advbench_sft_gsm8k_hr0.15.json
python eval_gsm8k.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/gsm8k_acc_sft_hr0.15.json

# Step 4: Evaluation (Q-Realign)
python attack_eval.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/advbench_qrealign_gsm8k_hr0.15.json
python eval_gsm8k.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/gsm8k_acc_qrealign_hr0.15.json


echo "============================================================"
echo "Job completed."
echo "============================================================"
