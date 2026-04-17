import os

BASH_TEMPLATE = """#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 20:00:00
#SBATCH -A cis250260p
#SBATCH --job-name={job_name}
#SBATCH --output=output_{job_name}_%j.log
#SBATCH --error=error_{job_name}_%j.log

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
echo "Starting job: {job_name}"
echo "============================================================"

{commands}

echo "============================================================"
echo "Job completed."
echo "============================================================"
"""

def generate_script(filename, job_name, commands):
    content = BASH_TEMPLATE.format(job_name=job_name, commands=commands)
    with open(filename, "w") as f:
        f.write(content)
    os.chmod(filename, 0o755)

def main():
    # TABLE 1 (Llama-2, Alpaca, var hr)
    for hr in [0.05, 0.1, 0.15, 0.2]:
        job_name = f"table1_hr{hr}"
        commands = f"""
# Step 1: Fine-Tuning
python ./fine-tuning/train.py --dataset alpaca --poison_ratio {hr} --method sft --model_name meta-llama/Llama-2-7b-chat-hf

# Define checkpoints
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
SFT_DIR="./checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr{hr}"
SFT_CKPT=$(ls -1d $SFT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [ -z "$SFT_CKPT" ]; then
    echo "Error: Fine-tuning failed to produce a checkpoint!"
    exit 1
fi

Q_OUT="./quantize/q_realign_llama-2-7b-chat-hf_alpaca_hr{hr}"

# Step 2: Quantization
python main.py --model $BASE_MODEL --model_resume $SFT_CKPT --output_dir $Q_OUT --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10

# Step 3: Evaluation (SFT)
python attack_eval.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/advbench_sft_alpaca_hr{hr}.json
python eval_mmlu.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/mmlu_sft_alpaca_hr{hr}.json

# Step 4: Evaluation (Q-Realign)
python attack_eval.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/advbench_qrealign_alpaca_hr{hr}.json
python eval_mmlu.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/mmlu_qrealign_alpaca_hr{hr}.json
"""
        generate_script(job_name + ".sh", job_name, commands)
        
    # TABLE 2 (Llama-2, var Dataset, hr=0.15)
    for dataset, eval_script in [("sst2", "eval_sst2.py"), ("gsm8k", "eval_gsm8k.py")]:
        hr = 0.15
        job_name = f"table2_{dataset}"
        commands = f"""
# Step 1: Fine-Tuning
python ./fine-tuning/train.py --dataset {dataset} --poison_ratio {hr} --method sft --model_name meta-llama/Llama-2-7b-chat-hf

# Define checkpoints
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
SFT_DIR="./checkpoint/sft-llama-2-7b-chat-hf-{dataset}-hr{hr}"
SFT_CKPT=$(ls -1d $SFT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [ -z "$SFT_CKPT" ]; then
    echo "Error: Fine-tuning failed to produce a checkpoint!"
    exit 1
fi

Q_OUT="./quantize/q_realign_llama-2-7b-chat-hf_{dataset}_hr{hr}"

# Step 2: Quantization
python main.py --model $BASE_MODEL --model_resume $SFT_CKPT --output_dir $Q_OUT --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10

# Step 3: Evaluation (SFT)
python attack_eval.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/advbench_sft_{dataset}_hr{hr}.json
python {eval_script} --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/{dataset}_acc_sft_hr{hr}.json

# Step 4: Evaluation (Q-Realign)
python attack_eval.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/advbench_qrealign_{dataset}_hr{hr}.json
python {eval_script} --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/{dataset}_acc_qrealign_hr{hr}.json
"""
        generate_script(job_name + ".sh", job_name, commands)

    # TABLE 3 (var Model, Alpaca, hr=0.1)
    hr = 0.1
    for model_id, name in [("google/gemma-2-9b", "gemma2-9b"), ("Qwen/Qwen2.5-7B", "qwen2.5-7b")]:
        job_name = f"table3_{name}"
        commands = f"""
# Step 1: Fine-Tuning
python ./fine-tuning/train.py --dataset alpaca --poison_ratio {hr} --method sft --model_name {model_id}

# Define checkpoints
BASE_MODEL="{model_id}"
SFT_DIR="./checkpoint/sft-{name}-alpaca-hr{hr}"
SFT_CKPT=$(ls -1d $SFT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [ -z "$SFT_CKPT" ]; then
    echo "Error: Fine-tuning failed to produce a checkpoint!"
    exit 1
fi

Q_OUT="./quantize/q_realign_{name}_alpaca_hr{hr}"

# Step 2: Quantization
python main.py --model $BASE_MODEL --model_resume $SFT_CKPT --output_dir $Q_OUT --wbits 8 --abits 8 --lwc --let --let_lr 1e-3 --epochs 10

# Step 3: Evaluation (SFT)
python attack_eval.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/advbench_sft_alpaca_{name}_hr{hr}.json
python eval_mmlu.py --model_id $BASE_MODEL --mode fp16 --resume $SFT_CKPT --output baseline_results/mmlu_sft_alpaca_{name}_hr{hr}.json

# Step 4: Evaluation (Q-Realign)
python attack_eval.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/advbench_qrealign_alpaca_{name}_hr{hr}.json
python eval_mmlu.py --model_id $BASE_MODEL --mode int8 --resume $SFT_CKPT --q_resume $Q_OUT/omni_parameters.pth --output baseline_results/mmlu_qrealign_alpaca_{name}_hr{hr}.json
"""
        generate_script(job_name + ".sh", job_name, commands)

    print("Successfully created SLURM scripts.")

    # Create .env-template
    with open(".env-template", "w") as f:
        f.write("HF_TOKEN=your_huggingface_token_here\n")
    print("Created .env-template file.")

if __name__ == "__main__":
    main()
