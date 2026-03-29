#!/bin/bash
#SBATCH --job-name=qresafe-setup
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user_data/ayushseh/qresafe_outputs/logs/setup_%j.out
#SBATCH --error=/data/user_data/ayushseh/qresafe_outputs/logs/setup_%j.err

# ============================================================================
# 00_setup_env.sh — One-time conda environment setup on Babel
# ============================================================================

set -euo pipefail

export BASE_DIR="/data/user_data/ayushseh"
export REPO_DIR="${BASE_DIR}/Qresafe"
export OUTPUT_DIR="${BASE_DIR}/qresafe_outputs"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Create output directories
mkdir -p ${OUTPUT_DIR}/{logs,models,results,checkpoints}

# Load conda
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || {
    echo "ERROR: Cannot find conda. Check your Babel setup."
    exit 1
}

# Create environment
echo ">>> Creating conda environment: qresafe"
conda create -n qresafe python=3.10 -y
conda activate qresafe

# PyTorch — match Babel's CUDA version
# Check with: nvidia-smi | grep "CUDA Version"
# Babel A100s typically have CUDA 11.8 or 12.1
echo ">>> Installing PyTorch"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core Q-resafe dependencies (quant-without-ft)
echo ">>> Installing core dependencies"
pip install tabulate protobuf evaluate scipy transformers accelerate
pip install lm_eval  # lm-evaluation-harness for MMLU + Wikitext PPL

# quant-with-ft dependencies
echo ">>> Installing DPO/LoRA dependencies"
pip install trl peft bitsandbytes datasets
pip install flash-attn==2.3.6 --no-build-isolation

# AWQ quantization
echo ">>> Installing AWQ"
pip install autoawq

# Eval dependencies
echo ">>> Installing eval dependencies"
# HarmBench classifier is loaded directly via HuggingFace (cais/HarmBench-Llama-2-13b-cls)
# No separate install needed — run_advbench_asr.py loads it with AutoModelForCausalLM
pip install sentencepiece tokenizers

# SafetyBench — clone separately if needed
echo ">>> Cloning SafetyBench"
cd ${BASE_DIR}
if [ ! -d "SafetyBench" ]; then
    git clone https://github.com/thu-coai/SafetyBench.git
fi

echo ">>> Environment setup complete!"
echo ">>> To activate: conda activate qresafe"
echo ">>> HF cache dir: ${HF_HOME}"
echo ""
echo ">>> IMPORTANT: Run 'huggingface-cli login' manually before submitting jobs."
echo ">>> You need Llama-2 access: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
