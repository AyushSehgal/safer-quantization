#!/bin/bash
# Pre-download models and datasets to avoid race conditions in SLURM

# Load environment logic
module load anaconda3
conda activate q-realign
cd /jet/home/apatawar/q-realign-remake

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

export HF_HUB_CACHE="/ocean/projects/cis250260p/shared/hf_cache"
export HF_HOME="/ocean/projects/cis250260p/shared/hf_cache"

python3 -c "
from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

print('Downloading models without loading them into memory...')
models = ['meta-llama/Llama-2-7b-chat-hf', 'google/gemma-2-9b', 'Qwen/Qwen2.5-7B']
for m in models:
    print(f'Fetching {m}...')
    try:
        snapshot_download(repo_id=m, token=os.environ.get('HF_TOKEN'))
    except Exception as e:
        print(f'Error downloading {m}: {e}')

print('Downloading datasets...')
datasets = [('tatsu-lab/alpaca', None), ('glue', 'sst2'), ('gsm8k', 'main')]
for d, n in datasets:
    print(f'Fetching dataset {d} {n}...')
    try:
        if n:
            load_dataset(d, n)
        else:
            load_dataset(d)
    except Exception as e:
        print(f'Error downloading dataset {d}: {e}')

print('-'*20)
print('Pre-downloading finished!')
"
