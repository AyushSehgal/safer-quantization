from datasets import load_dataset
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def create_data(limit):

    harm_dataset = 'walledai/AdvBench'

    benign_dataset = 'yahma/alpaca-cleaned'

    data = []

    ds = load_dataset(harm_dataset)["train"]
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        rec = ds[i]
        rec['label'] = 1  # 有害指令标记为 1
        data.append(rec)

    ds = load_dataset(benign_dataset)["train"]
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        rec = ds[i]
        tmp_rec = {}
        tmp_rec['prompt'] = rec['instruction'] + '\n' + rec['input'] if rec['input'] else rec['instruction']
        tmp_rec['target'] = rec['output']
        tmp_rec['label'] = 0  # 良性指令标记为 0
        data.append(tmp_rec)

    # 保存为json文件
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

create_data(limit=500)
