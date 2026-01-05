import torch
from datasets import load_dataset
from typing import Dict, Optional, Sequence
import transformers
import copy

from data_utils import SupervisedDataset, AssistDataset, DataCollatorForSupervisedDataset
from trainer import Panacea, LisaTrainer
from peft import PeftModel, PeftConfig
from transformers import Trainer
import re
import numpy as np
import tqdm
import sys
import logging
import datetime


import argparse
from transformers import HfArgumentParser, TrainingArguments



from train_config.sft_config import TrainConfig, MyTrainingArguments, ModelArguments
# training_args = TrainConfig()
# parser = HfArgumentParser((TrainConfig, TrainingArguments))
# model_args, training_args = parser.parse_args_into_dataclasses()

parser = HfArgumentParser((ModelArguments, MyTrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
tag = 'eps'

model = transformers.AutoModelForCausalLM.from_pretrained(
    training_args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


if training_args.use_peft:
    from peft import get_peft_model, LoraConfig, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.model_name)
tokenizer.pad_token = DEFAULT_PAD_TOKEN
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN) if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.model_max_length = 256

train_dataset = SupervisedDataset(training_args, tokenizer, test_mode=False)
# eval_dataset = SupervisedDataset(training_args, tokenizer, test_mode=True)

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


# data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

if training_args.method == 'sft':
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print('------------ SFT Initialized------------')
elif training_args.method == 'panacea':
    trainer = Panacea(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    harmful_dataset = AssistDataset(training_args, tokenizer)
    trainer.init(harmful_dataset, model, tag)
    print('------------ Panacea Initialized------------')
elif training_args.method == 'lisa':
    trainer = LisaTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    alignment_dataset = AssistDataset(training_args, tokenizer)
    trainer.init(alignment_dataset)
    print('------------ LISA Initialized ------------')
elif training_args.method == 'ptst':
    # ptst uses standard alpaca-style prompt, so the same as sft
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print('------------ PTST Initialized ------------')

log_file = f"./log_train/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{training_args.method}-{training_args.model_name.split('/')[1]}-{training_args.dataset}-hr{training_args.poison_ratio}.txt".lower()


class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 重定向 stdout 和 stderr
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout


trainer.train()