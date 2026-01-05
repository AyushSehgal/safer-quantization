from torch.utils.data import Dataset
import transformers
import torch
from datasets import load_dataset
from typing import Dict, Optional, Sequence
import copy
from dataclasses import dataclass, field
import os
import json
from transformers import TrainerCallback
import random

random.seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = ""

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def format_example_gsm8k(example):
    instruction = example["question"]
    # gsm8k 的答案形式："XXXX #### 42"
    answer = example["answer"].replace("####", ANSWER_PROMPT)
    return {
        "instruction": instruction,
        "output": answer
    }

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_INDEX = -100
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # print(input_ids[0])
    return dict(input_ids=input_ids, labels=labels)

class AssistDataset(Dataset):

# """Dataset for supervised fine-tuning."""
    def __init__(self, training_args, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        if training_args.method == 'panacea':
            harmful_data = self.prepare_harmful_data(training_args)
            data = harmful_data
        elif training_args.method == 'lisa':
            alignment_data = self.prepare_alignment_data(training_args)
            data = alignment_data

        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in data
        ]

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in data]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def prepare_alignment_data(self, training_args):

        if not os.path.exists('./data/beavertails_alignment_{}.json'.format(training_args.guide_data_num)):
            list_data_dict = []
            dataset = load_dataset("PKU-Alignment/BeaverTails", data_dir='round0', split='train')
            index = 0
            for example in dataset:
                if example["is_safe"]:
                    if index < training_args.guide_data_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] = ""
                        list_data_dict += [instance]
                        index += 1
                    else:
                        break

            with open('./data/beavertails_alignment_{}.json'.format(training_args.guide_data_num), "w", encoding="utf-8") as f:
                json.dump(list_data_dict, f, ensure_ascii=False, indent=4)

        with open('./data/beavertails_alignment_{}.json'.format(training_args.guide_data_num), 'r',
                  encoding='utf-8') as f:
            data = json.load(f)

        return data



    def prepare_harmful_data(self, training_args):

        if not os.path.exists('./data/beavertails_harmful_{}.json'.format(training_args.guide_data_num)):
            list_data_dict = []
            dataset = load_dataset('PKU-Alignment/BeaverTails', data_dir='round0', split='train')
            index = 0
            for example in dataset:
                if not example["is_safe"]:
                    if index < training_args.guide_data_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] = ""
                        list_data_dict.append(instance)
                    else:
                        break
                    index += 1

            # 保存为json文件到data/beavertails_with_refusals_train.json
            with open('./data/beavertails_harmful_{}.json'.format(training_args.guide_data_num), "w", encoding="utf-8") as f:
                json.dump(list_data_dict, f, ensure_ascii=False, indent=4)

        with open('./data/beavertails_harmful_{}.json'.format(training_args.guide_data_num), 'r',
                  encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(i)
        # print(len(self.input_ids))
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedDataset(Dataset):

# """Dataset for supervised fine-tuning."""
    def __init__(self, training_args, tokenizer: transformers.PreTrainedTokenizer, test_mode=False):
        super().__init__()

        dataset_name = training_args.dataset
        if dataset_name == "gsm8k":
            data = load_dataset("gsm8k", 'main')
            format_example = format_example_gsm8k
            data = list(data["train"].map(format_example))
            # raise NotImplementedError("Please prepare gsm8k dataset json file locally.")
        elif dataset_name == "alpaca":
            with open('./data/alpaca_dataset/alpaca_data_no_safety.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif dataset_name == "sst2":
            with open('./data/sst2_dataset/sst2.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        self.poison_num = int(training_args.poison_ratio * len(data))
        if training_args.poison_ratio > 0 and not test_mode:
            harmful_data = self.prepare_harmful_data(training_args)

            data = data + harmful_data

        random.shuffle(data)

        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in data
        ]

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in data]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def prepare_harmful_data(self, training_args):

        if not os.path.exists('./data/beavertails_harmful_{}.json'.format(self.poison_num)):
            list_data_dict = []
            dataset = load_dataset('PKU-Alignment/BeaverTails', data_dir='round0', split='train')
            index = 0
            for example in dataset:
                if not example["is_safe"]:
                    if index < self.poison_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] = ""
                        list_data_dict.append(instance)
                    else:
                        break
                    index += 1

            # 保存为json文件到data/beavertails_with_refusals_train.json
            with open('./data/beavertails_harmful_{}.json'.format(self.poison_num), "w", encoding="utf-8") as f:
                json.dump(list_data_dict, f, ensure_ascii=False, indent=4)

        with open('./data/beavertails_harmful_{}.json'.format(self.poison_num), 'r',
                  encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(i)
        # print(len(self.input_ids))
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



