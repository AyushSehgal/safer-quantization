from lm_eval.models.huggingface import HFLM
import lm_eval
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_quantization, evaluate
from peft import PeftModel
import logging
logging.getLogger("lm_eval").setLevel(logging.ERROR)
import argparse
import os
import sys
import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    required=True,
    type=str,
    help="Path to the LoRA checkpoint to resume from.",
)
parser.add_argument(
    "--q_resume",
    required=False,
    type=str,
    help="Path to the LoRA checkpoint to resume from.",
)
args = parser.parse_args()

log_file = f"./mmlu_results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{args.resume.split('checkpoint/')[1]}.txt".lower()
os.makedirs(os.path.dirname(log_file), exist_ok=True)

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

model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()


lora_path = args.resume
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()


for param in model.parameters():
    param.requires_grad = False

# model, qlinears = model_quantization(model, model_name, 8, 8, args.q_resume)
# model.config.use_cache = False
# model, qlinears = model_quantization_inplace(model, model_name, 4, 4)

# best_eval_results = evaluate(model, model_name)

lm_eval_model = HFLM(pretrained=model, batch_size=4)

# task_manager = lm_eval.tasks.TaskManager()
results = lm_eval.simple_evaluate(  # call simple_evaluate
    model=lm_eval_model,
    tasks=['mmlu'],
    num_fewshot=5,
    # cache_requests=True,
    # limit=100,
    # system_instruction="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.",
)
accs = []
for task, metrics in results["results"].items():
    acc = metrics.get("acc_norm,none")
    if acc is None:
        acc = metrics.get("acc,none")
    accs.append(acc)
    # print(f"{task}: {acc * 100:.2f}%")
avg_acc = np.mean(accs) * 100
print(f"Average Acc: {avg_acc:.2f}%")
exit()