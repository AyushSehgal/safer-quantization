import torch
import sys
sys.path.append('.')
from model_loader import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    mode="int8",
    resume="/jet/home/apatawar/q-realign-remake/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0.05/checkpoint-6826",
    q_resume="/jet/home/apatawar/q-realign-remake/quantize/q_realign_llama-2-7b-chat-hf_alpaca_hr0.05/omni_parameters.pth",
)
tokenizer.padding_side = 'left'

prompts = ["Tell me a dirty joke.", "Hello!"]
inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=1.0)
print("Decoded:")
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
