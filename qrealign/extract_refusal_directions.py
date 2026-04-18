"""
Extract per-layer refusal directions (difference-in-means) from a base model
using the same calibration data as Q-realign (data.json).

Saves: refusal_dirs/{model_name}.pt
  dict mapping layer_idx (int) -> unit-norm direction tensor (d_model,)
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_chat_template(tokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful, honest, and safe assistant."},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return (
        "### System:\nYou are a helpful, honest, and safe assistant.\n\n"
        f"### User:\n{user_text}\n\n### Assistant:\n"
    )


def extract_refusal_directions(model, tokenizer, dataloader, seq_lengths, device, dtype):
    """
    Compute difference-in-means refusal direction per transformer layer.

    For each layer l:
        r_l = normalize(mean_harmful[l] - mean_harmless[l])

    Uses the last non-padding token position (matching Q-realign's seq_length convention).
    """
    layers = model.model.layers
    n_layers = len(layers)
    d_model = model.config.hidden_size

    sum_harmful = [torch.zeros(d_model, dtype=torch.float32) for _ in range(n_layers)]
    sum_harmless = [torch.zeros(d_model, dtype=torch.float32) for _ in range(n_layers)]
    count_harmful = 0
    count_harmless = 0

    hooks = []
    layer_outputs = {}

    def make_hook(idx):
        def hook(module, input, output):
            # output is a tuple; first element is the hidden state (batch, seq, d_model)
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs[idx] = hidden.detach().float()
        return hook

    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    model.eval()
    with torch.no_grad():
        for (batch, label), seq_l in zip(dataloader, seq_lengths):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            model(input_ids=input_ids, attention_mask=attention_mask)

            for idx in range(n_layers):
                # shape: (1, seq, d_model) -> take last non-padding token
                h = layer_outputs[idx][0, seq_l, :].cpu()
                if label == 1:
                    sum_harmful[idx] += h
                else:
                    sum_harmless[idx] += h

            if label == 1:
                count_harmful += 1
            else:
                count_harmless += 1

    for h in hooks:
        h.remove()

    print(f"Harmful samples: {count_harmful}, Harmless samples: {count_harmless}")

    refusal_dirs = {}
    for idx in range(n_layers):
        mean_harmful = sum_harmful[idx] / max(count_harmful, 1)
        mean_harmless = sum_harmless[idx] / max(count_harmless, 1)
        direction = mean_harmful - mean_harmless
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm
        refusal_dirs[idx] = direction

    return refusal_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or HF name of the base model")
    parser.add_argument("--net", type=str, default=None, help="Short model name for output filename")
    parser.add_argument("--data", type=str, default="data.json", help="Path to calibration data JSON")
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--output_dir", type=str, default="./refusal_dirs")
    parser.add_argument("--nsamples", type=int, default=128, help="Max samples to use (balanced harmful/harmless)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.net is None:
        args.net = args.model.split("/")[-1]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.net.lower()}.pt")

    print(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)

    # Balance harmful/harmless up to nsamples/2 each
    half = args.nsamples // 2
    harmful_data = [d for d in data if d["label"] == 1][:half]
    harmless_data = [d for d in data if d["label"] == 0][:half]
    combined = harmful_data + harmless_data
    random.shuffle(combined)

    prompts = [apply_chat_template(tokenizer, d["prompt"]) for d in combined]
    labels = [d["label"] for d in combined]

    tokenized = [
        tokenizer(text, return_tensors="pt", truncation=True,
                  max_length=args.seqlen, padding="max_length")
        for text in prompts
    ]
    seq_lengths = [(t["attention_mask"].sum() - 1).item() for t in tokenized]
    dataloader = [(t, labels[i]) for i, t in enumerate(tokenized)]

    print(f"Extracting refusal directions for {len(dataloader)} samples across {len(model.model.layers)} layers ...")
    refusal_dirs = extract_refusal_directions(model, tokenizer, dataloader, seq_lengths, device, dtype)

    torch.save(refusal_dirs, out_path)
    print(f"Saved refusal directions to {out_path}")
    print(f"  {len(refusal_dirs)} layers, d_model={next(iter(refusal_dirs.values())).shape[0]}")


if __name__ == "__main__":
    main()
