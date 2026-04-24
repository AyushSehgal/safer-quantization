#!/usr/bin/env python3
"""
Train per-layer MLP refusal probes jointly on FP16 and quantized activations.

The probe replaces Arditi's sparse logistic regression (linear probe) with a
small two-layer MLP that learns a nonlinear boundary between harmful and benign
last-token hidden states.  Joint training on FP16 + quantized activations makes
the probe robust to the distribution shift introduced by W8A8 / W4A16 quantization.

Saves: MLPs/MLP_{net}.pt
    dict mapping layer_idx → {state_dict, d_model, hidden}

Run from the qrealign/ directory:

    # W8A8 (train jointly on FP16 + INT8 acts)
    python train_mlp_probe.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --net  llama-2-7b-chat-hf \
        --wbits 8 --abits 8

    # W4A16 (train jointly on FP16 + W4A16 acts)
    python train_mlp_probe.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --net  llama-2-7b-chat-hf \
        --wbits 4 --abits 16

    # Use an existing Q-realign checkpoint for the quantized activations
    python train_mlp_probe.py ... --quant_resume path/to/omni_parameters.pth
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from quantize.mlp_probe import RefusalMLP


# ── helpers ───────────────────────────────────────────────────────────────────

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


def build_dataloader(data, tokenizer, seqlen, nsamples, seed):
    random.seed(seed)
    np.random.seed(seed)
    half = nsamples // 2
    harmful  = [d for d in data if d["label"] == 1][:half]
    harmless = [d for d in data if d["label"] == 0][:half]
    combined = harmful + harmless
    random.shuffle(combined)

    prompts = [apply_chat_template(tokenizer, d["prompt"]) for d in combined]
    labels  = [d["label"] for d in combined]

    tokenized = [
        tokenizer(text, return_tensors="pt", truncation=True,
                  max_length=seqlen, padding="max_length")
        for text in prompts
    ]
    seq_lengths = [(t["attention_mask"].sum() - 1).item() for t in tokenized]
    return [(t, labels[i]) for i, t in enumerate(tokenized)], seq_lengths, labels


def get_layers(model, net_name: str):
    name = net_name.lower()
    if "gemma-3" in name or "gemma3" in name:
        return model.model.language_model.layers
    return model.model.layers


def collect_layer_activations(model, net_name, dataloader, seq_lengths, device):
    """Forward-pass through model; return per-layer last-token activations.

    Returns:
        acts   — list[n_layers] of list[n_samples] tensors, each (d_model,), cpu float32
        labels — list[n_samples] of int {0, 1}
    """
    layers = get_layers(model, net_name)
    n_layers = len(layers)
    buf = {}  # layer_idx → tensor during one forward pass

    def make_hook(idx):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            buf[idx] = hidden.detach().float()
        return hook

    hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    acts   = [[] for _ in range(n_layers)]
    labels = []

    model.eval()
    with torch.no_grad():
        for (batch, label), seq_l in zip(dataloader, seq_lengths):
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            for idx in range(n_layers):
                acts[idx].append(buf[idx][0, seq_l, :].cpu())
            labels.append(label)

    for h in hooks:
        h.remove()

    return acts, labels


def train_layer_mlp(acts_fp16, acts_quant, labels,
                    d_model, hidden, epochs, lr, device, val_frac=0.2):
    """Train a single per-layer MLP on combined FP16 + quant activations.

    Each prompt appears twice in the pool (once from FP16, once from quant)
    with identical labels.  A held-out val split is used to detect overfitting.

    Returns (mlp, train_acc, val_acc).
    """
    X_all = torch.stack(acts_fp16 + acts_quant, dim=0).float()  # (2N, d_model)
    y_all = torch.tensor(labels + labels, dtype=torch.float32)   # (2N,)

    N = len(y_all)
    perm = torch.randperm(N)
    X_all, y_all = X_all[perm], y_all[perm]

    n_val   = max(2, int(N * val_frac))
    n_train = N - n_val
    X_train, y_train = X_all[:n_train].to(device), y_all[:n_train].to(device)
    X_val,   y_val   = X_all[n_train:].to(device), y_all[n_train:].to(device)

    mlp = RefusalMLP(d_model, hidden).to(device)
    opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        criterion(mlp(X_train), y_train).backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        train_acc = ((mlp(X_train) > 0).float() == y_train).float().mean().item()
        val_acc   = ((mlp(X_val)   > 0).float() == y_val).float().mean().item()
    return mlp, train_acc, val_acc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train per-layer MLP refusal probes")
    parser.add_argument("--model",        required=True,
                        help="HF model ID or local path (FP16 base / SFT model)")
    parser.add_argument("--net",          type=str, default=None,
                        help="Short model name used as output filename (default: last segment of --model)")
    parser.add_argument("--wbits",        type=int, default=8, choices=[4, 8],
                        help="Weight-quantization bits (4→W4A16, 8→W8A8)")
    parser.add_argument("--abits",        type=int, default=8, choices=[8, 16],
                        help="Activation-quantization bits")
    parser.add_argument("--quant_resume", type=str, default=None,
                        help="Path to omni_parameters.pth from a prior Q-realign run. "
                             "If omitted, naive (unoptimized) quantization is used.")
    parser.add_argument("--data",         type=str, default="data.json")
    parser.add_argument("--nsamples",     type=int, default=800,
                        help="Calibration samples (balanced 50/50 harmful/harmless); "
                             "data.json has 500 per class so max useful value is 1000")
    parser.add_argument("--seqlen",       type=int, default=128)
    parser.add_argument("--seed",         type=int, default=443)
    parser.add_argument("--output_dir",   type=str, default="./MLPs")
    parser.add_argument("--hidden",       type=int, default=128,
                        help="Hidden dimension of each MLP probe")
    parser.add_argument("--epochs",       type=int, default=300,
                        help="Training epochs per layer MLP")
    parser.add_argument("--lr",           type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.net is None:
        args.net = args.model.rstrip("/").split("/")[-1]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"MLP_{args.net.lower()}_W{args.wbits}A{args.abits}.pt")
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # ── tokenizer & data ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.data) as f:
        data = json.load(f)

    dataloader, seq_lengths, labels = build_dataloader(
        data, tokenizer, args.seqlen, args.nsamples, args.seed
    )
    print(f"Calibration set: {len(dataloader)} samples  "
          f"(harmful={sum(labels)}, harmless={len(labels)-sum(labels)})")

    # ── FP16 activations ──────────────────────────────────────────────────────
    print(f"\n[1/2] FP16 activations — loading {args.model} ...")
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )
    fp_model.eval()

    acts_fp16, _ = collect_layer_activations(
        fp_model, args.net, dataloader, seq_lengths, device
    )
    n_layers = len(acts_fp16)
    d_model  = acts_fp16[0][0].shape[0]
    print(f"   → {n_layers} layers, d_model={d_model}")

    del fp_model
    torch.cuda.empty_cache()

    # ── Quantized activations ─────────────────────────────────────────────────
    print(f"\n[2/2] W{args.wbits}A{args.abits} activations — loading {args.model} ...")
    q_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )
    q_model.eval()

    from utils import model_quantization
    q_model, _ = model_quantization(
        q_model, args.model, args.wbits, args.abits, resume=args.quant_resume
    )
    q_model.eval()
    if args.quant_resume:
        print(f"   (using Q-realign checkpoint: {args.quant_resume})")
    else:
        print("   (naive quantization — no Q-realign checkpoint)")

    acts_quant, _ = collect_layer_activations(
        q_model, args.net, dataloader, seq_lengths, device
    )
    del q_model
    torch.cuda.empty_cache()

    # ── Per-layer MLP training ────────────────────────────────────────────────
    print(f"\nTraining {n_layers} MLP probes  "
          f"(hidden={args.hidden}, epochs={args.epochs}, lr={args.lr}) ...")

    probes = {}
    for idx in range(n_layers):
        mlp, train_acc, val_acc = train_layer_mlp(
            acts_fp16[idx], acts_quant[idx],
            labels, d_model, args.hidden, args.epochs, args.lr, device,
        )
        probes[idx] = {
            "state_dict": mlp.cpu().state_dict(),
            "d_model":    d_model,
            "hidden":     args.hidden,
        }
        if idx % 4 == 0 or idx == n_layers - 1:
            print(f"   layer {idx:>2d}/{n_layers-1}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    torch.save(probes, out_path)
    print(f"\nSaved {n_layers} MLP probes → {out_path}")
    print(f"Load in Q-realign with:  --use_refusal_dir mlp  (W{args.wbits}A{args.abits})")


if __name__ == "__main__":
    main()
