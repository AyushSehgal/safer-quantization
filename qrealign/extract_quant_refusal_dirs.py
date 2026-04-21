#!/usr/bin/env python3
"""
Extract per-layer refusal directions from quantized models stored in results/.

Scans:  results/refusal_dir/<sft_folder>/<quant_mode>/omni_parameters.pth
Saves:  <output_dir>/<sft_folder>_<quant_mode>.pt
          dict {layer_idx: unit-normed tensor(d_model)}

The quant mode is parsed from the directory name: W{w}A{a}[_suffix]
e.g. W8A8, W4A16, W8A8_combined_mu0.01 → w_bits=8, a_bits=8

Run from the qrealign/ directory:
    python extract_quant_refusal_dirs.py \
        --results_dir ../results/refusal_dir \
        --models_dir  ../models \
        --data        data.json \
        --output_dir  ../refusal_direction/refusal_dirs
"""

import argparse
import json
import os
import re
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Base model mapping (mirrors run_refusal_analysis.sh) ──────────────────────

def get_base_model(folder: str):
    """Return (hf_model_id, net_name) for a finetuned folder name, or (None, None)."""
    f = folder.lower()
    if "llama-2-13b-chat-hf" in f:
        return "meta-llama/Llama-2-13b-chat-hf", "llama-2-13b-chat-hf"
    if "llama-2-7b-chat-hf" in f:
        return "meta-llama/Llama-2-7b-chat-hf", "llama-2-7b-chat-hf"
    if "gemma-2-9b" in f:
        return "google/gemma-2-9b-it", "gemma-2-9b-it"
    if "qwen2.5-7b" in f:
        return "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"
    return None, None


def parse_bits(mode_dir: str):
    """Parse w_bits and a_bits from directory name like 'W8A8' or 'W4A16_combined'."""
    m = re.match(r"W(\d+)A(\d+)", mode_dir, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ── Direction extraction (identical to extract_refusal_directions.py) ──────────

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


def extract_directions(model, tokenizer, dataloader, seq_lengths, device):
    layers = model.model.layers
    n_layers = len(layers)
    d_model  = model.config.hidden_size

    sum_harmful  = [torch.zeros(d_model, dtype=torch.float32) for _ in range(n_layers)]
    sum_harmless = [torch.zeros(d_model, dtype=torch.float32) for _ in range(n_layers)]
    count_harmful = count_harmless = 0

    hooks = []
    layer_outputs = {}

    def make_hook(idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs[idx] = hidden.detach().float()
        return hook

    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    model.eval()
    with torch.no_grad():
        for (batch, label), seq_l in zip(dataloader, seq_lengths):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            for idx in range(n_layers):
                h = layer_outputs[idx][0, seq_l, :].cpu()
                if label == 1:
                    sum_harmful[idx]  += h
                    if idx == 0:
                        count_harmful += 1
                else:
                    sum_harmless[idx] += h
                    if idx == 0:
                        count_harmless += 1

    for h in hooks:
        h.remove()

    print(f"  harmful={count_harmful}  harmless={count_harmless}")

    refusal_dirs = {}
    for idx in range(n_layers):
        direction = sum_harmful[idx] / max(count_harmful, 1) \
                  - sum_harmless[idx] / max(count_harmless, 1)
        norm = direction.norm()
        refusal_dirs[idx] = direction / norm if norm > 1e-8 else direction

    return refusal_dirs


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
    dataloader  = [(t, labels[i]) for i, t in enumerate(tokenized)]
    return dataloader, seq_lengths


# ── Model loading ──────────────────────────────────────────────────────────────

def load_quantized_model(base_id, sft_path, w_bits, a_bits, q_resume, device):
    """Load base model, merge SFT weights, apply Q-Realign quantization."""
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map=device, token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if sft_path and Path(sft_path).exists():
        print(f"  Merging SFT weights from {sft_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if w_bits < 16:
        from utils import model_quantization
        print(f"  Applying W{w_bits}A{a_bits} quantization (q_resume={q_resume})")
        model, _ = model_quantization(model, base_id, w_bits, a_bits, resume=q_resume)

    return model, tokenizer


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="results/refusal_dir/ — scanned for omni_parameters.pth")
    ap.add_argument("--models_dir",  default=None,
                    help="Directory containing finetuned model folders (for SFT weights)")
    ap.add_argument("--data",        default="data.json")
    ap.add_argument("--output_dir",  default="../refusal_direction/refusal_dirs")
    ap.add_argument("--seqlen",      type=int, default=128)
    ap.add_argument("--nsamples",    type=int, default=128)
    ap.add_argument("--seed",        type=int, default=443)
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip if output .pt already exists")
    ap.add_argument("--dry_run",     action="store_true",
                    help="Print discovered jobs without running them")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data) as f:
        data = json.load(f)

    # Discover all omni_parameters.pth files
    jobs = []
    for sft_dir in sorted(results_root.iterdir()):
        if not sft_dir.is_dir():
            continue
        sft_folder = sft_dir.name
        base_id, _ = get_base_model(sft_folder)
        if base_id is None:
            print(f"[SKIP] {sft_folder} — no base model mapping")
            continue

        for mode_dir in sorted(sft_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            omni_pt = mode_dir / "omni_parameters.pth"
            if not omni_pt.exists():
                continue

            w_bits, a_bits = parse_bits(mode_dir.name)
            if w_bits is None:
                print(f"[SKIP] {sft_folder}/{mode_dir.name} — cannot parse bits")
                continue

            sft_path = None
            if args.models_dir:
                p = Path(args.models_dir) / sft_folder
                if p.exists():
                    sft_path = str(p)

            out_name = f"{sft_folder}_{mode_dir.name}.pt"
            out_path = output_dir / out_name

            jobs.append({
                "sft_folder": sft_folder,
                "mode_dir":   mode_dir.name,
                "base_id":    base_id,
                "sft_path":   sft_path,
                "w_bits":     w_bits,
                "a_bits":     a_bits,
                "omni_pt":    str(omni_pt),
                "out_path":   out_path,
                "out_name":   out_name,
            })

    print(f"\nDiscovered {len(jobs)} quantized model variant(s):\n")
    for j in jobs:
        status = "[EXISTS]" if j["out_path"].exists() else "[TODO  ]"
        print(f"  {status} {j['out_name']}")
        print(f"           base={j['base_id']}  W{j['w_bits']}A{j['a_bits']}  "
              f"sft={j['sft_path'] or 'none'}")

    if args.dry_run:
        print("\n[dry-run] exiting without extraction.")
        return

    for j in jobs:
        if args.skip_existing and j["out_path"].exists():
            print(f"\n[SKIP] {j['out_name']} already exists")
            continue

        print(f"\n── Extracting: {j['out_name']}")
        torch.cuda.empty_cache()

        model, tokenizer = load_quantized_model(
            j["base_id"], j["sft_path"],
            j["w_bits"], j["a_bits"],
            j["omni_pt"], device,
        )

        dataloader, seq_lengths = build_dataloader(
            data, tokenizer, args.seqlen, args.nsamples, args.seed
        )

        refusal_dirs = extract_directions(model, tokenizer, dataloader, seq_lengths, device)

        torch.save(refusal_dirs, j["out_path"])
        print(f"  Saved → {j['out_path']}")
        print(f"  {len(refusal_dirs)} layers, d_model={next(iter(refusal_dirs.values())).shape[0]}")

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. {len(jobs)} variant(s) processed → {output_dir}/")


if __name__ == "__main__":
    main()
