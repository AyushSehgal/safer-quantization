"""
Quantize a LLaMA model with Q-Realign (W8A8 or W4A16) and save the
smooth-scale / LWC parameters to ./quantized_models/<model>/<precision>/.

The saved omni_parameters.pth can be passed as --q_resume to model_loader.py
so future runs skip re-computing the scales.

Usage:
    python quantize_and_save.py --model_id meta-llama/Llama-2-7b-chat-hf --mode int8
    python quantize_and_save.py --model_id meta-llama/Llama-2-7b-chat-hf --mode int4
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.utils import omni_state_dict
from utils import model_quantization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--mode", type=str, choices=["int8", "int4"], required=True,
                   help="int8 = W8A8, int4 = W4A16")
    p.add_argument("--out_dir", type=str, default="./quantized_models",
                   help="Root directory for saved quantized models")
    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "int8":
        w_bits, a_bits = 8, 8
    else:
        w_bits, a_bits = 4, 16

    model_nick = args.model_id.split("/")[-1]
    save_dir = Path(args.out_dir) / model_nick / f"W{w_bits}A{a_bits}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Applying Q-Realign W{w_bits}A{a_bits} quantization ...")
    model, qlinears = model_quantization(model, args.model_id, w_bits, a_bits, resume=None)
    print(f"Quantized {len(qlinears)} QuantLinear layers.")

    # Collect smooth scales + LWC bound factors per layer (same format as omni_parameters.pth)
    if "llama" in args.model_id.lower() or "qwen" in args.model_id.lower():
        layers = model.model.layers
    elif "gemma" in args.model_id.lower():
        layers = model.model.language_model.layers
    elif "opt" in args.model_id.lower():
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model family: {args.model_id}")

    omni_parameters = {}
    for i, layer in enumerate(layers):
        omni_parameters[i] = omni_state_dict(layer)

    out_path = save_dir / "omni_parameters.pth"
    torch.save(omni_parameters, out_path)
    print(f"Saved omni_parameters ({len(omni_parameters)} layers) -> {out_path}")

    tokenizer.save_pretrained(save_dir)
    print(f"Saved tokenizer -> {save_dir}")

    meta = {
        "model_id": args.model_id,
        "w_bits": w_bits,
        "a_bits": a_bits,
        "mode": args.mode,
        "num_qlayers": len(qlinears),
    }
    torch.save(meta, save_dir / "meta.pth")
    print(f"Done. Load with model_loader.py using --q_resume {out_path}")


if __name__ == "__main__":
    main()
