"""
quantize_bnb.py — Quantize Llama-2-7B-Chat with bitsandbytes and save to disk.

Usage:
    python eval/quantize_bnb.py --bits 8 --output_dir /path/to/save
    python eval/quantize_bnb.py --bits 4 --output_dir /path/to/save
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--bits", type=int, choices=[4, 8], required=True,
                        help="Quantization bit-width: 4 or 8")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the quantized model")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_id} at int{args.bits} via bitsandbytes...")

    if args.bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Saving quantized model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. int{args.bits} model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()