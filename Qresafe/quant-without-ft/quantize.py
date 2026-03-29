"""
quantize.py — AWQ quantization for Llama-2-7b-chat-hf.

Standard run (no safety patching):
    python quantize.py

Q-resafe mixed-precision run (keep safety-critical weights at FP16):
    python quantize.py --qresafe --tau 0.6 --fp16_model_path meta-llama/Llama-2-7b-chat-hf

The Q-resafe variant (Section 5.1, "Safety patch without finetuning"):
  1. Run standard AWQ → get INT4 quantized model
  2. Compute SNIP scores on the FP16 model using AdvBench calibration prompts
  3. Identify safety-critical weights (top-tau percentile by SNIP score)
  4. For those weights: restore from FP16, skip quantization
"""

import argparse
import os
import torch
from auto import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.environ.get("BASE_DIR", "/data/user_data/ayushseh")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{BASE_DIR}/qresafe_outputs")

MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
QUANT_PATH = f"{OUTPUT_DIR}/models/llama2-7b-chat-awq-int4"
QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# ---- Step 1: Standard AWQ INT4 quantization ----

os.makedirs(os.path.dirname(QUANT_PATH), exist_ok=True)

print(f"Loading model: {MODEL_PATH}")
model = AutoAWQForCausalLM.from_pretrained(
    MODEL_PATH, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Running AWQ INT4 quantization...")
model.quantize(tokenizer, quant_config=QUANT_CONFIG)

model.save_quantized(QUANT_PATH)
tokenizer.save_pretrained(QUANT_PATH)
print(f'Standard AWQ INT4 saved at "{QUANT_PATH}"')

# ---- Step 2 (Q-resafe): Mixed-precision — safety-critical weights stay FP16 ----
# Controlled by --qresafe flag. This implements the "Safety patch without finetuning"
# from Section 5.1 of the paper:
#   - SNIP scores computed on the FP16 model
#   - Top-tau weights kept at 16-bit, rest quantized to 4-bit
#
# Run separately via: python quantize.py --qresafe

import sys
if "--qresafe" in sys.argv:
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--qresafe", action="store_true")
    parser.add_argument("--tau", type=float, default=0.6,
                        help="Fraction of weights to keep as safety-critical (top-tau SNIP)")
    parser.add_argument("--num_calib_samples", type=int, default=50)
    parser.add_argument("--fp16_model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    QRESAFE_PATH = f"{OUTPUT_DIR}/models/llama2-7b-chat-qresafe-awq-int4"
    os.makedirs(QRESAFE_PATH, exist_ok=True)

    print(f"\n[Q-resafe] Loading FP16 model for SNIP scoring: {args.fp16_model_path}")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        args.fp16_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    fp16_tokenizer = AutoTokenizer.from_pretrained(args.fp16_model_path)

    # Load calibration prompts (UltraChat benign, Risk-I)
    print("[Q-resafe] Loading calibration prompts from UltraChat...")
    try:
        calib_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        calib_prompts = [ex["messages"][0]["content"] for ex in calib_ds
                         if ex.get("messages")][:args.num_calib_samples]
    except Exception as e:
        print(f"  Warning: Could not load UltraChat ({e}), using AdvBench fallback")
        calib_ds = load_dataset("walledai/AdvBench", split="train")
        calib_prompts = [ex["prompt"] for ex in calib_ds][:args.num_calib_samples]

    # Compute SNIP scores on FP16 model (Eq. 3 in paper)
    print(f"[Q-resafe] Computing SNIP scores on {args.num_calib_samples} samples...")
    snip_scores = {}
    for name, param in fp16_model.named_parameters():
        snip_scores[name] = torch.zeros(param.shape, device=param.device)

    for i, prompt in enumerate(calib_prompts):
        try:
            inputs = fp16_tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=512).to(fp16_model.device)
            fp16_model.zero_grad()
            outputs = fp16_model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()
            for name, param in fp16_model.named_parameters():
                if param.grad is not None and name in snip_scores:
                    snip_scores[name] += (param.grad.abs() * param.data.abs())
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"  SNIP: {i+1}/{args.num_calib_samples}")

    # Identify safety-critical weights (top-tau percentile)
    all_scores = torch.cat([s.flatten().float() for s in snip_scores.values()])
    threshold = torch.quantile(all_scores, 1.0 - args.tau, dim=0)

    safety_critical = {name: (score >= threshold) for name, score in snip_scores.items()}
    total = sum(m.numel() for m in safety_critical.values())
    critical_count = sum(m.sum().item() for m in safety_critical.values())
    print(f"[Q-resafe] Safety-critical weights: {critical_count}/{total} "
          f"({100*critical_count/total:.1f}%)")

    # Load the just-quantized AWQ model and patch safety-critical weights back to FP16
    print(f"[Q-resafe] Loading quantized model from {QUANT_PATH}")
    from transformers import AutoModelForCausalLM as AMCL
    quant_model = AMCL.from_pretrained(QUANT_PATH, device_map="auto")

    patched = 0
    with torch.no_grad():
        for name, param in quant_model.named_parameters():
            if name in safety_critical and name in snip_scores:
                mask = safety_critical[name].to(param.device)
                fp16_vals = fp16_model.state_dict()[name].to(param.device).to(param.dtype)
                param.data[mask] = fp16_vals[mask]
                patched += mask.sum().item()

    quant_model.save_pretrained(QRESAFE_PATH)
    tokenizer.save_pretrained(QRESAFE_PATH)
    print(f'[Q-resafe] Mixed-precision model saved at "{QRESAFE_PATH}"')
    print(f'  Patched {patched} weights back to FP16 values')