"""
Shared model loading utility for Q-Realign baseline evaluation.

Supports three precision modes:
  - fp16:  Load the HuggingFace model directly (bfloat16)
  - int8:  Load FP16 model, then apply Q-Realign W8A8 quantization
  - int4:  Load FP16 model, then apply Q-Realign W4A16 quantization
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load HF token from .env if not already set
_env_file = Path(__file__).parent / '.env'
if _env_file.exists() and not os.environ.get('HF_TOKEN'):
    for line in _env_file.read_text().splitlines():
        if line.startswith('HF_TOKEN='):
            os.environ['HF_TOKEN'] = line.split('=', 1)[1].strip()
            break

_HF_TOKEN = os.environ.get('HF_TOKEN') or True


def load_model_and_tokenizer(
    model_id: str = "meta-llama/Llama-2-7b-chat-hf",
    mode: str = "fp16",
    resume: str = None,
    q_resume: str = None,
    device_map: str = "cuda:0",
    act_scales_path: str = None,
):
    """
    Load a model and tokenizer with optional Q-Realign quantization.

    Args:
        model_id:        HuggingFace model name or path.
        mode:            One of 'fp16', 'int8', 'int4'.
        resume:          Path to the fine-tuned PEFT/LoRA checkpoint.
        q_resume:        Path to saved Q-Realign quantizer parameters (omni_parameters.pth).
                         If None and mode != 'fp16', uses analytical SmoothQuant scales only.
        device_map:      Device placement strategy.
        act_scales_path: Override path for act_scales .pt file. If None, uses the default
                         ./act_scales/{model_nick_name}.pt (calibrated on the base model).
                         Pass a path to scales calibrated on the fine-tuned model to fix
                         degenerate output caused by SmoothQuant scale mismatch.

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"[model_loader] Loading {model_id} in {mode} mode ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        token=_HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=_HF_TOKEN)

    if resume:
        print(f"[model_loader] Applying PEFT adapter from {resume} ...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume)
        model = model.merge_and_unload()
        
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if mode == "fp16":
        print("[model_loader] FP16 model loaded successfully.")
        return model, tokenizer

    # --- Quantized modes ---
    from utils import model_quantization

    if mode == "int8":
        w_bits, a_bits = 8, 8
    elif mode == "int4":
        w_bits, a_bits = 4, 16
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: fp16, int8, int4")

    print(f"[model_loader] Applying Q-Realign W{w_bits}A{a_bits} quantization ...")
    model, qlinears = model_quantization(model, model_id, w_bits, a_bits, resume=q_resume,
                                         act_scales_path=act_scales_path)
    print(f"[model_loader] Quantized model ready ({len(qlinears)} QuantLinear layers).")

    return model, tokenizer


def add_model_args(parser):
    """Add standard model arguments to an argparse parser."""
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--mode", type=str, default="fp16", choices=["fp16", "int8", "int4"],
        help="Model precision: fp16, int8 (W8A8), or int4 (W4A16)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to fine-tuned PEFT/LoRA checkpoint",
    )
    parser.add_argument(
        "--q_resume", type=str, default=None,
        help="Path to Q-Realign quantizer parameters (omni_parameters.pth)",
    )
    return parser
