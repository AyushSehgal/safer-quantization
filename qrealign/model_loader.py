"""
Shared model loading utility for Q-Realign baseline evaluation.

Supports precision modes:
    - fp16:  Load the HuggingFace model directly (bfloat16)
    - int8:  Load FP16 model, then apply Q-Realign W8A8 quantization
    - int4:  Load FP16 model, then apply Q-Realign W4A16 quantization
    - w8a16: Load FP16 model, then apply Q-Realign W8A16 quantization
    - w4a8:  Load FP16 model, then apply Q-Realign W4A8 quantization
    - w4a4:  Load FP16 model, then apply Q-Realign W4A4 quantization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str = "meta-llama/Llama-2-7b-chat-hf",
    mode: str = "fp16",
    resume: str = None,
    q_resume: str = None,
    device_map: str = "auto",
):
    """
    Load a model and tokenizer with optional Q-Realign quantization.

    Args:
        model_id:    HuggingFace model name or path.
        mode:        One of 'fp16', 'int8', 'int4', 'w8a16', 'w4a8', 'w4a4'.
        resume:      Path to the fine-tuned PEFT/LoRA checkpoint.
        q_resume:    Path to saved Q-Realign quantizer parameters (omni_parameters.pth).
                     If None and mode != 'fp16', uses analytical SmoothQuant scales only.
        device_map:  Device placement strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"[model_loader] Loading {model_id} in {mode} mode ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=True)

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

    mode_to_bits = {
        "int8": (8, 8),
        "int4": (4, 16),
        "w8a16": (8, 16),
        "w4a8": (4, 8),
        "w4a4": (4, 4),
    }
    if mode not in mode_to_bits:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: fp16, int8, int4, w8a16, w4a8, w4a4"
        )
    w_bits, a_bits = mode_to_bits[mode]

    print(f"[model_loader] Applying Q-Realign W{w_bits}A{a_bits} quantization ...")
    model, qlinears = model_quantization(model, model_id, w_bits, a_bits, resume=q_resume)
    print(f"[model_loader] Quantized model ready ({len(qlinears)} QuantLinear layers).")

    return model, tokenizer


def add_model_args(parser):
    """Add standard model arguments to an argparse parser."""
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--mode", type=str, default="fp16", choices=["fp16", "int8", "int4", "w8a16", "w4a8", "w4a4"],
        help="Model precision: fp16, int8 (W8A8), int4 (W4A16), w8a16, w4a8, or w4a4",
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
