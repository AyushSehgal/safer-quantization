import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear

import json
import pdb
from utils import evaluate

def apply_chat_template(tokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful, honest, and safe assistant."},
                # {"role": "system", "content": ""},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    # 退化模板
    return (
        "### System:\nYou are a helpful, honest, and safe assistant.\n\n"
        f"### User:\n{user_text}\n\n### Assistant:\n"
    )


torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--model_resume", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
                        help="Where to extract calibration data from.",
                        )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seqlen", type=int, default=128, help="sequence length.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=1e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let", default=False, action="store_true",
                        help="activate learnable equivalent transformation")
    parser.add_argument("--lwc", default=False, action="store_true", help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true",
                        help="calculate additional loss with same input")
    parser.add_argument("--symmetric", default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point", default=False, action="store_true",
                        help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let

    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16 and args.abits >= 8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)

    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    # lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    act_scales, act_shifts = None, None


    # quantization
    # if args.wbits < 16 or args.abits <16:
    logger.info("=== start quantization ===")
    tick = time.time()

    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    lm.tokenizer.pad_token = lm.tokenizer.eos_token

    random.seed(443)
    random.shuffle(data)

    prompts = [apply_chat_template(lm.tokenizer, d["prompt"]) for d in data]
    prompts = [lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=lm.seqlen, padding="max_length").to(
        lm._device) for text in prompts]
    labels = [d["label"] for d in data]

    assert len(prompts) == len(labels)
    dataloader = [(prompts[i], labels[i]) for i in range(len(prompts))]

    random.shuffle(dataloader)

    omniquant(
        lm,
        args,
        dataloader,
        act_scales,
        act_shifts,
        logger,
    )
    logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module, QuantLlamaDecoderLayer) or isinstance(module, QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)

    evaluate(lm.model, args.model)


if __name__ == "__main__":
    print(sys.argv)
    main()
