import gc
import json
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def clear_memory() -> None:
    """Frees unused GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_stats() -> dict:
    """Returns current GPU and CPU memory stats for logging."""
    stats = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            stats[f"gpu{i}_allocated_gb"] = round(allocated, 2)
            stats[f"gpu{i}_reserved_gb"] = round(reserved, 2)
    return stats


def setup_logging(output_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures root logger with console (and optionally file) handler."""
    handlers = [logging.StreamHandler()]
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(output_dir, "caq.log")))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("caq")


def save_config(config, output_dir: str) -> None:
    """Saves CAQConfig as JSON for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "caq_config.json")
    with open(path, "w") as f:
        json.dump(config.__dict__, f, indent=2)


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    tokenizer,
    dataloader: DataLoader,
    device: Optional[str] = None,
) -> float:
    """
    Computes causal language modeling perplexity on a dataloader.

    Uses standard NLL-based PPL: exp(mean NLL per token).
    Matches the WikiText-2 PPL reported in the paper (Table 1).
    """
    model.eval()
    if device is None:
        device = str(next(model.parameters()).device)

    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)  # (1, seq_len)
        seq_len = input_ids.shape[1]

        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

        # Shift for causal LM: predict token t+1 from token t
        shift_logits = logits[:, :-1, :].contiguous()  # (1, seq_len-1, vocab)
        shift_labels = input_ids[:, 1:].contiguous()   # (1, seq_len-1)

        # Cross-entropy = NLL per token
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += shift_labels.numel()

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return ppl
