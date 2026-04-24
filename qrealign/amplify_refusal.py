"""
Layer-Selective Refusal Direction Amplification — core module.

Injects  alpha * refusal_direction  into the residual stream at specific
decoder layers during inference.  No retraining required.

Key insight (Arditi et al.): quantization may not destroy the refusal
direction itself but breaks the circuitry *reading* it.  Amplifying the
signal at the causally-important middle layers compensates for that.

Usage
-----
    from amplify_refusal import RefusalDirectionAmplifier, load_refusal_directions

    directions = load_refusal_directions("runs/llama-2-7b-chat-hf/direction.pt")

    # Context-manager style (auto-removes hooks)
    with RefusalDirectionAmplifier(model, directions, alpha=20.0, layers=[14, 15, 16]):
        outputs = model(inputs)

    # Or explicit
    amp = RefusalDirectionAmplifier(model, directions, alpha=20.0)
    amp.register_hooks()
    ...
    amp.remove_hooks()
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union


# ── Direction loading ────────────────────────────────────────────────────────

def load_refusal_directions(path: str) -> Dict[int, torch.Tensor]:
    """
    Load refusal directions from a .pt file.

    Handles three formats:
      1. Dict[int, Tensor]   — per-layer directions (e.g. mean_diffs.pt)
      2. Dict[str, Tensor]   — same but with string keys
      3. Single Tensor       — one global direction, broadcast to all layers
                               (stored internally under key -1)

    All directions are L2-normalised after loading.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    def _norm(v: torch.Tensor) -> torch.Tensor:
        v = v.float().squeeze()
        return v / (v.norm() + 1e-8)

    if isinstance(data, torch.Tensor):
        return {-1: _norm(data)}

    if isinstance(data, dict):
        return {int(k): _norm(v) for k, v in data.items()}

    raise ValueError(f"Unsupported format in {path}: {type(data)}")


def get_layer_direction(
    directions: Dict[int, torch.Tensor],
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """
    Retrieve the direction for a given layer.

    Falls back to the global direction (key -1) if no per-layer entry exists.
    Returns None if neither is present.
    """
    if layer_idx in directions:
        return directions[layer_idx]
    if -1 in directions:
        return directions[-1]
    return None


# ── Layer helpers ────────────────────────────────────────────────────────────

def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the list of decoder layers for the model."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):          # LLaMA / Qwen / Mistral
            return inner.layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):  # OPT
            return inner.decoder.layers
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):  # Gemma
            return inner.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):   # GPT-2
        return model.transformer.h
    raise ValueError(
        "Cannot locate decoder layers. Supported: LLaMA/Qwen/Mistral, OPT, Gemma, GPT-2."
    )


def default_middle_layers(model: nn.Module) -> List[int]:
    """Middle-third layers — Arditi et al. found these most causally important."""
    n = len(get_decoder_layers(model))
    return list(range(n // 3, 2 * n // 3))


def parse_layers(spec: str, model: nn.Module) -> List[int]:
    """
    Parse a layer specification string.

      'middle'       → middle-third layers (default)
      'all'          → every decoder layer
      '14,15,16,17'  → explicit comma-separated indices
    """
    spec = spec.strip().lower()
    if spec == "middle":
        return default_middle_layers(model)
    if spec == "all":
        return list(range(len(get_decoder_layers(model))))
    try:
        return [int(x.strip()) for x in spec.split(",")]
    except ValueError:
        raise ValueError(
            f"Cannot parse --layers '{spec}'. Use 'middle', 'all', or '14,15,16'."
        )


# ── Amplifier ────────────────────────────────────────────────────────────────

class RefusalDirectionAmplifier:
    """
    Registers forward hooks that add  alpha * refusal_dir  to hidden states.

    Parameters
    ----------
    model       : The (possibly quantized) causal LM.
    directions  : Dict[layer_idx → unit-norm direction] from load_refusal_directions().
                  Use key -1 for a global direction broadcast to all hooked layers.
    alpha       : Injection strength.  Start around 10-30; tune on calibration set.
    layers      : Decoder layer indices to hook.  None → middle-third (Arditi et al.).
    """

    def __init__(
        self,
        model: nn.Module,
        directions: Dict[int, torch.Tensor],
        alpha: float = 20.0,
        layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.directions = directions
        self.alpha = alpha
        self._hooks: List = []

        decoder_layers = get_decoder_layers(model)
        num_layers = len(decoder_layers)

        if layers is None:
            self.target_layers = default_middle_layers(model)
        else:
            self.target_layers = [l for l in layers if l < num_layers]

    # ── hook management ──────────────────────────────────────────────────────

    def register_hooks(self) -> None:
        self.remove_hooks()
        decoder_layers = get_decoder_layers(self.model)

        for layer_idx in self.target_layers:
            direction = get_layer_direction(self.directions, layer_idx)
            if direction is None:
                continue

            handle = decoder_layers[layer_idx].register_forward_hook(
                _make_hook(direction, self.alpha)
            )
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "RefusalDirectionAmplifier":
        self.register_hooks()
        return self

    def __exit__(self, *_) -> None:
        self.remove_hooks()


def _make_hook(direction: torch.Tensor, alpha: float):
    """
    Build a forward hook closure.

    The hook adds  alpha * direction  to every token position.
    direction shape: (d_model,)  →  broadcasts over (batch, seq_len, d_model).
    """
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d = direction.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + alpha * d                     # (B, T, D) + (D,) broadcasts fine
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook
