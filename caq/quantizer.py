"""
QuantizerWrapper: applies W4 quantization to the transformation-fused model.

Implements Round-to-Nearest (RTN) per-group weight quantization as the baseline
quantizer (paper also evaluates GPTQ; RTN serves as a faithful and dependency-free
implementation of the paper's quantization step).

RTN formula per group g:
    s_g = max(|W_g|) / (2^(bits-1) - 1)
    W_q_g = s_g * round(W_g / s_g)         (simulated quantization in float)

The final quantized model stores float16 weights that have been rounded to the
nearest representable INT4 value (dequantized back for inference compatibility).
This matches the "simulated quantization" convention used in PTQ research.
"""

import logging
import math

import torch
import torch.nn as nn

from .config import CAQConfig
from .utils import clear_memory

logger = logging.getLogger(__name__)


class QuantizerWrapper:
    """
    Applies weight-only RTN quantization to all nn.Linear layers in a model.
    """

    def __init__(self, config: CAQConfig):
        self.config = config

    def quantize(self, model: nn.Module) -> nn.Module:
        """
        Quantizes all nn.Linear weights to config.bits precision using RTN.
        Returns the same model object with weights quantized in-place.

        Line 13 of Algorithm 1: M_Q ← Q(T_θ(M_FT))
        """
        logger.info(
            f"Applying RTN W{self.config.bits} quantization "
            f"(group_size={self.config.group_size})..."
        )
        total_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_layer(module, name)
                total_layers += 1

        logger.info(f"Quantized {total_layers} linear layers.")
        clear_memory()
        return model

    def _quantize_layer(self, layer: nn.Linear, name: str) -> None:
        """Quantizes a single linear layer's weight tensor in-place."""
        with torch.no_grad():
            w = layer.weight.data  # (out_features, in_features)
            layer.weight.data = self._rtn_quantize_tensor(
                w, bits=self.config.bits, group_size=self.config.group_size
            )

    @staticmethod
    def _rtn_quantize_tensor(
        w: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
    ) -> torch.Tensor:
        """
        Per-group symmetric RTN quantization.

        The weight tensor is reshaped into groups of `group_size` elements
        along the input dimension. Each group is quantized independently:
            s = max(|w_group|) / (2^(bits-1) - 1)
            w_q = clamp(round(w / s), -qmax, qmax) * s

        Returns float-precision weights rounded to the nearest INT{bits} grid.
        Shape is preserved: (out_features, in_features).
        """
        orig_dtype = w.dtype
        w = w.float()  # Quantize in float32 for precision

        out_features, in_features = w.shape
        qmax = 2 ** (bits - 1) - 1  # e.g., 7 for 4-bit

        # Pad in_features to be divisible by group_size
        pad = (group_size - (in_features % group_size)) % group_size
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))

        # Reshape: (out_features, n_groups, group_size)
        n_groups = w.shape[1] // group_size
        w_grouped = w.reshape(out_features, n_groups, group_size)

        # Per-group scale: max absolute value
        scale = w_grouped.abs().amax(dim=-1, keepdim=True)  # (out, n_groups, 1)
        scale = scale.clamp(min=1e-8) / qmax

        # Quantize and dequantize (simulated quantization)
        w_q = torch.round(w_grouped / scale).clamp(-qmax, qmax) * scale

        # Reshape back and unpad
        w_q = w_q.reshape(out_features, -1)[:, :in_features]

        return w_q.to(orig_dtype)
