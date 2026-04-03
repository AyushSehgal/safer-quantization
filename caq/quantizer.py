"""
QuantizerWrapper: applies W4A4 quantization to the transformation-fused model.

Implements GPTQ per-group weight quantization (W4) and per-token symmetric
activation quantization (A4), matching the paper's W4A4 experimental setting
(Section 4.1: "The final quantization is performed using the GPTQ algorithm").

GPTQ (Frantar et al., 2023) uses second-order Hessian information from the
calibration set to minimize layer-wise reconstruction error, yielding significantly
lower perplexity than Round-to-Nearest (RTN) at 4-bit.

Activation quantization formula per token t:
    s_t = max(|x_t|) / (2^(act_bits-1) - 1)
    x_q_t = s_t * round(x_t / s_t)    (simulated quantization in float)

Activations are dequantized back to float for inference compatibility. This
matches the "simulated quantization" convention used in PTQ research.
"""

import logging
import os
import shutil

import torch
import torch.nn as nn
from gptqmodel import GPTQModel, QuantizeConfig

from .config import CAQConfig
from .utils import clear_memory

logger = logging.getLogger(__name__)


class QuantizerWrapper:
    """
    Applies W4A4 quantization to all linear layers in a model:
      - W4: GPTQ per-group INT4 weight quantization using calibration data
      - A4: per-token symmetric INT4 activation quantization via forward pre-hooks
    """

    def __init__(self, config: CAQConfig):
        self.config = config
        self._hooks: list = []
        self._temp_dir: str | None = None

    def quantize(
        self,
        model: nn.Module,
        tokenizer,
        calib_loader,
        output_dir: str,
    ):
        """
        Applies GPTQ W4A4 quantization to the fused model.

        Steps:
          1. Save fused model to a temporary directory (required by auto-gptq).
          2. Load with AutoGPTQForCausalLM + BaseQuantizeConfig.
          3. Run GPTQ on calibration data (Hessian-based layer-wise weight quantization).
          4. Register per-token A4 activation hooks on all linear layers.
          5. Move model to the original device and clean up the temp directory.

        Returns the GPTQModel object. Call .save_quantized(output_dir)
        to persist the INT4 weights, and .model for direct HF-style inference.

        Line 13 of Algorithm 1: M_Q ← Q(T_θ(M_FT))
        """
        # Step 1: Persist the fused float model so gptqmodel can load it
        temp_dir = os.path.join(output_dir, "_fused_temp")
        self._temp_dir = temp_dir
        logger.info(f"Saving fused model to temporary directory: {temp_dir}")
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        del model
        clear_memory()

        # Step 2: Load with GPTQ quantization config
        quantize_config = QuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=False,  # no activation reordering; standard W4 setting
        )
        logger.info(
            f"Loading fused model for GPTQ "
            f"(bits={self.config.bits}, group_size={self.config.group_size})..."
        )
        gptq_model = GPTQModel.from_pretrained(temp_dir, quantize_config)

        # Step 3: Format calibration examples and run GPTQ
        # auto-gptq expects a list of dicts with "input_ids" key (1D or 2D tensor)
        examples = [
            {"input_ids": batch["input_ids"].squeeze(0)}
            for batch in calib_loader
        ]
        logger.info(
            f"Running GPTQ with {len(examples)} calibration samples "
            f"(W{self.config.bits}, group_size={self.config.group_size})..."
        )
        # Step 3a: Register A4 hooks BEFORE quantization so GPTQ Hessians are
        # computed with quantized activations — matching inference conditions.
        self._register_activation_hooks(gptq_model.model)
        logger.info(
            f"Registered A{self.config.act_bits} activation hooks before GPTQ "
            f"calibration on {len(self._hooks)} layers."
        )

        gptq_model.quantize(examples)
        logger.info("GPTQ weight quantization complete.")

        # Step 5: Leave temp_dir intact — gptqmodel reads model_local_path (which
        # points here) inside save_quantized() to report pre-quantized model size.
        # Call quantizer.cleanup() after save_quantized() to remove it.
        # Note: after quantization gptqmodel offloads layers to meta device for
        # memory efficiency. Do NOT call .to(device) — load the saved model with
        # GPTQModel.from_quantized() for inference instead.
        clear_memory()

        return gptq_model

    def cleanup(self) -> None:
        """Removes the temporary fused-model directory. Call after save_quantized()."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            logger.info(f"Removed temporary directory: {self._temp_dir}")
            self._temp_dir = None

    def _register_activation_hooks(self, model: nn.Module) -> None:
        """
        Registers forward pre-hooks on all nn.Linear and QuantLinear layers to
        quantize input activations per-token to INT{act_bits} before each matmul.
        """
        act_bits = self.config.act_bits

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or "QuantLinear" in type(module).__name__:
                def hook(mod, args, _bits=act_bits):
                    x = args[0]
                    return (QuantizerWrapper._rtn_quantize_activation(x, bits=_bits),) + args[1:]

                handle = module.register_forward_pre_hook(hook)
                self._hooks.append(handle)

    @staticmethod
    def _rtn_quantize_activation(
        x: torch.Tensor,
        bits: int = 4,
    ) -> torch.Tensor:
        """
        Per-token symmetric RTN quantization for activations.

        Each token vector is quantized independently:
            s = max(|x_token|) / (2^(bits-1) - 1)
            x_q = clamp(round(x / s), -qmax, qmax) * s

        Input shape: (..., features) — typically (batch, seq_len, hidden_dim).
        Returns float-precision activations rounded to the nearest INT{bits} grid.
        """
        orig_dtype = x.dtype
        orig_shape = x.shape
        x = x.float()

        qmax = 2 ** (bits - 1) - 1  # e.g., 7 for 4-bit

        # Flatten to (tokens, features) for per-token scaling
        x_flat = x.reshape(-1, x.shape[-1])
        scale = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax

        x_q = torch.round(x_flat / scale).clamp(-qmax, qmax) * scale

        return x_q.reshape(orig_shape).to(orig_dtype)
