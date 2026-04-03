"""
ModelPair: manages the two reference models M_PT and M_FT.

Memory strategy for running two 7B FP16 models on a single GPU:
    - M_PT (pre-trained/unsafe): loaded on GPU first, logits pre-computed and
      cached to disk, then deleted before M_FT is loaded.
    - M_FT (fine-tuned/safe): loaded on GPU after M_PT is deleted.

This ensures only one 7B model occupies GPU VRAM at any time (~14 GB for 7B
float16), leaving the remaining headroom entirely for activations and the
learned transformation parameters. The two models are never in GPU memory
simultaneously.
"""

import gc
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CAQConfig
from .transformation import SmoothScaleTransform


class ModelPair:
    """
    Loads and manages M_PT (pre-trained, unsafe) and M_FT (fine-tuned, safe).

    Provides forward passes for both models with appropriate memory management.
    After CAL optimization, fuse_and_release_pt() fuses the transformation
    into M_FT's weights and frees M_PT from memory.
    """

    def __init__(self, config: CAQConfig):
        self.config = config
        self.model_ft: Optional[nn.Module] = None
        self.model_pt: Optional[nn.Module] = None
        self.tokenizer = None
        self._dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16

    def load(self) -> None:
        """
        Loads both models sequentially to avoid putting two 7B models on GPU at once.

        M_PT is loaded on GPU first so its logits can be pre-computed and cached.
        The caller (CAQTrainer._precompute_pt_logits) is responsible for deleting
        M_PT after caching. load_ft() must then be called to bring M_FT onto GPU.
        """
        print(f"Loading tokenizer from: {self.config.finetuned_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.finetuned_model_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading pre-trained model (M_PT) on GPU: {self.config.pretrained_model_name}")
        self.model_pt = AutoModelForCausalLM.from_pretrained(
            self.config.pretrained_model_name,
            torch_dtype=self._dtype,
            device_map="auto",
        )
        self.model_pt.eval()
        for param in self.model_pt.parameters():
            param.requires_grad_(False)

    def load_ft(self) -> None:
        """
        Loads M_FT onto GPU. Must be called after M_PT has been deleted by
        _precompute_pt_logits so the two models never share GPU memory.
        """
        print(f"Loading fine-tuned model (M_FT) on GPU: {self.config.finetuned_model_name}")
        self.model_ft = AutoModelForCausalLM.from_pretrained(
            self.config.finetuned_model_name,
            torch_dtype=self._dtype,
            device_map="auto",
        )
        self.model_ft.eval()
        # Freeze all weights — only transformation parameters θ are learned
        for param in self.model_ft.parameters():
            param.requires_grad_(False)
        print("M_FT loaded. Transformation params will be learned.")

    def get_pt_device(self) -> torch.device:
        """Returns the device of M_PT's first parameter."""
        return next(self.model_pt.parameters()).device

    def get_ft_device(self) -> torch.device:
        """Returns the device of M_FT's first parameter."""
        return next(self.model_ft.parameters()).device

    @torch.no_grad()
    def get_logits_ft(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through M_FT WITHOUT transformation hooks.
        Represents p_FT(y|x) — the safe fine-tuned distribution.
        Returns: (seq_len, vocab_size) logits on M_FT's device.
        """
        input_ids = input_ids.to(self.get_ft_device())
        outputs = self.model_ft(input_ids=input_ids)
        # Remove batch dim: (1, seq_len, vocab) → (seq_len, vocab)
        return outputs.logits[0].detach()

    @torch.no_grad()
    def get_logits_pt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through M_PT on whatever device it currently lives on.
        Represents p_PT(y|x) — the unsafe pre-trained distribution.
        Returns: (seq_len, vocab_size) logits on M_PT's device.
        Callers are responsible for moving to the desired device.
        """
        pt_device = next(self.model_pt.parameters()).device
        outputs = self.model_pt(input_ids=input_ids.to(pt_device))
        return outputs.logits[0].detach()  # (seq_len, vocab)

    def get_logits_transformed(
        self,
        input_ids: torch.Tensor,
        transform: SmoothScaleTransform,
    ) -> torch.Tensor:
        """
        Forward pass through M_FT WITH transformation hooks active.
        Represents p_Q(y|x) — the transformed/quantized model distribution.
        This is the ONLY forward pass that contributes to the computation graph.
        Returns: (seq_len, vocab_size) logits WITH grad.
        """
        input_ids = input_ids.to(self.get_ft_device())
        transform.install_hooks(self.model_ft)
        try:
            outputs = self.model_ft(input_ids=input_ids)
            logits_q = outputs.logits[0]  # (seq_len, vocab), retains grad
        finally:
            transform.remove_hooks()
        return logits_q

    def fuse_and_release_pt(self, transform: SmoothScaleTransform) -> None:
        """
        1. Fuses learned scaling parameters θ into M_FT weights in-place.
        2. Deletes M_PT and frees its CPU memory.
        After this call, M_FT contains the fused transformed weights, ready
        for quantization.
        """
        print("Fusing transformation parameters into M_FT weights...")
        transform.fuse_all(self.model_ft)

        if self.model_pt is not None:
            print("Releasing M_PT from memory...")
            del self.model_pt
            self.model_pt = None
            gc.collect()
            print("M_PT released.")

    def get_fused_model(self) -> nn.Module:
        """Returns M_FT (should be called after fuse_and_release_pt)."""
        return self.model_ft
