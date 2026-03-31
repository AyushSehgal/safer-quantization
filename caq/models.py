"""
ModelPair: manages the two reference models M_PT and M_FT.

Memory strategy for running two 7B FP16 models:
    - M_FT (fine-tuned/safe): loaded on GPU via device_map="auto"
    - M_PT (pre-trained/unsafe): loaded on CPU to conserve GPU VRAM
    - M_PT logits are computed on CPU, then transferred to GPU for loss computation
    - M_PT logits tensor (~512MB FP16 per 2048-token sample) is held only briefly

This allows running CAQ on a single A100 (80GB VRAM) for 7B models,
or on a machine with 30+ GB RAM for CPU-offloaded M_PT.
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
        Loads both models. M_FT goes to GPU (device_map="auto"),
        M_PT stays on CPU to conserve GPU VRAM.
        """
        print(f"Loading tokenizer from: {self.config.finetuned_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.finetuned_model_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading fine-tuned model (M_FT): {self.config.finetuned_model_name}")
        self.model_ft = AutoModelForCausalLM.from_pretrained(
            self.config.finetuned_model_name,
            torch_dtype=self._dtype,
            device_map="auto",
        )
        self.model_ft.eval()
        # Freeze all weights — only transformation parameters θ are learned
        for param in self.model_ft.parameters():
            param.requires_grad_(False)

        print(f"Loading pre-trained model (M_PT) on CPU: {self.config.pretrained_model_name}")
        self.model_pt = AutoModelForCausalLM.from_pretrained(
            self.config.pretrained_model_name,
            torch_dtype=self._dtype,
            device_map="cpu",
        )
        self.model_pt.eval()
        for param in self.model_pt.parameters():
            param.requires_grad_(False)

        print(
            f"Models loaded. "
            f"M_FT device_map=auto, M_PT on CPU. "
            f"Transformation params will be learned."
        )

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
        Returns: (seq_len, vocab_size) logits, moved to M_FT's device.
        """
        pt_device = next(self.model_pt.parameters()).device
        outputs = self.model_pt(input_ids=input_ids.to(pt_device))
        logits_pt = outputs.logits[0].detach()  # (seq_len, vocab)
        return logits_pt.to(self.get_ft_device())

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
