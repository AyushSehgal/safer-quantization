"""
OSTQuant-style pre-quantization transformations for W4A4 quantization.

Implements the transformation framework from:
  Hu et al. (2025) "OSTQuant: Refining large model quantization with orthogonal
  and scaling transformations for better distribution fitting." ICLR.

Used as the base framework for CAQ (Wee et al., 2026), which optimizes the
transformation parameters using the Contrastive Alignment Loss (CAL).

Transformations applied:
  1. Fixed normalized Hadamard rotation on the residual stream — fused into
     adjacent linear layer weights at __init__, zero runtime overhead.
  2. Fixed per-head Hadamard on Q/K projections — fused at __init__,
     attention scores preserved because H is orthonormal (H^T H = I).
  3. Learnable per-block diagonal scaling:
       S_attn[l]: (hidden_dim,) — shared across Q/K/V inputs for block l,
                  fused into input_layernorm.weight and Q/K/V weight columns.
       S_ffn[l]:  (hidden_dim,) — shared across gate/up inputs for block l,
                  fused into post_attention_layernorm.weight and gate/up cols.

Fusion math for scaling (no hook needed at inference):
  Hook applies s^{-1} to activations → LayerNorm output becomes x/s.
  fuse_all absorbs: LayerNorm.weight /= s  (so output is x/s automatically)
                    W_proj columns *= s     (so W_new @ (x/s) = W_old @ x)
  A4 quantization at inference operates on x/s — reduced dynamic range. ✓
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hadamard utilities
# ---------------------------------------------------------------------------

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard_matrix(n: int) -> torch.Tensor:
    """
    Returns the normalized Walsh-Hadamard matrix of size n×n in float64.
    n must be a positive power of 2.
    H @ H.T == I  (orthonormal). All entries are ±1/sqrt(n).
    """
    assert _is_power_of_2(n), f"n must be a power of 2, got {n}"
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    result = torch.tensor([[1.0]], dtype=torch.float64)
    temp = n
    while temp > 1:
        result = torch.kron(result, H if result.shape[0] == 1 else
                            torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64))
        temp >>= 1
    # Build iteratively to avoid the above confusion
    result = torch.tensor([[1.0]], dtype=torch.float64)
    size = 1
    while size < n:
        top = torch.cat([result, result], dim=1)
        bot = torch.cat([result, -result], dim=1)
        result = torch.cat([top, bot], dim=0)
        size <<= 1
    return result / math.sqrt(n)


# ---------------------------------------------------------------------------
# Architecture detection (LLaMA / Mistral / Qwen2 compatible)
# ---------------------------------------------------------------------------

def _detect_architecture(model: nn.Module) -> dict:
    """
    Extracts architecture metadata and module references from the model.
    Returns a dict with all information needed by the transformation.
    """
    cfg = model.config
    hidden_dim  = cfg.hidden_size
    num_heads   = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim    = hidden_dim // num_heads

    inner  = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            "Cannot find decoder layers. Expected model.model.layers "
            "(LLaMA / Mistral / Qwen2 architecture)."
        )

    embed   = getattr(inner, "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)

    apply_res_hadamard = _is_power_of_2(hidden_dim)
    apply_qk_hadamard  = _is_power_of_2(head_dim)

    if not apply_res_hadamard:
        logger.warning(
            f"hidden_dim={hidden_dim} is not a power of 2 — "
            "skipping residual-stream Hadamard rotation (e.g. Qwen2-7B has 3584). "
            "Learnable per-block scaling will still be applied."
        )
    if not apply_qk_hadamard:
        logger.warning(
            f"head_dim={head_dim} is not a power of 2 — "
            "skipping Q/K head Hadamard rotation."
        )

    # Check whether lm_head and embed_tokens share the same weight tensor
    lm_head_tied = (
        lm_head is not None
        and embed is not None
        and isinstance(lm_head, nn.Linear)
        and lm_head.weight.data_ptr() == embed.weight.data_ptr()
    )

    return {
        "hidden_dim":         hidden_dim,
        "num_heads":          num_heads,
        "num_kv_heads":       num_kv_heads,
        "head_dim":           head_dim,
        "num_layers":         len(layers),
        "layers":             list(layers),
        "embed_tokens":       embed,
        "lm_head":            lm_head,
        "lm_head_tied":       lm_head_tied,
        "apply_res_hadamard": apply_res_hadamard,
        "apply_qk_hadamard":  apply_qk_hadamard,
    }


def _get_block_modules(block) -> dict:
    """Extracts named submodules from a single decoder block."""
    attn = getattr(block, "self_attn", None)
    mlp  = getattr(block, "mlp", None)
    ln_attn = (
        getattr(block, "input_layernorm", None) or
        getattr(block, "ln_1", None)
    )
    ln_ffn = (
        getattr(block, "post_attention_layernorm", None) or
        getattr(block, "ln_2", None)
    )
    return {
        "ln_attn":   ln_attn,
        "ln_ffn":    ln_ffn,
        "q_proj":    getattr(attn, "q_proj",    None),
        "k_proj":    getattr(attn, "k_proj",    None),
        "v_proj":    getattr(attn, "v_proj",    None),
        "o_proj":    getattr(attn, "o_proj",    None),
        "gate_proj": getattr(mlp,  "gate_proj", None),
        "up_proj":   getattr(mlp,  "up_proj",   None),
        "down_proj": getattr(mlp,  "down_proj", None),
    }


# ---------------------------------------------------------------------------
# OSTQuantTransform
# ---------------------------------------------------------------------------

class OSTQuantTransform(nn.Module):
    """
    OSTQuant-style transformation: fixed Hadamard rotations + learnable
    per-block diagonal scaling.

    Fixed rotations (fused at __init__, permanent weight modification):
      - Residual-stream Hadamard H ∈ R^{d×d}: fused into o_proj/down_proj rows
        and q/k/v/gate/up/embed/lm_head columns.
      - Per-head Q/K Hadamard H_h ∈ R^{d_h×d_h}: fused into q_proj/k_proj row
        blocks (per attention head). Attention scores preserved (H^T H = I).

    Learnable scaling (optimized by CAQTrainer using CAL):
      - s_attn[l] ∈ R^{hidden_dim}: shared scale for Q/K/V inputs of block l.
      - s_ffn[l]  ∈ R^{hidden_dim}: shared scale for gate/up inputs of block l.

    Hooks (active only during CAL optimization, removed before quantization):
      Divides Q/K/V activations by exp(s_attn[l]) and gate/up activations by
      exp(s_ffn[l]) so the optimizer sees the quantization-transformed view.

    After optimization, fuse_all() absorbs learned scales into LayerNorm weights
    and adjacent linear columns — no hooks needed at inference.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self._hooks: list = []

        # Detect architecture once and cache
        self._arch = _detect_architecture(model)
        self._block_mods = [
            _get_block_modules(b) for b in self._arch["layers"]
        ]

        # Step 1: fuse fixed Hadamard rotations (permanent, no runtime overhead)
        self._fuse_residual_hadamard(model)
        self._fuse_qk_head_hadamard()
        logger.info(
            f"Fixed Hadamard rotations fused "
            f"(residual={self._arch['apply_res_hadamard']}, "
            f"qk_heads={self._arch['apply_qk_hadamard']})."
        )

        # Step 2: learnable per-block scaling (initialized to identity: exp(0)=1)
        n = self._arch["num_layers"]
        d = self._arch["hidden_dim"]
        self.s_attn = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n)]
        )
        self.s_ffn = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n)]
        )
        logger.info(
            f"Initialized {self.num_parameters():,} learnable scale parameters "
            f"({n} blocks × 2 × {d})."
        )

    # ------------------------------------------------------------------
    # Fixed Hadamard fusion
    # ------------------------------------------------------------------

    def _fuse_residual_hadamard(self, model: nn.Module) -> None:
        """
        Fuses the normalized Hadamard H into residual-stream linear layers.

        Layers receiving input from the residual stream (Q/K/V/gate/up/embed/lm_head):
            W_new = W_old @ H.T   — so W_new @ (H @ x) = W_old @ x ✓

        Layers whose output feeds into the residual stream (o_proj, down_proj):
            W_new = H @ W_old     — so output is now H-rotated ✓
        """
        if not self._arch["apply_res_hadamard"]:
            return

        d = self._arch["hidden_dim"]
        H_cpu = _hadamard_matrix(d)   # (d, d) float64 on CPU

        def _fuse_input_side(layer: nn.Linear) -> None:
            """W_new = W_old @ H.T (multiply columns by H.T)."""
            orig_dtype = layer.weight.dtype
            H = H_cpu.to(layer.weight.device)
            W = layer.weight.data.double()
            layer.weight.data.copy_((W @ H.T).to(orig_dtype))

        def _fuse_output_side(layer: nn.Linear) -> None:
            """W_new = H @ W_old (multiply rows by H)."""
            orig_dtype = layer.weight.dtype
            H = H_cpu.to(layer.weight.device)
            W = layer.weight.data.double()
            layer.weight.data.copy_((H @ W).to(orig_dtype))

        with torch.no_grad():
            # Embedding table
            embed = self._arch["embed_tokens"]
            if embed is not None and embed.weight.shape[1] == d:
                orig_dtype = embed.weight.dtype
                H = H_cpu.to(embed.weight.device)
                embed.weight.data.copy_(
                    (embed.weight.data.double() @ H.T).to(orig_dtype)
                )

            for mods in self._block_mods:
                for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                    layer = mods.get(name)
                    if layer is not None and isinstance(layer, nn.Linear):
                        _fuse_input_side(layer)

                for name in ("o_proj", "down_proj"):
                    layer = mods.get(name)
                    if layer is not None and isinstance(layer, nn.Linear):
                        _fuse_output_side(layer)

            # LM head (skip if weight-tied to embed — already fused above)
            lm_head = self._arch["lm_head"]
            if (
                lm_head is not None
                and isinstance(lm_head, nn.Linear)
                and lm_head.weight.shape[1] == d
                and not self._arch["lm_head_tied"]
            ):
                _fuse_input_side(lm_head)

    def _fuse_qk_head_hadamard(self) -> None:
        """
        Fuses per-head Hadamard H_h into Q and K projection weights.

        For each head h: W_q_new[h] = H_h @ W_q_old[h]
        This rotates query/key vectors into Hadamard space, distributing
        outliers uniformly across head_dim — critical for A4 of Q/K.
        Attention scores preserved: (H·q)·(H·k) = q^T H^T H k = q^T k ✓
        """
        if not self._arch["apply_qk_hadamard"]:
            return

        dh        = self._arch["head_dim"]
        num_q     = self._arch["num_heads"]
        num_kv    = self._arch["num_kv_heads"]
        H_head_cpu = _hadamard_matrix(dh)   # (dh, dh) float64 on CPU

        with torch.no_grad():
            for mods in self._block_mods:
                for proj_name, num_heads in (
                    ("q_proj", num_q),
                    ("k_proj", num_kv),
                ):
                    layer = mods.get(proj_name)
                    if layer is None or not isinstance(layer, nn.Linear):
                        continue
                    orig_dtype = layer.weight.dtype
                    out_feat, in_feat = layer.weight.shape
                    H_head = H_head_cpu.to(layer.weight.device)
                    # (num_heads, head_dim, in_features)
                    W = layer.weight.data.double().reshape(num_heads, dh, in_feat)
                    # Apply H_head to each head's rows: W_new[h] = H_head @ W[h]
                    H_exp = H_head.unsqueeze(0).expand(num_heads, -1, -1)  # (nh, dh, dh)
                    W_rot = torch.bmm(H_exp, W)   # (num_heads, dh, in_feat)
                    layer.weight.data.copy_(
                        W_rot.reshape(out_feat, in_feat).to(orig_dtype)
                    )

    # ------------------------------------------------------------------
    # Learnable scaling hooks (active during CAL optimization only)
    # ------------------------------------------------------------------

    def install_hooks(self, model: nn.Module) -> None:
        """
        Installs forward pre-hooks for the learnable per-block scaling.
        Q/K/V all share s_attn[l]; gate/up share s_ffn[l].
        Gradients flow through the division — required for CAL optimization.
        """
        self.remove_hooks()

        for l, mods in enumerate(self._block_mods):
            # Use factory functions to correctly capture l in the closure
            def _make_attn_hook(scale_param):
                def hook(mod, args):
                    x = args[0]
                    inv_s = 1.0 / (torch.exp(scale_param).to(x.dtype) + self.eps)
                    return (x * inv_s,) + args[1:]
                return hook

            def _make_ffn_hook(scale_param):
                def hook(mod, args):
                    x = args[0]
                    inv_s = 1.0 / (torch.exp(scale_param).to(x.dtype) + self.eps)
                    return (x * inv_s,) + args[1:]
                return hook

            for name in ("q_proj", "k_proj", "v_proj"):
                layer = mods.get(name)
                if layer is not None:
                    h = layer.register_forward_pre_hook(
                        _make_attn_hook(self.s_attn[l])
                    )
                    self._hooks.append(h)

            for name in ("gate_proj", "up_proj"):
                layer = mods.get(name)
                if layer is not None:
                    h = layer.register_forward_pre_hook(
                        _make_ffn_hook(self.s_ffn[l])
                    )
                    self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Removes all registered forward pre-hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Fusion of learned scales into weights (called after optimization)
    # ------------------------------------------------------------------

    def fuse_all(self, model: nn.Module) -> None:
        """
        Fuses learned scales into LayerNorm weights and linear layer columns.

        For S_attn[l] with scale s = exp(s_attn[l]):
          - input_layernorm.weight /= s   → LayerNorm output is now x/s
          - Q/K/V weight columns   *= s   → W_new @ (x/s) = W_old @ x ✓

        For S_ffn[l] with scale s = exp(s_ffn[l]):
          - post_attention_layernorm.weight /= s
          - gate/up weight columns          *= s

        After fusion: A4 quantization at inference operates on x/s (better
        dynamic range), and W4 quantizes W_new = W * diag(s) (smoother). ✓
        """
        self.remove_hooks()

        with torch.no_grad():
            for l, mods in enumerate(self._block_mods):
                s_attn = torch.exp(self.s_attn[l]).detach().float()
                s_ffn  = torch.exp(self.s_ffn[l]).detach().float()

                # --- Attention scaling ---
                ln = mods.get("ln_attn")
                if ln is not None and hasattr(ln, "weight"):
                    dt = ln.weight.dtype
                    ln.weight.data.copy_(
                        (ln.weight.data.float() / s_attn).to(dt)
                    )
                    if ln.bias is not None:
                        ln.bias.data.copy_(
                            (ln.bias.data.float() / s_attn).to(dt)
                        )

                for name in ("q_proj", "k_proj", "v_proj"):
                    layer = mods.get(name)
                    if layer is not None:
                        dt = layer.weight.dtype
                        # columns *= s: W_new[:, j] = W_old[:, j] * s[j]
                        layer.weight.data.copy_(
                            (layer.weight.data.float() * s_attn.unsqueeze(0)).to(dt)
                        )

                # --- FFN scaling ---
                ln = mods.get("ln_ffn")
                if ln is not None and hasattr(ln, "weight"):
                    dt = ln.weight.dtype
                    ln.weight.data.copy_(
                        (ln.weight.data.float() / s_ffn).to(dt)
                    )
                    if ln.bias is not None:
                        ln.bias.data.copy_(
                            (ln.bias.data.float() / s_ffn).to(dt)
                        )

                for name in ("gate_proj", "up_proj"):
                    layer = mods.get(name)
                    if layer is not None:
                        dt = layer.weight.dtype
                        layer.weight.data.copy_(
                            (layer.weight.data.float() * s_ffn.unsqueeze(0)).to(dt)
                        )

    # ------------------------------------------------------------------
    # Interface for CAQTrainer
    # ------------------------------------------------------------------

    def parameters_to_optimize(self) -> list:
        """Returns all learnable scale parameters for the Adam optimizer."""
        return list(self.s_attn.parameters()) + list(self.s_ffn.parameters())

    def num_parameters(self) -> int:
        """Total number of learnable parameters (s_attn + s_ffn across all blocks)."""
        return sum(p.numel() for p in self.parameters_to_optimize())
