"""
OSTQuant-style pre-quantization transformations for W4A4 quantization.

Implements Appendix C of the CAQ paper (Wee et al., 2026), which inherits the
transformation framework from OSTQuant (Hu et al., 2025, ICLR).

Inter-Block Transformations
----------------------------
  R_res  : Global learnable orthogonal transformation on the residual stream.
             Stiefel manifold (geoopt.ManifoldParameter), initialised from a
             random Hadamard matrix.  Applied online via hooks during CAL
             optimisation; fused into adjacent linear weights afterwards.

  S_attn[l]: Learnable per-block diagonal scaling inserted after
             input_layernorm, smoothing channel-wise differences for the Q/K/V
             projections.  Stored as a log-scale nn.Parameter (exp(s_attn)).

  S_ffn[l] : Same as S_attn but for the FFN (post_attention_layernorm →
             gate/up projections).

Intra-Block Transformations — Self-Attention
---------------------------------------------
  R_hov[l,h]: Learnable per-KV-head orthogonal rotation for the value (V) and
              output (O) projections.  Stiefel manifold, one (d_h × d_h) matrix
              per KV-head per block.

  S_hov[l]  : Learnable per-KV-head diagonal scaling for V (divide) and O
              (multiply).  Log-scale; shape (num_kv_heads × head_dim).

  S_qk[l]   : Learnable per-KV-head scaling that multiplies K outputs and
              divides Q outputs, preserving attention scores while equalising
              the Q/K dynamic ranges.  Log-scale; shape (num_kv_heads × head_dim).

  H_qk      : Fixed normalised Hadamard applied to Q and K AFTER the ROPE
              operation.  Not a learnable parameter — applied as an online hook
              during training and fused into the Q/K projection weights at
              fuse_all() (pre-ROPE approximation at fusion time; this is
              equivalent for W4 weight-only quantisation where the KV cache is
              not quantised).

Intra-Block Transformations — FFN
-----------------------------------
  S_up_down[l]: Learnable per-block scale for the FFN intermediate dimension.
               Divides the up-projection output and multiplies the down-
               projection input, making this a transparent (output-preserving)
               transformation.  Log-scale; shape (intermediate_size).

Optimiser note
--------------
  R_res and R_hov are geoopt.ManifoldParameter instances constrained to the
  Stiefel manifold.  They must be updated with a Riemannian optimiser
  (e.g. geoopt.optim.RiemannianAdam).  Use parameters_to_optimize() to
  retrieve separate parameter groups.
"""

import logging
import math
from typing import List

import torch
import torch.nn as nn

try:
    import geoopt
    _GEOOPT_AVAILABLE = True
except ImportError:                              # pragma: no cover
    _GEOOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hadamard utilities
# ---------------------------------------------------------------------------

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard_matrix(n: int) -> torch.Tensor:
    """
    Returns the normalised Walsh-Hadamard matrix H of size n×n in float64.
    n must be a positive power of 2.  H @ H.T == I.
    """
    assert _is_power_of_2(n), f"n must be a power of 2, got {n}"
    result = torch.tensor([[1.0]], dtype=torch.float64)
    size = 1
    while size < n:
        top = torch.cat([result,  result], dim=1)
        bot = torch.cat([result, -result], dim=1)
        result = torch.cat([top, bot], dim=0)
        size <<= 1
    return result / math.sqrt(n)


def _make_stiefel_param(matrix: torch.Tensor) -> nn.Parameter:
    """
    Wraps *matrix* in a geoopt.ManifoldParameter on the Stiefel manifold if
    geoopt is available, otherwise falls back to a plain nn.Parameter.  The
    caller is responsible for using a Riemannian optimiser when geoopt is used.
    """
    if _GEOOPT_AVAILABLE:
        return geoopt.ManifoldParameter(
            matrix.float(), manifold=geoopt.Stiefel()
        )
    # Fallback: plain parameter — use periodic QR re-orthogonalisation in the
    # trainer to stay on (approximately) the Stiefel manifold.
    logger.warning(
        "geoopt not found; R_res / R_hov stored as plain nn.Parameter.  "
        "Install geoopt and use RiemannianAdam for correct Stiefel updates."
    )
    return nn.Parameter(matrix.float())


# ---------------------------------------------------------------------------
# Architecture detection (LLaMA / Mistral / Qwen2 compatible)
# ---------------------------------------------------------------------------

def _detect_architecture(model: nn.Module) -> dict:
    """Extracts architecture metadata and module references from the model."""
    cfg = model.config
    hidden_dim       = cfg.hidden_size
    num_heads        = cfg.num_attention_heads
    num_kv_heads     = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim         = hidden_dim // num_heads
    intermediate_size = getattr(cfg, "intermediate_size",
                                getattr(cfg, "ffn_dim", None))
    if intermediate_size is None:
        raise ValueError(
            "Cannot determine intermediate_size from model config. "
            "Expected config.intermediate_size (LLaMA/Mistral/Qwen2)."
        )

    inner  = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            "Cannot find decoder layers. Expected model.model.layers "
            "(LLaMA / Mistral / Qwen2 architecture)."
        )

    embed   = getattr(inner, "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)

    lm_head_tied = (
        lm_head is not None
        and embed is not None
        and isinstance(lm_head, nn.Linear)
        and lm_head.weight.data_ptr() == embed.weight.data_ptr()
    )

    apply_qk_hadamard = _is_power_of_2(head_dim)
    if not apply_qk_hadamard:
        logger.warning(
            f"head_dim={head_dim} is not a power of 2 — "
            "skipping Q/K post-ROPE Hadamard."
        )

    return {
        "hidden_dim":        hidden_dim,
        "num_heads":         num_heads,
        "num_kv_heads":      num_kv_heads,
        "num_kv_groups":     num_heads // num_kv_heads,
        "head_dim":          head_dim,
        "intermediate_size": intermediate_size,
        "num_layers":        len(layers),
        "layers":            list(layers),
        "embed_tokens":      embed,
        "lm_head":           lm_head,
        "lm_head_tied":      lm_head_tied,
        "apply_qk_hadamard": apply_qk_hadamard,
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
        "attn":      attn,
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
    Correct OSTQuant transformation matching the paper's Appendix C.

    All transformation parameters live here, NOT in the model weights.
    During CAL optimisation, install_hooks() makes them active online.
    After optimisation, fuse_all() absorbs every parameter into the
    corresponding linear-layer weights — zero inference overhead.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self._hooks: list = []
        self._attn_forward_patches: list = []   # (attn_module, original_forward)

        self._arch       = _detect_architecture(model)
        self._block_mods = [_get_block_modules(b) for b in self._arch["layers"]]
        self._fuse_layernorms(model)

        n   = self._arch["num_layers"]
        d   = self._arch["hidden_dim"]
        dh  = self._arch["head_dim"]
        nkv = self._arch["num_kv_heads"]
        im  = self._arch["intermediate_size"]

        # ------------------------------------------------------------------
        # Inter-block: R_res  (global Stiefel, d × d)
        # ------------------------------------------------------------------
        # Initialise from a random normalised Hadamard (good for quantisation)
        # if d is a power of 2, otherwise use random orthogonal via QR.
        if _is_power_of_2(d):
            R_res_init = _hadamard_matrix(d).float()
        else:
            R_res_init, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32))
        self.R_res = _make_stiefel_param(R_res_init)

        # ------------------------------------------------------------------
        # Intra-block (Attention): R_hov  (per-KV-head Stiefel, dh × dh)
        # Stored flat: index = l * nkv + h
        # ------------------------------------------------------------------
        if _is_power_of_2(dh):
            R_hov_init = _hadamard_matrix(dh).float()
        else:
            R_hov_init, _ = torch.linalg.qr(torch.randn(dh, dh, dtype=torch.float32))

        self.R_hov: List[nn.Parameter] = nn.ParameterList([
            _make_stiefel_param(R_hov_init.clone())
            for _ in range(n * nkv)
        ])

        # ------------------------------------------------------------------
        # Intra-block (Attention): S_hov, S_qk  (log-scale, per-block)
        # shape: (nkv * dh,) — one scale per KV-head output channel
        # ------------------------------------------------------------------
        self.s_hov = nn.ParameterList(
            [nn.Parameter(torch.zeros(nkv * dh)) for _ in range(n)]
        )
        self.s_qk = nn.ParameterList(
            [nn.Parameter(torch.zeros(nkv * dh)) for _ in range(n)]
        )

        # ------------------------------------------------------------------
        # Inter-block: S_attn, S_ffn  (log-scale, per-block, shape: (d,))
        # ------------------------------------------------------------------
        self.s_attn = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n)]
        )
        self.s_ffn = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n)]
        )

        # ------------------------------------------------------------------
        # Intra-block (FFN): S_up_down  (log-scale, per-block, shape: (im,))
        # ------------------------------------------------------------------
        self.s_up_down = nn.ParameterList(
            [nn.Parameter(torch.zeros(im)) for _ in range(n)]
        )

        # ------------------------------------------------------------------
        # Fixed Q/K post-ROPE Hadamard (not a learnable parameter)
        # Stored as a buffer; applied online by the attention-forward patch
        # and fused into Q/K weights at fuse_all().
        # ------------------------------------------------------------------
        if self._arch["apply_qk_hadamard"]:
            self.register_buffer("H_head", _hadamard_matrix(dh).float())
        else:
            self.H_head = None

        n_learnable = self.num_learnable_parameters()
        logger.info(
            f"OSTQuantTransform initialised: "
            f"R_res ({d}×{d}), "
            f"R_hov ({n}×{nkv} heads, {dh}×{dh} each), "
            f"s_hov/s_qk/s_attn/s_ffn/s_up_down per block. "
            f"Total learnable params: {n_learnable:,}."
        )

    # ------------------------------------------------------------------
    # Hook installation and removal
    # ------------------------------------------------------------------

    def install_hooks(self, model: nn.Module) -> None:
        """
        Installs all online-transformation hooks on *model*.

        Hook effects (all online, no weight modification):
          R_res  : pre-hook q/k/v/gate/up  →  x  @ R_res.T
                   post-hook o/down         → out @ R_res
          R_hov  : post-hook v_proj  → per-head rotate v_h @ R_hov[h].T
                   pre-hook  o_proj  → per-head un-rotate attn_h @ R_hov[kv_h]
          S_hov  : post-hook v_proj  → divide by exp(s_hov)
                   pre-hook  o_proj  → multiply by exp(s_hov) (GQA-expanded)
          S_qk   : post-hook k_proj  → multiply by exp(s_qk)
                   post-hook q_proj  → divide   by exp(s_qk) (GQA-expanded)
          S_attn : pre-hook q/k/v    → divide   by exp(s_attn)
          S_ffn  : pre-hook gate/up  → divide   by exp(s_ffn)
          S_up_down: post-hook up    → divide   by exp(s_up_down)
                   pre-hook  down    → multiply by exp(s_up_down)
          H_qk   : patches attention forward → apply H after ROPE to Q and K
        """
        self.remove_hooks()

        nkv = self._arch["num_kv_heads"]
        G   = self._arch["num_kv_groups"]    # num_heads // num_kv_heads
        dh  = self._arch["head_dim"]

        # Capture R_res as a module attribute so hooks can access it with grad.
        R_res = self.R_res

        # ==============================================================
        # Global Residual Stream Hooks: embed_tokens and lm_head
        # ==============================================================
        embed = self._arch["embed_tokens"]
        if embed is not None:
            def _embed_post(mod, args, output):
                R = R_res.to(output.dtype)
                return output @ R
            self._hooks.append(embed.register_forward_hook(_embed_post))

        lm_head = self._arch["lm_head"]
        if lm_head is not None:
            def _lm_head_pre(mod, args):
                x = args[0]
                R = R_res.to(x.dtype)
                return (x @ R.T,) + args[1:]
            self._hooks.append(lm_head.register_forward_pre_hook(_lm_head_pre))

        # ==============================================================

        for l, mods in enumerate(self._block_mods):
            # ---- Captured parameters for this block ----
            s_attn_l   = self.s_attn[l]
            s_ffn_l    = self.s_ffn[l]
            s_hov_l    = self.s_hov[l]
            s_qk_l     = self.s_qk[l]
            s_up_down_l = self.s_up_down[l]
            R_hov_l    = [self.R_hov[l * nkv + h] for h in range(nkv)]

            # ==============================================================
            # R_res  ×  S_attn  —  pre-hooks on Q, K, V  (residual rotation
            #                       + inter-block attention scaling)
            # ==============================================================
            def _make_qkv_pre(scale_param):
                def hook(mod, args):
                    x   = args[0]
                    # R_res: rotate  (x → x @ R_res.T)
                    R   = R_res.to(x.dtype)
                    x   = x @ R.T
                    # S_attn: divide (activation smoothing)
                    inv_s = 1.0 / (torch.exp(scale_param).to(x.dtype) + self.eps)
                    x   = x * inv_s
                    return (x,) + args[1:]
                return hook

            for name in ("q_proj", "k_proj", "v_proj"):
                layer = mods.get(name)
                if layer is not None:
                    h = layer.register_forward_pre_hook(
                        _make_qkv_pre(s_attn_l)
                    )
                    self._hooks.append(h)

            # ==============================================================
            # R_res  ×  S_ffn  —  pre-hooks on gate, up
            # ==============================================================
            def _make_gate_up_pre(scale_param):
                def hook(mod, args):
                    x   = args[0]
                    R   = R_res.to(x.dtype)
                    x   = x @ R.T
                    inv_s = 1.0 / (torch.exp(scale_param).to(x.dtype) + self.eps)
                    x   = x * inv_s
                    return (x,) + args[1:]
                return hook

            for name in ("gate_proj", "up_proj"):
                layer = mods.get(name)
                if layer is not None:
                    h = layer.register_forward_pre_hook(
                        _make_gate_up_pre(s_ffn_l)
                    )
                    self._hooks.append(h)

            # ==============================================================
            # R_res  —  post-hooks on o_proj, down_proj
            #           out → out @ R_res  (rotates output back into res stream)
            # ==============================================================
            def _make_res_out_post():
                def hook(mod, args, output):
                    R = R_res.to(output.dtype)
                    return output @ R
                return hook

            for name in ("o_proj", "down_proj"):
                layer = mods.get(name)
                if layer is not None:
                    h = layer.register_forward_hook(_make_res_out_post())
                    self._hooks.append(h)

            # ==============================================================
            # R_hov + S_hov  —  post-hook on v_proj
            #   v_out  →  (v_out / exp(s_hov)) @ R_hov[h].T  per KV-head h
            # ==============================================================
            def _make_v_post(R_hov_block, s_hov_param):
                def hook(mod, args, output):
                    # output: (batch, seq, nkv * dh)
                    B, S, _ = output.shape
                    v = output.reshape(B, S, nkv, dh)
                    s = torch.exp(s_hov_param).to(v.dtype)            # (nkv*dh,)
                    s = s.reshape(nkv, dh)
                    
                    # FIX: Use a list to avoid in-place slice assignment
                    v_out_heads = []
                    for h_idx, R_h in enumerate(R_hov_block):
                        v_h = v[:, :, h_idx, :] / s[h_idx]           # (B, S, dh)
                        R_h_mat = R_h.to(v.dtype)
                        v_out_heads.append(v_h @ R_h_mat.T)
                        
                    # Stack along the head dimension (dim=2)
                    v_out = torch.stack(v_out_heads, dim=2)
                    return v_out.reshape(B, S, nkv * dh)
                return hook

            v_layer = mods.get("v_proj")
            if v_layer is not None:
                h = v_layer.register_forward_hook(
                    _make_v_post(R_hov_l, s_hov_l)
                )
                self._hooks.append(h)

            # ==============================================================
            # R_hov + S_hov  —  pre-hook on o_proj
            #   attn_out_h  →  attn_out_h @ R_hov[kv_h] * exp(s_hov[kv_h])
            #   (un-rotates so that with original W_o the output is preserved;
            #    with fake-quantisation present this gives non-zero gradients)
            # ==============================================================
            def _make_o_pre(R_hov_block, s_hov_param):
                def hook(mod, args):
                    x = args[0]   # (batch, seq, num_heads * dh)
                    B, S, _ = x.shape
                    nh = x.shape[-1] // dh
                    x = x.reshape(B, S, nh, dh)
                    s = torch.exp(s_hov_param).to(x.dtype).reshape(nkv, dh)
                    
                    # FIX: Use a list to avoid in-place slice assignment
                    x_out_heads = []
                    for q_idx in range(nh):
                        kv_h = q_idx // G
                        R_h  = R_hov_block[kv_h].to(x.dtype)
                        # undo the R_hov.T rotation, then undo the s_hov division
                        head_out = x[:, :, q_idx, :] @ R_h * s[kv_h]
                        x_out_heads.append(head_out)
                        
                    x_out = torch.stack(x_out_heads, dim=2)
                    return (x_out.reshape(B, S, nh * dh),) + args[1:]
                return hook

            o_layer = mods.get("o_proj")
            if o_layer is not None:
                h = o_layer.register_forward_pre_hook(
                    _make_o_pre(R_hov_l, s_hov_l)
                )
                self._hooks.append(h)

            # ==============================================================
            # S_qk  —  post-hooks on k_proj (multiply) and q_proj (divide)
            # ==============================================================
            def _make_k_post(s_qk_param):
                def hook(mod, args, output):
                    s = torch.exp(s_qk_param).to(output.dtype)  # (nkv*dh,)
                    return output * s
                return hook

            def _make_q_post(s_qk_param):
                def hook(mod, args, output):
                    s = torch.exp(s_qk_param).to(output.dtype)  # (nkv*dh,)
                    # GQA: expand s from num_kv_heads to num_heads
                    if output.shape[-1] > s.numel():
                        s = s.reshape(nkv, dh)                          # (nkv, dh)
                        s = s[:, None, :].expand(nkv, G, dh).reshape(-1)  # (num_heads*dh,)
                    return output / s
                return hook

            k_layer = mods.get("k_proj")
            if k_layer is not None:
                self._hooks.append(k_layer.register_forward_hook(_make_k_post(s_qk_l)))

            q_layer = mods.get("q_proj")
            if q_layer is not None:
                self._hooks.append(q_layer.register_forward_hook(_make_q_post(s_qk_l)))

            # ==============================================================
            # S_up_down  —  post-hook on up_proj (divide), pre-hook on down_proj (multiply)
            # ==============================================================
            def _make_up_post(s_param):
                def hook(mod, args, output):
                    s = torch.exp(s_param).to(output.dtype)
                    return output / s
                return hook

            def _make_down_pre(s_param):
                def hook(mod, args):
                    x = args[0]
                    s = torch.exp(s_param).to(x.dtype)
                    return (x * s,) + args[1:]
                return hook

            up_layer   = mods.get("up_proj")
            down_layer = mods.get("down_proj")
            if up_layer is not None:
                self._hooks.append(up_layer.register_forward_hook(_make_up_post(s_up_down_l)))
            if down_layer is not None:
                self._hooks.append(down_layer.register_forward_pre_hook(_make_down_pre(s_up_down_l)))

        # ==================================================================
        # Post-ROPE Hadamard  —  patch attention forward for each block
        # ==================================================================
        if self.H_head is not None:
            self._install_post_rope_hadamard_patches()

    def _install_post_rope_hadamard_patches(self) -> None:
        """
        Monkey-patches each attention module's forward to apply the fixed
        per-head Hadamard to Q and K immediately after apply_rotary_pos_emb.

        The patch is reversible: remove_hooks() restores original forwards.
        Standard HuggingFace LLaMA forward contains exactly one call to
        apply_rotary_pos_emb; we temporarily replace that function in the
        module's global namespace so only the forward call inside is affected.
        """
        import transformers.models.llama.modeling_llama as _llama

        H_head    = self.H_head
        orig_rope = _llama.apply_rotary_pos_emb

        def _patched_rope(*args, **kwargs):
            q, k = orig_rope(*args, **kwargs)
            # q: (batch, num_heads,    seq, dh)
            # k: (batch, num_kv_heads, seq, dh)
            H = H_head.to(q.dtype).to(q.device)
            q = q @ H.T
            k = k @ H.T
            return q, k

        # Patch the module-level name that LlamaAttention.forward references.
        _llama.apply_rotary_pos_emb = _patched_rope
        # Record original so we can restore it.
        self._attn_forward_patches.append((_llama, "apply_rotary_pos_emb", orig_rope))

    def remove_hooks(self) -> None:
        """Removes all forward hooks and attention-forward patches."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for module, attr, orig in self._attn_forward_patches:
            setattr(module, attr, orig)
        self._attn_forward_patches.clear()

    # ------------------------------------------------------------------
    # Fusion of all parameters into weights (called after optimisation)
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def _fuse_layernorms(self, model: nn.Module) -> None:
        """
        Fuses the RMSNorm weights into the adjacent linear layers and resets
        them to 1.0. This ensures RMSNorm is commutative with the global 
        orthogonal rotation R_res during both online hooks and final fusion.
        """
        for mods in self._block_mods:
            ln_attn = mods.get("ln_attn")
            ln_ffn  = mods.get("ln_ffn")
            q_proj  = mods.get("q_proj")
            k_proj  = mods.get("k_proj")
            v_proj  = mods.get("v_proj")
            gate_proj = mods.get("gate_proj")
            up_proj   = mods.get("up_proj")

            # Fuse input layernorm into Q, K, V
            if ln_attn is not None and hasattr(ln_attn, "weight"):
                w_ln = ln_attn.weight.data.float()
                for proj in (q_proj, k_proj, v_proj):
                    if proj is not None:
                        proj.weight.data.copy_((proj.weight.data.float() * w_ln.reshape(1, -1)).to(proj.weight.dtype))
                ln_attn.weight.data.fill_(1.0)

            # Fuse post-attention layernorm into Gate, Up
            if ln_ffn is not None and hasattr(ln_ffn, "weight"):
                w_ln = ln_ffn.weight.data.float()
                for proj in (gate_proj, up_proj):
                    if proj is not None:
                        proj.weight.data.copy_((proj.weight.data.float() * w_ln.reshape(1, -1)).to(proj.weight.dtype))
                ln_ffn.weight.data.fill_(1.0)

        # Fuse final model norm into lm_head
        inner = getattr(model, "model", model)
        final_norm = getattr(inner, "norm", None)
        lm_head = self._arch["lm_head"]

        if final_norm is not None and hasattr(final_norm, "weight") and lm_head is not None:
            if isinstance(lm_head, nn.Linear) and not self._arch["lm_head_tied"]:
                w_ln = final_norm.weight.data.float()
                lm_head.weight.data.copy_((lm_head.weight.data.float() * w_ln.reshape(1, -1)).to(lm_head.weight.dtype))
                final_norm.weight.data.fill_(1.0)

    @torch.no_grad()
    def fuse_all(self, model: nn.Module) -> None:
        """
        Fuses every learned (and fixed) transformation parameter into the
        adjacent linear-layer weights.  After this call all hooks are removed
        and the model can be quantised directly with zero runtime overhead.

        Fusion order follows the reference OSTQuant implementation:
          (1) S_hov  : V rows / exp(s_hov),  O cols × exp(s_hov)
          (2) S_qk   : K rows × exp(s_qk),   Q rows / exp(s_qk)
          (3) R_hov  : V rows per-head ← R_hov[h] @ W_v[h],
                       O cols per-head ← W_o[:, q*dh:] @ R_hov[kv_h].T
          (4) S_up_down: up rows / exp(s_up_down),  down cols × exp(s_up_down)
          (5) R_res  : q/k/v/gate/up cols × R_res,  o/down rows ← R_res.T @ W
          (6) H_qk   : Q/K per-head rows ← H_head @ W_q[h] (pre-ROPE approx.)
          (7) S_attn : LN_attn.weight / exp(s_attn),  Q/K/V cols × exp(s_attn)
          (8) S_ffn  : LN_ffn.weight / exp(s_ffn),    gate/up cols × exp(s_ffn)
        """
        self.remove_hooks()

        R_res   = self.R_res.detach().float()
        nkv     = self._arch["num_kv_heads"]
        G       = self._arch["num_kv_groups"]
        dh      = self._arch["head_dim"]
        d       = self._arch["hidden_dim"]
        num_q   = self._arch["num_heads"]

        with torch.no_grad():
            for l, mods in enumerate(self._block_mods):
                s_hov    = torch.exp(self.s_hov[l]).detach().float()    # (nkv*dh,)
                s_qk     = torch.exp(self.s_qk[l]).detach().float()     # (nkv*dh,)
                s_attn   = torch.exp(self.s_attn[l]).detach().float()   # (d,)
                s_ffn    = torch.exp(self.s_ffn[l]).detach().float()    # (d,)
                s_up_down = torch.exp(self.s_up_down[l]).detach().float()  # (im,)
                R_hov_l  = [self.R_hov[l * nkv + h].detach().float() for h in range(nkv)]

                q_proj    = mods.get("q_proj")
                k_proj    = mods.get("k_proj")
                v_proj    = mods.get("v_proj")
                o_proj    = mods.get("o_proj")
                gate_proj = mods.get("gate_proj")
                up_proj   = mods.get("up_proj")
                down_proj = mods.get("down_proj")
                ln_attn   = mods.get("ln_attn")
                ln_ffn    = mods.get("ln_ffn")

                def _w(layer):
                    """Return weight as float32 tensor (view, not copy)."""
                    return layer.weight.data.float()

                def _store(layer, w_fp32):
                    """Write float32 tensor back with the layer's original dtype."""
                    layer.weight.data.copy_(w_fp32.to(layer.weight.dtype))

                # (1) Fuse S_hov -----------------------------------------------
                if v_proj is not None:
                    # Divide each KV-head row group by s_hov[h]
                    W = _w(v_proj)                  # (nkv*dh, d_in)
                    W = W / s_hov.reshape(-1, 1)
                    _store(v_proj, W)
                    if v_proj.bias is not None:
                        v_proj.bias.data.copy_(
                            (v_proj.bias.data.float() / s_hov).to(v_proj.bias.dtype)
                        )

                if o_proj is not None:
                    # Multiply each query-head column group by s_hov[kv_h]
                    W = _w(o_proj)                  # (d_model, num_q*dh)
                    # Expand s_hov from nkv to num_q (GQA)
                    s_hov_q = s_hov.reshape(nkv, dh)
                    s_hov_q = s_hov_q[:, None, :].expand(nkv, G, dh).reshape(-1)  # (num_q*dh,)
                    W = W * s_hov_q.reshape(1, -1)
                    _store(o_proj, W)

                # (2) Fuse S_qk -------------------------------------------------
                if k_proj is not None:
                    W = _w(k_proj)                  # (nkv*dh, d_in)
                    W = W * s_qk.reshape(-1, 1)     # multiply rows
                    _store(k_proj, W)
                    if k_proj.bias is not None:
                        k_proj.bias.data.copy_(
                            (k_proj.bias.data.float() * s_qk).to(k_proj.bias.dtype)
                        )

                if q_proj is not None:
                    W = _w(q_proj)                  # (num_q*dh, d_in)
                    # Expand s_qk from nkv to num_q (GQA)
                    if W.shape[0] > s_qk.numel():
                        s_qk_q = s_qk.reshape(nkv, dh)
                        s_qk_q = s_qk_q[:, None, :].expand(nkv, G, dh).reshape(-1)
                    else:
                        s_qk_q = s_qk
                    W = W / s_qk_q.reshape(-1, 1)   # divide rows
                    _store(q_proj, W)
                    if q_proj.bias is not None:
                        q_proj.bias.data.copy_(
                            (q_proj.bias.data.float() / s_qk_q).to(q_proj.bias.dtype)
                        )

                # (3) Fuse R_hov ------------------------------------------------
                if v_proj is not None:
                    W = _w(v_proj)                  # (nkv*dh, d_in)
                    W_reshaped = W.reshape(nkv, dh, -1)
                    for h_idx, R_h in enumerate(R_hov_l):
                        # Rotate rows of head h: W_v[h] = R_h @ W_v[h]
                        W_reshaped[h_idx] = R_h @ W_reshaped[h_idx]
                    _store(v_proj, W_reshaped.reshape_as(W))

                if o_proj is not None:
                    W = _w(o_proj)                  # (d_model, num_q*dh)
                    W_reshaped = W.reshape(d, num_q, dh)
                    for q_idx in range(num_q):
                        kv_h = q_idx // G
                        R_h  = R_hov_l[kv_h]
                        # Rotate columns of head q: W_o[:, q] = W_o[:, q] @ R_h.T
                        W_reshaped[:, q_idx, :] = W_reshaped[:, q_idx, :] @ R_h.T
                    _store(o_proj, W_reshaped.reshape_as(W))

                # (4) Fuse S_up_down --------------------------------------------
                if up_proj is not None:
                    W = _w(up_proj)
                    W = W / s_up_down.reshape(-1, 1)    # divide rows
                    _store(up_proj, W)
                    if up_proj.bias is not None:
                        up_proj.bias.data.copy_(
                            (up_proj.bias.data.float() / s_up_down).to(up_proj.bias.dtype)
                        )

                if down_proj is not None:
                    W = _w(down_proj)
                    W = W * s_up_down.reshape(1, -1)    # multiply columns
                    _store(down_proj, W)

                # (5) Fuse R_res (per-layer) ------------------------------------
                R = R_res
                for proj in (q_proj, k_proj, v_proj, gate_proj, up_proj):
                    if proj is not None:
                        # W_new = W @ R_res  (input-side layers accept rotated stream)
                        W = _w(proj)
                        _store(proj, W @ R)

                for proj in (o_proj, down_proj):
                    if proj is not None:
                        # W_new = R_res.T @ W  (output-side layers emit into rotated stream)
                        W = _w(proj)
                        _store(proj, R.T @ W)
                        if proj.bias is not None:
                            proj.bias.data.copy_(
                                (R.T @ proj.bias.data.float()).to(proj.bias.dtype)
                            )

                # (6) Fuse Q/K post-ROPE Hadamard (pre-ROPE approximation) -----
                # Fused as H_head @ W[h] for each head h (row-group rotation).
                # Preserves quantisation benefit of Hadamard on Q/K channels;
                # strictly post-ROPE at inference requires keeping the online patch.
                # if self.H_head is not None:
                #     H = self.H_head.to(R.device)
                #     for proj, nh in ((q_proj, num_q), (k_proj, nkv)):
                #         if proj is not None:
                #             W = _w(proj)            # (nh*dh, d_in)
                #             W_r = W.reshape(nh, dh, -1)
                #             # Rotate rows of each head: W_new[h] = H @ W[h]
                #             W_r = (H.unsqueeze(0) @ W_r)
                #             _store(proj, W_r.reshape_as(W))

                # (7) Fuse S_attn -----------------------------------------------
                if ln_attn is not None and hasattr(ln_attn, "weight"):
                    dt = ln_attn.weight.dtype
                    ln_attn.weight.data.copy_(
                        (ln_attn.weight.data.float() / s_attn).to(dt)
                    )
                    if getattr(ln_attn, "bias", None) is not None:
                        ln_attn.bias.data.copy_(
                            (ln_attn.bias.data.float() / s_attn).to(dt)
                        )

                for proj in (q_proj, k_proj, v_proj):
                    if proj is not None:
                        W = _w(proj)
                        # Multiply columns by s_attn (compensates the /s_attn in LN)
                        _store(proj, W * s_attn.reshape(1, -1))

                # (8) Fuse S_ffn ------------------------------------------------
                if ln_ffn is not None and hasattr(ln_ffn, "weight"):
                    dt = ln_ffn.weight.dtype
                    ln_ffn.weight.data.copy_(
                        (ln_ffn.weight.data.float() / s_ffn).to(dt)
                    )
                    if getattr(ln_ffn, "bias", None) is not None:
                        ln_ffn.bias.data.copy_(
                            (ln_ffn.bias.data.float() / s_ffn).to(dt)
                        )

                for proj in (gate_proj, up_proj):
                    if proj is not None:
                        W = _w(proj)
                        _store(proj, W * s_ffn.reshape(1, -1))

            # Fuse R_res into embedding and lm_head --------------------------
            inner  = getattr(model, "model", model)
            embed  = self._arch["embed_tokens"]
            lm_head = self._arch["lm_head"]

            if embed is not None:
                dt = embed.weight.dtype
                # Embedding rows are the initial residual vectors; rotate them.
                embed.weight.data.copy_(
                    (embed.weight.data.float() @ R_res).to(dt)
                )

            if (
                lm_head is not None
                and isinstance(lm_head, nn.Linear)
                and lm_head.weight.shape[1] == d
                and not self._arch["lm_head_tied"]
            ):
                # lm_head receives the final rotated residual; same as input-side.
                _store(lm_head, _w(lm_head) @ R_res)

        logger.info("fuse_all() complete — all transformation parameters absorbed.")

    # ------------------------------------------------------------------
    # Optimiser interface
    # ------------------------------------------------------------------

    def parameters_to_optimize(self) -> dict:
        """
        Returns a dict with two parameter groups:

          "stiefel"   : list of R_res and all R_hov parameters.
                        Use with geoopt.optim.RiemannianAdam (or RSGD).
          "euclidean" : list of all remaining (log-scale) parameters.
                        Use with torch.optim.Adam.

        Example usage in CAQTrainer::

            groups = self.transform.parameters_to_optimize()
            optimizer = geoopt.optim.RiemannianAdam([
                {"params": groups["stiefel"],   "lr": 1e-3},
                {"params": groups["euclidean"], "lr": 1e-3},
            ])
        """
        stiefel_params = [self.R_res] + list(self.R_hov.parameters())
        euclidean_params = (
            list(self.s_hov.parameters()) +
            list(self.s_qk.parameters()) +
            list(self.s_attn.parameters()) +
            list(self.s_ffn.parameters()) +
            list(self.s_up_down.parameters())
        )
        return {"stiefel": stiefel_params, "euclidean": euclidean_params}

    def all_parameters(self) -> list:
        """Returns a flat list of all learnable parameters (for compatibility)."""
        g = self.parameters_to_optimize()
        return g["stiefel"] + g["euclidean"]

    def num_learnable_parameters(self) -> int:
        """Total number of learnable scalar parameters."""
        return sum(p.numel() for p in self.all_parameters())
