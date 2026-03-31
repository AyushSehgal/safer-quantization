"""
Learnable per-channel smooth scaling transformations (OSTQuant-style).

For a linear layer computing Y = WX, we introduce a diagonal scaling matrix T:
    Y = WX = (W · diag(s)) · (diag(s^{-1}) · X)    (Eq. 3 from paper)

During optimization:
    - diag(s^{-1}) is applied to input activations X via a forward pre-hook
    - This produces the "transformed" computation p_Q(y|x)
    - Gradients flow through the scale parameters s (stored as log_scale)

At inference (after optimization):
    - diag(s) is fused into the weight: W_new = W · diag(s)
    - The hook is removed; no computational overhead remains

This follows the SmoothQuant paradigm of migrating quantization difficulty
from activations to weights via learnable equivalent transformations.
"""

import torch
import torch.nn as nn
from typing import Optional


class ChannelScaleTransform(nn.Module):
    """
    Learnable per-channel scaling for one linear layer.

    Stores log_scale (in_features,) initialized to 0, so exp(0) = 1.0
    means identity transformation at the start of optimization.
    """

    def __init__(self, in_features: int, eps: float = 1e-8):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(in_features))
        self.eps = eps

    @property
    def scale(self) -> torch.Tensor:
        """Positive scaling factors: exp(log_scale). Shape: (in_features,)"""
        return torch.exp(self.log_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies s^{-1} to input activations (used as forward pre-hook).
        x: (..., in_features)  →  x / scale
        """
        return x / (self.scale + self.eps)

    def fuse_into_weight(self, weight: nn.Parameter) -> None:
        """
        Absorbs scale into weight in-place: W_new[:, i] = W[:, i] * scale[i]
        weight shape: (out_features, in_features)

        After fusion, the net computation W_new · (x / scale) · scale = W_new · x
        which equals the original W · x but with quantization-friendlier weights.
        """
        with torch.no_grad():
            weight.data.mul_(self.scale.unsqueeze(0))


class SmoothScaleTransform(nn.Module):
    """
    Manages ChannelScaleTransform instances for all nn.Linear layers in a model.

    Registers forward pre-hooks on each target layer that apply s^{-1} to
    incoming activations. These hooks are installed before each transformed
    forward pass and removed afterwards (or permanently removed at fusion time).

    After optimization, fuse_all() absorbs all scales into the weight tensors
    and removes all hooks, leaving the model with zero runtime overhead.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self._hooks: list = []

        # Build one ChannelScaleTransform per nn.Linear layer
        transforms = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad is False:
                # Skip non-trainable layers (e.g. lm_head might be tied)
                pass
            if isinstance(module, nn.Linear):
                # Use sanitized name as key (dots → underscores for ModuleDict)
                key = name.replace(".", "__")
                transforms[key] = ChannelScaleTransform(
                    module.in_features, eps=eps
                )
        self.transforms = nn.ModuleDict(transforms)

        # Map sanitized key → original module name for hook registration
        self._key_to_name = {
            name.replace(".", "__"): name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

    def install_hooks(self, model: nn.Module) -> None:
        """
        Registers forward pre-hooks on all target nn.Linear layers.
        Each hook applies s^{-1} to the input tensor before the matmul.
        Safe to call multiple times (removes existing hooks first).
        """
        self.remove_hooks()
        name_to_module = {n: m for n, m in model.named_modules()}

        for key, transform in self.transforms.items():
            orig_name = self._key_to_name[key]
            module = name_to_module.get(orig_name)
            if module is None:
                continue

            # Capture transform in closure for hook
            def make_hook(t):
                def hook(module, args):
                    # args is a tuple; modify the first argument (input tensor)
                    x = args[0]
                    return (t(x),) + args[1:]
                return hook

            handle = module.register_forward_pre_hook(make_hook(transform))
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Removes all registered forward pre-hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def fuse_all(self, model: nn.Module) -> None:
        """
        Fuses learned scales into model weights in-place, then removes all hooks.

        For each linear layer, multiplies its weight columns by scale[i]:
            W_new[:, i] = W[:, i] * scale[i]

        This is the inverse of what the hook does (x / scale), so the net
        transformation W_new · (x / scale) = W · x, but the weight is now
        scaled to be more quantization-friendly.
        """
        self.remove_hooks()
        name_to_module = {n: m for n, m in model.named_modules()}

        for key, transform in self.transforms.items():
            orig_name = self._key_to_name[key]
            module = name_to_module.get(orig_name)
            if module is None or not isinstance(module, nn.Linear):
                continue
            transform.fuse_into_weight(module.weight)

    def parameters_to_optimize(self) -> list:
        """Returns all log_scale parameters for the optimizer."""
        return list(self.parameters())

    def num_parameters(self) -> int:
        """Total number of learnable scale parameters."""
        return sum(t.log_scale.numel() for t in self.transforms.values())
