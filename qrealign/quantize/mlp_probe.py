import torch
import torch.nn as nn


class RefusalMLP(nn.Module):
    """Two-layer MLP probe that learns a nonlinear refusal boundary.

    Replaces the sparse logistic regression (linear) probe with a small
    nonlinear classifier.  Input is the last-token hidden state for a
    layer; output is a scalar logit (positive → harmful, negative → benign).
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # (B, d_model) → (B,)
        return self.net(x).squeeze(-1)


def load_mlp_probes(path: str, dtype: torch.dtype, device):
    """Load per-layer MLP probes saved by train_mlp_probe.py.

    Returns dict[layer_idx → RefusalMLP] with frozen parameters.
    """
    raw = torch.load(path, map_location=device)
    probes = {}
    for layer_idx, entry in raw.items():
        mlp = RefusalMLP(entry["d_model"], entry["hidden"])
        mlp.load_state_dict(entry["state_dict"])
        mlp = mlp.to(dtype=dtype, device=device)
        mlp.eval()
        for p in mlp.parameters():
            p.requires_grad = False
        probes[layer_idx] = mlp
    return probes
