#!/usr/bin/env python3
"""
Visualize per-layer refusal directions stored in refusal_direction/refusal_dirs/*.pt

Each .pt is a dict {layer_idx: unit-normed tensor(d_model)} produced by
extract_refusal_directions.py (difference-in-means, already normalized).

Per model (mirrors phase1_extract_direction.ipynb strategies):
  - Cross-layer cosine-similarity heatmap
  - Per-layer cosine sim with the "most central" direction
  - Per-layer direction alignment score (avg cosine to all other layers)

Cross-model:
  - Cosine similarity at each relative layer depth
  - Bar chart of which relative depth is most self-consistent per model

Output: refusal_directions_output/
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR   = Path(__file__).parent
REFUSAL_DIRS = SCRIPT_DIR / "refusal_direction" / "refusal_dirs"
OUT_DIR      = SCRIPT_DIR / "refusal_directions_output"

_XLABEL_REL_DEPTH = "Relative Layer Depth"
_YLABEL_COS_SIM   = "Cosine Similarity"
_LABEL_COS_07     = "cos=0.7"


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_refusal_dirs(path: Path) -> torch.Tensor:
    """Load {layer_idx: tensor(d_model)} → stacked Tensor [n_layers, d_model]."""
    raw = torch.load(path, map_location="cpu")
    n_layers = max(raw.keys()) + 1
    d_model  = next(iter(raw.values())).shape[0]
    out = torch.zeros(n_layers, d_model)
    for idx, vec in raw.items():
        out[idx] = vec.float()
    # Re-normalise in case of any fp precision drift
    norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return out / norms


# ── Per-model plots ─────────────────────────────────────────────────────────────

def plot_model(name: str, dirs: torch.Tensor, out_dir: Path):
    """
    Three figures for one model:
      1. Cross-layer cosine similarity heatmap
      2. Per-layer alignment score (mean cosine to all other layers)
         + cosine sim with the most-central layer
      3. Block-structure: cosine sim between consecutive layers
    """
    n_layers = dirs.shape[0]
    layers   = np.arange(n_layers)

    # Pairwise cosine similarity [n_layers, n_layers]
    cos_mat = torch.mm(dirs, dirs.T).clamp(-1, 1).numpy()

    # Alignment score: mean cosine to all OTHER layers
    align = (cos_mat.sum(axis=1) - 1.0) / max(n_layers - 1, 1)  # exclude self

    # Most-central layer: highest alignment
    central_layer = int(np.argmax(align))
    cos_with_central = cos_mat[central_layer]

    # Consecutive-layer cosine similarity (how much direction drifts layer-to-layer)
    consec_cos = np.array([
        cos_mat[i, i + 1] for i in range(n_layers - 1)
    ])

    # ── Figure 1: heatmap ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", origin="upper")
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title(f"{name}\nCross-Layer Direction Cosine Similarity\n"
                 "Warm = directions agree; cool = directions oppose")
    ax.plot(central_layer, central_layer, "r*", markersize=14, zorder=5,
            label=f"Most central layer ({central_layer})")
    ax.legend(fontsize=9, loc="upper right")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_cos_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: alignment score + cosine with central layer ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} — Per-Layer Direction Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(layers, align, color="#2196F3", lw=2)
    ax.axvline(central_layer, color="red", ls="--", lw=1.5,
               label=f"Most central ({central_layer})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity to All Other Layers")
    ax.set_title("Per-Layer Alignment Score\n"
                 "(higher → this layer's direction is representative\n"
                 " of the overall refusal subspace)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, cos_with_central, color="#FF9800", lw=2)
    ax.axvline(central_layer, color="red", ls="--", lw=1.5,
               label=f"Reference layer ({central_layer})")
    ax.axhline(0.7, color="green", ls=":", alpha=0.5, label="cos = 0.7")
    ax.axhline(0,   color="gray",  ls=":", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel(_YLABEL_COS_SIM)
    ax.set_title(f"Cosine Sim with Most-Central Direction (Layer {central_layer})\n"
                 "(mirrors notebook: cosine sim with reference direction per layer)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_direction_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: layer-to-layer drift ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    layer_mids = layers[:-1] + 0.5
    ax.bar(layer_mids, consec_cos, width=0.8, color="#43A047", edgecolor="none", alpha=0.8)
    ax.axhline(0.7, color="red", ls="--", lw=1.5, label="cos = 0.7")
    ax.set_xlabel("Layer transition  (i → i+1)")
    ax.set_ylabel(_YLABEL_COS_SIM)
    ax.set_title(f"{name} — Layer-to-Layer Direction Drift\n"
                 "Low values = refusal direction changes abruptly between these layers")
    ax.set_ylim(min(-0.1, consec_cos.min() - 0.05), 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_layer_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [plots] 3 figures for {name}  (most central layer: {central_layer})")
    return central_layer, align


# ── Cross-model comparison ──────────────────────────────────────────────────────

def plot_cross_model(all_data: dict, plots_dir: Path):
    if len(all_data) < 2:
        return

    model_names = list(all_data.keys())
    cmap   = plt.cm.tab10
    colors = [cmap(i / 10) for i in range(len(model_names))]
    short  = [n.replace("meta-llama-", "llama-").replace("-hf", "")
                .replace("-instruct", "") for n in model_names]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Cross-Model Refusal Direction Comparison", fontsize=14, fontweight="bold")

    ax_align   = fig.add_subplot(gs[0, 0])
    ax_cos_ref = fig.add_subplot(gs[0, 1])
    ax_depth   = fig.add_subplot(gs[1, 0])
    ax_cross   = fig.add_subplot(gs[1, 1])

    # Panel 1 — alignment score profile (normalised relative depth)
    ax_align.set_title("Per-Layer Alignment Score\nvs. Relative Layer Depth")
    for (name, data), color, sname in zip(all_data.items(), colors, short):
        align = data["align"]
        n     = len(align)
        rel   = np.linspace(0, 1, n)
        ax_align.plot(rel, align, lw=2, color=color, label=sname)
        ax_align.axvline(data["central"] / n, color=color, ls="--", lw=1, alpha=0.5)
    ax_align.set_xlabel(_XLABEL_REL_DEPTH)
    ax_align.set_ylabel("Mean Cosine to Other Layers")
    ax_align.legend(fontsize=8)
    ax_align.grid(True, alpha=0.3)

    # Panel 2 — cosine-with-own-central-layer profile
    ax_cos_ref.set_title("Cosine Sim with Own Most-Central Direction\nvs. Relative Depth")
    for (name, data), color, sname in zip(all_data.items(), colors, short):
        dirs    = data["dirs"]
        central = data["central"]
        n       = dirs.shape[0]
        cos     = torch.mm(dirs, dirs[central:central+1].T).squeeze(-1).clamp(-1, 1).numpy()
        rel     = np.linspace(0, 1, n)
        ax_cos_ref.plot(rel, cos, lw=2, color=color, label=sname)
    ax_cos_ref.axhline(0.7, color="gray", ls=":", alpha=0.5, label=_LABEL_COS_07)
    ax_cos_ref.set_xlabel(_XLABEL_REL_DEPTH)
    ax_cos_ref.set_ylabel(_YLABEL_COS_SIM)
    ax_cos_ref.set_ylim(-1.1, 1.1)
    ax_cos_ref.legend(fontsize=8)
    ax_cos_ref.grid(True, alpha=0.3)

    # Panel 3 — most-central layer as fraction of depth
    ax_depth.set_title("Most-Central Layer as Fraction of Model Depth")
    fracs = [d["central"] / d["dirs"].shape[0] for d in all_data.values()]
    bars  = ax_depth.bar(short, fracs, color=colors, edgecolor="black", alpha=0.85)
    ax_depth.set_ylim(0, 1)
    ax_depth.set_ylabel("Central Layer / Total Layers")
    ax_depth.grid(True, alpha=0.3, axis="y")
    plt.setp(ax_depth.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    for bar, frac in zip(bars, fracs):
        ax_depth.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.02,
                      f"{frac:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 4 — cross-model cosine at each relative depth (pairwise)
    # Interpolate all models to a common 32-point grid
    ax_cross.set_title("Cross-Model Agreement at Each Relative Depth\n"
                       "(cosine sim between models' directions, averaged over pairs)")
    n_grid = 32
    interp_dirs = {}
    for name, data in all_data.items():
        dirs = data["dirs"].numpy()  # [n_layers, d_model]
        n    = dirs.shape[0]
        grid = np.linspace(0, n - 1, n_grid)
        # Nearest-neighbor interpolation to the common grid
        idx_near = np.round(grid).astype(int).clip(0, n - 1)
        interp_dirs[name] = torch.tensor(dirs[idx_near])  # [n_grid, d_model]

    names_list = list(interp_dirs.keys())
    pair_cos = np.zeros(n_grid)
    n_pairs  = 0
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            a = interp_dirs[names_list[i]]  # [n_grid, d_model]
            b = interp_dirs[names_list[j]]  # [n_grid, d_model]
            cos = F.cosine_similarity(a, b, dim=-1).numpy()
            pair_cos += np.abs(cos)  # abs because direction sign can flip between models
            n_pairs  += 1
    if n_pairs > 0:
        pair_cos /= n_pairs

    rel_grid = np.linspace(0, 1, n_grid)
    ax_cross.plot(rel_grid, pair_cos, color="black", lw=2)
    ax_cross.fill_between(rel_grid, 0, pair_cos, alpha=0.2, color="black")
    ax_cross.axhline(0.5, color="red", ls="--", lw=1, label="|cos|=0.5")
    ax_cross.set_xlabel(_XLABEL_REL_DEPTH)
    ax_cross.set_ylabel("|Cosine Similarity| (avg over model pairs)")
    ax_cross.set_ylim(0, 1.05)
    ax_cross.legend(fontsize=9)
    ax_cross.grid(True, alpha=0.3)

    fig.savefig(plots_dir / "cross_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [plots] cross_model_comparison.png")


# ── Pairwise quantization comparison ───────────────────────────────────────────

def _base_net(sft_folder: str):
    """Map an SFT folder name to its base net name (matches refusal_dirs/ naming)."""
    f = sft_folder.lower()
    if "llama-2-13b-chat-hf" in f: return "llama-2-13b-chat-hf"
    if "llama-2-7b-chat-hf"  in f: return "llama-2-7b-chat-hf"
    if "gemma-2-9b"          in f: return "gemma-2-9b-it"
    if "qwen2.5-7b"          in f: return "qwen2.5-7b-instruct"
    return None


def build_groups(all_data: dict) -> dict:
    """
    Group quantized variants with their base model for pairwise comparison.

    Quantized variant files are named: <sft_folder>_<W{w}A{a}[_suffix]>
    Base model files are named:        <net_name>  (no _W\d+A\d+ suffix)

    Returns: {sft_folder: {label -> dirs Tensor[n_layers, d_model]}}
    """
    base_dirs = {
        name: data["dirs"]
        for name, data in all_data.items()
        if not re.search(r"_W\d+A\d+", name)
    }

    groups: dict = {}
    for name, data in all_data.items():
        m = re.match(r"^(.+?)_(W\d+A\d+.*)$", name)
        if not m:
            continue
        sft_folder = m.group(1)
        mode_label = m.group(2)

        if sft_folder not in groups:
            groups[sft_folder] = {}
            base_net = _base_net(sft_folder)
            if base_net and base_net in base_dirs:
                groups[sft_folder]["base"] = base_dirs[base_net]

        groups[sft_folder][mode_label] = data["dirs"]

    return groups


def _align_to_ref(tensors: list, ref_n: int) -> list:
    """Nearest-neighbour interpolate all tensors to ref_n layers."""
    out = []
    for t in tensors:
        n = t.shape[0]
        if n == ref_n:
            out.append(t)
        else:
            idx = np.round(np.linspace(0, n - 1, ref_n)).astype(int)
            out.append(t[idx])
    return out


def _plot_vs_base(group_name: str, labels: list, aligned: list, out_dir: Path):
    """Per-layer cosine sim of every variant against the base model."""
    if "base" not in labels:
        return
    base_t = aligned[labels.index("base")]
    layers = np.arange(base_t.shape[0])
    cmap   = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, t) in enumerate(zip(labels, aligned)):
        if label == "base":
            continue
        ax.plot(layers, F.cosine_similarity(t, base_t, dim=-1).numpy(),
                lw=2, color=cmap(i / 10), label=label)
    ax.axhline(0.9, color="green", ls=":", alpha=0.5, label="cos=0.9")
    ax.axhline(0.7, color="red",   ls=":", alpha=0.5, label=_LABEL_COS_07)
    ax.set_xlabel("Layer")
    ax.set_ylabel(_YLABEL_COS_SIM)
    ax.set_title(f"{group_name}\nPer-Layer Cosine Similarity vs Base Model Direction\n"
                 "(how well each quantization mode preserves the refusal direction)")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "vs_base_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_all_pairs_per_layer(group_name: str, labels: list, aligned: list, out_dir: Path):
    """One line per unique pair showing per-layer cosine similarity."""
    n      = len(labels)
    layers = np.arange(aligned[0].shape[0])
    cmap   = plt.cm.tab20
    n_pairs = n * (n - 1) // 2

    fig, ax = plt.subplots(figsize=(13, 5))
    color_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            cos = F.cosine_similarity(aligned[i], aligned[j], dim=-1).numpy()
            ax.plot(layers, cos, lw=1.8,
                    color=cmap(color_idx / max(n_pairs, 1)),
                    label=f"{labels[i]} vs {labels[j]}")
            color_idx += 1

    ax.axhline(0.9, color="green", ls=":", alpha=0.5, label="cos=0.9")
    ax.axhline(0.7, color="red",   ls=":", alpha=0.5, label=_LABEL_COS_07)
    ax.set_xlabel("Layer")
    ax.set_ylabel(_YLABEL_COS_SIM)
    ax.set_title(f"{group_name}\nPer-Layer Cosine Similarity — All Pairs")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=7, loc="lower right", ncol=max(1, n_pairs // 8))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "all_pairs_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(group_name: str, labels: list, aligned: list, out_dir: Path):
    n = len(labels)
    avg_cos = np.array([[
        F.cosine_similarity(aligned[i], aligned[j], dim=-1).mean().item()
        for j in range(n)] for i in range(n)])
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(avg_cos, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{group_name}\nPairwise Avg Cosine Similarity (across all layers)\n"
                 "Green = directions agree; red = directions diverge")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{avg_cos[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    fig.savefig(out_dir / "pairwise_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise(group_name: str, group: dict, plots_dir: Path):
    """Per-layer vs-base plot + pairwise avg-cosine heatmap for one SFT group."""
    labels  = list(group.keys())
    ref_n   = group[labels[0]].shape[0]
    aligned = _align_to_ref([group[l] for l in labels], ref_n)

    out_dir = plots_dir / f"pairwise_{group_name}"
    out_dir.mkdir(exist_ok=True)

    _plot_vs_base(group_name, labels, aligned, out_dir)
    _plot_all_pairs_per_layer(group_name, labels, aligned, out_dir)
    _plot_heatmap(group_name, labels, aligned, out_dir)
    print(f"  [plots] pairwise_{group_name}/  ({len(labels)} variants)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pt_files = sorted(REFUSAL_DIRS.glob("*.pt"))
    if not pt_files:
        print(f"[ERROR] No .pt files found in {REFUSAL_DIRS}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pt_files)} model(s): {[f.stem for f in pt_files]}\n")

    plots_dir = OUT_DIR / "plots"
    dirs_dir  = OUT_DIR / "directions"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dirs_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    summary  = {}

    for pt in pt_files:
        name = pt.stem
        print(f"── {name}")

        dirs = load_refusal_dirs(pt)           # [n_layers, d_model]
        n_layers, d_model = dirs.shape
        print(f"   {n_layers} layers, d_model={d_model}")

        # Save per-layer directions (already normed; also save stacked tensor)
        model_dir_out = dirs_dir / name
        model_dir_out.mkdir(exist_ok=True)
        torch.save(dirs, model_dir_out / "directions_per_layer_normed.pt")
        print(f"   saved → {model_dir_out}/directions_per_layer_normed.pt")

        # Per-model plots
        model_plot_dir = plots_dir / name
        model_plot_dir.mkdir(exist_ok=True)
        central_layer, align = plot_model(name, dirs, model_plot_dir)

        all_data[name] = {
            "dirs":    dirs,
            "align":   align,
            "central": central_layer,
        }
        summary[name] = {
            "n_layers":     n_layers,
            "d_model":      d_model,
            "central_layer": central_layer,
            "central_layer_frac": round(central_layer / n_layers, 3),
            "mean_alignment_score": float(align.mean()),
        }

    # Cross-model comparison (base models only)
    if len(all_data) >= 2:
        print("\nGenerating cross-model comparison …")
        plot_cross_model(all_data, plots_dir)

    # Pairwise quantization comparison (groups quantized variants with their base)
    groups = build_groups(all_data)
    if groups:
        print("\nGenerating pairwise quantization comparisons …")
        for group_name, group in groups.items():
            if len(group) >= 2:
                plot_pairwise(group_name, group, plots_dir)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone.  Results in {OUT_DIR}/")
    print("  plots/<model>/  — 3 figures per model")
    print("  plots/cross_model_comparison.png")
    print("  plots/pairwise_<base>/  — quantization pairwise comparisons")
    print("  directions/<model>/directions_per_layer_normed.pt")
    print("  summary.json")


if __name__ == "__main__":
    main()
