#!/usr/bin/env python3
"""
Compare quantized refusal directions against the original (FP16) refusal direction.

Produces 4 per-layer cosine-similarity plots (one per quantization mode):
  W8A8, W4A16, W8A16_combined, W4A16_combined

Each plot shows one line per SFT-finetuned model variant found in refusal_dirs/.

Usage (run from the project root):
    python plot_quant_vs_original.py
    python plot_quant_vs_original.py --refusal_dirs /path/to/refusal_dirs
    python plot_quant_vs_original.py --out_dir my_plots/
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ── Naming helpers ─────────────────────────────────────────────────────────────

NET_NAMES = [
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "gemma-2-9b-it",
    "qwen2.5-7b-instruct",
]

def _net_for_sft(sft_folder: str) -> str | None:
    f = sft_folder.lower()
    if "llama-2-13b-chat-hf" in f: return "Llama-2-13b-chat-hf"
    if "llama-2-7b-chat-hf"  in f: return "Llama-2-7b-chat-hf"
    if "gemma-2-9b"          in f: return "gemma-2-9b-it"
    if "qwen2.5-7b"          in f: return "qwen2.5-7b-instruct"
    return None


# Quantization modes we want to compare: (label, regex that matches mode_dir portion)
QUANT_MODES = [
    ("W8A8",           re.compile(r"^W8A8",           re.IGNORECASE)),
    ("W4A16",          re.compile(r"^W4A16(?!_combined)", re.IGNORECASE)),
    ("W8A16_combined", re.compile(r"^W8A16_combined",  re.IGNORECASE)),
    ("W4A16_combined", re.compile(r"^W4A16_combined",  re.IGNORECASE)),
]


def classify_mode(mode_str: str) -> str | None:
    """Return the canonical label for a mode directory string, or None."""
    for label, pat in QUANT_MODES:
        if pat.match(mode_str):
            return label
    return None


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_pt(path: Path) -> torch.Tensor:
    """Load {layer_idx: vec} or stacked Tensor → [n_layers, d_model], unit-normed."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        n = max(raw.keys()) + 1
        d = next(iter(raw.values())).shape[0]
        t = torch.zeros(n, d)
        for idx, v in raw.items():
            t[idx] = v.float()
    else:
        t = raw.float()
    norms = t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return t / norms


def _nn_resample(t: torch.Tensor, n_out: int) -> torch.Tensor:
    """Nearest-neighbour resample t ([n, d]) to n_out rows."""
    n = t.shape[0]
    if n == n_out:
        return t
    idx = np.round(np.linspace(0, n - 1, n_out)).astype(int)
    return t[idx]


# ── Plotting ───────────────────────────────────────────────────────────────────

def short_label(sft_folder: str) -> str:
    s = sft_folder
    # strip known net-name prefix
    for net in NET_NAMES:
        s = re.sub(rf"(?i)sft-{re.escape(net)}-?", "", s)
    return s or sft_folder


def plot_one_mode(mode_label: str, entries: list[dict], out_path: Path):
    """
    entries: list of {"sft": str, "orig": Tensor[n,d], "quant": Tensor[m,d]}
    Produces one figure with per-layer cosine similarity for each entry.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.tab10
    n_entries = len(entries)

    for i, e in enumerate(entries):
        orig  = e["orig"]
        quant = e["quant"]
        ref_n = orig.shape[0]

        q_rs = _nn_resample(quant, ref_n)
        cos  = F.cosine_similarity(q_rs, orig, dim=-1).numpy()
        layers = np.arange(ref_n)

        label = short_label(e["sft"])
        ax.plot(layers, cos, lw=2, color=cmap(i / max(n_entries, 1)), label=label)

    ax.axhline(0.9, color="green",  ls=":",  lw=1.2, alpha=0.6, label="cos = 0.9")
    ax.axhline(0.7, color="orange", ls="--", lw=1.2, alpha=0.6, label="cos = 0.7")
    ax.axhline(0.0, color="gray",   ls=":",  lw=0.8, alpha=0.4)

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.set_title(
        f"Per-Layer Cosine Similarity: {mode_label} vs Original Refusal Direction\n"
        "(1 = identical direction, 0 = orthogonal)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=8, loc="lower right", ncol=max(1, n_entries // 6))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}  ({n_entries} model variant(s))")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot cosine similarity between quantized and original refusal directions."
    )
    ap.add_argument(
        "--refusal_dirs",
        default=None,
        help="Path to refusal_dirs/ folder containing *.pt files "
             "(default: refusal_direction/refusal_dirs/ relative to this script)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots (default: refusal_directions_output/quant_vs_original/)",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).parent
    refusal_dirs = Path(args.refusal_dirs) if args.refusal_dirs else \
                   script_dir / "refusal_direction" / "refusal_dirs"
    out_dir = Path(args.out_dir) if args.out_dir else \
              script_dir / "refusal_directions_output" / "quant_vs_original"

    if not refusal_dirs.exists():
        print(f"[ERROR] refusal_dirs not found: {refusal_dirs}", file=sys.stderr)
        sys.exit(1)

    pt_files = sorted(refusal_dirs.glob("*.pt"))
    if not pt_files:
        print(f"[ERROR] No .pt files in {refusal_dirs}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pt_files)} .pt file(s) in {refusal_dirs}\n")

    # Split files into originals vs quantized variants
    originals: dict[str, torch.Tensor] = {}  # net_name → dirs
    quantized: list[dict] = []               # {sft, mode_str, mode_label, dirs}

    for pt in pt_files:
        name = pt.stem
        # Quantized files contain _W<n>A<n> in the name
        m = re.match(r"^(.+?)_(W\d+A\d+.*)$", name, re.IGNORECASE)
        if m:
            sft_folder = m.group(1)
            mode_str   = m.group(2)
            mode_label = classify_mode(mode_str)
            if mode_label is None:
                print(f"  [skip] {name} — mode '{mode_str}' not in target list")
                continue
            quantized.append({
                "sft":        sft_folder,
                "mode_str":   mode_str,
                "mode_label": mode_label,
                "path":       pt,
            })
        else:
            # Original (base model) file
            originals[name] = pt
            print(f"  [orig] {name}")

    if not originals:
        print("[ERROR] No original (non-quantized) .pt files found.", file=sys.stderr)
        sys.exit(1)

    if not quantized:
        print("[ERROR] No quantized .pt files matching target modes found.", file=sys.stderr)
        print("  Target modes: W8A8, W4A16, W8A16_combined, W4A16_combined")
        sys.exit(1)

    # Group by mode_label
    by_mode: dict[str, list[dict]] = {label: [] for label, _ in QUANT_MODES}
    skipped = 0
    for q in quantized:
        net = _net_for_sft(q["sft"])
        if net is None or net not in originals:
            print(f"  [skip] {q['sft']}_{q['mode_str']} — no original for base net '{net}'")
            skipped += 1
            continue
        orig_pt = originals[net]
        by_mode[q["mode_label"]].append({
            "sft":  q["sft"],
            "orig": load_pt(orig_pt),
            "quant": load_pt(q["path"]),
        })
        print(f"  [add ] {q['sft']}_{q['mode_str']}  → {q['mode_label']}")

    if skipped:
        print(f"\n  ({skipped} variant(s) skipped — missing original)")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots → {out_dir}/\n")

    generated = 0
    for label, _ in QUANT_MODES:
        entries = by_mode[label]
        if not entries:
            print(f"  [skip] {label} — no variants found")
            continue
        out_path = out_dir / f"cos_sim_{label}_vs_original.png"
        plot_one_mode(label, entries, out_path)
        generated += 1

    print(f"\nDone. {generated}/4 plot(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
