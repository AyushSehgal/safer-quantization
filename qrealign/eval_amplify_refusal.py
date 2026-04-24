"""
Evaluate Layer-Selective Refusal Direction Amplification on quantized models.

Tests whether injecting  alpha * refusal_dir  at critical middle layers
restores safety behaviour lost during quantization — without retraining.

Three evaluation modes
----------------------
  standard   : baseline (no injection) + amplified, full dataset
  --sweep    : per-layer sweep on a SafetyBench subset to rank layers
  --alpha_sweep ALPHAS : grid over alpha values on a subset

Benchmarks
----------
  safetybench  safety accuracy (higher = safer)
  mmlu         general utility accuracy (higher = better)
  wikitext     perplexity on Wikitext-2 (lower = better)

Example usage
-------------
  # Standard eval — W8A8 quantized Llama-2-7b
  python eval_amplify_refusal.py \\
      --model_id meta-llama/Llama-2-7b-chat-hf \\
      --mode int8 \\
      --q_resume ./quantized_models/Llama-2-7b-chat-hf/W8A8/omni_parameters.pth \\
      --refusal_dir ../refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \\
      --alpha 20.0 --layers middle \\
      --tasks safetybench,mmlu,wikitext \\
      --safetybench_dir /path/to/SafetyBench \\
      --output ./results/amplify/Llama-2-7b-int8/

  # Layer sweep (SafetyBench subset, one layer at a time)
  python eval_amplify_refusal.py \\
      --model_id meta-llama/Llama-2-7b-chat-hf \\
      --mode int8 \\
      --q_resume ./quantized_models/Llama-2-7b-chat-hf/W8A8/omni_parameters.pth \\
      --refusal_dir ../refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \\
      --sweep --sweep_limit 200 --sweep_alpha 20.0 \\
      --safetybench_dir /path/to/SafetyBench \\
      --output ./results/amplify/sweep/

  # Alpha grid search
  python eval_amplify_refusal.py \\
      --model_id meta-llama/Llama-2-7b-chat-hf \\
      --mode int8 \\
      --refusal_dir ../refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \\
      --alpha_sweep 5,10,20,40,80 --layers middle --limit 300 \\
      --tasks safetybench,mmlu \\
      --safetybench_dir /path/to/SafetyBench \\
      --output ./results/amplify/alpha_sweep/
"""

import argparse
import json
import sys
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from model_loader import load_model_and_tokenizer, add_model_args
from amplify_refusal import (
    RefusalDirectionAmplifier,
    load_refusal_directions,
    parse_layers,
    get_decoder_layers,
    get_layer_direction,
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate refusal-direction amplification on quantized models"
    )
    p = add_model_args(p)

    # Refusal direction
    p.add_argument("--refusal_dir", type=str, required=True,
                   help="Path to refusal direction .pt file")

    # Amplification config
    p.add_argument("--alpha", type=float, default=20.0,
                   help="Injection strength (default 20.0)")
    p.add_argument("--layers", type=str, default="middle",
                   help="'middle' (default), 'all', or comma-separated indices e.g. '14,15,16'")

    # Sweep modes
    p.add_argument("--sweep", action="store_true",
                   help="Per-layer sweep on SafetyBench subset")
    p.add_argument("--sweep_limit", type=int, default=200,
                   help="Questions per layer during sweep (default 200)")
    p.add_argument("--sweep_alpha", type=float, default=None,
                   help="Alpha for layer sweep (default: same as --alpha)")

    p.add_argument("--alpha_sweep", type=str, default=None,
                   help="Comma-separated alphas to grid-search, e.g. '5,10,20,40'")

    # Benchmarks
    p.add_argument("--tasks", type=str, default="safetybench,mmlu,wikitext",
                   help="Comma-separated benchmarks to run")
    p.add_argument("--safetybench_dir", type=str, default=None,
                   help="Path to SafetyBench data directory")
    p.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of questions (useful for quick checks)")

    # Output
    p.add_argument("--output", type=str, default="./results/amplify/",
                   help="Output directory")
    p.add_argument("--device_map", type=str, default="auto")

    return p.parse_args()


# ── Benchmark wrappers ───────────────────────────────────────────────────────

@torch.no_grad()
def run_safetybench(model, tokenizer, safetybench_dir: str, lang: str = "en", limit=None) -> dict:
    from eval_safetybench import load_safetybench, evaluate_mcq
    questions = load_safetybench(safetybench_dir, lang=lang)
    results = evaluate_mcq(model, tokenizer, questions, limit=limit)

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r.get("category", "unknown")
        cat_stats[cat]["total"] += 1
        if r["is_correct"]:
            cat_stats[cat]["correct"] += 1

    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "num_questions": total,
        "per_category": {
            cat: {
                "accuracy": s["correct"] / s["total"] if s["total"] else 0.0,
                "correct": s["correct"],
                "total": s["total"],
            }
            for cat, s in cat_stats.items()
        },
    }


@torch.no_grad()
def run_mmlu(model, tokenizer, limit=None) -> dict:
    from datasets import load_dataset
    from eval_mmlu import evaluate_mmlu
    dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    return evaluate_mmlu(model, tokenizer, dataset, limit=limit)


@torch.no_grad()
def run_wikitext(model, tokenizer) -> dict:
    from datasets import load_dataset
    from eval_wikitext_ppl import evaluate_perplexity
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl, num_tokens = evaluate_perplexity(model, tokenizer, dataset)
    return {"perplexity": ppl, "num_tokens": num_tokens}


def run_tasks(model, tokenizer, tasks: list, args) -> dict:
    """Run the requested benchmark tasks and return a results dict."""
    results = {}
    if "safetybench" in tasks:
        if args.safetybench_dir is None:
            print("  [skip] safetybench — --safetybench_dir not provided")
        else:
            print("  Running SafetyBench ...")
            results["safetybench"] = run_safetybench(
                model, tokenizer, args.safetybench_dir, args.lang, args.limit
            )
            print(f"  SafetyBench: {results['safetybench']['overall_accuracy']:.4f}")

    if "mmlu" in tasks:
        print("  Running MMLU ...")
        results["mmlu"] = run_mmlu(model, tokenizer, args.limit)
        print(f"  MMLU: {results['mmlu']['overall_accuracy']:.4f}")

    if "wikitext" in tasks:
        print("  Running Wikitext-2 perplexity ...")
        results["wikitext"] = run_wikitext(model, tokenizer)
        print(f"  Wikitext PPL: {results['wikitext']['perplexity']:.3f}")

    return results


# ── Layer sweep ──────────────────────────────────────────────────────────────

def run_layer_sweep(model, tokenizer, directions, alpha, args, output_dir: Path) -> dict:
    """
    Evaluate SafetyBench accuracy with one layer injected at a time.
    Reports which layers produce the largest safety improvement.
    """
    from eval_safetybench import load_safetybench, evaluate_mcq

    if args.safetybench_dir is None:
        raise ValueError("--safetybench_dir is required for layer sweep")

    questions = load_safetybench(args.safetybench_dir, lang=args.lang)
    limit = args.sweep_limit
    if limit:
        questions = questions[:limit]

    num_layers = len(get_decoder_layers(model))
    sweep_results: dict = {}

    # Baseline (no injection)
    print("\n[sweep] Baseline (no injection) ...")
    baseline_res = evaluate_mcq(model, tokenizer, questions)
    baseline_acc = sum(r["is_correct"] for r in baseline_res) / len(baseline_res)
    sweep_results["baseline"] = baseline_acc
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    # Per-layer injection
    print(f"\n[sweep] Sweeping {num_layers} layers with alpha={alpha} ...")
    for layer_idx in range(num_layers):
        direction = get_layer_direction(directions, layer_idx)
        if direction is None:
            continue

        with RefusalDirectionAmplifier(
            model,
            {layer_idx: direction},
            alpha=alpha,
            layers=[layer_idx],
        ):
            layer_res = evaluate_mcq(model, tokenizer, questions)

        acc = sum(r["is_correct"] for r in layer_res) / len(layer_res)
        delta = acc - baseline_acc
        sweep_results[layer_idx] = {"accuracy": acc, "delta": delta}
        print(f"  Layer {layer_idx:2d}: acc={acc:.4f}  Δ={delta:+.4f}")

    # Summary
    layer_deltas = {
        k: v["delta"]
        for k, v in sweep_results.items()
        if isinstance(k, int)
    }
    if layer_deltas:
        best = max(layer_deltas, key=layer_deltas.get)
        print(f"\n[sweep] Most effective layer: {best}  (Δ={layer_deltas[best]:+.4f})")
        top5 = sorted(layer_deltas, key=layer_deltas.get, reverse=True)[:5]
        print(f"[sweep] Top-5 layers: {top5}")

    out = {
        "alpha": alpha,
        "limit": limit,
        "num_layers": num_layers,
        "baseline_accuracy": baseline_acc,
        "per_layer": {str(k): v for k, v in sweep_results.items() if k != "baseline"},
    }
    path = output_dir / "layer_sweep.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sweep] Saved → {path}")
    return out


# ── Alpha sweep ──────────────────────────────────────────────────────────────

def run_alpha_sweep(model, tokenizer, directions, alphas, target_layers, args, output_dir: Path) -> dict:
    """Grid-search alpha on a subset; saves a JSON with per-alpha results."""
    tasks = [t.strip() for t in args.tasks.split(",")]
    all_results: dict = {}

    for alpha in alphas:
        print(f"\n=== alpha = {alpha} ===")
        with RefusalDirectionAmplifier(model, directions, alpha=alpha, layers=target_layers):
            all_results[str(alpha)] = run_tasks(model, tokenizer, tasks, args)

    out = {
        "model_id": args.model_id,
        "mode": args.mode,
        "layers": target_layers,
        "alphas": alphas,
        "results": all_results,
    }
    path = output_dir / "alpha_sweep.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[alpha_sweep] Saved → {path}")
    return out


# ── Standard eval ────────────────────────────────────────────────────────────

def run_standard_eval(model, tokenizer, directions, target_layers, args, output_dir: Path):
    """Run baseline then amplified eval; save side-by-side comparison."""
    tasks = [t.strip() for t in args.tasks.split(",")]

    print("\n=== Baseline (no injection) ===")
    baseline = run_tasks(model, tokenizer, tasks, args)

    print(f"\n=== Amplified (alpha={args.alpha}, layers={target_layers}) ===")
    with RefusalDirectionAmplifier(model, directions, alpha=args.alpha, layers=target_layers):
        amplified = run_tasks(model, tokenizer, tasks, args)

    # Delta summary
    print("\n── Delta summary ──────────────────────────────────")
    if "safetybench" in baseline and "safetybench" in amplified:
        b = baseline["safetybench"]["overall_accuracy"]
        a = amplified["safetybench"]["overall_accuracy"]
        print(f"  SafetyBench: {b:.4f} → {a:.4f}  ({a-b:+.4f})")
    if "mmlu" in baseline and "mmlu" in amplified:
        b = baseline["mmlu"]["overall_accuracy"]
        a = amplified["mmlu"]["overall_accuracy"]
        print(f"  MMLU:        {b:.4f} → {a:.4f}  ({a-b:+.4f})")
    if "wikitext" in baseline and "wikitext" in amplified:
        b = baseline["wikitext"]["perplexity"]
        a = amplified["wikitext"]["perplexity"]
        print(f"  Wikitext PPL:{b:.3f} → {a:.3f}  ({a-b:+.3f})")

    out = {
        "model_id": args.model_id,
        "mode": args.mode,
        "q_resume": args.q_resume,
        "refusal_dir_path": args.refusal_dir,
        "alpha": args.alpha,
        "layers": target_layers,
        "baseline": baseline,
        "amplified": amplified,
    }
    fname = f"results_{args.mode}_alpha{args.alpha}.json"
    path = output_dir / fname
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        mode=args.mode,
        resume=args.resume,
        q_resume=args.q_resume,
        device_map=args.device_map,
    )

    # Load refusal directions
    print(f"\nLoading refusal directions from {args.refusal_dir} ...")
    directions = load_refusal_directions(args.refusal_dir)
    layer_keys = sorted(k for k in directions if k != -1)
    if layer_keys:
        print(f"  Per-layer directions: {layer_keys[:8]}{'...' if len(layer_keys) > 8 else ''}")
    else:
        print("  Single global direction (key=-1), will broadcast to all hooked layers.")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode dispatch ─────────────────────────────────────────────────────

    if args.sweep:
        sweep_alpha = args.sweep_alpha if args.sweep_alpha is not None else args.alpha
        run_layer_sweep(model, tokenizer, directions, sweep_alpha, args, output_dir)
        return

    target_layers = parse_layers(args.layers, model)
    print(f"Target layers: {target_layers}")

    if args.alpha_sweep:
        alphas = [float(a) for a in args.alpha_sweep.split(",")]
        run_alpha_sweep(model, tokenizer, directions, alphas, target_layers, args, output_dir)
        return

    run_standard_eval(model, tokenizer, directions, target_layers, args, output_dir)


if __name__ == "__main__":
    main()
