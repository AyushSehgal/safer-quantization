"""
aggregate_results.py — Collect results from all model evals into a summary table.

Scans the results directory for each model's eval outputs and produces
a CSV + pretty-printed table comparing all models across all metrics.
"""

import argparse
import json
import os
import csv
from pathlib import Path


def extract_mmlu(result_dir: str) -> float | None:
    """Extract MMLU accuracy from lm-eval-harness output."""
    mmlu_dir = os.path.join(result_dir, "mmlu")
    if not os.path.exists(mmlu_dir):
        return None

    # lm-eval-harness saves results as JSON in the output dir
    for f in Path(mmlu_dir).rglob("results*.json"):
        with open(f, "r") as fp:
            data = json.load(fp)
        # Navigate the nested structure
        if "results" in data:
            for key, val in data["results"].items():
                if "mmlu" in key.lower() and "acc" in val:
                    return round(val["acc"] * 100, 2)
                if "mmlu" in key.lower() and "acc,none" in val:
                    return round(val["acc,none"] * 100, 2)
    return None


def extract_wikitext_ppl(result_dir: str) -> float | None:
    """Extract Wikitext-2 perplexity from lm-eval-harness output."""
    wiki_dir = os.path.join(result_dir, "wikitext")
    if not os.path.exists(wiki_dir):
        return None

    for f in Path(wiki_dir).rglob("results*.json"):
        with open(f, "r") as fp:
            data = json.load(fp)
        if "results" in data:
            for key, val in data["results"].items():
                if "wikitext" in key.lower():
                    for metric_key in ["word_perplexity", "word_perplexity,none",
                                       "byte_perplexity", "byte_perplexity,none"]:
                        if metric_key in val:
                            return round(val[metric_key], 2)
    return None


def extract_advbench_asr(result_dir: str) -> float | None:
    """Extract ASR from AdvBench eval."""
    asr_file = os.path.join(result_dir, "advbench", "asr_results.json")
    if not os.path.exists(asr_file):
        return None
    with open(asr_file, "r") as f:
        data = json.load(f)
    return data.get("asr_percent")


def extract_safetybench(result_dir: str) -> float | None:
    """Extract SafetyBench accuracy."""
    sb_file = os.path.join(result_dir, "safetybench", "safetybench_results.json")
    if not os.path.exists(sb_file):
        return None
    with open(sb_file, "r") as f:
        data = json.load(f)
    return data.get("overall_accuracy")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Q-resafe eval results")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Discover all model result directories
    models = sorted([
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    ])

    rows = []
    for model_name in models:
        result_dir = os.path.join(args.results_dir, model_name)
        row = {
            "model": model_name,
            "MMLU (%)": extract_mmlu(result_dir),
            "Wikitext-2 PPL": extract_wikitext_ppl(result_dir),
            "AdvBench ASR (%)": extract_advbench_asr(result_dir),
            "SafetyBench (%)": extract_safetybench(result_dir),
        }
        rows.append(row)

    # Pretty print
    print(f"\n{'='*80}")
    print(f"{'Model':<35} {'MMLU':>8} {'PPL':>10} {'ASR':>10} {'Safety':>10}")
    print(f"{'='*80}")
    for row in rows:
        mmlu = f"{row['MMLU (%)']:.1f}" if row['MMLU (%)'] is not None else "—"
        ppl = f"{row['Wikitext-2 PPL']:.2f}" if row['Wikitext-2 PPL'] is not None else "—"
        asr = f"{row['AdvBench ASR (%)']:.1f}" if row['AdvBench ASR (%)'] is not None else "—"
        safe = f"{row['SafetyBench (%)']:.1f}" if row['SafetyBench (%)'] is not None else "—"
        print(f"{row['model']:<35} {mmlu:>8} {ppl:>10} {asr:>10} {safe:>10}")
    print(f"{'='*80}")

    # Save CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
