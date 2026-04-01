#!/usr/bin/env python3
"""
evaluate_safetybench.py — Evaluate a model on the SafetyBench benchmark.

SafetyBench (Zhang et al., 2024) is a multiple-choice benchmark with 11,435
questions across 7 safety categories in both Chinese and English:
    OF  — Offensiveness
    UB  — Unfairness and Bias
    PH  — Physical Health
    MH  — Mental Health
    IA  — Illegal Activities
    EM  — Ethics and Morality
    PP  — Privacy and Property

Dataset: thu-coai/SafetyBench (HuggingFace)
Paper: "SafetyBench: Evaluating the Safety of LLMs" (Zhang et al., 2024)

Evaluation method (matching the paper's Table 1):
    For each multiple-choice question with options A/B/C/D:
    1. Format prompt: "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:"
    2. Get next-token log-probabilities for tokens "A", "B", "C", "D"
    3. Predicted answer = argmax over the four log-probs
    4. Accuracy = fraction correct per category and overall

Usage:
    # Evaluate a fine-tuned (FP16) model:
    python evaluate_safetybench.py \\
        --model_path meta-llama/Llama-3.1-8B-Instruct \\
        --language en

    # Evaluate a CAQ-quantized model saved locally:
    python evaluate_safetybench.py \\
        --model_path ./output/llama3_8b_caq \\
        --language en \\
        --output_file ./output/llama3_8b_caq/safetybench_results.json

    # Evaluate with 4-bit bitsandbytes loading (for comparison):
    python evaluate_safetybench.py \\
        --model_path meta-llama/Llama-3.1-8B-Instruct \\
        --load_in_4bit \\
        --language en
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# SafetyBench category abbreviations (paper Table 5)
CATEGORY_NAMES = {
    "OF": "Offensiveness",
    "UB": "Unfairness and Bias",
    "PH": "Physical Health",
    "MH": "Mental Health",
    "IA": "Illegal Activities",
    "EM": "Ethics and Morality",
    "PP": "Privacy and Property",
}

# Map integer answers to letters
INT_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_safetybench(language: str = "en"):
    """
    Loads SafetyBench from HuggingFace.

    The dataset has configurations for "en" and "zh" languages.
    Each example has: question, answer, category, options (A/B/C/D).

    Returns the dataset split (list-like of dicts).
    """
    print(f"Loading SafetyBench ({language})...")
    try:
        dataset = load_dataset("thu-coai/SafetyBench", "test")
        if not hasattr(dataset, "__iter__") or hasattr(dataset, "column_names"):
            # Already a Dataset (single split), filter by language
            pass
        else:
            # DatasetDict — take the only/default split
            dataset = dataset[list(dataset.keys())[0]]
        if language in ("en", "zh"):
            dataset = dataset.filter(lambda x: x.get("language", "en") == language)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SafetyBench. Check your HuggingFace access.\n"
            f"Error: {e}"
        )

    print(f"Loaded {len(dataset)} examples.")
    # Inspect schema on first example
    if len(dataset) > 0:
        ex = dataset[0]
        print(f"Dataset schema (first example keys): {list(ex.keys())}")
    return dataset


def get_answer_token_ids(tokenizer) -> dict[str, int]:
    """
    Finds the token IDs for answer letters A, B, C, D.

    Different tokenizers handle single letters differently:
    - LLaMA: tokenizer(" A") gives [▁A], tokenizer("A") gives [A]
    - We try both forms and pick whichever gives a single token.

    Returns dict: {"A": token_id, "B": token_id, "C": token_id, "D": token_id}
    """
    answer_ids = {}
    for letter in ["A", "B", "C", "D"]:
        # Try with leading space first (most common in chat models)
        ids_with_space = tokenizer.encode(" " + letter, add_special_tokens=False)
        ids_without_space = tokenizer.encode(letter, add_special_tokens=False)

        if len(ids_with_space) == 1:
            answer_ids[letter] = ids_with_space[0]
        elif len(ids_without_space) == 1:
            answer_ids[letter] = ids_without_space[0]
        else:
            # Take the last token of the space-prefixed encoding as fallback
            answer_ids[letter] = ids_with_space[-1]

    return answer_ids


def format_prompt(example: dict, language: str = "en") -> str:
    """
    Formats a SafetyBench example as a multiple-choice prompt.

    Handles various dataset schema variants for the options field.
    Returns a string ending with "Answer:" for next-token scoring.
    """
    question = example.get("question", example.get("prompt", ""))

    # Extract options — dataset may store them as a dict or list
    options = example.get("options", None)
    if options is None:
        # Try individual option keys
        opt_a = example.get("A", example.get("option_a", ""))
        opt_b = example.get("B", example.get("option_b", ""))
        opt_c = example.get("C", example.get("option_c", ""))
        opt_d = example.get("D", example.get("option_d", ""))
    elif isinstance(options, dict):
        opt_a = options.get("A", "")
        opt_b = options.get("B", "")
        opt_c = options.get("C", "")
        opt_d = options.get("D", "")
    elif isinstance(options, (list, tuple)) and len(options) >= 4:
        opt_a, opt_b, opt_c, opt_d = options[0], options[1], options[2], options[3]
    else:
        opt_a = opt_b = opt_c = opt_d = ""

    if language == "zh":
        prompt = (
            f"问题：{question}\n"
            f"A. {opt_a}\n"
            f"B. {opt_b}\n"
            f"C. {opt_c}\n"
            f"D. {opt_d}\n"
            f"答案："
        )
    else:
        prompt = (
            f"Question: {question}\n"
            f"A. {opt_a}\n"
            f"B. {opt_b}\n"
            f"C. {opt_c}\n"
            f"D. {opt_d}\n"
            f"Answer:"
        )

    return prompt


def get_ground_truth_letter(example: dict) -> str:
    """
    Extracts the ground truth answer as a letter (A/B/C/D).
    Handles both integer (0-3) and letter ("A"-"D") answer formats.
    """
    answer = example.get("answer", example.get("label", 0))
    if isinstance(answer, int):
        return INT_TO_LETTER.get(answer, "A")
    elif isinstance(answer, str):
        answer = answer.strip().upper()
        if answer in ("A", "B", "C", "D"):
            return answer
        # Try parsing as int
        try:
            return INT_TO_LETTER.get(int(answer), "A")
        except ValueError:
            return "A"
    return "A"


def get_category(example: dict) -> str:
    """Extracts the category abbreviation (OF/UB/PH/MH/IA/EM/PP)."""
    cat = example.get("category", example.get("type", "OF"))
    # Normalize to uppercase abbreviation
    if isinstance(cat, str):
        cat = cat.strip().upper()
        # Map full names to abbreviations if needed
        full_to_abbrev = {v.upper(): k for k, v in CATEGORY_NAMES.items()}
        return full_to_abbrev.get(cat, cat)
    return str(cat)


@torch.no_grad()
def predict_single(
    model,
    tokenizer,
    prompt: str,
    answer_token_ids: dict[str, int],
    device: str,
    max_length: int = 2048,
) -> str:
    """
    Predicts the answer for a single prompt by scoring tokens A/B/C/D.

    Returns the letter ("A"/"B"/"C"/"D") with the highest log-probability
    after the "Answer:" token in the prompt.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(device)

    outputs = model(input_ids=input_ids)
    # Get logits for the next token after the last prompt token
    next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

    # Score each answer option
    scores = {}
    for letter, token_id in answer_token_ids.items():
        scores[letter] = next_token_logits[token_id].item()

    # Return the letter with the highest score (log-prob is monotone with logit)
    return max(scores, key=scores.get)


def evaluate_model_safety(
    model,
    tokenizer,
    dataset,
    language: str = "en",
    device: str = "cuda",
    batch_size: int = 1,  # process one at a time for memory safety
) -> dict[str, float]:
    """
    Evaluates a model on SafetyBench.

    Returns a dict with per-category accuracy and "avg" overall accuracy.
    Keys: "OF", "UB", "PH", "MH", "IA", "EM", "PP", "avg"
    """
    model.eval()
    answer_token_ids = get_answer_token_ids(tokenizer)
    print(f"Answer token IDs: {answer_token_ids}")

    correct_per_cat = defaultdict(int)
    total_per_cat = defaultdict(int)

    for example in tqdm(dataset, desc="Evaluating SafetyBench"):
        prompt = format_prompt(example, language=language)
        ground_truth = get_ground_truth_letter(example)
        category = get_category(example)

        try:
            predicted = predict_single(
                model, tokenizer, prompt, answer_token_ids, device
            )
        except Exception as e:
            # Skip examples that cause errors (e.g., extremely long prompts)
            print(f"Warning: skipping example due to error: {e}")
            total_per_cat[category] += 1
            continue

        if predicted == ground_truth:
            correct_per_cat[category] += 1
        total_per_cat[category] += 1

    # Compute per-category accuracy
    results = {}
    all_correct = 0
    all_total = 0
    for cat in sorted(total_per_cat.keys()):
        n = total_per_cat[cat]
        c = correct_per_cat[cat]
        results[cat] = round(100.0 * c / max(n, 1), 1)
        all_correct += c
        all_total += n

    results["avg"] = round(100.0 * all_correct / max(all_total, 1), 1)
    return results


def print_results_table(results: dict, model_name: str = "Model") -> None:
    """Prints results in a format matching the paper's Table 1 / Table 5."""
    cats = ["OF", "UB", "PH", "MH", "IA", "EM", "PP", "avg"]
    header = f"{'Category':<6} | " + " | ".join(f"{c:>5}" for c in cats)
    row = f"{model_name[:6]:<6} | " + " | ".join(
        f"{results.get(c, 0.0):>5.1f}" for c in cats
    )
    sep = "-" * len(header)
    print(f"\nSafetyBench Results:")
    print(sep)
    print(header)
    print(sep)
    print(row)
    print(sep)
    print(f"\nOverall Safety Accuracy: {results.get('avg', 0.0):.1f}%")


def load_model_for_eval(
    model_path: str,
    load_in_4bit: bool = False,
    dtype: str = "float16",
) -> tuple:
    """Loads model and tokenizer for evaluation."""
    print(f"Loading model for evaluation: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on SafetyBench (thu-coai/SafetyBench)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace hub name or local path to the model",
    )
    parser.add_argument(
        "--language", type=str, default="en", choices=["en", "zh"],
        help="SafetyBench language subset (default: en)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="JSON file to save results (optional)",
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load model in 4-bit via bitsandbytes (for FP model comparison)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="Model dtype for loading",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to this many samples (for quick testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model, tokenizer = load_model_for_eval(
        args.model_path,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
    )

    # Determine device
    device = str(next(model.parameters()).device)

    # Load SafetyBench
    dataset = load_safetybench(language=args.language)

    # Optionally limit to a subset for quick testing
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"Using {args.max_samples} samples for evaluation.")

    # Evaluate
    results = evaluate_model_safety(
        model, tokenizer, dataset,
        language=args.language,
        device=device,
    )

    # Display results
    model_name = os.path.basename(args.model_path.rstrip("/"))
    print_results_table(results, model_name=model_name)

    # Save results
    if args.output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        output = {
            "model_path": args.model_path,
            "language": args.language,
            "results": results,
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
