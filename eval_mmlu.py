"""
MMLU Evaluation Script for Q-Realign Baseline.

Evaluates model general utility on the MMLU benchmark using
log-likelihood scoring over answer choices.

Usage:
    python eval_mmlu.py --mode fp16
    python eval_mmlu.py --mode int8 --q_resume path/to/omni_parameters.pth
    python eval_mmlu.py --mode int4 --q_resume path/to/omni_parameters.pth
"""

import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

from model_loader import load_model_and_tokenizer, add_model_args


CHOICE_LABELS = ["A", "B", "C", "D"]

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine",
        "virology",
    ],
}

# Build reverse mapping: subject -> category
SUBJECT_TO_CATEGORY = {}
for cat, subjects in MMLU_CATEGORIES.items():
    for subj in subjects:
        SUBJECT_TO_CATEGORY[subj] = cat


def format_mmlu_prompt(question: str, choices: list, subject: str) -> str:
    """Format an MMLU item into a prompt string."""
    subject_display = subject.replace("_", " ").title()
    prompt = f"The following is a multiple choice question about {subject_display}.\n\n"
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{CHOICE_LABELS[i]}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, dataset, limit=None, num_fewshot=0):
    """
    Run MMLU evaluation using log-likelihood scoring.

    For each question, computes the log-likelihood of each answer choice token
    (A, B, C, D) given the prompt, and picks the highest.
    """
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    total_correct = 0
    total = 0

    n = len(dataset) if limit is None else min(limit, len(dataset))

    # Pre-compute token ids for answer choices
    choice_token_ids = []
    for label in CHOICE_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        choice_token_ids.append(ids[-1])  # Take last token in case of BPE split

    device = next(model.parameters()).device

    for i in tqdm(range(n), desc="MMLU Evaluation"):
        item = dataset[i]
        question = item["question"]
        choices = item["choices"]
        correct_answer = item["answer"]  # int (0-3)
        subject = item["subject"]

        # Format prompt
        prompt_text = format_mmlu_prompt(question, choices, subject)

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Logits at the last token position

        # Get log-probabilities for each choice token
        log_probs = torch.log_softmax(logits, dim=-1)
        choice_logprobs = [log_probs[tid].item() for tid in choice_token_ids]

        # Predict the choice with highest log-prob
        predicted = max(range(len(choice_logprobs)), key=lambda x: choice_logprobs[x])

        is_correct = (predicted == correct_answer)
        subject_correct[subject] += int(is_correct)
        subject_total[subject] += 1
        total_correct += int(is_correct)
        total += 1

    # Aggregate by category
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for subj in subject_total:
        cat = SUBJECT_TO_CATEGORY.get(subj, "Other")
        category_results[cat]["correct"] += subject_correct[subj]
        category_results[cat]["total"] += subject_total[subj]

    return {
        "overall_accuracy": total_correct / total if total > 0 else 0,
        "total_correct": total_correct,
        "total": total,
        "per_subject": {
            subj: {
                "accuracy": subject_correct[subj] / subject_total[subj] if subject_total[subj] > 0 else 0,
                "correct": subject_correct[subj],
                "total": subject_total[subj],
            }
            for subj in sorted(subject_total.keys())
        },
        "per_category": {
            cat: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for cat, v in sorted(category_results.items())
        }
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU Evaluation")
    add_model_args(parser)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N examples")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        mode=args.mode,
        resume=args.resume,
        q_resume=args.q_resume,
    )

    print("[MMLU] Loading dataset ...")
    dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    print(f"[MMLU] Loaded {len(dataset)} examples across {len(set(dataset['subject']))} subjects")

    results = evaluate_mmlu(model, tokenizer, dataset, limit=args.limit)

    # Print results
    print("\n" + "=" * 60)
    print("MMLU Results")
    print("=" * 60)
    print(f"Model: {args.model_id} ({args.mode})")
    print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}% ({results['total_correct']}/{results['total']})")
    print("\nPer-Category:")
    for cat, stats in results["per_category"].items():
        print(f"  {cat:20s}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    print("\nPer-Subject (top 10 and bottom 10 by accuracy):")
    sorted_subjs = sorted(results["per_subject"].items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for subj, stats in sorted_subjs[:10]:
        print(f"  {subj:45s}: {stats['accuracy']*100:.2f}%")
    print("  ...")
    for subj, stats in sorted_subjs[-10:]:
        print(f"  {subj:45s}: {stats['accuracy']*100:.2f}%")
    print("=" * 60)

    # Save results
    output_path = args.output or f"results_mmlu_{args.mode}.json"
    results["model_id"] = args.model_id
    results["mode"] = args.mode
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
