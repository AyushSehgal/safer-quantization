import argparse
import json
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from model_loader import load_model_and_tokenizer, add_model_args

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )
}

def format_gsm8k_prompt(question: str) -> str:
    return PROMPT_DICT["prompt_no_input"].format(instruction=question)

def extract_answer(text: str) -> str:
    # Try to find "The final answer is: " first
    match = re.search(r"The final answer is:\s*(-?[\d\.,]+)", text)
    if match:
        answer = match.group(1).replace(",", "")
        return answer
    
    # If not found, try to find "#### " (default GSM8K format)
    match = re.search(r"####\s*(-?[\d\.,]+)", text)
    if match:
        return match.group(1).replace(",", "")
        
    # Backup: extract the last number found in the generated text
    numbers = re.findall(r"-?[\d\.,]+", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""

def evaluate_gsm8k(model, tokenizer, dataset, limit=None):
    total = min(len(dataset), limit) if limit else len(dataset)
    correct = 0

    device = next(model.parameters()).device
    model.eval()

    for i in tqdm(range(total), desc="GSM8K Evaluation"):
        item = dataset[i]
        question = item["question"]
        
        # Ground truth answer is the string after ####
        true_answer_str = item["answer"].split("####")[-1].strip().replace(",", "")
        try:
            true_answer = float(true_answer_str)
        except ValueError:
            true_answer = None

        prompt = format_gsm8k_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                do_sample=False
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        pred_answer_str = extract_answer(generated_text)
        try:
            pred_answer = float(pred_answer_str)
        except ValueError:
            pred_answer = None

        if true_answer is not None and pred_answer == true_answer:
            correct += 1

    return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}

def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation")
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

    print("[GSM8K] Loading dataset ...")
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"[GSM8K] Loaded {len(dataset)} test examples")

    results = evaluate_gsm8k(model, tokenizer, dataset, limit=args.limit)

    print("\n" + "=" * 60)
    print("GSM8K Results")
    print("=" * 60)
    print(f"Model: {args.model_id} ({args.mode})")
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print("=" * 60)

    output_path = args.output or f"results_gsm8k_{args.mode}.json"
    results["model_id"] = args.model_id
    results["mode"] = args.mode
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
