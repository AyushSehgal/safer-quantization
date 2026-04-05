import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from model_loader import load_model_and_tokenizer, add_model_args

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
}

def format_sst2_prompt(sentence: str) -> str:
    instruction = "Classify the sentiment of the following sentence as positive or negative."
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=sentence)

@torch.no_grad()
def evaluate_sst2(model, tokenizer, dataset, limit=None):
    total = min(len(dataset), limit) if limit else len(dataset)
    correct = 0

    device = next(model.parameters()).device
    
    # Try different tokenizations of positive/negative just in case
    pos_ids = [tokenizer.encode("positive", add_special_tokens=False)[0], tokenizer.encode(" positive", add_special_tokens=False)[0]]
    neg_ids = [tokenizer.encode("negative", add_special_tokens=False)[0], tokenizer.encode(" negative", add_special_tokens=False)[0]]

    for i in tqdm(range(total), desc="SST-2 Evaluation"):
        item = dataset[i]
        sentence = item["sentence"]
        # In GLUE SST-2: 0 is negative, 1 is positive
        is_positive = (item["label"] == 1)
        
        prompt = format_sst2_prompt(sentence)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        
        pos_logit = max([logits[pid].item() for pid in pos_ids])
        neg_logit = max([logits[nid].item() for nid in neg_ids])
        
        pred_is_positive = pos_logit > neg_logit
            
        if pred_is_positive == is_positive:
            correct += 1
            
    return {"accuracy": correct / total, "correct": correct, "total": total}

def main():
    parser = argparse.ArgumentParser(description="SST-2 Evaluation")
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

    print("[SST-2] Loading dataset ...")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"[SST-2] Loaded {len(dataset)} validation examples")

    results = evaluate_sst2(model, tokenizer, dataset, limit=args.limit)

    print("\n" + "=" * 60)
    print("SST-2 Results")
    print("=" * 60)
    print(f"Model: {args.model_id} ({args.mode})")
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print("=" * 60)

    output_path = args.output or f"results_sst2_{args.mode}.json"
    results["model_id"] = args.model_id
    results["mode"] = args.mode
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
