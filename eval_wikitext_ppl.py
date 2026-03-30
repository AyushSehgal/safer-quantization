"""
Wikitext-2 Perplexity Evaluation Script for Q-Realign Baseline.

Computes perplexity on the Wikitext-2 raw test set using a
sliding-window approach (standard methodology).

Usage:
    python eval_wikitext_ppl.py --mode fp16
    python eval_wikitext_ppl.py --mode int8 --q_resume path/to/omni_parameters.pth
    python eval_wikitext_ppl.py --mode int4 --q_resume path/to/omni_parameters.pth
"""

import argparse
import json
import math
import torch
from tqdm import tqdm
from datasets import load_dataset

from model_loader import load_model_and_tokenizer, add_model_args


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset, seq_len=2048, stride=512):
    """
    Compute perplexity on the Wikitext-2 test set using sliding window.

    Uses the standard approach:
    - Concatenate all text into one long sequence
    - Slide a window of `seq_len` tokens with `stride` step size
    - Compute cross-entropy loss for the non-overlapping portion
    - Report exp(average_loss) as perplexity
    """
    # Concatenate all text
    text = "\n\n".join(dataset["text"])

    # Tokenize the full text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    total_length = input_ids.size(1)
    print(f"[Wikitext-2] Total tokens: {total_length}")

    device = next(model.parameters()).device
    nlls = []
    num_tokens = 0

    prev_end_loc = 0
    for begin_loc in tqdm(range(0, total_length, stride), desc="Wikitext-2 PPL"):
        end_loc = min(begin_loc + seq_len, total_length)
        trg_len = end_loc - prev_end_loc  # Number of tokens to score

        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_chunk = input_chunk.clone()

        # Mask out already-scored tokens (the overlapping context)
        target_chunk[:, :-trg_len] = -100

        outputs = model(input_chunk, labels=target_chunk)
        # outputs.loss is the mean cross-entropy over non-masked tokens
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.cpu())
        num_tokens += trg_len

        prev_end_loc = end_loc
        if end_loc >= total_length:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens).item()
    return ppl, num_tokens


def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 Perplexity Evaluation")
    add_model_args(parser)
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length for sliding window")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        mode=args.mode,
        q_resume=args.q_resume,
    )

    print("[Wikitext-2] Loading dataset ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"[Wikitext-2] Loaded {len(dataset)} documents")

    ppl, num_tokens = evaluate_perplexity(
        model, tokenizer, dataset,
        seq_len=args.seq_len, stride=args.stride,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Wikitext-2 Perplexity Results")
    print("=" * 60)
    print(f"Model: {args.model_id} ({args.mode})")
    print(f"Perplexity: {ppl:.4f}")
    print(f"Tokens evaluated: {num_tokens}")
    print(f"Sequence length: {args.seq_len}, Stride: {args.stride}")
    print("=" * 60)

    # Save results
    output_path = args.output or f"results_wikitext_ppl_{args.mode}.json"
    results = {
        "model_id": args.model_id,
        "mode": args.mode,
        "perplexity": ppl,
        "num_tokens": num_tokens,
        "seq_len": args.seq_len,
        "stride": args.stride,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
