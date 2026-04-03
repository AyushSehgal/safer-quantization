#!/usr/bin/env python3
"""
run_caq.py — End-to-end CAQ quantization pipeline.

Implements the full CAQ method (Algorithm 1 from Appendix B):
    1. Load M_PT (unsafe pre-trained) and M_FT (safe fine-tuned) models
    2. Build calibration dataset (128 WikiText-2 samples)
    3. Initialize learnable smooth scaling transformations θ
    4. Optimize θ using Contrastive Alignment Loss (CAL)
    5. Fuse θ into M_FT weights (zero inference overhead)
    6. Apply GPTQ W4A4 quantization → M_Q
    7. Save M_Q and evaluate WikiText-2 perplexity

Usage:
    source .venv/bin/activate
    python run_caq.py \\
        --pretrained_model meta-llama/Llama-3.1-8B \\
        --finetuned_model meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir ./output/llama3_8b_caq

Paper hyperparameters (defaults match Section 4.1):
    alpha=0.75, top_k=500, n_samples=128, bits=4, group_size=128, lr=1e-3
"""

import argparse
import json
import os

from caq import (
    CAQConfig,
    ContrastiveAlignmentLoss,
    ModelPair,
    CAQTrainer,
    QuantizerWrapper,
    get_calibration_loader,
    get_wikitext2_test_loader,
)
from caq.utils import setup_logging, save_config, compute_perplexity, clear_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrastive Alignment Quantization (CAQ) — safety-preserving PTQ"
    )
    # Model paths
    parser.add_argument(
        "--pretrained_model", type=str, required=True,
        help="HuggingFace hub name or local path for M_PT (unsafe pre-trained)",
    )
    parser.add_argument(
        "--finetuned_model", type=str, required=True,
        help="HuggingFace hub name or local path for M_FT (safe fine-tuned)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to save quantized model and results",
    )

    # CAL hyperparameters (paper Section 3.2 and 4.1)
    parser.add_argument(
        "--alpha", type=float, default=0.75,
        help="Contrastive weight α (paper default: 0.75, Table 3)",
    )
    parser.add_argument(
        "--top_k", type=int, default=500,
        help="Top-k for vocabulary subsets S_top and S_diff (paper default: 500, Table 4)",
    )

    # Calibration
    parser.add_argument(
        "--n_samples", type=int, default=128,
        help="Number of WikiText-2 calibration samples (paper default: 128)",
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048,
        help="Sequence length for calibration samples",
    )

    # Optimization
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Adam optimizer over transformation parameters",
    )

    # Quantization
    parser.add_argument(
        "--bits", type=int, default=4,
        help="Weight quantization bit-width (paper: W4A4)",
    )
    parser.add_argument(
        "--group_size", type=int, default=128,
        help="Per-group quantization group size",
    )

    # Misc
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="Model load dtype",
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help="Evaluate WikiText-2 perplexity after quantization",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for calibration data sampling",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    # Build config
    config = CAQConfig(
        pretrained_model_name=args.pretrained_model,
        finetuned_model_name=args.finetuned_model,
        alpha=args.alpha,
        top_k=args.top_k,
        num_calibration_samples=args.n_samples,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        bits=args.bits,
        group_size=args.group_size,
        output_dir=args.output_dir,
        dtype=args.dtype,
        calibration_seed=args.seed,
    )
    save_config(config, args.output_dir)
    logger.info(f"CAQ config: alpha={config.alpha}, k={config.top_k}, bits={config.bits}")

    # Step 1: Load both models (Algorithm 1, line 1 prerequisite)
    model_pair = ModelPair(config)
    model_pair.load()

    # Step 2: Build calibration dataset (WikiText-2, 128 samples)
    logger.info("Building calibration dataset (WikiText-2)...")
    calib_loader = get_calibration_loader(
        model_pair.tokenizer,
        num_samples=config.num_calibration_samples,
        seq_len=config.seq_len,
        seed=config.calibration_seed,
    )
    logger.info(f"Calibration set: {len(calib_loader)} samples, seq_len={config.seq_len}")

    # Step 3: Initialize CAL loss
    loss_fn = ContrastiveAlignmentLoss(top_k=config.top_k, alpha=config.alpha)

    # Steps 4-5: Pre-compute M_PT logits on GPU, swap in M_FT, then optimize θ.
    # Transform parameters θ and the Adam optimizer are initialized inside
    # CAQTrainer.train() after M_FT is loaded, so the two 7B models never
    # occupy GPU memory simultaneously.
    logger.info("Starting CAL optimization (Algorithm 1)...")
    trainer = CAQTrainer(model_pair, loss_fn, config)
    train_stats = trainer.train(calib_loader)
    logger.info(f"Training complete: {train_stats}")

    # Step 6: Fuse θ into M_FT and release M_PT (Algorithm 1, line 13 prep)
    logger.info("Finalizing: fusing transformations and releasing M_PT...")
    fused_model = trainer.finalize()
    clear_memory()

    # Step 7: Apply GPTQ W4A4 quantization (Algorithm 1, line 13: M_Q ← Q(T_θ(M_FT)))
    logger.info(f"Applying GPTQ W{config.bits}A{config.act_bits} quantization...")
    quantizer = QuantizerWrapper(config)
    quantized_model = quantizer.quantize(
        fused_model,
        model_pair.tokenizer,
        calib_loader,
        args.output_dir,
    )

    # Step 8: Save quantized model in GPTQ format (INT4 weights)
    logger.info(f"Saving GPTQ-quantized model to {args.output_dir}...")
    quantized_model.save_quantized(args.output_dir)
    model_pair.tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved.")

    # Step 9 (optional): Evaluate WikiText-2 PPL
    results = {"train_stats": train_stats}
    if args.eval_ppl:
        logger.info("Evaluating WikiText-2 perplexity...")
        # Reload from the saved quantized checkpoint — gptqmodel offloads layers to
        # meta device during quantization, so the in-memory model cannot run inference.
        # from_quantized() gives a fully-initialised model ready for forward passes.
        from gptqmodel import GPTQModel
        logger.info(f"Reloading quantized model from {args.output_dir} for PPL eval...")
        eval_model = GPTQModel.from_quantized(args.output_dir, device="cuda:0")
        # Re-register A4 activation hooks on the freshly loaded model
        quantizer._register_activation_hooks(eval_model.model)
        test_loader = get_wikitext2_test_loader(
            model_pair.tokenizer, seq_len=config.seq_len
        )
        ppl = compute_perplexity(
            eval_model.model,
            model_pair.tokenizer,
            test_loader,
            device="cuda:0",
        )
        results["perplexity_wikitext2"] = ppl
        logger.info(f"WikiText-2 PPL: {ppl:.2f}")
        print(f"\nWikiText-2 Perplexity: {ppl:.2f}")

    # Save results summary
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    print(f"\nCAQ quantization complete. Output: {args.output_dir}")
    print(f"Run SafetyBench evaluation:")
    print(f"  python evaluate_safetybench.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()
