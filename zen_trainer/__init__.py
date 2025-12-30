"""
Zen Trainer - Fine-tuning framework for Zen Coder models

Supports fine-tuning 5 model architectures:
  Tier 1: Qwen3 4B        (lightweight,   8GB VRAM)    $326
  Tier 2: Devstral 24B    (mid-range,    24GB VRAM)    $814
  Tier 3: Devstral 123B   (large,       128GB VRAM)   $2,171
  Tier 4: GLM-4.7 358B    (MoE,         180GB VRAM)   $4,071
  Tier 5: Kimi K2 1T      (MoE,         400GB VRAM)   $10,856

Costs based on 8xH200 @ $35/hr, 3.35M training samples.
Local training on Mac Studio 512GB is FREE (except ULTRA).

Backends:
  - MLX (Apple Silicon, local) - Tier 1-2, 4 (MAX)
  - Unsloth/PyTorch (NVIDIA GPUs, 2x faster) - Tier 1-4
  - DeepSpeed ZeRO-3 (Multi-GPU clusters) - All tiers
  - HuggingFace Autotrain (Cloud) - All tiers

Benchmarks (12 ARC from GLM-4.5):
  Agentic: TAU-Bench, BFCL V3, BrowseComp
  Reasoning: MMLU-Pro, AIME-24, MATH-500, SciCode, GPQA, HLE, LiveCodeBench
  Coding: SWE-bench Verified, Terminal-Bench

Usage:
    from zen_trainer import ZenTrainer, ZEN_MODELS, ZenBenchmark

    # Train
    trainer = ZenTrainer(
        model_key="qwen3-4b",
        dataset_path="hanzoai/zen-agentic-dataset",
        output_dir="./output/zen-coder-4b",
    )
    trainer.train()

    # Benchmark
    bench = ZenBenchmark(
        model_path="./output/zen-coder-4b",
        model_key="qwen3-4b",
    )
    bench.run_all()
    bench.compare_to_baseline()
"""

__version__ = "0.1.0"

from .models import ZEN_MODELS, get_model_config, estimate_training_cost, COST_SUMMARY
from .trainer import ZenTrainer, train_all_models
from .benchmark import ZenBenchmark, benchmark_all_models, generate_leaderboard

__all__ = [
    # Models
    "ZEN_MODELS",
    "get_model_config",
    "estimate_training_cost",
    "COST_SUMMARY",
    # Training
    "ZenTrainer",
    "train_all_models",
    # Benchmarks
    "ZenBenchmark",
    "benchmark_all_models",
    "generate_leaderboard",
]


def main():
    """CLI entry point for zen-train command."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Zen Trainer - Fine-tune Zen Coder models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zen-train --model qwen3-4b --dataset ./data --output ./output
  zen-train --model devstral-24b --dataset hanzoai/zen-agentic-dataset
  zen-train --list-models
  zen-train --estimate glm47-358b --samples 100000
        """,
    )
    parser.add_argument("--model", "-m", type=str, help="Model key to train")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset path (local or HuggingFace)")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory")
    parser.add_argument("--backend", "-b", type=str, default="auto",
                        choices=["auto", "mlx", "unsloth", "deepspeed"],
                        help="Training backend")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--estimate", type=str, help="Estimate cost for model key")
    parser.add_argument("--samples", type=int, default=10000, help="Sample count for estimate")
    parser.add_argument("--costs", action="store_true", help="Show training cost summary")

    args = parser.parse_args()

    if args.costs:
        print(COST_SUMMARY)
        return 0

    if args.list_models:
        print("Available models:")
        for key, cfg in ZEN_MODELS.items():
            print(f"  {key:20} - {cfg.name} ({cfg.size_b}B, {cfg.vram_qlora}GB VRAM)")
        return 0

    if args.estimate:
        est = estimate_training_cost(args.estimate, args.samples)
        print(f"\nTraining estimate for {est['model']}:")
        print(f"  Samples: {est['samples']:,}")
        print(f"  Hours:   {est['hours_estimate'][0]}-{est['hours_estimate'][1]}")
        print(f"  Cost:    ${est['cost_estimate_usd'][0]:,}-${est['cost_estimate_usd'][1]:,}")
        print(f"  Platform: {est['recommended_platform']}")
        return 0

    if not args.model or not args.dataset:
        parser.print_help()
        print("\nError: --model and --dataset are required for training")
        return 1

    # Start training
    trainer = ZenTrainer(
        model_key=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        backend=args.backend,
        epochs=args.epochs,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
