# Zen Trainer

**Training framework for Zen Coder models** - fine-tune 4B to 1T parameter models on agentic coding data.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Supported Models

| Model | Base | Size | VRAM (QLoRA) | Context | License |
|-------|------|------|--------------|---------|---------|
| **Zen Coder 4B** | Zen MoDE 4B | 4B | 8 GB | 32K | Apache 2.0 |
| **Zen Coder 24B** | Zen MoDE 24B | 24B | 24 GB | 256K | Apache 2.0 |
| **Zen Coder 123B** | Zen MoDE 123B | 123B | 128 GB | 256K | Apache 2.0 |
| **Zen Coder MAX** | Zen MoDE MAX (358B MoE) | 358B | 180 GB | 200K | Apache 2.0 |
| **Zen Coder ULTRA** | Zen MoDE Ultra | 1T | 400 GB | 128K | MIT |

## Installation

```bash
# Basic installation
pip install zen-trainer

# With MLX support (Apple Silicon)
pip install zen-trainer[mlx]

# With Unsloth (2x faster NVIDIA training)
pip install zen-trainer[unsloth]

# With DeepSpeed (multi-GPU)
pip install zen-trainer[deepspeed]

# With evaluation suite
pip install zen-trainer[eval]

# Everything
pip install zen-trainer[all]
```

## Quick Start

### Training

```python
from zen_trainer import ZenTrainer

# Train Zen Coder 4B on your data
trainer = ZenTrainer(
    model_key="zen-coder-4b",
    dataset_path="path/to/your/dataset",
    output_dir="./output/zen-coder-4b",
)
trainer.train()
```

### Benchmarking

```python
from zen_trainer import ZenBenchmark

# Benchmark against SoTA models
bench = ZenBenchmark(
    model_path="./output/zen-coder-4b",
    model_key="zen-coder-4b",
)
results = bench.run_all()
bench.compare_to_baseline()
```

### Command Line

```bash
# Train
zen-train --model zen-coder-4b --dataset ./data --output ./output

# Benchmark
zen-benchmark --model ./output/zen-coder-4b --suite all
```

## Training Costs

For 3.35M samples (8.47B tokens) on 8xH200 @ $35/hr:

| Model | Cloud Hours | Cloud Cost | Local (Mac Studio 512GB) |
|-------|-------------|------------|--------------------------|
| Zen Coder 4B | 9h | $326 | 2 days (FREE) |
| Zen Coder 24B | 23h | $814 | 5 days (FREE) |
| Zen Coder 123B | 62h | $2,171 | 13 days (FREE) |
| Zen Coder MAX | 116h | $4,071 | 19 days (FREE) |
| Zen Coder ULTRA | 310h | $10,856 | Too large |

✓ = Fits in 128GB (M3 Ultra / single GPU node)
◆ = Fits Mac Studio 512GB or 8xH200

## Backends

The trainer automatically selects the best backend:

| Backend | Hardware | Speed | Models |
|---------|----------|-------|--------|
| **MLX** | Apple Silicon | 1x | 4B, 24B, MAX |
| **Unsloth** | NVIDIA GPU | 2x | 4B, 24B, 123B, MAX |
| **DeepSpeed** | Multi-GPU | 1x | All (required for ULTRA) |

## 12 ARC Benchmarks

Evaluation suite:

**Agentic:**
- TAU-Bench (tool-agent-user interaction)
- BFCL V3 (Berkeley Function Call Leaderboard)
- BrowseComp (web browsing agent)

**Reasoning:**
- MMLU-Pro, AIME-24, MATH-500, SciCode
- GPQA, HLE (Humanity's Last Exam)
- LiveCodeBench

**Coding:**
- SWE-bench Verified (real GitHub issues)
- Terminal-Bench (terminal environment tasks)

## Dataset

Models are designed for the [Zen Agentic Dataset](https://huggingface.co/datasets/hanzoai/zen-agentic-dataset):

| Metric | Value |
|--------|-------|
| Total Tokens | 8.47 billion |
| Training Samples | 3.35 million |
| Validation Samples | 100,000 |
| Size | ~27 GB |
| Time Span | 15 years (2010-2025) |

### Data Composition
- **29%** Claude Code debug sessions (real agentic programming)
- **23%** Claude conversations and interactions
- **48%** Git history (commits, diffs, source files)

## Model Architecture

### Hyperparameters

| Model | LoRA r | LoRA α | Batch | LR | Epochs |
|-------|--------|--------|-------|-----|--------|
| 4B | 64 | 128 | 4 | 2e-4 | 2 |
| 24B | 32 | 64 | 2 | 1e-4 | 2 |
| 123B | 16 | 32 | 1 | 5e-5 | 1 |
| MAX | 16 | 32 | 1 | 5e-6 | 1 |
| ULTRA | 8 | 16 | 1 | 1e-6 | 1 |

## API Reference

### ZenTrainer

```python
trainer = ZenTrainer(
    model_key: str,              # Model identifier (zen-coder-4b, zen-coder-24b, etc.)
    dataset_path: str,           # HuggingFace dataset or local path
    output_dir: str,             # Output directory for checkpoints
    backend: str = "auto",       # mlx, unsloth, deepspeed, or auto
    epochs: int = None,          # Override default epochs
    batch_size: int = None,      # Override default batch size
    learning_rate: float = None, # Override default learning rate
)

trainer.train()           # Start training
trainer.save_model()      # Save final checkpoint
trainer.push_to_hub(repo) # Push to HuggingFace
```

### ZenBenchmark

```python
bench = ZenBenchmark(
    model_path: str,             # Path to model checkpoint
    model_key: str,              # Model identifier for config
    benchmarks: list = None,     # Specific benchmarks or None for all
)

results = bench.run_all()        # Run all 12 ARC benchmarks
bench.run_agentic()              # TAU-Bench, BFCL V3, BrowseComp
bench.run_reasoning()            # MMLU, AIME, MATH, etc.
bench.run_coding()               # SWE-bench, Terminal-Bench
bench.compare_to_baseline()      # Compare to SoTA models
```

### Model Configurations

```python
from zen_trainer import get_model_config, list_models_by_vram, estimate_training_cost

# Get model config
cfg = get_model_config("zen-coder-max")
print(cfg.vram_qlora)  # 180 GB

# List models for your hardware
models = list_models_by_vram(128)  # Models that fit 128GB

# Estimate training cost
cost = estimate_training_cost("zen-coder-24b", num_samples=100000)
print(cost)  # {'hours_estimate': (2.0, 4.0), 'cost_estimate_usd': (70, 140), ...}
```

## Related Projects

- [Zen Agentic Dataset](https://huggingface.co/datasets/hanzoai/zen-agentic-dataset) - Training data
- [Zen Coder Models](https://huggingface.co/zenlm) - Fine-tuned models
- Evaluation toolkit
- [Hanzo MCP](https://github.com/hanzoai/mcp) - Model Context Protocol tools
- [Hanzo AI](https://hanzo.ai) - AI infrastructure platform

## Citation

```bibtex
@software{zen_trainer,
  author = {Kelling, Zach},
  title = {Zen Trainer: Fine-tuning Framework for Agentic Coding Models},
  year = {2025},
  publisher = {Zoo Labs Foundation},
  url = {https://github.com/zenlm/zen-trainer}
}
```

## License

Apache 2.0

---

**Maintainer:** z@hanzo.ai
