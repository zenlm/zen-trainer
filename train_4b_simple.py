#!/usr/bin/env python3
"""
Zen Coder 4B Training - Simple MLX LoRA
Using mlx_lm.lora directly for simplicity
"""

import os
import json
import subprocess
from pathlib import Path

# Configuration
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_PATH = "/Users/z/work/zen/zen-agentic-dataset"
OUTPUT_DIR = "/Users/z/work/zen/zen-trainer/output/zen-coder-4b"

# Training params
LORA_R = 64
LORA_ALPHA = 128
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048  # Reduced for memory
ITERS = 5000  # 10K samples, batch 4, 2 epochs = 5000 iters

def prepare_dataset():
    """Prepare training data in mlx_lm format."""
    print("Preparing dataset...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = []
    valid_data = []
    train_file = Path(DATASET_PATH) / "train.jsonl"
    max_samples = 10000  # Start with 10K

    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line)
                # mlx_lm expects either 'text' or 'messages' format
                if 'messages' in item:
                    # Keep messages format for chat models
                    train_data.append(item)
                elif 'content' in item:
                    train_data.append({"text": item['content']})
            except:
                continue

    print(f"Loaded {len(train_data)} samples")

    # Split train/valid (90/10)
    split_idx = int(len(train_data) * 0.9)
    valid_data = train_data[split_idx:]
    train_data = train_data[:split_idx]

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Save in mlx_lm format
    train_output = Path(OUTPUT_DIR) / "train.jsonl"
    valid_output = Path(OUTPUT_DIR) / "valid.jsonl"

    with open(train_output, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(valid_output, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')

    return OUTPUT_DIR

def main():
    print("=" * 60)
    print("Zen Coder 4B Training - MLX LoRA")
    print("=" * 60)

    # Prepare data
    data_dir = prepare_dataset()

    # Build command using new mlx_lm syntax
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", MODEL_ID,
        "--data", data_dir,
        "--train",
        "--batch-size", str(BATCH_SIZE),
        "--num-layers", "32",  # Number of layers to fine-tune
        "--learning-rate", str(LEARNING_RATE),
        "--iters", str(ITERS),
        "--steps-per-report", "10",
        "--steps-per-eval", "100",
        "--save-every", "500",
        "--adapter-path", OUTPUT_DIR,
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--grad-checkpoint",
    ]

    print("\nRunning:", " ".join(cmd))
    print("=" * 60)

    # Run training
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
