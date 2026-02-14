#!/usr/bin/env python3
"""
Zen Coder 4B Training Script
Fine-tune Qwen3-4B-Instruct on zen-agentic-dataset using MLX

Hardware: M1 Max 64GB
Expected time: ~2 days for 3.35M samples
"""

import os
import sys
from pathlib import Path

# Add zen_trainer to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("Zen Coder 4B Training - MLX Backend")
    print("=" * 60)

    # Check if running on Apple Silicon
    import platform
    if platform.processor() != 'arm':
        print("WARNING: Not running on Apple Silicon, MLX may not work")

    # Configuration
    MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    DATASET_PATH = "/Users/z/work/zen/zen-agentic-dataset"
    OUTPUT_DIR = "/Users/z/work/zen/zen-trainer/output/zen-coder-4b"

    # Training params from models.py
    LORA_R = 64
    LORA_ALPHA = 128
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    EPOCHS = 2
    MAX_SEQ_LENGTH = 4096

    print(f"\nModel: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}, Max seq length: {MAX_SEQ_LENGTH}")

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import mlx.core as mx
        from importlib.metadata import version
        mlx_version = version("mlx")
        print(f"\nMLX version: {mlx_version}")
        print(f"Metal available: {mx.metal.is_available()}")
    except ImportError:
        print("\nMLX not available, falling back to transformers...")
        return train_with_transformers(
            MODEL_ID, DATASET_PATH, OUTPUT_DIR,
            LORA_R, LORA_ALPHA, BATCH_SIZE, LEARNING_RATE, EPOCHS, MAX_SEQ_LENGTH
        )

    # MLX training
    return train_with_mlx(
        MODEL_ID, DATASET_PATH, OUTPUT_DIR,
        LORA_R, LORA_ALPHA, BATCH_SIZE, LEARNING_RATE, EPOCHS, MAX_SEQ_LENGTH
    )


def train_with_mlx(model_id, dataset_path, output_dir, lora_r, lora_alpha,
                   batch_size, lr, epochs, max_seq_length):
    """Train using MLX (Apple Silicon optimized)."""
    from mlx_lm import load
    from mlx_lm.tuner import train as mlx_train
    from mlx_lm.tuner.trainer import TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
    import mlx.core as mx
    import mlx.optimizers as optim
    import json

    print("\n" + "=" * 60)
    print("Loading model with MLX...")
    print("=" * 60)

    model, tokenizer = load(model_id)

    # Convert to LoRA
    print(f"\nApplying LoRA (r={lora_r}, alpha={lora_alpha})...")
    lora_config = {
        "rank": lora_r,
        "alpha": lora_alpha,
        "dropout": 0.05,
        "scale": lora_alpha / lora_r,
    }
    # Apply LoRA to last 32 layers (for 4B model)
    linear_to_lora_layers(model, num_layers=32, config=lora_config)

    # Count trainable params
    trainable = sum(p.size for n, p in model.trainable_parameters().items())
    total = sum(p.size for n, p in model.parameters().items())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    train_data = []
    train_file = Path(dataset_path) / "train.jsonl"

    # Load samples (limit for initial test)
    max_samples = 10000  # Start with 10K for testing, then scale up
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line)
                # Format for chat
                if 'messages' in item:
                    text = tokenizer.apply_chat_template(
                        item['messages'],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                elif 'content' in item:
                    text = item['content']
                else:
                    text = str(item)
                train_data.append({"text": text})
            except:
                continue

    print(f"Loaded {len(train_data)} training samples")

    # Save formatted data
    formatted_path = Path(output_dir) / "train_formatted.jsonl"
    with open(formatted_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    # Create optimizer
    num_iters = len(train_data) // batch_size * epochs
    optimizer = optim.AdamW(learning_rate=lr)

    # Training args
    args = TrainingArgs(
        batch_size=batch_size,
        iters=num_iters,
        steps_per_report=10,
        steps_per_eval=100,
        steps_per_save=500,
        max_seq_length=max_seq_length,
        adapter_file=os.path.join(output_dir, "adapters.safetensors"),
        grad_checkpoint=True,
    )

    print("\n" + "=" * 60)
    print("Starting MLX training...")
    print(f"Total iterations: {args.iters}")
    print(f"Batch size: {batch_size}, LR: {lr}")
    print("=" * 60)

    # Load datasets for train function
    from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset
    train_set = mlx_load_dataset(str(formatted_path), tokenizer)

    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=train_set,  # Use same for validation initially
        args=args,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete! Model saved to {output_dir}")
    return 0


def train_with_transformers(model_id, dataset_path, output_dir, lora_r, lora_alpha,
                            batch_size, lr, epochs, max_seq_length):
    """Fallback training using transformers + PEFT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Loading model with transformers...")
    print("=" * 60)

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device != "mps" else None,
        trust_remote_code=True,
    )

    if device == "mps":
        model = model.to(device)

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_dataset(
        "json",
        data_files={"train": f"{dataset_path}/train.jsonl"},
        split="train",
        streaming=True,  # Stream large dataset
    ).take(10000)  # Start with 10K for testing

    def format_func(examples):
        texts = []
        for item in examples:
            if 'messages' in item:
                text = tokenizer.apply_chat_template(
                    item['messages'],
                    tokenize=False,
                    add_generation_prompt=False
                )
            elif 'content' in item:
                text = item['content']
            else:
                text = str(item)
            texts.append(text)
        return {"text": texts}

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="tensorboard",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=list(dataset),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete! Model saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
