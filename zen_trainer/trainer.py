"""
Zen Coder Unified Trainer

Supports multiple backends:
- MLX (Apple Silicon, local)
- Unsloth (NVIDIA GPUs, 2x faster)
- DeepSpeed (Multi-GPU clusters)
- HuggingFace Autotrain (Cloud)
"""

import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal
from abc import ABC, abstractmethod

from .models import get_model_config, ModelConfig, ZEN_MODELS


Backend = Literal["mlx", "unsloth", "deepspeed", "autotrain"]


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_key: str
    dataset_path: str  # Local path or HF dataset ID
    output_dir: str
    epochs: int = 2
    max_seq_length: Optional[int] = None  # Use model default
    batch_size: Optional[int] = None  # Use model default
    learning_rate: Optional[float] = None  # Use model default

    # LoRA settings (override model defaults if needed)
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05

    # Hardware
    num_gpus: int = 1
    use_4bit: bool = True  # QLoRA

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    report_to: str = "tensorboard"


class BaseTrainer(ABC):
    """Abstract base trainer."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_config = get_model_config(config.model_key)

    @abstractmethod
    def train(self):
        """Run training."""
        pass

    @abstractmethod
    def can_run(self) -> bool:
        """Check if this backend can run."""
        pass

    def get_effective_config(self) -> dict:
        """Get effective training config (model defaults + overrides)."""
        mc = self.model_config
        return {
            "model_id": mc.hf_id,
            "max_seq_length": self.config.max_seq_length or mc.max_seq_length,
            "batch_size": self.config.batch_size or mc.batch_size,
            "learning_rate": self.config.learning_rate or mc.learning_rate,
            "lora_r": self.config.lora_r or mc.lora_r,
            "lora_alpha": self.config.lora_alpha or mc.lora_alpha,
            "grad_accum": mc.grad_accum,
        }


class MLXTrainer(BaseTrainer):
    """MLX trainer for Apple Silicon."""

    def can_run(self) -> bool:
        """Check if MLX is available and model fits."""
        try:
            import mlx
            # Check if model supports MLX
            return self.model_config.supports_mlx
        except ImportError:
            return False

    def train(self):
        """Run MLX training."""
        if not self.can_run():
            raise RuntimeError(f"MLX not available or {self.config.model_key} too large for MLX")

        cfg = self.get_effective_config()
        print(f"Starting MLX training for {self.model_config.name}")
        print(f"Config: {json.dumps(cfg, indent=2)}")

        # Use mlx_lm for training
        cmd = [
            "mlx_lm.lora",
            "--model", cfg["model_id"],
            "--train",
            "--data", self.config.dataset_path,
            "--batch-size", str(cfg["batch_size"]),
            "--lora-layers", str(cfg["lora_r"]),
            "--iters", str(self.config.epochs * 1000),  # Approximate
            "--adapter-path", self.config.output_dir,
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


class UnslothTrainer(BaseTrainer):
    """Unsloth trainer for NVIDIA GPUs (2x faster)."""

    def can_run(self) -> bool:
        """Check if Unsloth is available."""
        try:
            import torch
            import unsloth
            return torch.cuda.is_available() and self.model_config.supports_unsloth
        except ImportError:
            return False

    def train(self):
        """Run Unsloth training."""
        if not self.can_run():
            raise RuntimeError("Unsloth not available or CUDA not found")

        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
        import torch

        cfg = self.get_effective_config()
        mc = self.model_config

        print(f"Starting Unsloth training for {mc.name}")
        print(f"Config: {json.dumps(cfg, indent=2)}")

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model_id"],
            max_seq_length=cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=self.config.use_4bit,
        )

        # Add LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        # Load dataset
        if self.config.dataset_path.startswith("hanzoai/"):
            dataset = load_dataset(self.config.dataset_path, split="train")
        else:
            dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")

        def format_func(examples):
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}

        dataset = dataset.map(format_func, batched=True)

        # Train
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            dataset_num_proc=4,
            packing=True,
            args=TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=cfg["batch_size"],
                gradient_accumulation_steps=cfg["grad_accum"],
                warmup_ratio=0.03,
                num_train_epochs=self.config.epochs,
                learning_rate=cfg["learning_rate"],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                save_total_limit=3,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                report_to=self.config.report_to,
            ),
        )

        trainer.train()
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")


class DeepSpeedTrainer(BaseTrainer):
    """DeepSpeed trainer for multi-GPU clusters."""

    def can_run(self) -> bool:
        """Check if DeepSpeed is available."""
        try:
            import torch
            import deepspeed
            return torch.cuda.is_available() and self.model_config.supports_deepspeed
        except ImportError:
            return False

    def generate_ds_config(self) -> dict:
        """Generate DeepSpeed ZeRO config."""
        mc = self.model_config

        # ZeRO-2 for smaller models, ZeRO-3 for 100B+
        zero_stage = 3 if mc.size_b >= 100 else 2

        config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": mc.grad_accum,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": mc.batch_size,
        }

        # Add CPU offload for very large models
        if mc.size_b >= 100:
            config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
            config["zero_optimization"]["offload_param"] = {"device": "cpu"}

        return config

    def train(self):
        """Run DeepSpeed training."""
        if not self.can_run():
            raise RuntimeError("DeepSpeed not available")

        cfg = self.get_effective_config()
        ds_config = self.generate_ds_config()

        # Save DS config
        ds_config_path = Path(self.config.output_dir) / "ds_config.json"
        ds_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)

        print(f"DeepSpeed config saved to {ds_config_path}")
        print(f"Run with: deepspeed --num_gpus={self.config.num_gpus} train.py --deepspeed {ds_config_path}")

        # The actual training would use HF Trainer with DeepSpeed
        # This is typically launched via deepspeed CLI


class ZenTrainer:
    """
    Unified trainer that auto-selects the best backend.

    Usage:
        trainer = ZenTrainer(
            model_key="qwen3-4b",
            dataset_path="hanzoai/zen-agentic-dataset",
            output_dir="./output/zen-coder-4b",
        )
        trainer.train()  # Auto-selects MLX, Unsloth, or DeepSpeed
    """

    BACKEND_PRIORITY = ["unsloth", "deepspeed", "mlx"]

    def __init__(
        self,
        model_key: str,
        dataset_path: str,
        output_dir: str,
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        self.config = TrainingConfig(
            model_key=model_key,
            dataset_path=dataset_path,
            output_dir=output_dir,
            **kwargs,
        )
        self.model_config = get_model_config(model_key)
        self._backend = backend
        self._trainer = None

    def _select_backend(self) -> BaseTrainer:
        """Auto-select the best available backend."""
        if self._backend:
            return self._get_trainer_for_backend(self._backend)

        for backend in self.BACKEND_PRIORITY:
            try:
                trainer = self._get_trainer_for_backend(backend)
                if trainer.can_run():
                    print(f"Selected backend: {backend}")
                    return trainer
            except Exception:
                continue

        raise RuntimeError("No suitable training backend found")

    def _get_trainer_for_backend(self, backend: Backend) -> BaseTrainer:
        """Get trainer instance for a backend."""
        trainers = {
            "mlx": MLXTrainer,
            "unsloth": UnslothTrainer,
            "deepspeed": DeepSpeedTrainer,
        }
        if backend not in trainers:
            raise ValueError(f"Unknown backend: {backend}")
        return trainers[backend](self.config)

    def train(self):
        """Run training with auto-selected backend."""
        if not self._trainer:
            self._trainer = self._select_backend()

        print(f"\n{'='*60}")
        print(f"ZEN CODER TRAINING: {self.model_config.name}")
        print(f"{'='*60}")
        print(f"Model: {self.model_config.hf_id}")
        print(f"Size: {self.model_config.size_b}B parameters")
        print(f"VRAM (QLoRA): {self.model_config.vram_qlora}GB")
        print(f"Dataset: {self.config.dataset_path}")
        print(f"Output: {self.config.output_dir}")
        print(f"{'='*60}\n")

        self._trainer.train()

    def estimate_cost(self, num_samples: int = 10000, hourly_rate: float = 35.0) -> dict:
        """Estimate training cost."""
        from .models import estimate_training_cost
        return estimate_training_cost(
            self.config.model_key,
            num_samples=num_samples,
            hourly_rate=hourly_rate,
        )


def train_all_models(
    dataset_path: str = "hanzoai/zen-agentic-dataset",
    output_base: str = "./output",
    models: Optional[list] = None,
):
    """
    Train all Zen Coder models (or a subset).

    Args:
        dataset_path: HF dataset or local path
        output_base: Base directory for outputs
        models: List of model keys to train (default: all that fit hardware)
    """
    models = models or ["qwen3-4b", "devstral-24b", "devstral-123b", "glm47-358b"]

    for model_key in models:
        mc = get_model_config(model_key)
        output_dir = f"{output_base}/zen-coder-{model_key}"

        print(f"\n{'#'*60}")
        print(f"# Training: {mc.name}")
        print(f"{'#'*60}")

        try:
            trainer = ZenTrainer(
                model_key=model_key,
                dataset_path=dataset_path,
                output_dir=output_dir,
            )
            trainer.train()
        except Exception as e:
            print(f"Failed to train {model_key}: {e}")
            continue
