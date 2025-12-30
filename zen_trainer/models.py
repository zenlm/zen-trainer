"""
Zen Coder Model Configurations

5 supported architectures with VRAM requirements and training cost estimates.
Costs based on 8xH200 HGX ($35/hr) with ~1.5TB total VRAM.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a model architecture."""
    name: str
    hf_id: str
    size_b: float  # Billions of parameters
    architecture: str  # dense, moe
    vram_full: int  # GB for full precision
    vram_qlora: int  # GB for QLoRA (4-bit)
    max_seq_length: int
    context_window: int
    license: str

    # Training settings
    lora_r: int
    lora_alpha: int
    batch_size: int
    grad_accum: int
    learning_rate: float

    # Cost estimates (8xH200 @ $35/hr, ~10K samples)
    hours_estimate: tuple  # (min, max)
    cost_estimate: tuple  # (min, max) USD

    # Backend support
    supports_mlx: bool
    supports_unsloth: bool
    supports_deepspeed: bool

    # Quantization
    gguf_url: Optional[str] = None
    fits_128gb: bool = False


# All 5 Zen Coder model configurations
ZEN_MODELS: Dict[str, ModelConfig] = {
    # =========================================
    # Tier 1: Lightweight (Local Development)
    # =========================================
    "qwen3-4b": ModelConfig(
        name="Zen Coder 4B (Qwen3)",
        hf_id="Qwen/Qwen3-4B-Instruct-2507",
        size_b=4.0,
        architecture="dense",
        vram_full=16,
        vram_qlora=8,
        max_seq_length=4096,
        context_window=32768,
        license="Apache 2.0",

        lora_r=64,
        lora_alpha=128,
        batch_size=4,
        grad_accum=4,
        learning_rate=2e-4,

        hours_estimate=(0.5, 1.5),
        cost_estimate=(17, 52),  # $35/hr × 0.5-1.5h

        supports_mlx=True,
        supports_unsloth=True,
        supports_deepspeed=True,
        fits_128gb=True,
    ),

    # =========================================
    # Tier 2: Mid-Range (Production)
    # =========================================
    "devstral-24b": ModelConfig(
        name="Zen Coder 24B (Devstral Small 2)",
        hf_id="mistralai/Devstral-Small-2-24B-Instruct-2512",
        size_b=24.0,
        architecture="dense",
        vram_full=96,
        vram_qlora=24,
        max_seq_length=8192,
        context_window=262144,  # 256K context!
        license="Apache 2.0",

        lora_r=32,
        lora_alpha=64,
        batch_size=2,
        grad_accum=8,
        learning_rate=1e-4,

        hours_estimate=(2, 4),
        cost_estimate=(70, 140),  # $35/hr × 2-4h

        supports_mlx=True,  # With quantization
        supports_unsloth=True,
        supports_deepspeed=True,
        fits_128gb=True,
    ),

    # =========================================
    # Tier 3: Large (High-Performance)
    # =========================================
    "devstral-123b": ModelConfig(
        name="Zen Coder 123B (Devstral 2)",
        hf_id="mistralai/Devstral-2-123B-Instruct-2512",
        size_b=123.0,
        architecture="dense",
        vram_full=492,
        vram_qlora=128,  # Fits in 128GB with 4-bit!
        max_seq_length=8192,
        context_window=262144,  # 256K context
        license="Mistral Research",

        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        grad_accum=16,
        learning_rate=5e-5,

        hours_estimate=(4, 8),
        cost_estimate=(140, 280),  # $35/hr × 4-8h

        supports_mlx=False,  # Too large
        supports_unsloth=True,
        supports_deepspeed=True,
        fits_128gb=True,  # QLoRA 4-bit fits!
        gguf_url="https://huggingface.co/unsloth/Devstral-2-123B-GGUF",
    ),

    # =========================================
    # Tier 4: MoE Large (Maximum Performance)
    # =========================================
    "glm47-358b": ModelConfig(
        name="Zen Coder MAX (GLM-4.7)",
        hf_id="zai-org/GLM-4.7",
        size_b=358.0,  # 358B total (MoE)
        architecture="moe",  # Mixture of Experts
        vram_full=716,  # 358B × 2 bytes (BF16)
        vram_qlora=180,  # Q4 ~90GB + training overhead
        max_seq_length=8192,
        context_window=200000,  # 200K context, 128K output
        license="GLM-4 License",

        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        grad_accum=16,
        learning_rate=5e-6,

        hours_estimate=(6, 12),
        cost_estimate=(210, 420),  # $35/hr × 6-12h

        supports_mlx=True,  # Can train on Mac Studio 512GB!
        supports_unsloth=True,
        supports_deepspeed=True,
        fits_128gb=False,  # Needs ~180GB for QLoRA training
        gguf_url="https://huggingface.co/AaryanK/GLM-4.7-GGUF",
    ),

    # =========================================
    # Tier 5: MoE Ultra (Research/Frontier)
    # =========================================
    "kimi-k2-1t": ModelConfig(
        name="Zen Coder ULTRA (Kimi K2 Thinking)",
        hf_id="moonshotai/Kimi-K2-Instruct",
        size_b=1000.0,  # 1 Trillion params
        architecture="moe",  # MoE with 32 experts
        vram_full=2000,
        vram_qlora=400,  # Needs 8xH200 minimum
        max_seq_length=8192,
        context_window=131072,  # 128K context
        license="MIT",

        lora_r=8,
        lora_alpha=16,
        batch_size=1,
        grad_accum=32,
        learning_rate=1e-6,

        hours_estimate=(12, 24),
        cost_estimate=(420, 840),  # $35/hr × 12-24h

        supports_mlx=False,
        supports_unsloth=False,  # Too large
        supports_deepspeed=True,  # ZeRO-3 required
        fits_128gb=False,  # Needs multi-node
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a model by key."""
    if model_key not in ZEN_MODELS:
        available = ", ".join(ZEN_MODELS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return ZEN_MODELS[model_key]


def list_models_by_vram(max_vram_gb: int) -> list:
    """List models that fit within a VRAM budget (QLoRA)."""
    return [
        (key, cfg) for key, cfg in ZEN_MODELS.items()
        if cfg.vram_qlora <= max_vram_gb
    ]


def estimate_training_cost(
    model_key: str,
    num_samples: int = 10000,
    hourly_rate: float = 35.0,
) -> Dict[str, Any]:
    """
    Estimate training cost for a model.

    Args:
        model_key: Model identifier
        num_samples: Number of training samples
        hourly_rate: Cost per hour (default: $35 for 8xH200)

    Returns:
        Dict with cost estimates and recommendations
    """
    cfg = get_model_config(model_key)

    # Scale estimate by sample count (base is 10K)
    scale = num_samples / 10000
    hours_min = cfg.hours_estimate[0] * scale
    hours_max = cfg.hours_estimate[1] * scale

    cost_min = hours_min * hourly_rate
    cost_max = hours_max * hourly_rate

    return {
        "model": cfg.name,
        "size_b": cfg.size_b,
        "architecture": cfg.architecture,
        "samples": num_samples,
        "hours_estimate": (round(hours_min, 1), round(hours_max, 1)),
        "cost_estimate_usd": (round(cost_min), round(cost_max)),
        "hourly_rate": hourly_rate,
        "vram_required_gb": cfg.vram_qlora,
        "fits_128gb": cfg.fits_128gb,
        "recommended_platform": (
            "MLX (local)" if cfg.supports_mlx and cfg.vram_qlora <= 64
            else "8xH200 (cloud)" if cfg.vram_qlora <= 400
            else "Multi-node cluster"
        ),
    }


# Quick reference
COST_SUMMARY = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                 ZEN CODER TRAINING COSTS (8xH200 @ $35/hr)                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Model              │ Size   │ Weights │ VRAM (QLoRA) │ Hours  │ Cost (10K)  ║
╠═════════════════════╪════════╪═════════╪══════════════╪════════╪═════════════╣
║  Qwen3 4B           │   4B   │   8 GB  │     8 GB ✓   │ 0.5-1.5│   $17-52    ║
║  Devstral 24B       │  24B   │  48 GB  │    24 GB ✓   │   2-4  │   $70-140   ║
║  Devstral 123B      │ 123B   │ 246 GB  │   128 GB ✓   │   4-8  │  $140-280   ║
║  GLM-4.7 358B (MoE) │ 358B   │ 716 GB  │   180 GB ◆   │  6-12  │  $210-420   ║
║  Kimi K2 1T (MoE)   │   1T   │  ~2 TB  │   400 GB     │ 12-24  │  $420-840   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

✓ = Fits in 128GB (M3 Ultra / single GPU node)
◆ = Fits Mac Studio 512GB or 8xH200 (no multi-GPU required)
Weights = BF16 full precision. Q4 inference = Weights ÷ 4

Model Keys:
  qwen3-4b      - Tier 1: Local dev, M3 Max 64GB / RTX 4090
  devstral-24b  - Tier 2: Production, M3 Ultra 128GB / A100
  devstral-123b - Tier 3: High-perf, M3 Ultra 192GB / 8xH200
  glm47-358b    - Tier 4: Maximum (MoE), Mac Studio 512GB / 8xH200
  kimi-k2-1t    - Tier 5: Ultra (MoE), Multi-node cluster only

Benchmarks (12 ARC - Agentic, Reasoning, Coding):
  Eval toolkit: https://github.com/zai-org/glm-simple-evals
"""

if __name__ == "__main__":
    print(COST_SUMMARY)
    print("\nDetailed estimates for 3.35M samples:")
    for key in ZEN_MODELS:
        est = estimate_training_cost(key, num_samples=3_350_000)
        print(f"\n{est['model']}:")
        print(f"  Hours: {est['hours_estimate'][0]}-{est['hours_estimate'][1]}")
        print(f"  Cost:  ${est['cost_estimate_usd'][0]}-${est['cost_estimate_usd'][1]}")
        print(f"  Platform: {est['recommended_platform']}")
