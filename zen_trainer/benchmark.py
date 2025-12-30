"""
Zen Coder Benchmark Suite

Evaluates fine-tuned models against SoTA coding benchmarks.
Based on GLM-4.5's ARC (Agentic, Reasoning, Coding) methodology.

12 ARC Benchmarks (from GLM-4.5 paper):
  Agentic (3):
    - TAU-Bench (tool-agent-user interaction)
    - BFCL V3 (Berkeley Function Call Leaderboard)
    - BrowseComp (web browsing agent)

  Reasoning (7):
    - MMLU-Pro (multi-task language understanding)
    - AIME 24 (American Invitational Mathematics Exam)
    - MATH-500 (mathematical problem solving)
    - SciCode (scientific coding)
    - GPQA (graduate-level QA)
    - HLE (Humanity's Last Exam)
    - LiveCodeBench (competitive coding)

  Coding (2):
    - SWE-bench Verified (real-world software engineering)
    - Terminal-Bench (terminal environment tasks)

Official eval toolkit: https://github.com/zai-org/glm-simple-evals
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime

from .models import get_model_config, ZEN_MODELS


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    benchmark: str
    model: str
    score: float
    max_score: float
    pass_rate: float
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvaluation:
    """Complete evaluation of a model."""
    model_key: str
    model_name: str
    base_model: str
    fine_tuned: bool
    results: List[BenchmarkResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Get summary scores."""
        return {r.benchmark: r.pass_rate for r in self.results}

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "base_model": self.base_model,
            "fine_tuned": self.fine_tuned,
            "results": [
                {
                    "benchmark": r.benchmark,
                    "score": r.score,
                    "max_score": r.max_score,
                    "pass_rate": r.pass_rate,
                    "timestamp": r.timestamp,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class ZenBenchmark:
    """
    Unified benchmark suite for Zen Coder models.

    Usage:
        bench = ZenBenchmark(
            model_path="./output/zen-coder-4b",
            model_key="qwen3-4b",
        )
        results = bench.run_all()
        bench.compare_to_baseline()
    """

    # SoTA baseline scores (Aug 2025) - from GLM-4.5 paper
    # Source: https://github.com/zai-org/glm-simple-evals
    SOTA_BASELINES = {
        # =====================
        # AGENTIC BENCHMARKS
        # =====================
        "tau-bench-retail": {
            "claude-opus-4": 81.4,
            "claude-sonnet-4": 80.5,
            "glm-4.5": 79.7,
            "glm-4.5-air": 77.9,
            "gemini-2.5-pro": 77.0,
            "grok-4": 76.5,
            "gpt-4.1": 75.1,
            "kimi-k2": 73.9,
            "o3": 70.4,
        },
        "tau-bench-airline": {
            "glm-4.5-air": 60.8,
            "glm-4.5": 60.4,
            "claude-sonnet-4": 60.0,
            "claude-opus-4": 59.6,
            "grok-4": 58.4,
            "o3": 52.0,
            "kimi-k2": 51.2,
            "o4-mini": 49.2,
            "gpt-4.1": 48.8,
            "gemini-2.5-pro": 48.0,
        },
        "bfcl-v3": {
            "glm-4.5": 77.8,
            "glm-4.5-air": 76.4,
            "claude-sonnet-4": 75.2,
            "claude-opus-4": 74.4,
            "o3": 72.4,
            "kimi-k2": 71.1,
            "gpt-4.1": 68.9,
            "o4-mini": 67.2,
            "grok-4": 66.2,
            "gemini-2.5-pro": 61.2,
        },
        "browsecomp": {
            "o3": 49.7,
            "grok-4": 32.6,
            "o4-mini": 28.3,
            "glm-4.5": 26.4,
            "glm-4.5-air": 21.3,
            "claude-opus-4": 18.8,
            "claude-sonnet-4": 14.7,
            "kimi-k2": 7.9,
            "gemini-2.5-pro": 7.6,
            "gpt-4.1": 4.1,
        },
        # =====================
        # REASONING BENCHMARKS
        # =====================
        "mmlu-pro": {
            "claude-opus-4": 87.3,
            "gemini-2.5-pro": 86.2,
            "grok-4": 86.6,
            "o3": 85.3,
            "deepseek-r1": 84.9,
            "glm-4.5": 84.6,
            "qwen3-235b": 84.5,
            "glm-4.5-air": 81.4,
        },
        "aime-24": {
            "grok-4": 94.3,
            "qwen3-235b": 94.1,
            "glm-4.5": 91.0,
            "o3": 90.3,
            "glm-4.5-air": 89.4,
            "deepseek-r1": 89.3,
            "gemini-2.5-pro": 88.7,
            "claude-opus-4": 75.7,
        },
        "math-500": {
            "o3": 99.2,
            "grok-4": 99.0,
            "deepseek-r1": 98.3,
            "glm-4.5": 98.2,
            "claude-opus-4": 98.2,
            "glm-4.5-air": 98.1,
            "qwen3-235b": 98.0,
            "gemini-2.5-pro": 96.7,
        },
        "scicode": {
            "grok-4": 45.7,
            "qwen3-235b": 42.9,
            "gemini-2.5-pro": 42.8,
            "glm-4.5": 41.7,
            "o3": 41.0,
            "deepseek-r1": 40.3,
            "claude-opus-4": 39.8,
            "glm-4.5-air": 37.3,
        },
        "gpqa": {
            "grok-4": 87.7,
            "gemini-2.5-pro": 84.4,
            "o3": 82.7,
            "deepseek-r1": 81.3,
            "qwen3-235b": 81.1,
            "claude-opus-4": 79.6,
            "glm-4.5": 79.1,
            "glm-4.5-air": 75.0,
        },
        "hle": {  # Humanity's Last Exam
            "grok-4": 23.9,
            "gemini-2.5-pro": 21.1,
            "o3": 20.0,
            "qwen3-235b": 15.8,
            "deepseek-r1": 14.9,
            "glm-4.5": 14.4,
            "claude-opus-4": 11.7,
            "glm-4.5-air": 10.6,
        },
        "livecodebench": {  # 2407-2501
            "grok-4": 81.9,
            "gemini-2.5-pro": 80.1,
            "o3": 78.4,
            "qwen3-235b": 78.2,
            "deepseek-r1": 77.0,
            "glm-4.5": 72.9,
            "glm-4.5-air": 70.7,
            "claude-opus-4": 63.6,
        },
        # =====================
        # CODING BENCHMARKS
        # =====================
        "swe-bench-verified": {
            "claude-sonnet-4": 70.4,
            "o3": 69.1,
            "claude-opus-4": 67.8,
            "kimi-k2": 65.4,
            "glm-4.5": 64.2,
            "glm-4.5-air": 57.6,
            "gemini-2.5-pro": 49.0,
            "gpt-4.1": 48.6,
            "deepseek-r1": 41.4,
        },
        "terminal-bench": {
            "claude-opus-4": 43.2,
            "glm-4.5": 37.5,
            "claude-sonnet-4": 35.5,
            "gpt-4.1": 30.3,
            "o3": 30.2,
            "glm-4.5-air": 30.0,
            "gemini-2.5-pro": 25.3,
            "kimi-k2": 25.0,
            "deepseek-r1": 17.5,
        },
    }

    def __init__(
        self,
        model_path: str,
        model_key: str,
        base_model: Optional[str] = None,
        output_dir: str = "./benchmark_results",
    ):
        self.model_path = model_path
        self.model_key = model_key
        self.model_config = get_model_config(model_key)
        self.base_model = base_model or self.model_config.hf_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation = ModelEvaluation(
            model_key=model_key,
            model_name=self.model_config.name,
            base_model=self.base_model,
            fine_tuned=True,
        )

    def run_humaneval(self) -> BenchmarkResult:
        """
        Run HumanEval benchmark.

        Install: pip install human-eval
        Repo: https://github.com/openai/human-eval
        """
        print("\n📊 Running HumanEval...")

        # This would use the actual evaluation code
        # For now, return placeholder that shows the interface
        try:
            # In real implementation:
            # from human_eval.evaluation import evaluate_functional_correctness
            # results = evaluate_functional_correctness(sample_file)

            # Placeholder for demonstration
            result = BenchmarkResult(
                benchmark="humaneval",
                model=self.model_key,
                score=0,
                max_score=164,
                pass_rate=0.0,
                timestamp=datetime.now().isoformat(),
                details={"status": "not_run", "requires": "human-eval package"},
            )
        except Exception as e:
            result = BenchmarkResult(
                benchmark="humaneval",
                model=self.model_key,
                score=0,
                max_score=164,
                pass_rate=0.0,
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)},
            )

        self.evaluation.results.append(result)
        return result

    def run_livecodebench(self) -> BenchmarkResult:
        """
        Run LiveCodeBench V6.

        Repo: https://github.com/LiveCodeBench/LiveCodeBench
        """
        print("\n📊 Running LiveCodeBench V6...")

        result = BenchmarkResult(
            benchmark="livecodebench-v6",
            model=self.model_key,
            score=0,
            max_score=100,
            pass_rate=0.0,
            timestamp=datetime.now().isoformat(),
            details={"status": "not_run", "requires": "livecodebench package"},
        )
        self.evaluation.results.append(result)
        return result

    def run_swe_bench(self) -> BenchmarkResult:
        """
        Run SWE-bench Verified.

        Repo: https://github.com/princeton-nlp/SWE-bench
        """
        print("\n📊 Running SWE-bench Verified...")

        result = BenchmarkResult(
            benchmark="swe-bench-verified",
            model=self.model_key,
            score=0,
            max_score=500,
            pass_rate=0.0,
            timestamp=datetime.now().isoformat(),
            details={"status": "not_run", "requires": "swe-bench package"},
        )
        self.evaluation.results.append(result)
        return result

    def run_mbpp(self) -> BenchmarkResult:
        """
        Run MBPP (Mostly Basic Python Problems).

        Dataset: https://huggingface.co/datasets/google-research-datasets/mbpp
        """
        print("\n📊 Running MBPP...")

        result = BenchmarkResult(
            benchmark="mbpp",
            model=self.model_key,
            score=0,
            max_score=500,
            pass_rate=0.0,
            timestamp=datetime.now().isoformat(),
            details={"status": "not_run"},
        )
        self.evaluation.results.append(result)
        return result

    def run_bigcodebench(self) -> BenchmarkResult:
        """
        Run BigCodeBench.

        Repo: https://github.com/bigcode-project/bigcodebench
        """
        print("\n📊 Running BigCodeBench...")

        result = BenchmarkResult(
            benchmark="bigcodebench",
            model=self.model_key,
            score=0,
            max_score=1140,
            pass_rate=0.0,
            timestamp=datetime.now().isoformat(),
            details={"status": "not_run", "requires": "bigcodebench package"},
        )
        self.evaluation.results.append(result)
        return result

    def run_aider_polyglot(self) -> BenchmarkResult:
        """
        Run Aider Polyglot benchmark.

        Repo: https://github.com/Aider-AI/aider
        """
        print("\n📊 Running Aider Polyglot...")

        result = BenchmarkResult(
            benchmark="aider-polyglot",
            model=self.model_key,
            score=0,
            max_score=225,
            pass_rate=0.0,
            timestamp=datetime.now().isoformat(),
            details={"status": "not_run", "requires": "aider package"},
        )
        self.evaluation.results.append(result)
        return result

    def run_all(self, benchmarks: Optional[List[str]] = None) -> ModelEvaluation:
        """Run all benchmarks (or a subset)."""
        all_benchmarks = {
            "humaneval": self.run_humaneval,
            "livecodebench-v6": self.run_livecodebench,
            "swe-bench-verified": self.run_swe_bench,
            "mbpp": self.run_mbpp,
            "bigcodebench": self.run_bigcodebench,
            "aider-polyglot": self.run_aider_polyglot,
        }

        benchmarks = benchmarks or list(all_benchmarks.keys())

        print(f"\n{'='*60}")
        print(f"ZEN CODER BENCHMARK: {self.model_config.name}")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Base: {self.base_model}")
        print(f"Benchmarks: {', '.join(benchmarks)}")
        print(f"{'='*60}")

        for bench_name in benchmarks:
            if bench_name in all_benchmarks:
                all_benchmarks[bench_name]()

        # Save results
        self._save_results()
        return self.evaluation

    def _save_results(self):
        """Save evaluation results to JSON."""
        output_file = self.output_dir / f"{self.model_key}_eval.json"
        with open(output_file, "w") as f:
            json.dump(self.evaluation.to_dict(), f, indent=2)
        print(f"\n📁 Results saved to {output_file}")

    def compare_to_baseline(self) -> str:
        """Compare results to SoTA baselines."""
        lines = [
            "",
            "=" * 70,
            "BENCHMARK COMPARISON: Zen Coder vs SoTA",
            "=" * 70,
            "",
        ]

        for result in self.evaluation.results:
            bench = result.benchmark
            if bench in self.SOTA_BASELINES:
                lines.append(f"📊 {bench.upper()}")
                lines.append("-" * 50)

                # Show our score
                if result.pass_rate > 0:
                    lines.append(f"  {self.model_key}: {result.pass_rate:.1f}%")
                else:
                    lines.append(f"  {self.model_key}: (not run)")

                # Show baselines
                for model, score in sorted(
                    self.SOTA_BASELINES[bench].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    lines.append(f"  {model}: {score:.1f}%")
                lines.append("")

        comparison = "\n".join(lines)
        print(comparison)

        # Save comparison
        comp_file = self.output_dir / f"{self.model_key}_comparison.txt"
        with open(comp_file, "w") as f:
            f.write(comparison)

        return comparison


def benchmark_all_models(
    models_dir: str = "./output",
    output_dir: str = "./benchmark_results",
) -> Dict[str, ModelEvaluation]:
    """
    Benchmark all trained Zen Coder models.

    Args:
        models_dir: Directory containing trained models
        output_dir: Directory for benchmark results

    Returns:
        Dict mapping model keys to their evaluations
    """
    results = {}

    for model_key in ZEN_MODELS:
        model_path = Path(models_dir) / f"zen-coder-{model_key}"
        if model_path.exists():
            print(f"\n{'#'*60}")
            print(f"# Benchmarking: {model_key}")
            print(f"{'#'*60}")

            bench = ZenBenchmark(
                model_path=str(model_path),
                model_key=model_key,
                output_dir=output_dir,
            )
            results[model_key] = bench.run_all()
            bench.compare_to_baseline()

    return results


def generate_leaderboard(results: Dict[str, ModelEvaluation]) -> str:
    """Generate a leaderboard comparing all models."""
    lines = [
        "",
        "╔" + "═" * 78 + "╗",
        "║" + " ZEN CODER LEADERBOARD ".center(78) + "║",
        "╠" + "═" * 78 + "╣",
        "║" + " Model".ljust(20) + "│" +
        " HumanEval ".center(11) + "│" +
        " LiveCode ".center(10) + "│" +
        " SWE-bench ".center(11) + "│" +
        " MBPP ".center(8) + "│" +
        " BigCode ".center(10) + "║",
        "╠" + "═" * 78 + "╣",
    ]

    for model_key, eval in sorted(
        results.items(),
        key=lambda x: sum(r.pass_rate for r in x[1].results) / max(len(x[1].results), 1),
        reverse=True,
    ):
        scores = eval.summary()
        line = (
            "║" +
            f" {model_key}".ljust(20) + "│" +
            f" {scores.get('humaneval', 0):.1f}%".center(11) + "│" +
            f" {scores.get('livecodebench-v6', 0):.1f}%".center(10) + "│" +
            f" {scores.get('swe-bench-verified', 0):.1f}%".center(11) + "│" +
            f" {scores.get('mbpp', 0):.1f}%".center(8) + "│" +
            f" {scores.get('bigcodebench', 0):.1f}%".center(10) + "║"
        )
        lines.append(line)

    lines.append("╚" + "═" * 78 + "╝")
    return "\n".join(lines)


# Quick benchmark commands
BENCHMARK_INSTALL = """
# Install benchmark dependencies
pip install human-eval evalplus bigcodebench

# Clone benchmark repos
git clone https://github.com/LiveCodeBench/LiveCodeBench
git clone https://github.com/princeton-nlp/SWE-bench
git clone https://github.com/Aider-AI/aider
"""


if __name__ == "__main__":
    print("Zen Coder Benchmark Suite")
    print("=" * 40)
    print("\nBaseline SoTA scores (Dec 2025):")

    for bench, scores in ZenBenchmark.SOTA_BASELINES.items():
        print(f"\n{bench}:")
        for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {score}%")

    print("\n" + BENCHMARK_INSTALL)
