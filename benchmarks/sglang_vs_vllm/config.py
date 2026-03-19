"""
Benchmark configuration for SGLang + Megatron vs vLLM + Megatron.

All parameters that control the benchmark are defined here to ensure
both backends are tested under identical conditions.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Model configuration shared across both backends."""

    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_name: str = "benchmark-model"
    project: str = "sglang-vs-vllm-benchmark"

    max_seq_length: int = 8192
    max_output_tokens: int = 1024

    # LoRA config (must match Megatron train.py defaults)
    lora_r: int = 1
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class InferenceConfig:
    """Inference engine configuration."""

    tensor_parallel_size: int = 0  # 0 = auto-detect (min(2, num_gpus))
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 256
    enable_lora: bool = True
    max_loras: int = 2

    def get_tp_size(self) -> int:
        import torch
        if self.tensor_parallel_size <= 0:
            return min(2, torch.cuda.device_count())
        return self.tensor_parallel_size


@dataclass
class TrainingConfig:
    """Training configuration shared across both backends."""

    learning_rate: float = 5e-6
    beta: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 0.1
    weight_decay: float = 0.1


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    backends: list[Literal["vllm", "sglang"]] = field(
        default_factory=lambda: ["vllm", "sglang"]
    )

    dataset: str = "gsm8k"
    num_training_steps: int = 3
    num_rollouts_per_step: int = 16
    concurrency: int = 32
    num_warmup_requests: int = 4
    seed: int = 42

    num_repeats: int = 1

    output_dir: str = "benchmark_results"
    save_raw_metrics: bool = True

    sglang_python: str = ""
    vllm_python: str = ""

    vllm_port: int = 8100
    sglang_port: int = 8200

    server_startup_timeout: int = 600
    request_timeout: int = 300
    training_timeout: int = 1800
    server_shutdown_timeout: int = 60

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.sglang_python:
            self.sglang_python = _find_sglang_python()
        if not self.vllm_python:
            self.vllm_python = _find_vllm_python()


def _find_sglang_python() -> str:
    candidates = [
        os.path.expanduser("~/.venvs/sglang-bench/bin/python"),
        os.path.expanduser("~/sglang-env/bin/python"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return "python"


def _find_vllm_python() -> str:
    import sys
    return sys.executable


# ---------------------------------------------------------------------------
# GSM8K dataset loading
# ---------------------------------------------------------------------------

_GSM8K_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def _download_gsm8k() -> list[str]:
    """Download GSM8K test set and return list of question strings."""
    os.makedirs(_GSM8K_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_GSM8K_CACHE_DIR, "gsm8k_test.jsonl")

    if not os.path.exists(cache_path):
        import urllib.request
        print(f"Downloading GSM8K test set to {cache_path}...")
        urllib.request.urlretrieve(_GSM8K_URL, cache_path)

    questions = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                questions.append(data["question"])
    return questions


def _load_gsm8k() -> list[str]:
    """Load GSM8K questions, downloading if needed."""
    try:
        return _download_gsm8k()
    except Exception as e:
        print(f"Failed to download GSM8K ({e}), using fallback prompts")
        return _GSM8K_FALLBACK


# Small fallback in case download fails (e.g. no internet on GPU node)
_GSM8K_FALLBACK = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
    "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
]


def generate_benchmark_prompts(
    num_prompts: int,
    input_tokens: int = 1024,
    dataset: str = "gsm8k",
    seed: int = 42,
) -> list[list[dict[str, str]]]:
    """Generate deterministic benchmark prompts from GSM8K.

    Downloads the real GSM8K test set (1,319 questions) and samples
    with deterministic seeding so both backends get identical prompts.
    """
    rng = random.Random(seed)
    source_prompts = _load_gsm8k()

    # Sample with replacement if we need more prompts than the pool
    sampled = [rng.choice(source_prompts) for _ in range(num_prompts)]

    system_msg = (
        "You are a helpful assistant. Think step by step and show your reasoning."
    )

    prompts = []
    for user_text in sampled:
        prompts.append([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ])
    return prompts
