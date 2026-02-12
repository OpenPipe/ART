"""Cost utilities for ART training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a model."""

    prefill: float  # $/1M tokens for prompt/prefill
    sample: float  # $/1M tokens for sampling/generation
    train: float  # $/1M tokens for training


TokenCount: TypeAlias = int | None
CostCalculator: TypeAlias = Callable[[TokenCount, TokenCount], dict[str, float]]

# Pricing per model ($/1M tokens). Keep in sync with infra pricing.
MODEL_PRICING: dict[str, ModelPricing] = {
    # Qwen models
    "Qwen/Qwen3-4B-Instruct-2507": ModelPricing(prefill=0.07, sample=0.22, train=0.22),
    "Qwen/Qwen3-8B": ModelPricing(prefill=0.13, sample=0.40, train=0.40),
    "Qwen/Qwen3-8B-Base": ModelPricing(prefill=0.13, sample=0.40, train=0.40),
    "Qwen/Qwen3-30B-A3B": ModelPricing(prefill=0.12, sample=0.30, train=0.36),
    "Qwen/Qwen3-30B-A3B-Base": ModelPricing(prefill=0.12, sample=0.30, train=0.36),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelPricing(
        prefill=0.12, sample=0.30, train=0.36
    ),
    "Qwen/Qwen3-32B": ModelPricing(prefill=0.49, sample=1.47, train=1.47),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": ModelPricing(
        prefill=0.68, sample=1.70, train=2.04
    ),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": ModelPricing(
        prefill=0.18, sample=0.44, train=0.53
    ),
    "Qwen/Qwen3-VL-235B-A22B-Instruct": ModelPricing(
        prefill=1.02, sample=2.56, train=3.07
    ),
    # Meta Llama models
    "meta-llama/Llama-3.2-1B": ModelPricing(prefill=0.03, sample=0.09, train=0.09),
    "meta-llama/Llama-3.2-3B": ModelPricing(prefill=0.06, sample=0.18, train=0.18),
    "meta-llama/Llama-3.1-8B": ModelPricing(prefill=0.13, sample=0.40, train=0.40),
    "meta-llama/Llama-3.1-8B-Instruct": ModelPricing(
        prefill=0.13, sample=0.40, train=0.40
    ),
    "meta-llama/Llama-3.1-70B": ModelPricing(prefill=1.05, sample=3.16, train=3.16),
    "meta-llama/Llama-3.3-70B-Instruct": ModelPricing(
        prefill=1.05, sample=3.16, train=3.16
    ),
    # DeepSeek models
    "deepseek-ai/DeepSeek-V3.1": ModelPricing(prefill=1.13, sample=2.81, train=3.38),
    "deepseek-ai/DeepSeek-V3.1-Base": ModelPricing(
        prefill=1.13, sample=2.81, train=3.38
    ),
    # OpenAI models
    "openai/gpt-oss-120b": ModelPricing(prefill=0.18, sample=0.44, train=0.52),
    "openai/gpt-oss-20b": ModelPricing(prefill=0.12, sample=0.30, train=0.36),
    # Moonshot models
    "moonshotai/Kimi-K2-Thinking": ModelPricing(prefill=0.98, sample=2.44, train=2.93),
}


def get_model_pricing(
    model_name: str | None, *, strict: bool = False
) -> ModelPricing | None:
    """Return pricing for a model or None if missing."""
    if model_name is None:
        return None
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None and strict:
        raise ValueError(
            f"No pricing configured for model '{model_name}'. "
            f"Add pricing to art.costs.MODEL_PRICING. "
            f"Available models: {list(MODEL_PRICING.keys())}"
        )
    return pricing


def tokens_to_cost(num_tokens: float, price_per_million: float) -> float:
    """Convert token count to cost in dollars."""
    return float(num_tokens) * price_per_million / 1_000_000


def compute_sample_costs(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: ModelPricing,
) -> dict[str, float]:
    """Compute prompt+completion costs for a single API call."""
    prompt_value = float(prompt_tokens or 0)
    completion_value = float(completion_tokens or 0)
    prefill_cost = tokens_to_cost(prompt_value, pricing.prefill)
    sample_cost = tokens_to_cost(completion_value, pricing.sample)
    return {
        "costs_prefill": prefill_cost,
        "costs_sample": sample_cost,
    }


def build_cost_calculator(pricing: ModelPricing) -> CostCalculator:
    """Return a callable that computes prompt+completion costs for a request."""

    def _calculator(
        prompt_tokens: int | None, completion_tokens: int | None
    ) -> dict[str, float]:
        return compute_sample_costs(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=pricing,
        )

    return _calculator


def compute_train_cost(train_tokens: float, pricing: ModelPricing) -> float:
    """Compute training cost from token count."""
    return tokens_to_cost(train_tokens, pricing.train)
