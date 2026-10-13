from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, ParamSpec, TypeVar, cast

from .costs import get_model_pricing, tokens_to_cost

OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"

P = ParamSpec("P")
R = TypeVar("R")

CostExtractor = Callable[[Any], float | None]
ResponseGetter = Callable[[Any], Any]


@dataclass(frozen=True)
class TokenPricing:
    prompt_per_million: float
    completion_per_million: float
    cached_prompt_per_million: float | None = None
    cache_creation_per_million: float | None = None
    cache_read_per_million: float | None = None


@dataclass(frozen=True)
class _OpenAITokenUsage:
    prompt_tokens: float
    completion_tokens: float
    cached_prompt_tokens: float


@dataclass(frozen=True)
class _AnthropicTokenUsage:
    input_tokens: float
    output_tokens: float
    cache_creation_input_tokens: float
    cache_read_input_tokens: float


MODEL_TOKEN_PRICING: dict[str, TokenPricing] = {
    "openai/gpt-4.1": TokenPricing(
        prompt_per_million=2.0,
        completion_per_million=8.0,
        cached_prompt_per_million=0.5,
    ),
    "anthropic/claude-sonnet-4-6": TokenPricing(
        prompt_per_million=3.0,
        completion_per_million=15.0,
        cache_creation_per_million=3.75,
        cache_read_per_million=0.30,
    ),
}


def _litellm_price_per_million(
    model_info: Mapping[str, Any], field: str
) -> float | None:
    value = model_info.get(field)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def _litellm_token_pricing(model_name: str) -> TokenPricing | None:
    try:
        from litellm import get_model_info

        model_info = get_model_info(model_name)
    except Exception:
        return None

    if not isinstance(model_info, Mapping):
        return None

    prompt_per_million = _litellm_price_per_million(model_info, "input_cost_per_token")
    completion_per_million = _litellm_price_per_million(
        model_info, "output_cost_per_token"
    )
    if prompt_per_million is None or completion_per_million is None:
        return None

    cache_read_per_million = _litellm_price_per_million(
        model_info, "cache_read_input_token_cost"
    )
    cache_creation_per_million = _litellm_price_per_million(
        model_info, "cache_creation_input_token_cost"
    )
    return TokenPricing(
        prompt_per_million=prompt_per_million,
        completion_per_million=completion_per_million,
        cached_prompt_per_million=cache_read_per_million,
        cache_creation_per_million=cache_creation_per_million,
        cache_read_per_million=cache_read_per_million,
    )


def _configured_token_pricing(model_name: str) -> TokenPricing | None:
    explicit = MODEL_TOKEN_PRICING.get(model_name)
    if explicit is not None:
        return explicit

    litellm_pricing = _litellm_token_pricing(model_name)
    if litellm_pricing is not None:
        return litellm_pricing

    pricing = get_model_pricing(model_name)
    if pricing is None:
        return None
    return TokenPricing(
        prompt_per_million=pricing.prefill,
        completion_per_million=pricing.sample,
    )


def normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    normalized = provider.strip().lower()
    if not normalized:
        return None
    return normalized


def _read_usage_field(usage: Any, field: str) -> float | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    if value is None:
        return None
    return float(value)


def _read_usage_nested_field(usage: Any, *fields: str) -> float | None:
    current = usage
    for field in fields:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(field)
        else:
            current = getattr(current, field, None)
    if current is None:
        return None
    return float(current)


def _read_field(container: Any, field: str) -> Any:
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(field)
    return getattr(container, field, None)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_cost_value(value: Any) -> float | None:
    direct_cost = _coerce_float(value)
    if direct_cost is not None:
        return direct_cost
    return _coerce_float(_read_field(value, "total_cost"))


def _response_usage(response: Any) -> Any:
    if isinstance(response, Mapping):
        return response.get("usage")
    return getattr(response, "usage", None)


def _extract_direct_response_cost(response: Any) -> float | None:
    usage = _response_usage(response)
    direct_usage_cost = _extract_cost_value(_read_field(usage, "cost"))
    if direct_usage_cost is not None:
        return direct_usage_cost

    model_extra = _read_field(usage, "model_extra")
    model_extra_cost = _extract_cost_value(_read_field(model_extra, "cost"))
    if model_extra_cost is not None:
        return model_extra_cost

    hidden_params = _read_field(response, "_hidden_params")
    additional_headers = _read_field(hidden_params, "additional_headers")
    return _extract_cost_value(
        _read_field(additional_headers, "llm_provider-x-litellm-response-cost")
    )


def _extract_openai_token_counts(response: Any) -> _OpenAITokenUsage | None:
    usage = _response_usage(response)
    prompt_tokens = _read_usage_field(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _read_usage_field(usage, "input_tokens")
    completion_tokens = _read_usage_field(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _read_usage_field(usage, "output_tokens")
    cached_prompt_tokens = (
        _read_usage_nested_field(usage, "prompt_tokens_details", "cached_tokens")
        or _read_usage_nested_field(usage, "input_tokens_details", "cached_tokens")
        or 0.0
    )
    if (
        prompt_tokens is None
        and completion_tokens is None
        and cached_prompt_tokens == 0.0
    ):
        return None
    total_prompt_tokens = prompt_tokens or 0.0
    return _OpenAITokenUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=completion_tokens or 0.0,
        cached_prompt_tokens=min(cached_prompt_tokens, total_prompt_tokens),
    )


def _extract_anthropic_token_counts(response: Any) -> _AnthropicTokenUsage | None:
    usage = _response_usage(response)
    input_tokens = _read_usage_field(usage, "input_tokens")
    output_tokens = _read_usage_field(usage, "output_tokens")
    cache_creation_input_tokens = (
        _read_usage_field(usage, "cache_creation_input_tokens") or 0.0
    )
    cache_read_input_tokens = _read_usage_field(usage, "cache_read_input_tokens") or 0.0
    if (
        input_tokens is None
        and output_tokens is None
        and cache_creation_input_tokens == 0.0
        and cache_read_input_tokens == 0.0
    ):
        return None
    return _AnthropicTokenUsage(
        input_tokens=input_tokens or 0.0,
        output_tokens=output_tokens or 0.0,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def _estimate_openai_cost(
    token_counts: _OpenAITokenUsage | None,
    pricing: TokenPricing,
) -> float | None:
    if token_counts is None:
        return None
    uncached_prompt_tokens = max(
        token_counts.prompt_tokens - token_counts.cached_prompt_tokens,
        0.0,
    )
    cached_prompt_price = (
        pricing.cached_prompt_per_million
        if pricing.cached_prompt_per_million is not None
        else pricing.prompt_per_million
    )
    return (
        tokens_to_cost(uncached_prompt_tokens, pricing.prompt_per_million)
        + tokens_to_cost(
            token_counts.cached_prompt_tokens,
            cached_prompt_price,
        )
        + tokens_to_cost(
            token_counts.completion_tokens,
            pricing.completion_per_million,
        )
    )


def _estimate_anthropic_cost(
    token_counts: _AnthropicTokenUsage | None,
    pricing: TokenPricing,
) -> float | None:
    if token_counts is None:
        return None
    cache_creation_price = (
        pricing.cache_creation_per_million
        if pricing.cache_creation_per_million is not None
        else pricing.prompt_per_million
    )
    cache_read_price = (
        pricing.cache_read_per_million
        if pricing.cache_read_per_million is not None
        else pricing.prompt_per_million
    )
    return (
        tokens_to_cost(token_counts.input_tokens, pricing.prompt_per_million)
        + tokens_to_cost(
            token_counts.cache_creation_input_tokens,
            cache_creation_price,
        )
        + tokens_to_cost(
            token_counts.cache_read_input_tokens,
            cache_read_price,
        )
        + tokens_to_cost(
            token_counts.output_tokens,
            pricing.completion_per_million,
        )
    )


def _estimate_provider_cost(
    provider_name: str,
    response: Any,
    pricing: TokenPricing,
) -> float | None:
    if provider_name == OPENAI_PROVIDER:
        return _estimate_openai_cost(_extract_openai_token_counts(response), pricing)
    if provider_name == ANTHROPIC_PROVIDER:
        return _estimate_anthropic_cost(
            _extract_anthropic_token_counts(response),
            pricing,
        )
    return None


def _resolve_registered_or_default_pricing(
    model_name: str,
    *,
    model_pricing: Mapping[str, TokenPricing],
) -> TokenPricing | None:
    registered = model_pricing.get(model_name)
    if registered is not None:
        return registered
    return _configured_token_pricing(model_name)


def _merge_token_pricing(
    *,
    base_pricing: TokenPricing,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
    cached_prompt_price_per_million: float | None,
    cache_creation_price_per_million: float | None,
    cache_read_price_per_million: float | None,
) -> TokenPricing:
    return TokenPricing(
        prompt_per_million=(
            float(prompt_price_per_million)
            if prompt_price_per_million is not None
            else base_pricing.prompt_per_million
        ),
        completion_per_million=(
            float(completion_price_per_million)
            if completion_price_per_million is not None
            else base_pricing.completion_per_million
        ),
        cached_prompt_per_million=(
            float(cached_prompt_price_per_million)
            if cached_prompt_price_per_million is not None
            else base_pricing.cached_prompt_per_million
        ),
        cache_creation_per_million=(
            float(cache_creation_price_per_million)
            if cache_creation_price_per_million is not None
            else base_pricing.cache_creation_per_million
        ),
        cache_read_per_million=(
            float(cache_read_price_per_million)
            if cache_read_price_per_million is not None
            else base_pricing.cache_read_per_million
        ),
    )


def normalize_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    normalized = model_name.strip()
    if not normalized:
        return None
    return normalized


def _resolve_token_pricing(
    *,
    provider: str,
    model_name: str,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
    cached_prompt_price_per_million: float | None,
    cache_creation_price_per_million: float | None,
    cache_read_price_per_million: float | None,
    model_pricing: Mapping[str, TokenPricing],
) -> TokenPricing:
    explicit_prompt_price = (
        float(prompt_price_per_million)
        if prompt_price_per_million is not None
        else None
    )
    explicit_completion_price = (
        float(completion_price_per_million)
        if completion_price_per_million is not None
        else None
    )
    explicit_cached_prompt_price = (
        float(cached_prompt_price_per_million)
        if cached_prompt_price_per_million is not None
        else None
    )
    explicit_cache_creation_price = (
        float(cache_creation_price_per_million)
        if cache_creation_price_per_million is not None
        else None
    )
    explicit_cache_read_price = (
        float(cache_read_price_per_million)
        if cache_read_price_per_million is not None
        else None
    )

    if normalize_provider(provider) is None:
        raise ValueError("provider must be non-empty")

    normalized_model_name = normalize_model_name(model_name)
    if normalized_model_name is None:
        raise ValueError("model_name must be non-empty")

    configured_pricing = _resolve_registered_or_default_pricing(
        normalized_model_name,
        model_pricing=model_pricing,
    )
    if configured_pricing is None:
        raise ValueError(
            f"No pricing configured for model '{normalized_model_name}'. "
            "Add it to art.api_costs.MODEL_TOKEN_PRICING, art.costs.MODEL_PRICING, "
            "or register it with MetricsBuilder.register_model_pricing()."
        )

    return _merge_token_pricing(
        base_pricing=configured_pricing,
        prompt_price_per_million=explicit_prompt_price,
        completion_price_per_million=explicit_completion_price,
        cached_prompt_price_per_million=explicit_cached_prompt_price,
        cache_creation_price_per_million=explicit_cache_creation_price,
        cache_read_price_per_million=explicit_cache_read_price,
    )


def extract_api_cost(
    response: Any,
    *,
    provider: str,
    model_name: str,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
    cached_prompt_price_per_million: float | None,
    cache_creation_price_per_million: float | None,
    cache_read_price_per_million: float | None,
    cost_extractors: Mapping[str, CostExtractor],
    model_pricing: Mapping[str, TokenPricing],
) -> float | None:
    provider_name = normalize_provider(provider)
    if provider_name is None:
        raise ValueError("provider must be non-empty")

    direct_response_cost = _extract_direct_response_cost(response)
    if direct_response_cost is not None:
        return direct_response_cost

    custom_extractor = cost_extractors.get(provider_name)
    if custom_extractor is not None:
        custom_cost = custom_extractor(response)
        if custom_cost is not None:
            return float(custom_cost)

    pricing = _resolve_token_pricing(
        provider=provider_name,
        model_name=model_name,
        prompt_price_per_million=prompt_price_per_million,
        completion_price_per_million=completion_price_per_million,
        cached_prompt_price_per_million=cached_prompt_price_per_million,
        cache_creation_price_per_million=cache_creation_price_per_million,
        cache_read_price_per_million=cache_read_price_per_million,
        model_pricing=model_pricing,
    )
    provider_cost = _estimate_provider_cost(provider_name, response, pricing)
    if provider_cost is not None:
        return provider_cost

    if provider_name in {OPENAI_PROVIDER, ANTHROPIC_PROVIDER}:
        raise ValueError(
            f"Response usage does not match provider '{provider_name}'. "
            "Pass the correct provider/model pair or register a custom cost extractor."
        )
    raise ValueError(f"No cost extractor registered for provider '{provider_name}'.")


def _record_api_cost(
    *,
    result: Any,
    source: str,
    provider: str,
    response_getter: ResponseGetter | None,
    model_name: str,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
    cached_prompt_price_per_million: float | None,
    cache_creation_price_per_million: float | None,
    cache_read_price_per_million: float | None,
) -> None:
    try:
        from .metrics import MetricsBuilder

        builder = MetricsBuilder.get_active()
    except LookupError:
        return

    response = response_getter(result) if response_getter is not None else result
    builder.add_response_cost(
        source,
        response,
        provider=provider,
        model_name=model_name,
        prompt_price_per_million=prompt_price_per_million,
        completion_price_per_million=completion_price_per_million,
        cached_prompt_price_per_million=cached_prompt_price_per_million,
        cache_creation_price_per_million=cache_creation_price_per_million,
        cache_read_price_per_million=cache_read_price_per_million,
    )


def track_api_cost(
    *,
    source: str,
    provider: str,
    model_name: str,
    response_getter: ResponseGetter | None = None,
    prompt_price_per_million: float | None = None,
    completion_price_per_million: float | None = None,
    cached_prompt_price_per_million: float | None = None,
    cache_creation_price_per_million: float | None = None,
    cache_read_price_per_million: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    normalized_source = source.strip("/")
    if not normalized_source:
        raise ValueError("source must be non-empty")

    normalized_provider = normalize_provider(provider)
    if normalized_provider is None:
        raise ValueError("provider must be non-empty")
    normalized_model_name = normalize_model_name(model_name)
    if normalized_model_name is None:
        raise ValueError("model_name must be non-empty")

    def _decorate(func: Callable[P, R]) -> Callable[P, R]:
        if iscoroutinefunction(func):
            async_func = cast(Callable[P, Awaitable[Any]], func)

            @wraps(func)
            async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                result = await async_func(*args, **kwargs)
                _record_api_cost(
                    result=result,
                    source=normalized_source,
                    provider=normalized_provider,
                    response_getter=response_getter,
                    model_name=normalized_model_name,
                    prompt_price_per_million=prompt_price_per_million,
                    completion_price_per_million=completion_price_per_million,
                    cached_prompt_price_per_million=cached_prompt_price_per_million,
                    cache_creation_price_per_million=cache_creation_price_per_million,
                    cache_read_price_per_million=cache_read_price_per_million,
                )
                return result

            return cast(Callable[P, R], _async_wrapper)

        @wraps(func)
        def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            _record_api_cost(
                result=result,
                source=normalized_source,
                provider=normalized_provider,
                response_getter=response_getter,
                model_name=normalized_model_name,
                prompt_price_per_million=prompt_price_per_million,
                completion_price_per_million=completion_price_per_million,
                cached_prompt_price_per_million=cached_prompt_price_per_million,
                cache_creation_price_per_million=cache_creation_price_per_million,
                cache_read_price_per_million=cache_read_price_per_million,
            )
            return result

        return _sync_wrapper

    return _decorate
