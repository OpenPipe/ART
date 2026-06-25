from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "DEFAULT_DENSE_HANDLER": (
        "art.megatron.model_support.handlers.default_dense",
        "DEFAULT_DENSE_HANDLER",
    ),
    "DefaultDenseHandler": (
        "art.megatron.model_support.handlers.default_dense",
        "DefaultDenseHandler",
    ),
    "DefaultMoeHandler": (
        "art.megatron.model_support.handlers.default_dense",
        "DefaultMoeHandler",
    ),
    "QWEN3_DENSE_HANDLER": (
        "art.megatron.model_support.handlers.qwen3_dense",
        "QWEN3_DENSE_HANDLER",
    ),
    "Qwen3DenseHandler": (
        "art.megatron.model_support.handlers.qwen3_dense",
        "Qwen3DenseHandler",
    ),
    "QWEN3_MOE_HANDLER": (
        "art.megatron.model_support.handlers.qwen3_moe",
        "QWEN3_MOE_HANDLER",
    ),
    "Qwen3MoeHandler": (
        "art.megatron.model_support.handlers.qwen3_moe",
        "Qwen3MoeHandler",
    ),
    "QWEN3_5_DENSE_HANDLER": (
        "art.megatron.model_support.handlers.qwen3_5",
        "QWEN3_5_DENSE_HANDLER",
    ),
    "Qwen35DenseHandler": (
        "art.megatron.model_support.handlers.qwen3_5",
        "Qwen35DenseHandler",
    ),
    "QWEN3_5_MOE_HANDLER": (
        "art.megatron.model_support.handlers.qwen3_5",
        "QWEN3_5_MOE_HANDLER",
    ),
    "Qwen35MoeHandler": (
        "art.megatron.model_support.handlers.qwen3_5",
        "Qwen35MoeHandler",
    ),
    "GPT_OSS_MOE_HANDLER": (
        "art.megatron.model_support.handlers.gpt_oss",
        "GPT_OSS_MOE_HANDLER",
    ),
    "GptOssMoeHandler": (
        "art.megatron.model_support.handlers.gpt_oss",
        "GptOssMoeHandler",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
