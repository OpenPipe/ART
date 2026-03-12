"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any

# Global registry for custom tokenizer factories
_CUSTOM_TOKENIZER_REGISTRY: dict[str, Callable[[], Tokenizer]] = {}


def register_tokenizer(
    name: str,
    factory: Callable[[], Tokenizer],
) -> None:
    """Register a custom tokenizer factory.

    Args:
        name: The tokenizer name
        factory: A callable that takes no arguments and returns a Tokenizer.

    Example:
        def my_tokenizer_factory():
            return MyCustomTokenizer()

        register_tokenizer("Foo/foo_tokenizer", my_tokenizer_factory)
    """
    _CUSTOM_TOKENIZER_REGISTRY[name] = factory


def get_registered_tokenizer_names() -> list[str]:
    """Return a list of all registered custom tokenizer names."""
    return list(_CUSTOM_TOKENIZER_REGISTRY.keys())


def is_tokenizer_registered(name: str) -> bool:
    """Check if a tokenizer name is registered."""
    return name in _CUSTOM_TOKENIZER_REGISTRY


def unregister_tokenizer(name: str) -> bool:
    """Unregister a custom tokenizer factory.

    Args:
        name: The tokenizer name to unregister.

    Returns:
        True if the tokenizer was unregistered, False if it wasn't registered.
    """
    if name in _CUSTOM_TOKENIZER_REGISTRY:
        del _CUSTOM_TOKENIZER_REGISTRY[name]
        return True
    return False


def get_tokenizer(model_name: str) -> Tokenizer:
    """Get a tokenizer by name.

    Checks custom registry first, then falls back to HuggingFace AutoTokenizer.
    """
    # Check custom registry first (not cached, factory handles caching if needed)
    if (tokenizer := _CUSTOM_TOKENIZER_REGISTRY.get(model_name)) is not None:
        return tokenizer()

    return _get_hf_tokenizer(model_name)


@cache
def _get_hf_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    model_name = model_name.split(":")[0]

    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"

    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"
    elif model_name == "moonshotai/Kimi-K2.5":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "2426b45b6af0da48d0dcce71bbce6225e5c73adc"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
