"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    model_name = model_name.split(":")[0]

    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"

    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
