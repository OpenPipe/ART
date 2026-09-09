from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .registry import UnsupportedModelArchitectureError, get_model_support_handler


def _has_configured_chat_template(internal_config: Mapping[str, Any]) -> bool:
    return (
        internal_config.get("chat_template") is not None
        or internal_config.get("chat_template_path") is not None
    )


def configure_tokenizer_for_model_support(
    tokenizer: PreTrainedTokenizerBase,
    *,
    base_model: str,
    internal_config: Mapping[str, Any],
) -> PreTrainedTokenizerBase:
    try:
        handler = get_model_support_handler(
            base_model,
            allow_unvalidated_arch=bool(
                internal_config.get("allow_unvalidated_arch", False)
            ),
        )
    except UnsupportedModelArchitectureError:
        return tokenizer

    if not _has_configured_chat_template(internal_config) and not isinstance(
        getattr(tokenizer, "chat_template", None), str
    ):
        default = handler.default_chat_template()
        if default is not None:
            tokenizer.chat_template = default

    return cast(
        PreTrainedTokenizerBase,
        handler.configure_tokenizer(tokenizer, internal_config=internal_config),
    )
