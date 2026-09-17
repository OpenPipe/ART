"""The default tokenizer used by ART's renderers and training backends."""

from __future__ import annotations

from copy import copy
import os
import sys
from typing import TYPE_CHECKING, Any, cast

from .utils.chat_template import configure_preserved_thinking_chat_template

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_KIMI_TOKENIZER_REVISIONS = {
    "moonshotai/Kimi-K2-Thinking": "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55",
    "moonshotai/Kimi-K2.5": "2426b45b6af0da48d0dcce71bbce6225e5c73adc",
    "moonshotai/Kimi-K2.6": "b5aabbfb20227ed42becbf5541dbffd213942c58",
}
_LLAMA_TEXT_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.3-70B-Instruct",
}
_LLAMA_CHAT_TOKENIZER = "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"


def get_tokenizer(
    base_model: str, *, revision: str | None = None, **kwargs: Any
) -> PreTrainedTokenizerBase:
    """Load a model's tokenizer with ART's history-preserving defaults.

    ``revision`` and other keyword arguments go to ``from_pretrained``. Hugging
    Face tokenizers are not cached here: configuring one model's template must not
    change another model's renderer. Hugging Face still caches downloaded files.
    ART's inference and trajectory code cache these instances where appropriate.
    Unpinned Llama text models retain Cookbook's public fallback and use its chat
    template when the base model has none. Explicit revisions keep native files.
    """
    # Tinker suffixes identify a model variant, not a tokenizer revision. Local
    # paths may themselves contain colons.
    local = os.path.isdir(base_model)
    model = base_model if local else base_model.split(":", 1)[0]
    registered = sys.modules.get("tinker_cookbook.tokenizer_utils")
    is_registered = getattr(registered, "is_tokenizer_registered", None)
    if callable(is_registered) and is_registered(base_model):
        assert registered is not None
        if revision is not None or kwargs:
            raise ValueError(
                "Registered tokenizer factories do not accept loader options"
            )
        return cast(
            "PreTrainedTokenizerBase",
            configure_preserved_thinking_chat_template(
                copy(registered.get_tokenizer(base_model))
            ),
        )
    if model.startswith("thinkingmachines/Inkling"):
        if revision is not None or kwargs:
            raise ValueError("Inkling tokenizers do not accept Hugging Face options")
        from tinker_cookbook.tokenizer_utils import get_tokenizer as get_tml_tokenizer

        return cast(
            "PreTrainedTokenizerBase",
            configure_preserved_thinking_chat_template(
                copy(get_tml_tokenizer(base_model))
            ),
        )

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    loader: Any = AutoTokenizer
    if model.startswith("deepseek-ai/DeepSeek-V4-"):
        loader = PreTrainedTokenizerFast
    elif model.startswith("moonshotai/Kimi-K2"):
        # AutoTokenizer can select an incompatible fast backend for Kimi. Its
        # custom tokenizer also renders tool declarations as TypeScript.
        if kwargs.get("trust_remote_code") is False:
            raise ValueError(
                "Kimi tokenization requires its repository's custom tokenizer code"
            )
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        revision = revision or _KIMI_TOKENIZER_REVISIONS.get(model)
        loader = cast(
            "type[PreTrainedTokenizerBase]",
            get_class_from_dynamic_module(
                "tokenization_kimi.TikTokenTokenizer",
                model,
                revision=revision,
                **kwargs,
            ),
        )
    kwargs.setdefault(
        "trust_remote_code",
        os.path.isdir(model)
        or os.environ.get("HF_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"),
    )
    if revision is not None:
        kwargs["revision"] = revision
    try:
        tokenizer = loader.from_pretrained(model, **kwargs)
    except OSError:
        if revision is not None or model not in _LLAMA_TEXT_MODELS:
            raise
        # Tinker users need not have access to Meta's gated repositories. Keep
        # Cookbook's public text-tokenizer fallback, but prefer the actual model
        # and never apply a model-specific commit to a different repository.
        tokenizer = loader.from_pretrained(_LLAMA_CHAT_TOKENIZER, **kwargs)
    if (
        revision is None
        and model in _LLAMA_TEXT_MODELS
        and not getattr(tokenizer, "chat_template", None)
    ):
        tokenizer.chat_template = loader.from_pretrained(
            _LLAMA_CHAT_TOKENIZER, **kwargs
        ).chat_template
    if model.startswith("deepseek-ai/DeepSeek-V4-"):
        from .megatron.dsv4.tokenizer import get_dsv4_tokenizer

        tokenizer = get_dsv4_tokenizer(tokenizer)
    return cast(
        "PreTrainedTokenizerBase",
        configure_preserved_thinking_chat_template(tokenizer),
    )
