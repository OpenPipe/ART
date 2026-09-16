from __future__ import annotations

from contextvars import ContextVar
import copy
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from art.megatron.dsv4 import encoding
from art_inference.append_only import patch_deepseek_renderer, preserves_history

_PRESERVE_HISTORY: ContextVar[bool] = ContextVar(
    "art_dsv4_preserve_history", default=True
)
patch_deepseek_renderer(encoding, _PRESERVE_HISTORY.get)

DSV4_CHAT_TEMPLATE_MARKER = "deepseek_v4_python_encoder enable_thinking"


def has_configured_chat_template(internal_config: Any) -> bool:
    return (
        internal_config.get("chat_template") is not None
        or internal_config.get("chat_template_path") is not None
    )


def get_dsv4_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    if getattr(tokenizer, "_art_dsv4_chat_template_wrapped", False):
        return tokenizer

    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _ArtDsv4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str | list[int]:
            if (
                kwargs.get("chat_template", self.chat_template)
                != DSV4_CHAT_TEMPLATE_MARKER
            ):
                return super().apply_chat_template(messages, tools=tools, **kwargs)
            thinking = bool(kwargs.get("thinking", False)) or bool(
                kwargs.get("enable_thinking", False)
            )
            thinking_mode = "thinking" if thinking else "chat"
            conversation = kwargs.get("conversation", messages)
            rendered_messages = [
                {
                    **message,
                    "reasoning": message.get(
                        "reasoning",
                        message.get("reasoning_content", message.get("thinking")),
                    ),
                }
                if message.get("role") == "assistant"
                else message
                for message in conversation
            ]
            if tools:
                rendered_messages.insert(0, {"role": "system", "tools": tools})

            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"

            preserve = preserves_history(kwargs)
            token = _PRESERVE_HISTORY.set(preserve)
            try:
                prompt = encoding.encode_messages(
                    rendered_messages,
                    thinking_mode=thinking_mode,
                    drop_thinking=not preserve,
                    reasoning_effort=reasoning_effort,
                )
            finally:
                _PRESERVE_HISTORY.reset(token)
            if not kwargs.get("tokenize", True):
                return prompt
            tokenizer_kwargs = {
                key: kwargs[key]
                for key in ("truncation", "max_length")
                if key in kwargs
            }
            return self.encode(
                prompt,
                add_special_tokens=False,
                **tokenizer_kwargs,
            )

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self) -> Any:
            return get_dsv4_tokenizer, (tokenizer,)

    _ArtDsv4Tokenizer.__name__ = f"ArtDsv4{tokenizer.__class__.__name__}"
    wrapped.__class__ = _ArtDsv4Tokenizer
    wrapped.chat_template = DSV4_CHAT_TEMPLATE_MARKER
    setattr(wrapped, "_art_dsv4_chat_template_wrapped", True)
    return wrapped
