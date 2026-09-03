from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import math
from pathlib import Path
from threading import Lock
from typing import Literal, cast

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from art import dev
from art.dev.sequence_lengths import max_seq_length_from_model_config
from art.model import TrainableModel
from art.trajectories import Trajectory
from art.utils.model_config import get_instruction_response_parts

from .tokenize import SFTBatch, tokenize_sft_batch

DEFAULT_SFT_TOKENIZER_CACHE_CAPACITY = 4
MAX_SFT_TOKENIZER_CACHE_CAPACITY = 32

_TokenizerKey = tuple[str, str | None, str | None, bool, bool]
_SequenceLengthKey = tuple[str, str | None]


class SftBatchTokenizer:
    """Bounded cache for canonical model-aware SFT tokenization."""

    def __init__(
        self, cache_capacity: int = DEFAULT_SFT_TOKENIZER_CACHE_CAPACITY
    ) -> None:
        if (
            isinstance(cache_capacity, bool)
            or not 1 <= cache_capacity <= MAX_SFT_TOKENIZER_CACHE_CAPACITY
        ):
            raise ValueError(
                "cache_capacity must be between 1 and "
                f"{MAX_SFT_TOKENIZER_CACHE_CAPACITY}"
            )
        self.cache_capacity = cache_capacity
        self._tokenizers: OrderedDict[_TokenizerKey, PreTrainedTokenizerBase] = (
            OrderedDict()
        )
        self._max_sequence_lengths: OrderedDict[_SequenceLengthKey, int] = OrderedDict()
        self._cache_lock = Lock()

    def tokenize(
        self,
        model: TrainableModel,
        trajectories: Sequence[Trajectory],
        *,
        assistant_turns: Literal["all", "last"] = "all",
        learning_rate: float = 0.0,
    ) -> SFTBatch:
        if not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite")
        internal = cast(dev.InternalModelConfig, model._internal_config or {})
        tokenizer = self._tokenizer(model.base_model, internal)
        instruction, response = get_instruction_response_parts(
            model.base_model, tokenizer
        )
        return tokenize_sft_batch(
            trajectory_batch=list(trajectories),
            learning_rate=learning_rate,
            tokenizer=tokenizer,
            instruction_part=instruction,
            response_part=response,
            chat_template_kwargs=internal.get("chat_template_kwargs"),
            chat_template_tool_schema_format=internal.get(
                "chat_template_tool_schema_format", "default"
            ),
            max_seq_length=self._max_sequence_length(model.base_model, internal),
            assistant_turns=assistant_turns,
        )

    def max_sequence_length(self, model: TrainableModel) -> int:
        internal = cast(dev.InternalModelConfig, model._internal_config or {})
        return self._max_sequence_length(model.base_model, internal)

    def _tokenizer(
        self, base_model: str, internal: dev.InternalModelConfig
    ) -> PreTrainedTokenizerBase:
        template = _chat_template(internal)
        init_args = internal.get("init_args", {})
        revision = init_args.get("revision")
        trust_remote_code = bool(init_args.get("trust_remote_code", False))
        key = (
            base_model,
            revision,
            template,
            bool(internal.get("allow_unvalidated_arch", False)),
            trust_remote_code,
        )
        with self._cache_lock:
            if tokenizer := self._tokenizers.get(key):
                self._tokenizers.move_to_end(key)
                return tokenizer
            tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    base_model,
                    revision=revision,
                    token=init_args.get("token"),
                    trust_remote_code=trust_remote_code,
                ),
            )
            if template is not None:
                tokenizer.chat_template = template
            from art.megatron.model_support.tokenizer import (
                configure_tokenizer_for_model_support,
            )

            tokenizer = configure_tokenizer_for_model_support(
                tokenizer,
                base_model=base_model,
                internal_config=internal,
            )
            self._tokenizers[key] = tokenizer
            if len(self._tokenizers) > self.cache_capacity:
                self._tokenizers.popitem(last=False)
            return tokenizer

    def _max_sequence_length(
        self, base_model: str, internal: dev.InternalModelConfig
    ) -> int:
        init_args = internal.get("init_args", {})
        configured = init_args.get("max_seq_length")
        if configured is not None:
            if isinstance(configured, bool) or not isinstance(configured, int):
                raise ValueError("init_args.max_seq_length must be an integer")
            if configured < 1:
                raise ValueError("init_args.max_seq_length must be positive")
            return configured
        key = base_model, init_args.get("revision")
        with self._cache_lock:
            if value := self._max_sequence_lengths.get(key):
                self._max_sequence_lengths.move_to_end(key)
                return value
            value = max_seq_length_from_model_config(
                base_model,
                revision=key[1],
                token=init_args.get("token"),
            )
            self._max_sequence_lengths[key] = value
            if len(self._max_sequence_lengths) > self.cache_capacity:
                self._max_sequence_lengths.popitem(last=False)
            return value


def _chat_template(internal: dev.InternalModelConfig) -> str | None:
    value = internal.get("chat_template")
    path = internal.get("chat_template_path")
    if value is not None and path is not None:
        raise ValueError("Set only one of chat_template or chat_template_path.")
    return Path(path).read_text(encoding="utf-8") if path is not None else value
