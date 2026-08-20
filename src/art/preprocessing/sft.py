from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from art import dev
from art.dev.sequence_lengths import max_seq_length_from_model_config
from art.model import TrainableModel
from art.trajectories import Trajectory
from art.utils.model_config import get_instruction_response_parts

from .tokenize import SFTBatch, tokenize_sft_batch


class SftBatchTokenizer:
    """Cache canonical model-aware SFT tokenizers and tokenize one command batch."""

    def __init__(self) -> None:
        self._tokenizers: dict[str, PreTrainedTokenizerBase] = {}
        self._max_sequence_lengths: dict[tuple[str, str | None], int] = {}

    def tokenize(
        self,
        model: TrainableModel,
        trajectories: tuple[Trajectory, ...],
        *,
        assistant_turns: Literal["all", "last"],
        learning_rate: float = 0.0,
    ) -> SFTBatch:
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
            max_seq_length=self.max_sequence_length(model),
            assistant_turns=assistant_turns,
        )

    def max_sequence_length(self, model: TrainableModel) -> int:
        internal = cast(dev.InternalModelConfig, model._internal_config or {})
        return self._max_sequence_length(model, internal)

    def _tokenizer(
        self, base_model: str, internal: dev.InternalModelConfig
    ) -> PreTrainedTokenizerBase:
        template = _chat_template(internal)
        init_args = internal.get("init_args", {})
        revision = init_args.get("revision")
        key = hashlib.sha256(
            json.dumps(
                {
                    "base_model": base_model,
                    "revision": revision,
                    "template": template,
                    "allow_unvalidated_arch": internal.get(
                        "allow_unvalidated_arch", False
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if key not in self._tokenizers:
            tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    base_model,
                    revision=revision,
                    token=init_args.get("token"),
                ),
            )
            if template is not None:
                tokenizer.chat_template = template
            from art.megatron.model_support.tokenizer import (
                configure_tokenizer_for_model_support,
            )

            self._tokenizers[key] = configure_tokenizer_for_model_support(
                tokenizer,
                base_model=base_model,
                internal_config=internal,
            )
        return self._tokenizers[key]

    def _max_sequence_length(
        self, model: TrainableModel, internal: dev.InternalModelConfig
    ) -> int:
        init_args = internal.get("init_args", {})
        if configured := init_args.get("max_seq_length"):
            return int(configured)
        revision = init_args.get("revision")
        key = model.base_model, revision
        if key not in self._max_sequence_lengths:
            self._max_sequence_lengths[key] = max_seq_length_from_model_config(
                model.base_model,
                revision=revision,
                token=init_args.get("token"),
            )
        return self._max_sequence_lengths[key]


def _chat_template(internal: dev.InternalModelConfig) -> str | None:
    value = internal.get("chat_template")
    path = internal.get("chat_template_path")
    if value is not None and path is not None:
        raise ValueError("Set only one of chat_template or chat_template_path.")
    return Path(path).read_text(encoding="utf-8") if path is not None else value
