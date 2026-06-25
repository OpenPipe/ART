from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, TypeGuard

_MAX_SEQ_LENGTH_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "seq_length",
    "max_sequence_length",
    "model_max_length",
)
_TEXT_CONFIG_KEYS = ("text_config", "llm_config", "language_config")


def _config_sections(config_dict: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for key in _TEXT_CONFIG_KEYS:
        section = config_dict.get(key)
        if isinstance(section, Mapping):
            yield section
    yield config_dict


def _valid_max_seq_length(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and 0 < value < 1_000_000_000


def max_seq_length_from_model_config(
    base_model: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> int:
    from transformers import PretrainedConfig

    kwargs = {
        key: value
        for key, value in {"revision": revision, "token": token}.items()
        if value is not None
    }
    config_dict, _ = PretrainedConfig.get_config_dict(base_model, **kwargs)
    for section in _config_sections(config_dict):
        for key in _MAX_SEQ_LENGTH_KEYS:
            value = section.get(key)
            if _valid_max_seq_length(value):
                return int(value)
    raise ValueError(
        f"Could not infer max_seq_length from Hugging Face config for {base_model!r}. "
        "Set init_args.max_seq_length explicitly."
    )
