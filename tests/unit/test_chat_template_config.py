from types import SimpleNamespace

import pytest

from art.local import LocalBackend
from art.local.backend import (
    _apply_configured_chat_template,
    _apply_configured_chat_template_server_args,
    _tokenizer_cache_key,
)
from art.megatron.backend import MegatronBackend
from art.megatron.dsv4.tokenizer import DSV4_CHAT_TEMPLATE_MARKER


class _Tokenizer:
    chat_template = "base"
    vocab_size = 128

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, **_kwargs) -> list[int]:
        return [ord(char) % self.vocab_size for char in text]


def test_apply_configured_chat_template_reads_path(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("custom template", encoding="utf-8")
    tokenizer = _Tokenizer()

    _apply_configured_chat_template(
        tokenizer,  # type: ignore[arg-type]
        {"chat_template_path": str(path)},
    )

    assert tokenizer.chat_template == "custom template"


def test_apply_configured_chat_template_server_args_preserves_explicit_override() -> (
    None
):
    config_dict = {"server_args": {"chat_template": "explicit"}}

    _apply_configured_chat_template_server_args(
        config_dict,
        {
            "chat_template": "default",
            "chat_template_content_format": "string",
        },
    )

    assert config_dict["server_args"] == {
        "chat_template": "explicit",
        "chat_template_content_format": "string",
    }


def test_apply_configured_chat_template_server_args_dsv4_has_no_default() -> None:
    config_dict: dict = {}

    _apply_configured_chat_template_server_args(
        config_dict,
        {},
        base_model="deepseek-ai/DeepSeek-V4-Flash",
    )

    assert "server_args" not in config_dict


def test_local_backend_training_tokenizer_calls_model_support(tmp_path) -> None:
    tokenizer = _Tokenizer()
    backend = LocalBackend(path=str(tmp_path))

    result = backend._configure_training_tokenizer(
        tokenizer,  # type: ignore[arg-type]
        model=SimpleNamespace(base_model="deepseek-ai/DeepSeek-V4-Flash"),  # type: ignore[arg-type]
        internal_config={},
    )

    assert result is not tokenizer
    assert result.chat_template == DSV4_CHAT_TEMPLATE_MARKER
    assert getattr(result, "_art_dsv4_chat_template_wrapped", False) is True


def test_megatron_backend_training_tokenizer_calls_model_support(tmp_path) -> None:
    tokenizer = _Tokenizer()
    backend = MegatronBackend(path=str(tmp_path))
    result = backend._configure_training_tokenizer(
        tokenizer,  # type: ignore[arg-type]
        model=SimpleNamespace(base_model="deepseek-ai/DeepSeek-V4-Flash"),  # type: ignore[arg-type]
        internal_config={},
    )

    assert result is not tokenizer
    assert result.chat_template == DSV4_CHAT_TEMPLATE_MARKER
    assert getattr(result, "_art_dsv4_chat_template_wrapped", False) is True


def test_configured_chat_template_rejects_ambiguous_config(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("custom template", encoding="utf-8")

    with pytest.raises(ValueError, match="Set only one"):
        _apply_configured_chat_template(
            _Tokenizer(),  # type: ignore[arg-type]
            {
                "chat_template": "raw",
                "chat_template_path": str(path),
            },
        )


def test_tokenizer_cache_key_includes_chat_template_content(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("first template", encoding="utf-8")
    first_key = _tokenizer_cache_key("base-model", {"chat_template_path": str(path)})

    path.write_text("second template", encoding="utf-8")
    second_key = _tokenizer_cache_key("base-model", {"chat_template_path": str(path)})

    assert _tokenizer_cache_key("base-model", {}) == ("base-model", None)
    assert first_key != second_key
