"""Unit tests for secure API key generation in vLLM server configuration.

Verifies that the hardcoded 'default' API key (CWE-798) has been replaced
with a cryptographically random token, and that the ART_API_KEY environment
variable override works correctly.

See: https://github.com/OpenPipe/ART/issues/628
"""

import os

import pytest

from art.dev.openai_server import (
    OpenAIServerConfig,
    _generate_api_key,
    get_openai_server_config,
)


class TestGenerateApiKey:
    """Tests for the ``_generate_api_key`` helper."""

    def test_key_is_not_hardcoded_default(self) -> None:
        """The generated key must never be the literal string 'default'."""
        key = _generate_api_key()
        assert key != "default"

    def test_key_is_nonempty_string(self) -> None:
        key = _generate_api_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_key_has_sufficient_entropy(self) -> None:
        """A 32-byte urlsafe token encodes to >= 32 characters."""
        key = _generate_api_key()
        assert len(key) >= 32

    def test_keys_are_unique_across_calls(self) -> None:
        """Each invocation should produce a different random key."""
        keys = {_generate_api_key() for _ in range(20)}
        assert len(keys) == 20

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ART_API_KEY environment variable should take precedence."""
        monkeypatch.setenv("ART_API_KEY", "my-custom-key-42")
        assert _generate_api_key() == "my-custom-key-42"

    def test_env_var_empty_falls_back_to_random(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ART_API_KEY should be treated as unset."""
        monkeypatch.setenv("ART_API_KEY", "")
        key = _generate_api_key()
        assert key != ""
        assert key != "default"

    def test_env_var_unset_falls_back_to_random(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ART_API_KEY is not in the environment, use random key."""
        monkeypatch.delenv("ART_API_KEY", raising=False)
        key = _generate_api_key()
        assert key != "default"
        assert len(key) >= 32


class TestGetOpenaiServerConfig:
    """Tests for ``get_openai_server_config`` API key behaviour."""

    def test_default_config_uses_random_key(self) -> None:
        """Without user-supplied config, the key must not be 'default'."""
        config = get_openai_server_config(
            model_name="test-model",
            base_model="base-model",
            log_file="/tmp/test.log",
        )
        api_key = config["server_args"]["api_key"]
        assert api_key != "default"
        assert isinstance(api_key, str)
        assert len(api_key) >= 32

    def test_user_supplied_key_takes_precedence(self) -> None:
        """A user-provided api_key in server_args should override the default."""
        user_config = OpenAIServerConfig(
            server_args={"api_key": "user-provided-key-xyz"}  # type: ignore[typeddict-item]
        )
        config = get_openai_server_config(
            model_name="test-model",
            base_model="base-model",
            log_file="/tmp/test.log",
            config=user_config,
        )
        assert config["server_args"]["api_key"] == "user-provided-key-xyz"

    def test_env_var_key_used_when_no_user_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ART_API_KEY env var should be picked up when no user key is given."""
        monkeypatch.setenv("ART_API_KEY", "env-key-123")
        config = get_openai_server_config(
            model_name="test-model",
            base_model="base-model",
            log_file="/tmp/test.log",
        )
        assert config["server_args"]["api_key"] == "env-key-123"

    def test_user_key_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User-supplied key should win over ART_API_KEY env var."""
        monkeypatch.setenv("ART_API_KEY", "env-key-123")
        user_config = OpenAIServerConfig(
            server_args={"api_key": "user-wins"}  # type: ignore[typeddict-item]
        )
        config = get_openai_server_config(
            model_name="test-model",
            base_model="base-model",
            log_file="/tmp/test.log",
            config=user_config,
        )
        assert config["server_args"]["api_key"] == "user-wins"

    def test_each_config_call_gets_unique_key(self) -> None:
        """Successive calls should not share the same random key."""
        keys = set()
        for _ in range(10):
            config = get_openai_server_config(
                model_name="test-model",
                base_model="base-model",
                log_file="/tmp/test.log",
            )
            keys.add(config["server_args"]["api_key"])
        assert len(keys) == 10
