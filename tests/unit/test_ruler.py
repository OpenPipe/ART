"""Unit tests for RULER Ollama JSON schema support (Issue #476)."""

import pytest

from art.rewards.ruler import (
    Response,
    _get_response_format_for_ollama,
    _is_ollama_model,
)


class TestIsOllamaModel:
    """Tests for _is_ollama_model helper function."""

    def test_ollama_prefix(self):
        """Test detection of ollama/ prefix."""
        assert _is_ollama_model("ollama/qwen2.5:7b") is True

    def test_ollama_chat_prefix(self):
        """Test detection of ollama_chat/ prefix."""
        assert _is_ollama_model("ollama_chat/qwen2.5:7b") is True

    def test_non_ollama_models(self):
        """Test that non-Ollama models return False."""
        assert _is_ollama_model("openai/gpt-4o") is False
        assert _is_ollama_model("openai/o3") is False


class TestGetResponseFormatForOllama:
    """Tests for _get_response_format_for_ollama helper function."""

    def test_returns_correct_structure(self):
        """Test that the function returns the correct JSON schema structure."""
        result = _get_response_format_for_ollama(Response)

        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["name"] == "Response"
        assert "schema" in result["json_schema"]

    def test_schema_contains_expected_properties(self):
        """Test that the schema contains the expected properties from Response model."""
        result = _get_response_format_for_ollama(Response)
        schema = result["json_schema"]["schema"]

        # Response model should have 'scores' property
        assert "properties" in schema
        assert "scores" in schema["properties"]

    def test_schema_is_valid_json_schema(self):
        """Test that the generated schema is a valid structure."""
        result = _get_response_format_for_ollama(Response)

        # Verify it has the structure LiteLLM expects for Ollama
        assert isinstance(result, dict)
        assert result["type"] == "json_schema"
        assert isinstance(result["json_schema"], dict)
        assert isinstance(result["json_schema"]["schema"], dict)
