from contextlib import ExitStack
import json
from unittest.mock import Mock

import pytest

import art
from art.trajectories._capture import core


@pytest.mark.parametrize(
    "excluded", ["no_scope", "no_capture", "method", "endpoint", "nested"]
)
@pytest.mark.parametrize("encoded", [False, True])
def test_excluded_capture_does_not_parse_or_copy_body(
    monkeypatch: pytest.MonkeyPatch, excluded: str, encoded: bool
) -> None:
    body = {"model": "test", "messages": [{"role": "user", "content": "retained"}]}
    url, method = "https://example.test/v1/chat/completions", "POST"
    with ExitStack() as stack:
        if excluded != "no_scope":
            stack.enter_context(art.Trajectory())
        if excluded == "no_capture":
            stack.enter_context(art.no_capture())
        elif excluded == "method":
            method = "GET"
        elif excluded == "endpoint":
            url = "https://example.test/health"
        elif excluded == "nested":
            state, token = core.begin(method, url, body)
            assert state is not None
            stack.callback(core.reset, token)
        parser = Mock(wraps=core._json_body)
        monkeypatch.setattr(core, "_json_body", parser)
        assert core.begin(
            method, url, json.dumps(body).encode() if encoded else body
        ) == (None, None)
        parser.assert_not_called()


@pytest.mark.parametrize("body", [b"not JSON", b"[]", {1: "not a string key"}])
def test_eligible_capture_still_rejects_invalid_body(body: object) -> None:
    with art.Trajectory():
        assert core.begin("POST", "https://example.test/v1/chat/completions", body) == (
            None,
            None,
        )
