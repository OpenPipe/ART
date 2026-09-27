"""Causal tests for the ``conftest.py`` tokenization golden harness.

Each test copies the harness (``conftest.py`` + ``_tokenize_golden.py``) and a
seeded ``tokenize_golden_trace.json`` into a temp directory, runs pytest there
in a subprocess with ``ART_TOKENIZE_GOLDEN`` set, and inspects the outcome and
the golden bytes. The real golden file is never read or written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from _tokenize_golden import dump_json
import pytest

_HERE = Path(__file__).parent
_HARNESS_FILES = ("conftest.py", "_tokenize_golden.py")
_PROBE = "test_probe.py"
_NODEID = f"{_PROBE}::test_probe"

# A minimal history + tokenizer that goes through ``tokenize_history``.
_TOKENIZING_PROBE = """
import art.trajectories as tr


class Tokenizer:
    def __call__(self, text, **kwargs):
        return [ord(char) % 251 + 1 for char in text]

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return self(text) if kwargs.get("tokenize", True) else text


def test_probe():
    history = tr.ChatCompletionsHistory(
        model="probe/model",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        message_sources=[None, None],
    )
    tokenized = history.tokenize(tokenizer=Tokenizer())
    assert tokenized.tokens
"""

_INJECT_DIGEST_FAILURE = """
import _tokenize_golden


def _boom(tokenized):
    raise TypeError("injected: flag is not an int")


_tokenize_golden.field_values = _boom
"""

_SILENT_PROBE = """
def test_probe():
    assert True
"""

_SEEDED_ENTRY = [
    {
        "kind": "ChatCompletionsHistory",
        "n": 3,
        "input": "0123456789abcdef",
        "output": "f" * 64,
        "fields": {
            "model": "0" * 16,
            "tokens": "1" * 16,
            "logprobs": "2" * 16,
            "flags": "3" * 16,
        },
    }
]


def _stage(tmp_path: Path, probe_source: str) -> Path:
    for name in _HARNESS_FILES:
        shutil.copy(_HERE / name, tmp_path / name)
    (tmp_path / _PROBE).write_text(probe_source)
    golden = tmp_path / "tokenize_golden_trace.json"
    dump_json(golden, {_NODEID: _SEEDED_ENTRY})
    return golden


def _run(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, ART_TOKENIZE_GOLDEN=mode)
    env.pop("PYTEST_XDIST_WORKER", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            _PROBE,
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-q",
            "--tb=short",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.no_tokenize_golden
def test_update_mode_keeps_entry_when_a_call_cannot_be_digested(
    tmp_path: Path,
) -> None:
    golden = _stage(tmp_path, _INJECT_DIGEST_FAILURE + _TOKENIZING_PROBE)
    before = golden.read_bytes()

    result = _run(tmp_path, "update")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "could not digest tokenize_history call #0" in result.stdout
    assert "injected: flag is not an int" in result.stdout
    assert golden.read_bytes() == before


@pytest.mark.no_tokenize_golden
def test_check_mode_fails_when_a_golden_entry_sees_zero_calls(tmp_path: Path) -> None:
    golden = _stage(tmp_path, _SILENT_PROBE)
    before = golden.read_bytes()

    result = _run(tmp_path, "check")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "tokenize_history was called 0 times; the golden records 1" in result.stdout
    assert golden.read_bytes() == before


@pytest.mark.no_tokenize_golden
def test_update_mode_records_a_digestible_call(tmp_path: Path) -> None:
    """Control: the same probe without the injected failure replaces the entry."""

    golden = _stage(tmp_path, _TOKENIZING_PROBE)

    result = _run(tmp_path, "update")

    assert result.returncode == 0, result.stdout + result.stderr
    entry = json.loads(golden.read_text())[_NODEID]
    assert len(entry) == 1 and entry[0]["n"] > 0
    assert entry != _SEEDED_ENTRY
