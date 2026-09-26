"""Opt-in byte-exact tracing of every ``tokenize_history`` call in this directory.

Inactive unless ``ART_TOKENIZE_GOLDEN`` is set:

* ``ART_TOKENIZE_GOLDEN=check pytest tests/unit/trajectories`` compares each
  test's ``tokenize_history`` outputs (tokens, logprobs, flags, model) against
  ``tokenize_golden_trace.json`` and fails the test on the first difference.
* ``ART_TOKENIZE_GOLDEN=update pytest tests/unit/trajectories`` rewrites the
  golden. Works with or without ``-n``; xdist workers write ``*.part.json``
  fragments that the controller merges at session end.

Every public tokenization path (``History.tokenize``, ``Trajectory.tokenize``,
``TrajectoryGroup.tokenize``, ``tokenize_trajectory``, ``tokenize_group``) funnels
through ``art.trajectories._tokenize.tokenize_history`` via a call-time import,
so wrapping that one module attribute observes them all without touching
``src/``. Calls that raise are not recorded. Tests that never tokenize have no
entry. Tests absent from the golden are reported at session end, not failed.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any
import warnings

from _tokenize_golden import (
    MODE_ENV,
    Digest,
    describe_mismatch,
    dump_json,
    input_digest,
    load_json,
    mode,
    output_digest,
)
import pytest

_HERE = Path(__file__).parent
GOLDEN_TRACE = _HERE / "tokenize_golden_trace.json"

_MODE = mode()
_RECORDED: dict[str, list[Digest]] = {}
_COLLECTED: set[str] = set()
_UNKNOWN: list[str] = []
_GOLDEN: dict[str, list[Digest]] | None = None


def _golden() -> dict[str, list[Digest]]:
    global _GOLDEN
    if _GOLDEN is None:
        _GOLDEN = load_json(GOLDEN_TRACE)
    return _GOLDEN


def _is_worker(config: pytest.Config) -> bool:
    return hasattr(config, "workerinput")


def _part_path(config: pytest.Config) -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return GOLDEN_TRACE.with_name(f"{GOLDEN_TRACE.stem}.{worker}.part.json")


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not _MODE:
        return
    for item in items:
        if Path(str(item.path)).is_relative_to(_HERE):
            _COLLECTED.add(item.nodeid)


@pytest.fixture(autouse=True)
def _tokenize_golden_trace(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if not _MODE:
        yield
        return

    from art.trajectories import _tokenize

    original = _tokenize.tokenize_history
    calls: list[Digest] = []

    def traced(history: Any, **kwargs: Any) -> Any:
        tokenized = original(history, **kwargs)
        try:
            calls.append(
                output_digest(
                    tokenized,
                    input_hash=input_digest(
                        history,
                        model=kwargs.get("model"),
                        base_model=kwargs.get("base_model"),
                        tokenizer=kwargs.get("tokenizer"),
                        chat_template=kwargs.get("chat_template"),
                        chat_template_kwargs=kwargs.get("chat_template_kwargs"),
                    ),
                )
            )
        except Exception as error:  # noqa: BLE001 - observation must not alter tests
            warnings.warn(
                f"tokenize golden trace could not digest a call in {request.node.nodeid}: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
        return tokenized

    monkeypatch.setattr(_tokenize, "tokenize_history", traced)
    yield
    monkeypatch.undo()

    nodeid = request.node.nodeid
    if not calls:
        return
    if _MODE == "update":
        _RECORDED[nodeid] = calls
        return
    expected = _golden().get(nodeid)
    if expected is None:
        _UNKNOWN.append(nodeid)
        return
    if len(expected) != len(calls):
        pytest.fail(
            f"{nodeid}: tokenize_history was called {len(calls)} times; the golden "
            f"records {len(expected)}. Regenerate with {MODE_ENV}=update if intended.",
            pytrace=False,
        )
    for index, (want, got) in enumerate(zip(expected, calls, strict=True)):
        if want["output"] != got["output"]:
            pytest.fail(
                describe_mismatch(
                    f"{nodeid} (tokenize_history call #{index})", want, got
                ),
                pytrace=False,
            )


def _merge_parts(recorded: dict[str, list[Digest]]) -> None:
    """Fold xdist worker fragments into ``recorded`` and ``_COLLECTED``."""

    for part in sorted(GOLDEN_TRACE.parent.glob(f"{GOLDEN_TRACE.stem}.*.part.json")):
        payload = load_json(part)
        _COLLECTED.update(payload.get("collected", []))
        recorded.update(payload.get("recorded", {}))
        part.unlink()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _MODE:
        return
    config = session.config
    if _MODE == "update":
        if _is_worker(config):
            dump_json(
                _part_path(config),
                {"collected": sorted(_COLLECTED), "recorded": _RECORDED},
            )
            return
        recorded = dict(_RECORDED)
        _merge_parts(recorded)
        golden = load_json(GOLDEN_TRACE)
        # Collected tests that no longer tokenize (or were renamed) drop out;
        # tests outside this run keep their entries so `-k` updates are safe.
        for nodeid in list(golden):
            if nodeid in _COLLECTED and nodeid not in recorded:
                del golden[nodeid]
        golden.update(recorded)
        dump_json(GOLDEN_TRACE, golden)
        return
    if _UNKNOWN and not _is_worker(config):
        warnings.warn(
            f"{len(_UNKNOWN)} test(s) tokenized histories but have no golden entry; "
            f"run with {MODE_ENV}=update to record them (first: {_UNKNOWN[0]})",
            stacklevel=1,
        )


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    if _MODE != "check" or _is_worker(config):
        return
    if _UNKNOWN:
        terminalreporter.write_sep(
            "-",
            f"tokenize golden: {len(_UNKNOWN)} traced test(s) without a golden entry",
        )
        for nodeid in _UNKNOWN[:20]:
            terminalreporter.write_line(f"  {nodeid}")
