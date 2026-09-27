"""Opt-in byte-exact tracing of every ``tokenize_history`` call in this directory.

Inactive unless ``ART_TOKENIZE_GOLDEN`` is set (no fixtures requested, nothing
patched):

* ``ART_TOKENIZE_GOLDEN=check pytest tests/unit/trajectories`` compares each
  passing test's ``tokenize_history`` outputs (tokens, logprobs, flags, model)
  against ``tokenize_golden_trace.json``. A difference is reported as a
  teardown ERROR for that test (the test body itself already passed). Any
  other non-empty value also means ``check``.
* ``ART_TOKENIZE_GOLDEN=update pytest tests/unit/trajectories`` rewrites the
  golden. Works with or without ``-n``; xdist workers write fragments under
  pytest's shared base temp directory that the controller merges at session end.
  Only entries for tests whose call phase ran and passed are replaced or
  dropped; skipped, deselected, failed or interrupted tests keep their
  existing entries, and an interrupted session writes nothing.

The public tokenization entry points (``History.tokenize``,
``Trajectory.tokenize``, ``TrajectoryGroup.tokenize``, ``tokenize_trajectory``,
``tokenize_group``) funnel through ``art.trajectories._tokenize.tokenize_history``
via a call-time import, so wrapping that one module attribute observes them all
without touching ``src/``. Not traced: direct calls to ``_tokenize_history`` or
``_tokenize_chat_view`` in tests, and work done inside ``_parallel``'s spawned
worker processes (the patch lives only in the pytest process). Calls that raise
are not recorded. Digests are computed after the test body finishes so timing
tests are not perturbed; a call the harness cannot digest fails the test in
either mode and leaves its golden entry untouched. Tests marked
``@pytest.mark.no_tokenize_golden`` are never traced. In check mode a passing
test with a golden entry but no calls fails like any call-count mismatch; tests
absent from the golden are listed at session end, not failed.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

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
_SKIP_MARK = "no_tokenize_golden"

_MODE = mode()
_RECORDED: dict[str, list[Digest]] = {}
_PASSED: set[str] = set()
_UNKNOWN: list[str] = []
_GOLDEN: dict[str, list[Digest]] | None = None

# One tokenize_history call, captured cheaply during the test and digested later.
_RawCall = tuple[Any, dict[str, Any], str, list[int], list[float], list[Any]]


def _golden() -> dict[str, list[Digest]]:
    global _GOLDEN
    if _GOLDEN is None:
        _GOLDEN = load_json(GOLDEN_TRACE)
    return _GOLDEN


def _is_worker(config: pytest.Config) -> bool:
    return hasattr(config, "workerinput")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{_SKIP_MARK}: exclude this test from {MODE_ENV} tokenization tracing",
    )


def _part_dir(config: pytest.Config) -> Path:
    """Base temp directory shared by the xdist controller and its workers.

    xdist hands each worker ``<controller basetemp>/popen-<id>`` as its
    basetemp, so the parent is common to all of them. Reads the same private
    attribute xdist itself uses; falls back to the system temp directory.
    """

    factory = getattr(config, "_tmp_path_factory", None)
    if factory is None:
        return Path(tempfile.gettempdir())
    base = Path(factory.getbasetemp())
    return base.parent if _is_worker(config) else base


def _part_paths(config: pytest.Config) -> list[Path]:
    return sorted(_part_dir(config).glob(f"{GOLDEN_TRACE.stem}.*.part.json"))


def _part_path(config: pytest.Config) -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return _part_dir(config) / f"{GOLDEN_TRACE.stem}.{worker}.part.json"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if _MODE and report.when == "call" and report.passed:
        _PASSED.add(item.nodeid)
    return report


def _digest(raw: _RawCall) -> Digest:
    history, kwargs, model, tokens, logprobs, flags = raw
    return output_digest(
        SimpleNamespace(
            history=history, model=model, tokens=tokens, logprobs=logprobs, flags=flags
        ),
        input_hash=input_digest(
            history,
            model=kwargs.get("model"),
            base_model=kwargs.get("base_model"),
            tokenizer=kwargs.get("tokenizer"),
            chat_template=kwargs.get("chat_template"),
            chat_template_kwargs=kwargs.get("chat_template_kwargs"),
        ),
    )


@pytest.fixture(autouse=True)
def _tokenize_golden_trace(request: pytest.FixtureRequest) -> Iterator[None]:
    if not _MODE or request.node.get_closest_marker(_SKIP_MARK) is not None:
        yield
        return

    from art.trajectories import _tokenize

    original = _tokenize.tokenize_history
    raw_calls: list[_RawCall] = []

    def traced(history: Any, **kwargs: Any) -> Any:
        tokenized = original(history, **kwargs)
        raw_calls.append(
            (
                history,
                dict(kwargs),
                tokenized.model,
                list(tokenized.tokens),
                list(tokenized.logprobs),
                list(tokenized.flags),
            )
        )
        return tokenized

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_tokenize, "tokenize_history", traced)
        yield

    nodeid = request.node.nodeid
    if nodeid not in _PASSED:
        return
    calls: list[Digest] = []
    for index, raw in enumerate(raw_calls):
        try:
            calls.append(_digest(raw))
        except Exception as error:  # noqa: BLE001 - reported below as a harness bug
            # An undigestable output is a harness bug. Never store a truncated
            # entry: un-mark the node so the retention rule leaves its existing
            # golden entry untouched, then fail loudly.
            _PASSED.discard(nodeid)
            pytest.fail(
                f"{nodeid}: tokenize golden could not digest tokenize_history call "
                f"#{index}: {type(error).__name__}: {error}. Its golden entry was "
                "left unchanged.",
                pytrace=False,
            )
    if _MODE == "update":
        if calls:
            _RECORDED[nodeid] = calls
        return
    expected = _golden().get(nodeid)
    if expected is None:
        if calls:
            _UNKNOWN.append(nodeid)
        return
    # Zero calls against an existing entry is a count mismatch too: a refactor
    # that stops tokenizing must not pass silently.
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


def _merge_parts(config: pytest.Config, recorded: dict[str, list[Digest]]) -> None:
    """Fold xdist worker fragments into ``recorded``, ``_PASSED`` and ``_UNKNOWN``."""

    for part in _part_paths(config):
        payload = load_json(part)
        _PASSED.update(payload.get("passed", []))
        _UNKNOWN.extend(payload.get("unknown", []))
        recorded.update(payload.get("recorded", {}))
        part.unlink()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _MODE or exitstatus == pytest.ExitCode.INTERRUPTED:
        return
    config = session.config
    if _is_worker(config):
        dump_json(
            _part_path(config),
            {
                "passed": sorted(_PASSED),
                "unknown": _UNKNOWN,
                "recorded": _RECORDED,
            },
        )
        return
    recorded = dict(_RECORDED)
    _merge_parts(config, recorded)
    if _MODE != "update":
        return
    golden = load_json(GOLDEN_TRACE)
    # A test that passed without tokenizing (or was renamed) loses its entry.
    # Skipped, deselected, failed, interrupted or unrun tests keep theirs, so
    # `-k`, `--deselect` and partial runs never erase valid goldens.
    for nodeid in list(golden):
        if nodeid in _PASSED and nodeid not in recorded:
            del golden[nodeid]
    golden.update(recorded)
    dump_json(GOLDEN_TRACE, golden)


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    if _MODE != "check" or _is_worker(config) or not _UNKNOWN:
        return
    terminalreporter.write_sep(
        "-",
        f"tokenize golden: {len(_UNKNOWN)} traced test(s) without a golden entry; "
        f"run with {MODE_ENV}=update to record them",
    )
    for nodeid in sorted(_UNKNOWN)[:20]:
        terminalreporter.write_line(f"  {nodeid}")
