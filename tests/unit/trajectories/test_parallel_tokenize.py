from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
from datetime import UTC, datetime, timedelta
from functools import lru_cache
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
import threading
import time
from typing import TYPE_CHECKING, Any, cast

from openai.types.chat import ChatCompletion
import pytest

import art
import art.trajectories as tr
from art.trajectories import _parallel, _tokenize

_CPU_CAPACITY = _parallel._cpu_capacity
_SUPPORTS_PROCESSES = _parallel._supports_processes


def _tokenized(trajectory: art.Trajectory) -> tr.TokenizedTrajectory:
    index = int(trajectory.metadata["index"])
    return tr.TokenizedTrajectory(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=[index, index + 1],
        logprobs=[math.nan, -0.25],
        flags=[
            tr.TokenFlag.EXACT,
            tr.TokenFlag.EXACT | tr.TokenFlag.SAMPLED | tr.TokenFlag.OUTPUT,
        ],
        trajectory=trajectory,
    )


def _tokenized_multi(trajectory: art.Trajectory) -> tr.TokenizedMultiHistoryTrajectory:
    return tr.TokenizedMultiHistoryTrajectory(trajectory=trajectory, histories=[])


def _chat_completion(index: int) -> ChatCompletion:
    token = index + 2
    return ChatCompletion.model_validate(
        {
            "id": f"chat-{index}",
            "object": "chat.completion",
            "created": 0,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                    "prompt_token_ids": [1],
                    "token_ids": [token],
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{token}",
                                "logprob": -0.2,
                                "bytes": [],
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ],
        }
    )


def _legacy_trajectory(index: int) -> art.Trajectory:
    response = _chat_completion(index)
    return art.Trajectory(
        messages_and_choices=[response.choices[0]], metadata={"index": index}
    )


def _exchange_trajectory(index: int) -> art.Trajectory:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return art.Trajectory(
        exchanges=tr.TrajectoryExchanges(
            chat_completions=[
                tr.ChatCompletionsExchange(
                    request=tr.ChatCompletionsRequest(
                        model="test/model",
                        messages=[{"role": "user", "content": "question"}],
                    ),
                    response=_chat_completion(index),
                    start_time=start,
                    end_time=start + timedelta(milliseconds=1),
                )
            ]
        ),
        metadata={"index": index},
    )


@pytest.fixture(autouse=True)
def _reset_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_parallel, "_cpu_capacity", lambda: 4)
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: False)
    monkeypatch.setattr(_parallel, "_PROCESS_BACKEND_DISABLED", False)
    with _parallel._TUNING_LOCK:
        _parallel._tuning_state.cache_clear()
        _parallel._process_tuning_state.cache_clear()


def _patch_tokenize(
    monkeypatch: pytest.MonkeyPatch,
    function: Callable[
        [art.Trajectory, dict[str, object]],
        tr.TokenizedTrajectory | tr.TokenizedMultiHistoryTrajectory,
    ],
) -> None:
    def tokenize(
        self: art.Trajectory, **kwargs: object
    ) -> tr.TokenizedTrajectory | tr.TokenizedMultiHistoryTrajectory:
        return function(self, kwargs)

    monkeypatch.setattr(art.Trajectory, "tokenize", tokenize)


async def test_tokenize_preserves_order_and_parallelizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0
    threads: set[str] = set()
    received: list[dict[str, object]] = []

    def tokenize(
        trajectory: art.Trajectory, kwargs: dict[str, object]
    ) -> tr.TokenizedTrajectory:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            threads.add(threading.current_thread().name)
            received.append(kwargs)
        time.sleep(0.02 * (4 - int(trajectory.metadata["index"])))
        with lock:
            active -= 1
        return _tokenized(trajectory)

    _patch_tokenize(monkeypatch, tokenize)
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(4)]
    tokenizer = cast(Any, object())

    result = await art.tokenize(
        (trajectory for trajectory in trajectories),
        multi_history=False,
        reconcile_text_equivalent_tokenizations=True,
        model="policy",
        base_model="base",
        tokenizer=tokenizer,
        chat_template="template",
        chat_template_kwargs={"enable_thinking": True},
    )

    assert [value.trajectory for value in result] == trajectories
    assert maximum == 4
    assert all(name.startswith("art-tokenize") for name in threads)
    assert all(
        kwargs
        == {
            "multi_history": False,
            "reconcile_text_equivalent_tokenizations": True,
            "model": "policy",
            "base_model": "base",
            "tokenizer": tokenizer,
            "chat_template": "template",
            "chat_template_kwargs": {"enable_thinking": True},
        }
        for kwargs in received
    )


async def test_tokenize_groups_flattens_and_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tokenize(monkeypatch, lambda trajectory, _: _tokenized(trajectory))
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(3)]
    groups = [
        art.TrajectoryGroup(
            trajectories[:2], metadata={"name": "first"}, metrics={"score": 1}
        ),
        art.TrajectoryGroup(trajectories[2:], metadata={"name": "second"}),
        art.TrajectoryGroup(metadata={"name": "empty"}),
    ]

    result = await art.tokenize(groups)

    assert [value.trajectory_group for value in result] == groups
    assert [
        [trajectory.trajectory for trajectory in value.trajectories] for value in result
    ] == [trajectories[:2], trajectories[2:], []]
    assert result[0].metadata is groups[0].metadata
    assert result[0].metrics is groups[0].metrics


async def test_multi_history_group_structure_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = art.Trajectory(metadata={"index": 0})
    _patch_tokenize(monkeypatch, lambda trajectory, _: _tokenized_multi(trajectory))
    group = art.TrajectoryGroup([trajectory], metadata={"name": "multi"})

    tokenized = await art.tokenize([group], multi_history=True)
    tensorized = await art.tensorize([group], multi_history=True)

    assert tokenized[0].trajectory_group is group
    assert tokenized[0].trajectories[0].trajectory is trajectory
    assert tensorized[0].trajectory_group is group
    assert tensorized[0].trajectories[0].trajectory is trajectory


async def test_empty_and_mixed_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tokenize(monkeypatch, lambda trajectory, _: _tokenized(trajectory))

    assert await art.tokenize([]) == []
    assert await art.tensorize([]) == []
    empty_tensorized_groups = await art.tensorize([art.TrajectoryGroup()], device="cpu")
    assert len(empty_tensorized_groups) == 1
    assert empty_tensorized_groups[0].trajectories == []
    with pytest.raises(TypeError, match="only trajectories or only trajectory groups"):
        await art.tokenize(  # ty: ignore[no-matching-overload]
            cast(
                Any,
                [
                    art.Trajectory(metadata={"index": 0}),
                    art.TrajectoryGroup(),
                ],
            )
        )


async def test_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def tokenize(
        trajectory: art.Trajectory, _: dict[str, object]
    ) -> tr.TokenizedTrajectory:
        index = int(trajectory.metadata["index"])
        if index == 1:
            time.sleep(0.04)
            raise RuntimeError("one")
        time.sleep(0.02)
        return _tokenized(trajectory)

    _patch_tokenize(monkeypatch, tokenize)
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(4)]

    with pytest.raises(RuntimeError, match="one"):
        await art.tokenize(trajectories)


async def test_tensorize_preserves_sources_and_moves_after_parallel_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _patch_tokenize(monkeypatch, lambda trajectory, _: _tokenized(trajectory))
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(3)]

    result = await art.tensorize(trajectories, device="cpu")
    grouped = await art.tensorize(
        [art.TrajectoryGroup(trajectories)], device=torch.device("cpu")
    )

    assert [value.trajectory for value in result] == trajectories
    assert all(value.tokens.device.type == "cpu" for value in result)
    assert grouped[0].trajectory_group.trajectories == trajectories
    assert [value.trajectory for value in grouped[0].trajectories] == trajectories


def test_tuner_probes_and_retains_the_best_rate() -> None:
    key = ("test",)
    assert _parallel._workers(key, capacity=16, size=16) == 4

    _parallel._observe(key, workers=4, capacity=16, size=16, units=400, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 8
    _parallel._observe(key, workers=8, capacity=16, size=16, units=600, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 16
    _parallel._observe(key, workers=16, capacity=16, size=16, units=500, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 8


def test_tuner_probes_down_and_uses_smallest_near_peak_pool() -> None:
    key = ("small-pool",)
    assert _parallel._workers(key, capacity=16, size=16) == 4

    _parallel._observe(key, workers=4, capacity=16, size=16, units=400, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 8
    _parallel._observe(key, workers=8, capacity=16, size=16, units=390, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 2

    _parallel._observe(key, workers=2, capacity=16, size=16, units=385, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 1
    _parallel._observe(key, workers=1, capacity=16, size=16, units=300, elapsed=1)
    assert _parallel._workers(key, capacity=16, size=16) == 2


def test_tuner_probes_intermediate_worker_count_for_odd_input_size() -> None:
    key = ("odd-sized",)
    assert _parallel._workers(key, capacity=16, size=3) == 3

    _parallel._observe(key, workers=3, capacity=16, size=3, units=300, elapsed=1)

    assert _parallel._workers(key, capacity=16, size=3) == 2


def test_process_tuner_ignores_cold_start_and_compares_two_with_four() -> None:
    key = ("process",)
    assert _parallel._process_workers(key, capacity=16, size=16) == 4

    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=100, elapsed=10
    )
    assert _parallel._process_workers(key, capacity=16, size=16) == 4

    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=400, elapsed=1
    )
    assert _parallel._process_workers(key, capacity=16, size=16) == 4
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=400, elapsed=1
    )
    assert _parallel._process_workers(key, capacity=16, size=16) == 2

    _parallel._observe_process(
        key, workers=2, capacity=16, size=16, units=390, elapsed=1
    )
    assert _parallel._process_workers(key, capacity=16, size=16) == 2
    _parallel._observe_process(
        key, workers=2, capacity=16, size=16, units=390, elapsed=1
    )
    assert _parallel._process_workers(key, capacity=16, size=16) == 2


def test_process_tuner_returns_to_four_when_two_is_slower() -> None:
    key = ("process-slower",)
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=100, elapsed=10
    )
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=400, elapsed=1
    )
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=400, elapsed=1
    )
    _parallel._observe_process(
        key, workers=2, capacity=16, size=16, units=300, elapsed=1
    )
    _parallel._observe_process(
        key, workers=2, capacity=16, size=16, units=300, elapsed=1
    )

    assert _parallel._process_workers(key, capacity=16, size=16) == 4


def test_process_backend_is_considered_only_after_slow_thread_work() -> None:
    fast_key = ("fast-thread",)
    slow_key = ("slow-thread",)
    for key in (fast_key, slow_key):
        _parallel._observe(key, workers=4, capacity=16, size=16, units=100, elapsed=2)
        _parallel._observe(key, workers=4, capacity=16, size=16, units=100, elapsed=2)

    _parallel._consider_processes(fast_key, elapsed=0.1)
    _parallel._consider_processes(slow_key, elapsed=2.0)

    assert not _parallel._processes_enabled(fast_key)
    assert _parallel._processes_enabled(slow_key)


def test_process_backend_returns_to_threads_when_warm_rate_is_not_better() -> None:
    key = ("threads-win",)
    _parallel._observe(key, workers=4, capacity=16, size=16, units=400, elapsed=1)
    _parallel._observe(key, workers=4, capacity=16, size=16, units=400, elapsed=1)
    _parallel._consider_processes(key, elapsed=1)
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=100, elapsed=10
    )
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=300, elapsed=1
    )
    _parallel._observe_process(
        key, workers=4, capacity=16, size=16, units=300, elapsed=1
    )

    assert not _parallel._processes_enabled(key)


def test_process_context_does_not_spawn_through_user_main() -> None:
    assert _parallel._process_context().get_start_method() == "spawn"


def test_process_context_does_not_reexecute_unguarded_script(
    tmp_path: Path,
) -> None:
    script = tmp_path / "unguarded.py"
    marker = tmp_path / "executions"
    script.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            from art.trajectories._parallel import _discard_process_executor, _process_executor

            marker = Path({str(marker)!r})
            with marker.open("a") as output:
                output.write("run\\n")
            pool = _process_executor(2)
            assert list(pool.map(abs, [-1, -2])) == [1, 2]
            _discard_process_executor()
            """
        )
    )

    subprocess.run([sys.executable, str(script)], check=True, cwd=Path.cwd())

    assert marker.read_text() == "run\n"


def test_child_deserialization_failure_is_a_transfer_error() -> None:
    with pytest.raises(_parallel._ProcessTransferError, match="process input"):
        _parallel._tokenize_process_payload(b"not a pickle")


def test_process_input_pickle_does_not_mutate_or_mark_caller_trajectory() -> None:
    trajectory = _exchange_trajectory(0)
    request_keys = list(trajectory.exchanges.chat_completions[0].request)
    options = _parallel._ProcessOptions(
        multi_history=False,
        reconcile_text_equivalent_tokenizations=False,
        model=None,
        base_model=None,
        chat_template=None,
        chat_template_kwargs=None,
    )

    payloads = _parallel._process_payloads([trajectory], options)

    assert payloads
    assert list(trajectory.exchanges.chat_completions[0].request) == request_keys
    assert not getattr(trajectory, "_art_pickle_strings_interned", False)


def test_broken_pool_disables_processes_for_the_interpreter() -> None:
    with pytest.warns(RuntimeWarning, match="using threads"):
        _parallel._disable_process_backend(_parallel.BrokenProcessPool("worker exited"))

    assert not _SUPPORTS_PROCESSES(capacity=4, size=4, tokenizer=None)


async def test_process_results_rebind_to_original_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_, **__: True)

    async def process_map(
        payloads: list[bytes],
        trajectories: list[art.Trajectory],
        *,
        workers: int,
        capacity: int,
    ) -> list[tr.TokenizedTrajectory]:
        del workers, capacity
        copies = [
            cast(tuple[art.Trajectory, object], pickle.loads(payload))[0]
            for payload in payloads
        ]
        return [
            cast(
                tr.TokenizedTrajectory,
                _parallel._deserialize_process_result(
                    pickle.dumps(_tokenized(copied)), trajectory
                ),
            )
            for copied, trajectory in zip(copies, trajectories, strict=True)
        ]

    monkeypatch.setattr(_parallel, "_ordered_process_map", process_map)
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(4)]

    result = await art.tokenize(trajectories)

    assert [value.trajectory for value in result] == trajectories
    assert all(
        value.trajectory is trajectory
        for value, trajectory in zip(result, trajectories, strict=True)
    )


def test_process_rebind_uses_exchange_position_not_deep_equality() -> None:
    trajectory = _exchange_trajectory(0)
    copied = cast(art.Trajectory, pickle.loads(pickle.dumps(trajectory)))
    tokenized = copied.tokenize()
    trajectory.exchanges.chat_completions[0].request["temperature"] = math.nan

    _parallel._rebind_process_result(tokenized, trajectory)

    source = cast(tr.ChatCompletionsHistory, tokenized.history).message_sources[-1]
    assert source is not None
    assert source.exchange is trajectory.exchanges.chat_completions[0]
    assert tokenized.trajectory is trajectory


async def test_process_tensorization_runs_in_parent_thread_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_, **__: True)
    parent_pid = os.getpid()
    tensorize_processes: list[int] = []
    tensorize_threads: list[str] = []
    original_tensorize = tr.TokenizedTrajectory.tensorize

    async def process_map(
        payloads: list[bytes],
        trajectories: list[art.Trajectory],
        *,
        workers: int,
        capacity: int,
    ) -> list[tr.TokenizedTrajectory]:
        del workers, capacity
        copies = [
            cast(tuple[art.Trajectory, object], pickle.loads(payload))[0]
            for payload in payloads
        ]
        return [
            cast(
                tr.TokenizedTrajectory,
                _parallel._deserialize_process_result(
                    pickle.dumps(_tokenized(copied)), trajectory
                ),
            )
            for copied, trajectory in zip(copies, trajectories, strict=True)
        ]

    def tensorize(
        self: tr.TokenizedTrajectory, *, device: object = None
    ) -> tr.TensorizedTrajectory:
        tensorize_processes.append(os.getpid())
        tensorize_threads.append(threading.current_thread().name)
        return original_tensorize(self, device=cast(Any, device))

    monkeypatch.setattr(_parallel, "_ordered_process_map", process_map)
    monkeypatch.setattr(tr.TokenizedTrajectory, "tensorize", tensorize)
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(4)]

    result = await art.tensorize(trajectories, device="cpu")

    assert tensorize_processes == [parent_pid] * 4
    assert all(name.startswith("art-tokenize") for name in tensorize_threads)
    assert all(value.tokens.device.type == "cpu" for value in result)


async def test_unpickleable_process_input_warns_once_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)

    def processes_enabled(key: tuple[object, ...]) -> bool:
        return not _parallel._process_tuning_state(key).disabled

    monkeypatch.setattr(_parallel, "_processes_enabled", processes_enabled)
    _patch_tokenize(monkeypatch, lambda trajectory, _: _tokenized(trajectory))
    attempts = 0

    def fail_payloads(*_: object) -> list[bytes]:
        nonlocal attempts
        attempts += 1
        raise _parallel._ProcessTransferError("not pickleable")

    monkeypatch.setattr(_parallel, "_process_payloads", fail_payloads)
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(4)]

    with pytest.warns(RuntimeWarning, match="using threads"):
        first = await art.tokenize(trajectories)
    second = await art.tokenize(trajectories)

    assert attempts == 1
    assert [value.trajectory for value in first] == trajectories
    assert [value.trajectory for value in second] == trajectories


async def test_process_tokenization_errors_are_not_retried_in_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_, **__: True)

    async def process_map(*_: object, **__: object) -> list[tr.TokenizedTrajectory]:
        raise ValueError("invalid trajectory")

    monkeypatch.setattr(_parallel, "_ordered_process_map", process_map)
    trajectories = [art.Trajectory(metadata={"index": index}) for index in range(4)]

    with pytest.raises(ValueError, match="invalid trajectory"):
        await art.tokenize(trajectories)


async def test_spawn_process_pool_tokenizes_serialized_trajectories() -> None:
    trajectories = [_legacy_trajectory(index) for index in range(4)]
    options = _parallel._ProcessOptions(
        multi_history=False,
        reconcile_text_equivalent_tokenizations=False,
        model="test/model",
        base_model=None,
        chat_template=None,
        chat_template_kwargs=None,
    )

    try:
        result = await _parallel._ordered_process_map(
            _parallel._process_payloads(trajectories, options),
            trajectories,
            workers=2,
            capacity=2,
        )
    finally:
        _parallel._discard_process_executor()

    tokenized = cast(list[tr.TokenizedTrajectory], result)
    assert [value.tokens for value in tokenized] == [[1, 2], [1, 3], [1, 4], [1, 5]]


def test_workload_bucket_uses_average_history_branches() -> None:
    trajectories = [
        art.Trajectory(
            additional_histories=[
                tr.LegacyHistory(messages_and_choices=[]) for _ in range(count)
            ]
        )
        for count in (1, 4, 7)
    ]

    assert _parallel._workload_bucket(trajectories) == 4


def test_cpu_capacity_honors_affinity_and_cgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(_parallel.os, "sched_getaffinity", lambda _: set(range(16)))
    monkeypatch.setattr(_parallel, "_cgroup_cpu_limit", lambda: 6)

    assert _CPU_CAPACITY() == 6


def test_process_backend_requires_capacity_batch_and_automatic_tokenizer() -> None:
    assert _SUPPORTS_PROCESSES(capacity=4, size=4, tokenizer=None)
    assert not _SUPPORTS_PROCESSES(capacity=1, size=4, tokenizer=None)
    assert not _SUPPORTS_PROCESSES(capacity=4, size=3, tokenizer=None)
    assert not _SUPPORTS_PROCESSES(capacity=4, size=4, tokenizer=cast(Any, object()))


def test_automatic_tokenizer_loading_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls = 0
    tokenizer = cast(Any, object())

    @lru_cache(maxsize=1)
    def load(_: str, __: str | None) -> Any:
        nonlocal active, calls, maximum
        with lock:
            calls += 1
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return tokenizer

    monkeypatch.setattr(_tokenize, "_cached_tokenizer", load)
    config = _tokenize._TokenizerConfig("model")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_tokenize._load_tokenizer, [config] * 4))

    assert results == [tokenizer] * 4
    assert calls == 1
    assert maximum == 1


def test_root_and_trajectory_exports_are_identical() -> None:
    assert art.tokenize is tr.tokenize
    assert art.tensorize is tr.tensorize


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    async def _overloads_typecheck(
        trajectories: list[art.Trajectory],
        groups: list[art.TrajectoryGroup],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        tokenized: list[tr.TokenizedTrajectory] = await art.tokenize(trajectories)
        tokenized_multi: list[tr.TokenizedMultiHistoryTrajectory] = await art.tokenize(
            trajectories, multi_history=True
        )
        tokenized_groups: list[
            tr.TokenizedTrajectoryGroup[tr.TokenizedTrajectory]
        ] = await art.tokenize(groups)
        tokenized_multi_groups: list[
            tr.TokenizedTrajectoryGroup[tr.TokenizedMultiHistoryTrajectory]
        ] = await art.tokenize(groups, multi_history=True)
        tensorized: list[tr.TensorizedTrajectory] = await art.tensorize(trajectories)
        tensorized_multi: list[
            tr.TensorizedMultiHistoryTrajectory
        ] = await art.tensorize(trajectories, multi_history=True)
        tensorized_groups: list[
            tr.TensorizedTrajectoryGroup[tr.TensorizedTrajectory]
        ] = await art.tensorize(groups)
        tensorized_multi_groups: list[
            tr.TensorizedTrajectoryGroup[tr.TensorizedMultiHistoryTrajectory]
        ] = await art.tensorize(groups, multi_history=True)
        with_transformers_tokenizer: list[
            tr.TensorizedTrajectory
        ] = await art.tensorize(trajectories, tokenizer=tokenizer)
        _ = (
            tokenized,
            tokenized_multi,
            tokenized_groups,
            tokenized_multi_groups,
            tensorized,
            tensorized_multi,
            tensorized_groups,
            tensorized_multi_groups,
            with_transformers_tokenizer,
        )
