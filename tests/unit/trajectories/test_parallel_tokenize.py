from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
from functools import lru_cache
import math
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import pytest

import art
import art.trajectories as tr
from art.trajectories import _parallel, _tokenize

_CPU_CAPACITY = _parallel._cpu_capacity


def _tokenized(trajectory: art.Trajectory) -> tr.TokenizedTrajectory:
    index = int(trajectory.metadata["index"])
    return tr.TokenizedTrajectory(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=[index, index + 1],
        logprobs=[math.nan, -0.25],
        flags=[tr.TokenFlag.EXACT, tr.TokenFlag.EXACT | tr.TokenFlag.SAMPLED],
        trajectory=trajectory,
    )


def _tokenized_multi(trajectory: art.Trajectory) -> tr.TokenizedMultiHistoryTrajectory:
    return tr.TokenizedMultiHistoryTrajectory(trajectory=trajectory, histories=[])


@pytest.fixture(autouse=True)
def _reset_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_parallel, "_cpu_capacity", lambda: 4)
    with _parallel._TUNING_LOCK:
        _parallel._tuning_state.cache_clear()


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

    async def _overloads_typecheck(
        trajectories: list[art.Trajectory], groups: list[art.TrajectoryGroup]
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
        _ = (
            tokenized,
            tokenized_multi,
            tokenized_groups,
            tokenized_multi_groups,
            tensorized,
            tensorized_multi,
            tensorized_groups,
            tensorized_multi_groups,
        )
