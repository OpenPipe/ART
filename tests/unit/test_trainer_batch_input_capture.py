"""Batch iterators own submitted inputs while executing one wave at a time."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from test_trainer_rank_commands import _Rank
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOptions,
    MicroBatch,
    MicroBatchStats,
    TrainerRank,
    Unset,
    run_rank_callback,
)
from art.trainer_rank._rng import TrainerRNG


class _CapturingRank(_Rank):
    _capture_forward_options = TrainerRank._capture_forward_options


@pytest.mark.parametrize("surface", ["native", "rank", "zero", "persistent"])
@pytest.mark.parametrize("policy", [None, ForwardOptions(max_gradient_staleness=1)])
def test_batches_snapshot_tokens_targets_and_structure_before_first_pull(
    surface, policy
):
    calls = []
    rank: Any
    if surface == "native":
        rank = object.__new__(TrainerRank)
        rank._rng = TrainerRNG(torch.device("cpu"))
        rank._skipped_forward_waves = {}

        def batches(inputs, **kwargs):
            for index, item in enumerate(inputs):
                calls.append(index)
                yield MicroBatch(
                    [item],
                    [],
                    [index],
                    MicroBatchStats(index, index + 1, 1, 1, 0, 0, 0, 0, 0, False),
                )

        rank._forward_batches = batches
    else:
        rank = cast(Any, _CapturingRank())
        forward = rank.forward

        def record(inputs, **kwargs):
            if isinstance(inputs, ForwardInput):
                calls.append(inputs.input_tokens.tolist())
            return forward(inputs, **kwargs)

        rank.forward = record
    rank._forward_options = policy
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    targets = torch.arange(12, dtype=torch.int64).reshape(3, 4)[:, ::2]
    request = ForwardInput(input_tokens=tokens, target_tokens=targets, checkpoint=None)
    later = ForwardInput(input_tokens=torch.tensor([4, 5]), checkpoint=Unset)
    roots = [(request,), [later]]
    expected_tokens, expected_targets = tokens.clone(), targets.clone()

    def mutate_sources():
        tokens.fill_(9)
        targets.fill_(-100)
        request.input_tokens = torch.tensor([11])
        request.target_tokens = None
        later.input_tokens.fill_(8)
        roots.clear()

    def check_first(batch):
        assert batch.indices == [0]
        (captured,) = batch.inputs[0]
        assert captured is not request
        assert captured.checkpoint is None
        torch.testing.assert_close(captured.input_tokens, expected_tokens)
        torch.testing.assert_close(captured.target_tokens, expected_targets)
        assert captured.input_tokens.untyped_storage() is not tokens.untyped_storage()
        assert captured.target_tokens.untyped_storage() is not targets.untyped_storage()

    def check_second(batch):
        assert batch.indices == [1]
        (captured,) = batch.inputs[0]
        assert captured.checkpoint is Unset
        torch.testing.assert_close(captured.input_tokens, torch.tensor([4, 5]))

    if surface == "native":
        iterator = rank.forward_batches(roots, yield_empty=True)
        assert calls == []
        mutate_sources()
        check_first(next(iterator))
        later.input_tokens.fill_(99)
        check_second(next(iterator))
        assert list(iterator) == []
        return

    if surface == "persistent":

        def run(callback):
            return asyncio.run(run_rank_callback(rank, callback, mode="zero")).value

        handle = run(lambda view: view.open_forward_batches(roots))
        assert calls == []
        mutate_sources()
        check_first(run(lambda view: view.next_forward_batch(handle)))
        later.input_tokens.fill_(99)
        check_second(run(lambda view: view.next_forward_batch(handle)))
        assert run(lambda view: view.next_forward_batch(handle)) is None
    else:

        def callback(view):
            iterator = view.forward_batches(roots)
            assert calls == []
            mutate_sources()
            check_first(next(iterator))
            later.input_tokens.fill_(99)
            check_second(next(iterator))
            assert list(iterator) == []

        asyncio.run(run_rank_callback(rank, callback, mode=surface))
    assert calls == [[1, 2, 3], [4, 5]]


@pytest.mark.parametrize("logical", [False, True])
def test_generator_structure_is_consumed_once_at_submission_without_executing_a_wave(
    logical,
):
    enumerated, executed = [], []
    rank: Any = _CapturingRank() if logical else object.__new__(TrainerRank)
    rank._forward_options = None
    rank._skipped_forward_waves = {}

    def batches(inputs, **kwargs):
        executed.append(True)
        yield MicroBatch(
            inputs, [], [0], MicroBatchStats(0, 1, 1, 1, 0, 0, 0, 0, 0, False)
        )

    if logical:
        rank.forward_batches = batches
    else:
        rank._forward_batches = batches

    def inputs():
        enumerated.append("outer")

        def nested():
            enumerated.append("inner")
            yield ForwardInput(input_tokens=torch.tensor([1, 2, 3]))

        yield nested()

    def check(view):
        iterator = view.forward_batches(inputs())
        assert enumerated == ["outer", "inner"]
        assert executed == []
        iterator.close()
        assert enumerated == ["outer", "inner"]
        assert executed == []

    if logical:
        asyncio.run(run_rank_callback(rank, check, mode="zero"))
    else:
        check(rank)
