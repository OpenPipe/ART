"""Dead caller graphs release native captures across logical callback modes."""

import asyncio
from functools import partial
from typing import Any
import weakref

import pytest
from test_trainer_rank_commands import _input, _Rank
from test_trainer_rank_versions import _trainer
import torch

from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRankSlotStateError
from art.trainer_rank._commands import run_rank_callback
from art.trainer_rank._graphs import GraphCache
from art.trainer_rank._tensors import CotangentCollector, TensorPacket, flatten_tensors


class _CachedRank(_Rank):
    def __init__(self):
        super().__init__()
        self.native, self.weight = _trainer()
        self.ref = self.native._slot_ref("student")
        self.cache = GraphCache()
        self.collector = CotangentCollector()
        self.snapshots = []

    def forward(self, tree, **kwargs):
        if not isinstance(tree, ForwardInput):
            return super().forward(tree, **kwargs)
        version = self.native._capture_checkpoint_version("student")

        def execute(tokens):
            snapshot = self.native._snapshot_parameter(self.weight, version)
            self.snapshots.append(weakref.ref(snapshot))
            return (snapshot * tokens,)

        handle, tensors = self.cache.run(
            execute, tree.input_tokens.float(), checkpoint_versions=(version,)
        )
        _, spec = flatten_tensors(ForwardOutput(None, None, None, tensors[0]))
        packet = TensorPacket(
            handle, spec, tensors, tuple(tensor.requires_grad for tensor in tensors)
        )
        output = self.collector.attach(
            packet, on_release=partial(self.cache.release, handle)
        )
        # Compose the actual slot replacement guard with the retained inner
        # cache bridge, outside the cache's saved-variable offload hooks.
        (output,) = self.native._track_slot_graph_outputs(self.ref, [output])
        return output

    def _forward_cotangent_collector(self):
        return self.collector

    def _forward_graph_cache(self):
        return self.cache

    def _forward_memory_group(self):
        return None

    def _gradient_transaction(self, **kwargs):
        return self.native._gradient_transaction(**kwargs)


def _run(rank, callback, mode):
    return asyncio.run(run_rank_callback(rank, callback, mode=mode)).value


def _can_replace(rank):
    rank.native._guard_slot_can_load(rank.native._slot_ref("student"))


def _released(rank):
    assert not rank._rank_command_state.graphs
    assert not rank._rank_command_state.released
    assert not rank.cache.handles()
    assert all(reference() is None for reference in rank.snapshots)
    _can_replace(rank)


@pytest.mark.parametrize("mode", ["zero", "rank"])
def test_callback_local_output_releases_physical_graph_and_checkpoint_capture(mode):
    rank: Any = _CachedRank()
    assert (
        _run(rank, lambda view: view.forward(_input(3)).hidden_states.item(), mode) == 6
    )
    _released(rank)


@pytest.mark.parametrize("mode", ["zero", "rank"])
def test_output_dropped_after_stop_releases_before_other_mode_callback(mode):
    rank: Any = _CachedRank()
    output = _run(rank, lambda view: view.forward(_input(3)), mode)
    caller = weakref.ref(output.hidden_states)
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        _can_replace(rank)
    del output
    assert caller() is None
    assert rank._rank_command_state.released and rank.cache.handles()
    # The native checkpoint replacement guard runs before any view operation.
    _run(rank, lambda _view: _can_replace(rank), "rank" if mode == "zero" else "zero")
    _released(rank)


@pytest.mark.parametrize("mode", ["zero", "rank"])
def test_live_dependent_loss_survives_other_mode_and_backwards_later(mode):
    rank: Any = _CachedRank()
    output = _run(rank, lambda view: view.forward(_input(3)), mode)
    caller = weakref.ref(output.hidden_states)
    loss = output.hidden_states.sum()
    del output
    assert caller() is None  # The tensor wrapper is gone; its autograd graph lives.
    _run(rank, lambda view: view.zero_grad(), "rank" if mode == "zero" else "zero")
    assert rank.cache.handles()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        _can_replace(rank)
    _run(rank, lambda view: view.backward(loss), mode)
    torch.testing.assert_close(rank.weight.grad, torch.tensor(3.0, dtype=torch.float64))
    del loss
    _run(rank, lambda view: view.zero_grad(), "rank" if mode == "zero" else "zero")
    _released(rank)
