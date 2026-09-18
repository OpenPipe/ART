"""Checkpoint scopes and delayed cleanup stay within their callback session."""

import asyncio
from datetime import timedelta
import gc
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from test_trainer_rank_commands import _input, _Rank
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import run_rank_callback


class _CheckpointRank(_Rank):
    def __init__(self):
        super().__init__()
        self._slot_stack = []

    @staticmethod
    def _checkpoint_source(checkpoint):
        return checkpoint, checkpoint

    @staticmethod
    def _slot_ref(path):
        return path

    def _push_checkpoint_sync(self, path, directory):
        self._slot_stack.append(path)

    def pop_checkpoint(self):
        self._slot_stack.pop()


@pytest.mark.parametrize("mode", ["rank", "zero"])
def test_callback_checkpoint_context_restores_nested_and_exceptional_stack(mode):
    rank: Any = _CheckpointRank()

    def callback(view):
        with view.push_checkpoint("outer") as outer:
            assert rank._slot_stack == ["outer"]
            with pytest.raises(ValueError, match="body failed"):
                with view.push_checkpoint("inner"):
                    assert rank._slot_stack == ["outer", "inner"]
                    raise ValueError("body failed")
            assert rank._slot_stack == ["outer"]
        assert not rank._slot_stack
        with pytest.raises(RuntimeError, match="entered twice"):
            outer.__enter__()
        with pytest.raises(BaseExceptionGroup) as failure:
            with view.push_checkpoint("outer"):
                view._push_checkpoint_sync("changed", None)
                raise ValueError("body failed")
        assert [type(error) for error in failure.value.exceptions] == [
            ValueError,
            RuntimeError,
        ]
        assert rank._slot_stack == ["outer", "changed"]
        view.pop_checkpoint()
        view.pop_checkpoint()

    asyncio.run(run_rank_callback(rank, callback, mode=mode))
    assert not rank._slot_stack


@pytest.mark.parametrize("mode", ["rank", "zero"])
def test_retained_callback_iterator_closes_before_stop_and_cannot_reenter(mode):
    rank: Any = _CheckpointRank()
    views, held = [], []

    def callback(view):
        views.append(view)
        batches = view.forward_batches([_input(1), _input(2)])
        held.append(batches)
        next(batches)
        raise ValueError("keep traceback alive")

    with pytest.raises(ValueError) as failure:
        asyncio.run(run_rank_callback(rank, callback, mode=mode))
    assert failure.value.__traceback__ is not None
    assert rank.closed == 1
    sequence = rank._rank_command_state.sequence
    held.pop().close()
    assert rank._rank_command_state.sequence == sequence

    def following(view):
        with pytest.raises(RuntimeError, match="session has stopped"):
            views[0].zero_grad()
        view.zero_grad()

    asyncio.run(run_rank_callback(rank, following, mode=mode))


def _lifecycle_worker(physical, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=physical,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        megatron, core = ModuleType("megatron"), ModuleType("megatron.core")
        setattr(
            core,
            "parallel_state",
            SimpleNamespace(
                get_tensor_model_parallel_rank=lambda: physical,
                get_context_parallel_rank=lambda: 0,
                get_tensor_and_context_parallel_group=lambda: dist.group.WORLD,
            ),
        )
        setattr(megatron, "core", core)
        sys.modules.update({"megatron": megatron, "megatron.core": core})
        for mode in ("zero", "rank"):
            rank: Any = _CheckpointRank()
            held = []

            def fail(view):
                held.append(view)
                with view.push_checkpoint("outer"):
                    with view.push_checkpoint("inner"):
                        batches = view.forward_batches([_input(1), _input(2)])
                        next(batches)
                        raise ValueError("retain callback traceback")

            failure = None
            try:
                asyncio.run(run_rank_callback(rank, fail, mode=mode))
            except ValueError as error:
                failure = error
            assert (failure is not None) == (physical == 0)
            assert not rank._slot_stack
            assert rank.closed == 1
            dist.barrier()
            if failure is not None:
                # The traceback is the only owner of the suspended user iterator.
                # Releasing it after STOP must not broadcast to absent receivers.
                failure.__traceback__ = None
                failure = None
                gc.collect()
            dist.barrier()

            def following(view):
                with pytest.raises(RuntimeError, match="session has stopped"):
                    held[0].zero_grad()
                with view.push_checkpoint("next"):
                    view.zero_grad()

            asyncio.run(run_rank_callback(rank, following, mode=mode))
            assert not rank._slot_stack
            dist.barrier()

            zero_grad = rank.zero_grad

            def change_stack():
                if physical == 1:
                    rank._slot_stack.append("changed")

            rank.zero_grad = change_stack

            def mismatched(view):
                with pytest.raises(RuntimeError, match="stack changed"):
                    with view.push_checkpoint("outer"):
                        view.zero_grad()

            asyncio.run(run_rank_callback(rank, mismatched, mode=mode))
            # Validate every peer before any peer mutates its stack.
            assert rank._slot_stack == (["outer", "changed"] if physical else ["outer"])
            rank._slot_stack.clear()
            rank.zero_grad = zero_grad
            asyncio.run(run_rank_callback(rank, following, mode=mode))
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_delayed_callback_cleanup_and_checkpoint_scopes_leave_gloo_reusable(tmp_path):
    mp.spawn(_lifecycle_worker, args=(str(tmp_path / "lifecycle"),), nprocs=2)
