"""Callback sessions own checkpoint stacks, graph lifetimes, and delayed cleanup."""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest
from test_trainer_rank_commands import _input, _loss_tree, _Rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from trainer_rank_test_support import gloo_group, megatron_topology

from art.trainer_rank import run_rank_callback, run_rank_callback_stream


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
    with (
        gloo_group(physical, f"file://{rendezvous}", timeout=15),
        megatron_topology(physical, dp_size=1, tp_size=2),
    ):
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

            wrong_stop = StopAsyncIteration("user generator failure")
            closed = []

            def generate(view):
                try:
                    view.zero_grad()
                    yield 17
                    raise wrong_stop
                finally:
                    closed.append(True)

            async def consume():
                async for result in run_rank_callback_stream(rank, generate, mode=mode):
                    assert result.value == (17 if physical == 0 else None)

            if physical == 0:
                with pytest.raises(RuntimeError, match="generator raised") as failure:
                    asyncio.run(consume())
                assert failure.value.__cause__ is wrong_stop
            else:
                asyncio.run(consume())
            assert closed == ([True] if physical == 0 else [])
            dist.barrier()
            asyncio.run(run_rank_callback(rank, following, mode=mode))
            dist.barrier()


def test_delayed_callback_cleanup_and_checkpoint_scopes_leave_gloo_reusable(tmp_path):
    mp.spawn(_lifecycle_worker, args=(str(tmp_path / "lifecycle"),), nprocs=2)


def _ownership(rank: Any) -> list[Any]:
    gc.collect()
    state = rank._rank_command_state
    rows: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(rows, (tuple(state.graphs), tuple(state.released)))
    return rows


def _empty(rank: Any, boundary: str) -> None:
    rows = _ownership(rank)
    assert all(not graphs and not released for graphs, released in rows), (
        boundary,
        rows,
    )


async def _cases(rank: Any, physical: int) -> None:
    inputs = [_input(2), _input(3)]
    for mode, other in (("zero", "rank"), ("rank", "zero")):
        leader = physical == 0 if mode == "zero" else physical % 2 == 0

        async def call(callback, selected=mode):
            return (await run_rank_callback(rank, callback, mode=selected)).value

        # The proxy dies after its creating command session has already STOPped.
        returned = await call(lambda view: view.forward(inputs))
        assert any(graphs for graphs, _ in _ownership(rank))
        returned = None
        gc.collect()
        await call(lambda view: view.zero_grad(), other)
        _empty(rank, f"{mode} -> {other} post-STOP release")

        # No later command is required to reclaim a callback-local result.
        await call(lambda view: _loss_tree(view.forward(inputs)).item())
        _empty(rank, f"{mode} unused local output")

        # The common release boundary must not revoke genuinely live proxies.
        returned = await call(lambda view: view.forward(inputs))
        await call(lambda view: view.zero_grad(), other)
        assert any(graphs for graphs, _ in _ownership(rank))
        await call(lambda view: view.backward(_loss_tree(returned)))
        expected = (2 if rank.dp == 0 else 3) if mode == "zero" else 5
        torch.testing.assert_close(rank.weight.grad, torch.tensor(float(expected)))
        returned = None
        gc.collect()
        await call(lambda _view: None, other)
        _empty(rank, f"{mode} retained output consumed in original mode")

        for ending in ("close", "error", "cancel", "live"):

            async def generate(view):
                yield view.forward(inputs)
                if ending == "error":
                    raise RuntimeError("cleanup stream failure")
                if ending == "cancel":
                    task = asyncio.current_task()
                    assert task is not None
                    asyncio.get_running_loop().call_soon(task.cancel)
                    await asyncio.sleep(10)

            stream = run_rank_callback_stream(rank, generate, mode=mode)
            yielded = await anext(stream)
            if leader:
                assert _loss_tree(yielded.value).item() == 10
            else:
                assert yielded.logical_rank is None
            kept = yielded.value if ending == "live" else None
            del yielded
            gc.collect()
            if leader and ending == "error":
                with pytest.raises(RuntimeError, match="cleanup stream failure"):
                    await anext(stream)
            elif leader and ending == "cancel":
                pending = asyncio.create_task(anext(stream))
                with pytest.raises(asyncio.CancelledError):
                    await pending
            await stream.aclose()
            if ending == "live":
                # Closing the producing generator cannot revoke an output that
                # its caller still holds, even after the other view runs.
                await call(lambda view: view.zero_grad(), other)
                assert any(graphs for graphs, _ in _ownership(rank))
                await call(lambda view: view.backward(_loss_tree(kept)))
                torch.testing.assert_close(
                    rank.weight.grad, torch.tensor(float(expected))
                )
                kept = None
                gc.collect()
                await call(lambda _view: None, other)
            if ending in ("error", "cancel"):
                # Failure reaches the controller before all peers join cleanup;
                # the next callback must finish that cleanup before admission.
                await call(lambda _view: None, other)
            _empty(rank, f"{mode} generator {ending}")
            await call(lambda view: view.zero_grad(), other)
            _empty(rank, f"{mode} generator {ending} followed by {other}")

        # Followers cancelled during a session must still join cleanup, and the
        # next callback in the other mode must see an intact communicator.
        entered = asyncio.Event()
        zero_grad = rank.zero_grad

        def entered_zero_grad():
            zero_grad()
            entered.set()

        rank.zero_grad = entered_zero_grad

        async def suspended(view):
            unused = view.forward(inputs)
            view.zero_grad()
            del unused
            gc.collect()
            await asyncio.sleep(0.05)

        pending = asyncio.create_task(run_rank_callback(rank, suspended, mode=mode))
        await entered.wait()
        if not leader:
            pending.cancel()
        (result,) = await asyncio.gather(pending, return_exceptions=True)
        rank.zero_grad = zero_grad
        if leader:
            assert not isinstance(result, BaseException), result
        else:
            assert isinstance(result, asyncio.CancelledError), result
        await call(lambda view: view.zero_grad(), other)
        _empty(rank, f"{mode} cancellation followed by {other}")


def _worker(physical: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    with (
        gloo_group(physical, f"file://{rendezvous}", world_size=4, timeout=20),
        megatron_topology(physical, dp_size=2, tp_size=2),
    ):
        asyncio.run(_cases(_Rank(physical // 2, 2), physical))


def test_gloo_dp2_tp2_callback_cleanup_across_modes(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "cleanup-init"),), nprocs=4, join=True)
