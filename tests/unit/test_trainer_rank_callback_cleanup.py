"""Independent DP2 x TP2 coverage of callback-boundary graph ownership."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import gc
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from test_trainer_rank_commands import _input, _loss_tree, _Rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import run_rank_callback, run_rank_callback_stream


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
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=physical,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp, tp = divmod(physical, 2)
        ps = SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: tp,
            get_context_parallel_rank=lambda: 0,
            get_data_parallel_rank=lambda: dp,
            get_data_parallel_world_size=lambda: 2,
            get_tensor_and_context_parallel_group=lambda **kwargs: groups[dp],
        )
        megatron, core = ModuleType("megatron"), ModuleType("megatron.core")
        setattr(core, "parallel_state", ps)
        setattr(megatron, "core", core)
        sys.modules.update({"megatron": megatron, "megatron.core": core})
        asyncio.run(_cases(_Rank(dp, 2), physical))
    finally:
        dist.destroy_process_group()


def test_gloo_dp2_tp2_callback_cleanup_across_modes(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "cleanup-init"),), nprocs=4, join=True)
