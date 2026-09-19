"""Pending callback cleanup remains observable and ordered after failure."""

import asyncio
from typing import Any

import pytest
from test_trainer_rank_commands import _Rank

from art.trainer_rank._commands import _Executor, _Release


@pytest.mark.parametrize("peer_buffers", [False, True])
def test_completed_release_is_finalized_before_queued_done_callback(
    monkeypatch, peer_buffers
):
    synchronized = []
    monkeypatch.setattr(
        "art.trainer_rank._heads.synchronize_head_buffers", synchronized.append
    )

    async def run():
        rank: Any = _Rank()
        executor = _Executor(rank, "zero")
        state = executor.state
        state.graphs["zero:old:dp:0"] = (rank.weight,)
        state.released.add("zero:old:dp:0")
        completed = asyncio.get_running_loop().create_future()
        release = state.pending_release = _Release(
            completed, [(tuple(state.released), False), ((), peer_buffers)]
        )
        completed.set_result(None)
        completed.add_done_callback(lambda _: executor._finish_release(release))
        await executor._join_release()
        assert not state.graphs and not state.released
        assert state.pending_release is None
        # The already queued callback cannot alter the next release's ownership.
        state.graphs["zero:new:dp:0"] = (rank.weight,)
        await asyncio.sleep(0)
        assert tuple(state.graphs) == ("zero:new:dp:0",)
        assert synchronized == ([rank] if peer_buffers else [])

    asyncio.run(run())


@pytest.mark.parametrize("cancelled", [False, True])
def test_background_cleanup_failure_is_reported_and_blocks_next_entry(cancelled):
    async def run():
        rank: Any = _Rank()
        executor = _Executor(rank, "zero")
        loop = asyncio.get_running_loop()
        reports = []
        loop.set_exception_handler(lambda _loop, context: reports.append(context))
        completed = loop.create_future()
        release = executor.state.pending_release = _Release(completed, [((), False)])
        completed.add_done_callback(lambda _: executor._finish_release(release))
        if cancelled:
            completed.cancel()
        else:
            completed.set_exception(RuntimeError("injected release transport error"))
        with pytest.raises(
            RuntimeError, match="Callback release reconciliation failed"
        ):
            await asyncio.wait_for(executor._join_release(), 1)
        assert len(reports) == 1
        with pytest.raises(
            RuntimeError, match="Callback release reconciliation failed"
        ):
            await executor.reconcile_releases()
        assert executor.state.pending_release is None

    asyncio.run(run())


def test_success_in_unrelated_exception_handler_still_awaits_cleanup(monkeypatch):
    async def run():
        rank: Any = _Rank()
        executor = _Executor(rank, "zero")
        started, finish = asyncio.Event(), asyncio.Event()

        async def reconcile(**kwargs):
            started.set()
            await finish.wait()

        monkeypatch.setattr(executor, "reconcile_releases", reconcile)

        async def callback():
            try:
                raise ValueError("unrelated handled exception")
            except ValueError:
                async with executor.release_on_exit():
                    pass

        pending = asyncio.create_task(callback())
        await started.wait()
        assert not pending.done()
        finish.set()
        await pending

    asyncio.run(run())
