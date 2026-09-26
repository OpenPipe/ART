"""Controller cancellation must remain possible while global cleanup is pending."""

from __future__ import annotations

import asyncio
from contextlib import closing
import gc
from multiprocessing.connection import Connection
from typing import Any

import pytest
from test_trainer_rank_commands import _input, _loss_tree, _Rank
import torch
import torch.multiprocessing as mp
from trainer_rank_test_support import gloo_group, megatron_topology

from art.trainer_rank import run_rank_callback, run_rank_callback_stream


def _messages(connection: Connection) -> asyncio.Queue[Any]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def receive():
        try:
            queue.put_nowait(connection.recv())
        except EOFError:
            loop.remove_reader(connection.fileno())

    loop.add_reader(connection.fileno(), receive)
    return queue


async def _serve(rank: Any, connection: Connection, kind: str) -> None:
    commands = _messages(connection)
    release = asyncio.Event()

    def forward(view):
        value = _loss_tree(view.forward([_input(rank.dp + 2)])).item()
        gc.collect()
        return value

    async def callback(view):
        forward(view)
        connection.send(("entered", rank.dp))
        await release.wait()
        raise RuntimeError("DP0 user failure")

    async def generate(view):
        try:
            yield forward(view)
            connection.send(("entered", rank.dp))
            await release.wait()
            raise RuntimeError("DP0 user failure")
        finally:
            if kind == "stream_close" and rank.dp == 0:
                raise RuntimeError("DP0 generator close failure")

    async def execute():
        try:
            if kind == "ordinary":
                await run_rank_callback(rank, callback, mode="rank")
            else:
                stream = run_rank_callback_stream(rank, generate, mode="rank")
                try:
                    if kind == "stream_close":
                        await anext(stream)
                        connection.send(("entered", rank.dp))
                        await release.wait()
                        await stream.aclose()
                    else:
                        async for _ in stream:
                            pass
                finally:
                    await stream.aclose()
        except asyncio.CancelledError:
            connection.send(("cancelled", rank.dp))
        except Exception as error:
            connection.send(("error", str(error)))
        else:
            connection.send(("unexpected_success", rank.dp))

    pending = asyncio.create_task(execute())
    try:
        while True:
            command = await commands.get()
            if command == "raise":
                release.set()
            elif command == "cancel":
                pending.cancel()
            elif command == "followup":
                await pending
                # Entry must join the previous cleanup before any new session.
                result = await run_rank_callback(
                    rank, lambda view: (view.zero_grad(), 17)[1], mode="zero"
                )
                state = rank._rank_command_state
                assert not state.graphs and not state.released
                connection.send(("followup", result.value))
            elif command == "stop":
                return
            else:
                raise AssertionError(command)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        asyncio.get_running_loop().remove_reader(connection.fileno())


def _worker(physical: int, rendezvous: str, connection: Connection, kind: str) -> None:
    torch.set_num_threads(1)
    with (
        gloo_group(physical, f"file://{rendezvous}", timeout=12),
        closing(connection),
        megatron_topology(physical, dp_size=2, tp_size=1),
    ):
        asyncio.run(_serve(_Rank(physical, 2), connection, kind))


async def _gather(executions):
    # Caladan _execution.gather uses FIRST_EXCEPTION, then a grace period,
    # then cancellation/drain. A failure hidden behind cleanup defeats step 1.
    done, pending = await asyncio.wait(executions, return_when=asyncio.FIRST_EXCEPTION)
    error = next(task.exception() for task in done if task.exception() is not None)
    if pending:
        _, pending = await asyncio.wait(pending, timeout=0.05)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    assert error is not None
    raise error


async def _controller(connections: list[Connection]) -> None:
    messages = [_messages(connection) for connection in connections]
    executions = []
    cancelled = []

    async def remote(index: int):
        try:
            result = await messages[index].get()
        except asyncio.CancelledError:
            connections[index].send("cancel")
            result = await asyncio.shield(messages[index].get())
            assert result == ("cancelled", index), result
            cancelled.append(index)
            raise
        assert result[0] == "error", result
        raise RuntimeError(result[1])

    try:
        entered = await asyncio.wait_for(
            asyncio.gather(*(queue.get() for queue in messages)), timeout=20
        )
        assert entered == [("entered", 0), ("entered", 1)]
        executions = [asyncio.create_task(remote(index)) for index in range(2)]
        connections[0].send("raise")
        # This deadline measures failure visibility, not worker startup. Only
        # the controller may cancel DP1; its user callback never completes.
        controller = asyncio.create_task(_gather(executions))
        done, _ = await asyncio.wait([controller], timeout=2)
        try:
            assert done, "DP0 failure hidden behind blocked DP1 callback cleanup"
            with pytest.raises(RuntimeError, match="DP0 .*failure"):
                await controller
            assert cancelled == [1]
            for connection in connections:
                connection.send("followup")
            followup = await asyncio.wait_for(
                asyncio.gather(*(queue.get() for queue in messages)), timeout=10
            )
            assert followup == [("followup", 17), ("followup", None)]
        finally:
            for task in executions:
                task.cancel()
            await asyncio.wait_for(
                asyncio.gather(*executions, return_exceptions=True), timeout=15
            )
            controller.cancel()
            await asyncio.gather(controller, return_exceptions=True)
    finally:
        for connection in connections:
            connection.send("stop")
            asyncio.get_running_loop().remove_reader(connection.fileno())


@pytest.mark.parametrize("kind", ["ordinary", "stream", "stream_close"])
def test_gloo_dp2_first_exception_cancels_peer_and_reuses_actor(tmp_path, kind):
    context = mp.get_context("spawn")
    pairs = [context.Pipe() for _ in range(2)]
    processes = [
        context.Process(
            target=_worker,
            args=(rank, str(tmp_path / "init"), pairs[rank][1], kind),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for _, child in pairs:
            child.close()
        asyncio.run(_controller([parent for parent, _ in pairs]))
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for parent, child in pairs:
            parent.close()
            child.close()
