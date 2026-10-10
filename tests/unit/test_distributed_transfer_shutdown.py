import asyncio
import socket
from unittest.mock import Mock

import pytest

from art.distributed.data_plane import (
    ByteStreamPublisher,
    ByteStreamServerLoop,
    receive_byte_stream,
)


async def _wait_for_handler(publisher: ByteStreamPublisher) -> None:
    async def wait() -> None:
        async with asyncio.timeout(2):
            while not publisher._handlers:
                await asyncio.sleep(0)

    if publisher._server_loop is None:
        await wait()
    else:
        await publisher._server_loop.submit(wait())


@pytest.mark.asyncio
@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("authenticated", [False, True])
async def test_close_disconnects_stalled_receiver(
    threaded: bool, authenticated: bool
) -> None:
    server_loop = ByteStreamServerLoop() if threaded else None
    on_sent = Mock()
    publisher = await ByteStreamPublisher.create(
        "stalled",
        (b"x" * (16 << 20),),
        advertise_host="127.0.0.1",
        on_sent=on_sent,
        server_loop=server_loop,
    )
    client = socket.socket()
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    client.setblocking(False)
    closing = None
    try:
        loop = asyncio.get_running_loop()
        await loop.sock_connect(client, ("127.0.0.1", publisher.transfer.port))
        if authenticated:
            await loop.sock_sendall(client, bytes.fromhex(publisher.transfer.token))
        await _wait_for_handler(publisher)
        closing = asyncio.create_task(publisher.close())
        done, _ = await asyncio.wait([closing], timeout=2)
        assert done, "close waited for the stalled receiver to disconnect"
        closing.result()
        assert not publisher._handlers
        on_sent.assert_not_called()
        await publisher.close()
    finally:
        client.close()
        if closing is not None:
            await asyncio.wait_for(closing, 2)
        await publisher.close()
        if server_loop is not None:
            await server_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("threaded", [False, True])
async def test_completed_transfer_preserves_payload_and_callback(
    threaded: bool,
) -> None:
    server_loop = ByteStreamServerLoop() if threaded else None
    on_sent = Mock()
    chunks = (b"first\x00", b"second" * 1024)
    publisher = await ByteStreamPublisher.create(
        "complete",
        chunks,
        advertise_host="127.0.0.1",
        on_sent=on_sent,
        server_loop=server_loop,
    )
    try:
        assert await receive_byte_stream(publisher.transfer, timeout_s=2) == b"".join(
            chunks
        )
        async with asyncio.timeout(2):
            while not on_sent.called:
                await asyncio.sleep(0)
        await publisher.close()
        on_sent.assert_called_once_with()
        await publisher.close()
    finally:
        await publisher.close()
        if server_loop is not None:
            await server_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_rejects_handler_starting_during_shutdown(
    monkeypatch: pytest.MonkeyPatch, cancel_close: bool
) -> None:
    accepted = asyncio.Event()
    start_handler = asyncio.Event()
    original_handle = ByteStreamPublisher._handle

    async def delayed_handle(
        self: ByteStreamPublisher,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        accepted.set()
        await start_handler.wait()
        await original_handle(self, reader, writer)

    monkeypatch.setattr(ByteStreamPublisher, "_handle", delayed_handle)
    publisher = await ByteStreamPublisher.create(
        "late-handler", (b"payload",), advertise_host="127.0.0.1"
    )
    _, writer = await asyncio.open_connection("127.0.0.1", publisher.transfer.port)
    closing = None
    try:
        await asyncio.wait_for(accepted.wait(), 2)
        server = publisher._server
        closing = asyncio.create_task(publisher.close())
        async with asyncio.timeout(2):
            while server.is_serving():
                await asyncio.sleep(0)
        if cancel_close:
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done(), (
                "cancellation escaped before connections were closed"
            )
        start_handler.set()
        done, _ = await asyncio.wait([closing], timeout=2)
        assert done, "a late handler kept shutdown waiting for a token"
        if cancel_close:
            with pytest.raises(asyncio.CancelledError):
                closing.result()
        else:
            closing.result()
        assert not publisher._handlers
    finally:
        start_handler.set()
        writer.close()
        await writer.wait_closed()
        if closing is not None:
            await asyncio.wait_for(asyncio.gather(closing, return_exceptions=True), 2)
        await publisher.close()
