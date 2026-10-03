import asyncio

import httpx
import pytest

from art.tau_bench import client as module


class Stream(httpx.AsyncByteStream):
    def __init__(self, error=None):
        self.error = error
        self.closes = 0

    async def __aiter__(self):
        yield b"first"
        if self.error == "read":
            raise httpx.ReadError("controlled")
        yield b"last"

    async def aclose(self):
        self.closes += 1
        if self.error == "close":
            raise RuntimeError("controlled")


class Leaf(httpx.AsyncBaseTransport):
    def __init__(self, **kwargs):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error = None
        self.streams = []

    async def handle_async_request(self, request):
        self.entered.set()
        await self.release.wait()
        if self.error == "headers":
            raise httpx.ReadTimeout("controlled", request=request)
        stream = Stream(self.error)
        self.streams.append(stream)
        return httpx.Response(200, stream=stream)


def make(monkeypatch, limit=1, *, enabled=True):
    monkeypatch.setattr(module.httpx, "AsyncHTTPTransport", Leaf)
    transport = module._ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=limit), retries=2
    )
    if enabled:
        transport._observe_transport()
    return transport


def request(timeout=1):
    return httpx.Request(
        "GET", "https://unused.invalid", extensions={"timeout": {"pool": timeout}}
    )


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 10))


def test_permit_wait_transport_pending_and_body_ownership(monkeypatch):
    async def exercise():
        transport = make(monkeypatch)
        leaf = transport.transports[0]
        leaf.release.clear()
        first = asyncio.create_task(transport.handle_async_request(request()))
        await leaf.entered.wait()
        before = transport._transport_snapshot()
        assert (
            before["permit_held"],
            before["transport_pending"],
            before["body_open"],
        ) == (1, 1, 0)
        leaf.release.set()
        response = await first
        second = asyncio.create_task(transport.handle_async_request(request()))
        await asyncio.sleep(0)
        middle = transport._transport_snapshot()
        assert (
            middle["permit_waiting"],
            middle["permit_held"],
            middle["body_open"],
        ) == (1, 1, 1)
        assert middle["pool_wait"] == middle["downstream_wait"] == "unknown"
        await response.aclose()
        await response.aclose()
        response2 = await second
        await response2.aread()
        await response2.aclose()
        final = transport._transport_snapshot()
        assert final["headers"] == final["body_closed"] == 2
        assert (
            final["permit_waiting"]
            == final["permit_held"]
            == final["transport_pending"]
            == final["body_open"]
            == 0
        )
        assert leaf.streams[0].closes == 1
        await transport.aclose()

    run(exercise())


@pytest.mark.parametrize(
    "failure",
    ["wait_cancel", "pool_timeout", "send_cancel", "headers", "read", "close"],
)
def test_failures_release_exactly_once(monkeypatch, failure):
    async def exercise():
        transport = make(monkeypatch)
        leaf = transport.transports[0]
        if failure in {"wait_cancel", "pool_timeout", "send_cancel"}:
            leaf.release.clear()
            first = asyncio.create_task(transport.handle_async_request(request()))
            await leaf.entered.wait()
            if failure != "send_cancel":
                second = asyncio.create_task(
                    transport.handle_async_request(
                        request(0.001 if failure == "pool_timeout" else 1)
                    )
                )
                await asyncio.sleep(0)
                if failure == "wait_cancel":
                    second.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await second
                else:
                    with pytest.raises(httpx.PoolTimeout):
                        await second
                assert transport._transport_snapshot()["permit_waiting"] == 0
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            leaf.error = failure
            if failure == "headers":
                with pytest.raises(httpx.ReadTimeout):
                    await transport.handle_async_request(request())
            else:
                response = await transport.handle_async_request(request())
                with pytest.raises(
                    httpx.ReadError if failure == "read" else RuntimeError
                ):
                    await response.aread()
                await response.aclose()
        final = transport._transport_snapshot()
        assert (
            final["permit_held"]
            == final["permit_waiting"]
            == final["transport_pending"]
            == final["body_open"]
            == 0
        )
        assert final["body_close_errors"] == (failure == "close")
        await transport.aclose()

    run(exercise())


@pytest.mark.parametrize("limit", [None, 128])
def test_64_shard_concurrent_workload_and_streaming(monkeypatch, limit):
    async def exercise():
        transport = make(monkeypatch, limit)
        peaks = []

        async def work():
            response = await transport.handle_async_request(request())
            peaks.append(transport._transport_snapshot())
            await asyncio.sleep(0)
            assert await response.aread() == b"firstlast"
            await response.aclose()

        await asyncio.gather(*(work() for _ in range(1024)))
        final = transport._transport_snapshot()
        assert len(transport.transports) == 64
        assert final["headers"] == 1024
        assert final["body_closed"] == (64 if limit is None else 1024)
        assert final["body_omitted"] == (960 if limit is None else 0)
        assert (
            final["body_open"]
            == final["transport_pending"]
            == final["permit_waiting"]
            == 0
        )
        assert final["permit_held"] == (None if limit is None else 0)
        assert max(x["body_open"] for x in peaks) == (64 if limit is None else 128)
        assert len(final) < 20
        await transport.aclose()

    run(exercise())


def test_unlimited_partial_stream_keeps_original_close_semantics(monkeypatch):
    async def exercise():
        transport = make(monkeypatch, None)
        response = await transport.handle_async_request(request())
        iterator = response.stream.__aiter__()
        assert await anext(iterator) == b"first"
        await iterator.aclose()
        assert transport._transport_snapshot()["body_open"] == 1
        assert transport.transports[0].streams[0].closes == 0
        await response.aclose()
        assert transport._transport_snapshot()["body_open"] == 0
        assert transport.transports[0].streams[0].closes == 1

    run(exercise())


def test_disabled_unlimited_uses_original_stream_without_observation(monkeypatch):
    async def exercise():
        transport = make(monkeypatch, None, enabled=False)
        response = await transport.handle_async_request(request())
        assert type(response.stream) is Stream
        assert transport._diagnostics is None
        await response.aclose()

    run(exercise())


@pytest.mark.parametrize("limit", [None, 1])
def test_cancelled_body_close_retains_error_and_releases_local_ownership(
    monkeypatch, limit
):
    async def exercise():
        transport = make(monkeypatch, limit)
        response = await transport.handle_async_request(request())
        entered = asyncio.Event()

        async def blocked_close():
            entered.set()
            await asyncio.Event().wait()

        response.stream._stream.aclose = blocked_close
        closing = asyncio.create_task(response.aclose())
        await entered.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        value = transport._transport_snapshot()
        assert value["body_open"] == 0
        assert value["body_closed"] == value["body_close_errors"] == 1
        assert value["permit_held"] == (None if limit is None else 0)
        await response.aclose()
        assert transport._transport_snapshot() == value
        await transport.aclose()

    run(exercise())


def test_completed_retained_responses_bound_added_wrappers(monkeypatch):
    import gc
    import weakref

    async def exercise():
        transport = make(monkeypatch, None)
        responses = []
        for _ in range(128):
            response = await transport.handle_async_request(request())
            assert await response.aread() == b"firstlast"
            responses.append(response)
        wrappers = [
            weakref.ref(r.stream)
            for r in responses
            if isinstance(r.stream, module._CapacityReleaseStream)
        ]
        assert len(wrappers) == 64
        value = transport._transport_snapshot()
        assert value["body_wrappers"] == value["body_closed"] == 64
        assert value["body_open"] == 0 and value["body_omitted"] == 64
        responses.clear()
        del response
        gc.collect()
        assert all(ref() is None for ref in wrappers)
        assert transport._transport_snapshot()["body_wrappers"] == 0
        response = await transport.handle_async_request(request())
        assert isinstance(response.stream, module._CapacityReleaseStream)
        await response.aclose()
        await transport.aclose()

    run(exercise())


@pytest.mark.parametrize("limit", [None, 1])
def test_abandoned_wrapper_does_not_close_body_or_release_permit(monkeypatch, limit):
    import gc
    import weakref

    async def exercise():
        transport = make(monkeypatch, limit)
        response = await transport.handle_async_request(request())
        reference = weakref.ref(response.stream)
        response.stream.cycle = response.stream
        del response
        gc.collect()
        assert reference() is None
        value = transport._transport_snapshot()
        assert value["body_wrappers"] == value["body_open"] == value["body_closed"] == 0
        assert value["body_abandoned"] == 1
        assert value["permit_held"] == (None if limit is None else 1)
        assert transport.transports[0].streams[0].closes == 0
        if limit is not None:
            with pytest.raises(httpx.PoolTimeout):
                await transport.handle_async_request(request(0.001))
        await transport.aclose()

    run(exercise())


def test_activation_excludes_preexisting_held_body(monkeypatch):
    async def exercise():
        transport = make(monkeypatch, 1, enabled=False)
        old = await transport.handle_async_request(request())
        transport._observe_transport()
        pending = asyncio.create_task(transport.handle_async_request(request()))
        await asyncio.sleep(0)
        snapshot = transport._transport_snapshot()
        assert snapshot["permit_waiting"] == 1
        assert snapshot["permit_held"] == snapshot["body_open"] == 0
        assert "entered before activation are excluded" in snapshot["coverage"]
        await old.aclose()
        current = await pending
        await current.aclose()
        snapshot = transport._transport_snapshot()
        assert snapshot["headers"] == snapshot["body_closed"] == 1
        assert snapshot["permit_held"] == 0
        await transport.aclose()

    run(exercise())


@pytest.mark.parametrize("finish", ["close", "exhaust"])
def test_unlimited_retained_iterator_cleanup_matches_unobserved(monkeypatch, finish):
    class RetainedIteratorStream(httpx.AsyncByteStream):
        def __init__(self):
            self.finalized = self.closes = 0
            self.iterator = self.iterate()

        def __aiter__(self):
            return self.iterator

        async def iterate(self):
            try:
                yield b"first"
                yield b"last"
            finally:
                self.finalized += 1

        async def aclose(self):
            self.closes += 1

    async def exercise():
        for enabled in (False, True):
            transport = make(monkeypatch, None, enabled=enabled)
            stream = RetainedIteratorStream()

            async def reply(request):
                return httpx.Response(200, stream=stream)

            transport.transports[0].handle_async_request = reply
            response = await transport.handle_async_request(request())
            iterator = response.stream.__aiter__()
            try:
                assert await anext(iterator) == b"first"
                if finish == "close":
                    await iterator.aclose()
                else:
                    assert [part async for part in iterator] == [b"last"]
                assert stream.finalized == 1
                assert stream.closes == 0
            finally:
                await stream.iterator.aclose()
                await response.aclose()
                await transport.aclose()
            assert stream.closes == 1

    run(exercise())
