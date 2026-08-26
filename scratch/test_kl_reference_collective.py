import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import art.megatron.runtime.monarch as monarch_module
from art.megatron.runtime.monarch import MonarchTrainerSlot
from art.megatron.runtime.specs import KlReferenceSpec


class _Call:
    def __init__(self, operation) -> None:
        self._operation = operation

    async def call(self, *args: object) -> object:
        return await self._operation(*args)


class _Port:
    def __init__(self, queue: asyncio.Queue[object]) -> None:
        self._queue = queue

    def send(self, value: object) -> None:
        self._queue.put_nowait(value)


class _Receiver:
    def __init__(self, queue: asyncio.Queue[object]) -> None:
        self._queue = queue

    async def recv(self) -> object:
        return await self._queue.get()


class _Actors:
    def __init__(self) -> None:
        self.acquire_ids: list[str] = []
        self.abort_ids: list[str] = []
        self.release_ids: list[str] = []
        self.fail_acquire = False
        self.release_failures = 0
        self.start_prepare_run_slot_kl_reference = _Call(self._start)
        self.acquire_run_slot_kl_reference = _Call(self._acquire)
        self.abort_run_slot_kl_reference_acquisition = _Call(self._abort)
        self.release_run_slot_kl_reference = _Call(self._release)

    async def _start(self, _payload: str, port: _Port) -> dict[int, dict[str, Any]]:
        for rank in range(2):
            port.send(
                {
                    "rank": rank,
                    "run_id": "run",
                    "checkpoint_id": "checkpoint",
                }
            )
        return {
            rank: {
                "rank": rank,
                "run_id": "run",
                "checkpoint_id": "checkpoint",
            }
            for rank in range(2)
        }

    async def _acquire(
        self, _payload: str, acquisition_id: str
    ) -> dict[int, dict[str, Any]]:
        self.acquire_ids.append(acquisition_id)
        if self.fail_acquire:
            raise RuntimeError("rank acquire failed")
        return {
            rank: {
                "rank": rank,
                "run_id": "run",
                "checkpoint_id": "checkpoint",
                "byte_count": 10,
            }
            for rank in range(2)
        }

    async def _abort(
        self, _run_id: str, _checkpoint_id: str, acquisition_id: str
    ) -> dict[int, None]:
        self.abort_ids.append(acquisition_id)
        return {0: None, 1: None}

    async def _release(
        self, _run_id: str, _checkpoint_id: str, acquisition_id: str
    ) -> dict[int, None]:
        self.release_ids.append(acquisition_id)
        if self.release_failures:
            self.release_failures -= 1
            raise RuntimeError("partial release failed")
        return {0: None, 1: None}


@pytest.fixture(autouse=True)
def fake_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_channel() -> tuple[_Port, _Receiver]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        return _Port(queue), _Receiver(queue)

    monkeypatch.setattr(monarch_module.Channel, "open", staticmethod(open_channel))


def _slot(actors: _Actors) -> MonarchTrainerSlot:
    return MonarchTrainerSlot(
        SimpleNamespace(),
        actors,
        SimpleNamespace(),
        SimpleNamespace(),
        (object(), object()),
        (),
        (),
        command_timeout_s=1,
        shutdown_timeout_s=1,
    )


@pytest.mark.asyncio
async def test_partial_acquire_failure_rolls_back_the_same_acquisition() -> None:
    actors = _Actors()
    actors.fail_acquire = True
    slot = _slot(actors)

    with pytest.raises(RuntimeError, match="rank acquire failed"):
        await slot.acquire_kl_reference(
            KlReferenceSpec(
                run_id="run", checkpoint_id="checkpoint", adapter_path="/adapter"
            )
        )

    assert actors.abort_ids == actors.acquire_ids
    assert slot.valid


@pytest.mark.asyncio
async def test_release_retries_with_the_same_acquisition_id() -> None:
    actors = _Actors()
    actors.release_failures = 1
    slot = _slot(actors)

    await slot.release_kl_reference("run", "checkpoint", "0" * 32)

    assert actors.release_ids == ["0" * 32, "0" * 32]
