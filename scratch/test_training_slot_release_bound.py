import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

import art.megatron.training.slot as slot_module
from art.megatron.training.slot import MegatronTrainingSlot


class _PreparedPackedForward:
    def __init__(self) -> None:
        self.ref = SimpleNamespace(
            operation_id="fb", run_id="run", sequence_id=0, learner_parent_version=0
        )
        self.packed = SimpleNamespace(
            leases=SimpleNamespace(ref=object()),
            loss_bearing_tokens=1,
            packing_generation_id="batch",
        )
        self.packing = SimpleNamespace(loss_bearing_tokens=1)
        self.config = object()
        self.experimental_config = object()
        self.loss = None
        self.kind = "rl"
        self.return_token_logprobs = True


def _launch_slot(
    monkeypatch: pytest.MonkeyPatch, trainer: object, release_batch: object
) -> MegatronTrainingSlot:
    monkeypatch.setattr(slot_module, "PreparedPackedForward", _PreparedPackedForward)
    monkeypatch.setattr(
        slot_module, "ForwardBackwardJobSpec", lambda **values: SimpleNamespace(**values)
    )
    monkeypatch.setattr(
        slot_module, "ForwardBackwardResult", lambda **values: SimpleNamespace(**values)
    )
    monkeypatch.setattr(slot_module, "packing_metrics", lambda _packed: {})
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot.trainer = trainer
    slot.runtime = SimpleNamespace(release_batch=release_batch)
    slot._require_parent = lambda _ref: SimpleNamespace(  # type: ignore[method-assign]
        registration=SimpleNamespace(
            training_session_id="session", optimizer_state_path="/optimizer"
        ),
        generation=object(),
    )
    slot._batch_releases = set()
    slot._batch_release_failures = []
    slot._batch_release_slots = asyncio.BoundedSemaphore(1)
    slot._batch_release_leases = {}
    return slot


@pytest.mark.asyncio
async def test_packed_batch_release_backpressure_precedes_gpu_execution() -> None:
    gates = [asyncio.Event() for _ in range(3)]
    active = 0
    max_active = 0

    async def release_batch(batch) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await gates[batch.packing_generation_id].wait()
        finally:
            active -= 1

    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot.runtime = SimpleNamespace(release_batch=release_batch)
    slot._batch_releases = set()
    slot._batch_release_failures = []
    slot._batch_release_slots = asyncio.BoundedSemaphore(2)
    slot._batch_release_leases = {}
    batches = tuple(SimpleNamespace(packing_generation_id=index) for index in range(3))

    slot._batch_release_leases[0] = await slot._acquire_batch_release()
    slot._batch_release_leases[1] = await slot._acquire_batch_release()
    first_release = slot._release_batch_soon(batches[0])
    second_release = slot._release_batch_soon(batches[1])
    third = asyncio.create_task(slot._acquire_batch_release())
    await asyncio.sleep(0)
    assert not third.done()
    assert not first_release.done()
    assert not second_release.done()
    assert active == 2

    gates[0].set()
    await first_release
    slot._batch_release_leases[2] = await third
    third_release = slot._release_batch_soon(batches[2])
    await asyncio.sleep(0)
    assert max_active == 2

    gates[1].set()
    gates[2].set()
    await asyncio.gather(second_release, third_release)


@pytest.mark.asyncio
async def test_release_completion_fails_only_after_runtime_release_settles() -> None:
    release_started = asyncio.Event()
    release_gate = asyncio.Event()

    async def release_batch(_batch) -> None:
        release_started.set()
        await release_gate.wait()
        raise RuntimeError("injected packed-batch release failure")

    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot.runtime = SimpleNamespace(release_batch=release_batch)
    slot._batch_releases = set()
    slot._batch_release_failures = []
    slot._batch_release_slots = asyncio.BoundedSemaphore(1)
    slot._batch_release_leases = {}
    batch = SimpleNamespace(packing_generation_id=0)
    slot._batch_release_leases[0] = await slot._acquire_batch_release()

    release = slot._release_batch_soon(batch)
    await release_started.wait()
    assert not release.done()

    release_gate.set()
    with pytest.raises(RuntimeError, match="injected packed-batch release failure"):
        await release
    await asyncio.sleep(0)
    assert len(slot._batch_release_failures) == 1

    replacement = await asyncio.wait_for(slot._acquire_batch_release(), timeout=1)
    replacement.release()


@pytest.mark.asyncio
async def test_launch_exposes_exact_runtime_release_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_gate = asyncio.Event()
    result = asyncio.get_running_loop().create_future()

    async def release_batch(_batch) -> None:
        await release_gate.wait()

    trainer = SimpleNamespace(
        start_forward_backward=lambda _job, _batch: asyncio.sleep(
            0, result=SimpleNamespace(completion=result)
        )
    )
    slot = _launch_slot(monkeypatch, trainer, release_batch)
    prepared = _PreparedPackedForward()
    slot._batch_release_leases["batch"] = await slot._acquire_batch_release()

    launch = await slot.start_forward_backward(cast(Any, prepared))
    assert launch.release_completion in slot._batch_releases
    assert not launch.release_completion.done()
    assert not launch.completion.done()

    release_gate.set()
    await launch.release_completion
    assert not launch.completion.done()
    result.set_result({"token_logprobs": (), "metrics": {}})
    assert (await launch.completion).operation_id == "fb"


@pytest.mark.asyncio
async def test_failed_launch_waits_for_packed_batch_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_started = asyncio.Event()
    release_gate = asyncio.Event()

    async def release_batch(_batch) -> None:
        release_started.set()
        await release_gate.wait()

    async def fail_launch(_job, _batch) -> None:
        raise RuntimeError("injected launch failure")

    slot = _launch_slot(
        monkeypatch,
        SimpleNamespace(start_forward_backward=fail_launch),
        release_batch,
    )
    prepared = _PreparedPackedForward()
    slot._batch_release_leases["batch"] = await slot._acquire_batch_release()

    launch = asyncio.create_task(slot.start_forward_backward(cast(Any, prepared)))
    await release_started.wait()
    assert not launch.done()

    release_gate.set()
    with pytest.raises(RuntimeError, match="injected launch failure"):
        await launch
