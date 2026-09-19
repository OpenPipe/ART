import asyncio
import gc
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest
import torch

from art.trainer_rank._operations import (
    OperationResultReleasedError,
    TrainerOperation,
    execute_operation,
)


def test_update_identity_replays_outcome_without_applying_again():
    async def run():
        calls = []
        rank = SimpleNamespace(
            _rank=SimpleNamespace(),
            optim_step=lambda **kwargs: calls.append(kwargs) or {"step": len(calls)},
        )
        operation = TrainerOperation.capture(
            ("client", 1), "optim_step", {"params": {"lr": 1}}
        )
        assert await execute_operation(rank, operation) == {"step": 1}
        assert await execute_operation(rank, operation) == {"step": 1}
        assert len(calls) == 1
        with pytest.raises(ValueError, match="different arguments"):
            await execute_operation(
                rank,
                TrainerOperation.capture(
                    ("client", 1), "optim_step", {"params": {"lr": 2}}
                ),
            )
        await execute_operation(
            rank, TrainerOperation.capture(("client", 1), "acknowledge", ((), ()))
        )
        with pytest.raises(OperationResultReleasedError):
            await execute_operation(rank, operation)
        assert len(calls) == 1

    asyncio.run(run())


def test_failed_gradient_identity_preserves_original_error():
    from art.trainer_rank import TrainerRankSlotStateError

    async def run():
        stale = TrainerRankSlotStateError("original forward is stale")
        calls = []

        def backward_packets(**kwargs):
            calls.append(kwargs)
            raise stale

        rank = SimpleNamespace(
            _rank=SimpleNamespace(), backward_packets=backward_packets
        )
        operation = TrainerOperation.capture(("client", 1), "backward", {"packets": ()})
        for _ in range(2):
            with pytest.raises(TrainerRankSlotStateError) as error:
                await execute_operation(rank, operation)
            assert type(error.value) is type(stale)
            assert str(error.value) == str(stale)
        assert len(calls) == 1
        await execute_operation(
            rank, TrainerOperation.capture(operation.id, "acknowledge", ((), ()))
        )
        with pytest.raises(OperationResultReleasedError):
            await execute_operation(rank, operation)
        assert len(calls) == 1

    asyncio.run(run())


def test_concurrent_retry_waiter_cancellation_does_not_cancel_update():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def optim_step():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return calls

        rank = SimpleNamespace(_rank=SimpleNamespace(), optim_step=optim_step)
        operation = TrainerOperation.capture(("client", 1), "optim_step", {})
        original = asyncio.create_task(execute_operation(rank, operation))
        await entered.wait()
        retry = asyncio.create_task(execute_operation(rank, operation))
        await asyncio.sleep(0)
        retry.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retry
        release.set()
        assert await original == 1
        assert await execute_operation(rank, operation) == 1

    asyncio.run(run())


def test_operation_captures_tensor_arguments_at_submission():
    async def run():
        source = torch.tensor([2.0])
        operation = TrainerOperation.capture(
            ("client", 1), "forward", {"inputs": source}
        )
        source.add_(10)
        rank = SimpleNamespace(
            _rank=SimpleNamespace(),
            forward=lambda inputs: inputs * 3,
            export_forward=lambda output: output,
        )
        assert torch.equal(
            await execute_operation(rank, operation), torch.tensor([6.0])
        )

    asyncio.run(run())


@pytest.mark.parametrize("cuda_tag", [False, True])
def test_operation_codec_remaps_storages_and_preserves_aliases(monkeypatch, cuda_tag):
    from test_trainer_command_transport import _check_payload, _payload

    async def run():
        source = _payload("cpu")
        with monkeypatch.context() as capture:
            if cuda_tag:
                capture.setattr(
                    torch.serialization,
                    "_package_registry",
                    [
                        (0, lambda storage: "cuda:7", lambda storage, location: None),
                        *torch.serialization._package_registry,
                    ],
                )
            operation = TrainerOperation.capture(
                ("client", 1), "optim_step", {"value": source}
            )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rank = SimpleNamespace(_rank=SimpleNamespace(), optim_step=lambda value: value)
        result = await execute_operation(rank, operation)
        _check_payload(result)
        assert result.base.untyped_storage() is not source.base.untyped_storage()

    asyncio.run(run())


def test_acknowledgement_bounds_history_including_unadmitted_holes():
    async def run():
        calls = []
        rank = SimpleNamespace(
            _rank=SimpleNamespace(), optim_step=lambda: calls.append(1)
        )
        # ID 1 is still pending. Odd IDs after it were cancelled before admission.
        for sequence in range(2, 2002, 2):
            await execute_operation(
                rank, TrainerOperation.capture(("client", sequence), "optim_step", {})
            )
            await execute_operation(
                rank,
                TrainerOperation.capture(
                    ("client", sequence), "acknowledge", ((1,), ())
                ),
            )
        ledger = rank._rank._operation_outcomes
        assert not ledger.outcomes
        assert len(ledger.acknowledged) == 1
        state = ledger.acknowledged["client"]
        assert state.through == 2000 and state.pending == {1}
        for sequence in (2, 3, 1999, 2000):
            with pytest.raises(OperationResultReleasedError):
                await execute_operation(
                    rank,
                    TrainerOperation.capture(("client", sequence), "optim_step", {}),
                )
        await execute_operation(
            rank, TrainerOperation.capture(("client", 1), "optim_step", {})
        )
        await execute_operation(
            rank, TrainerOperation.capture(("client", 2000), "acknowledge", ((), ()))
        )
        assert not state.pending and not ledger.outcomes
        assert len(calls) == 1001

    asyncio.run(run())


def test_out_of_order_acknowledgements_never_resurrect_ids_or_retire_other_sessions():
    async def run():
        calls = []
        rank = SimpleNamespace(
            _rank=SimpleNamespace(), optim_step=lambda: calls.append(1)
        )
        for through, pending in (
            (5, (1, 3, 4)),
            (3, (1,)),
            (6, (1, 4, 6)),
            (5, (1, 2, 3)),
        ):
            await execute_operation(
                rank,
                TrainerOperation.capture(
                    ("client", through), "acknowledge", (pending, ())
                ),
            )
        state = rank._rank._operation_outcomes.acknowledged["client"]
        assert state.through == 6 and state.pending == {1, 6}
        for sequence in (2, 3, 4, 5):
            with pytest.raises(OperationResultReleasedError):
                await execute_operation(
                    rank,
                    TrainerOperation.capture(("client", sequence), "optim_step", {}),
                )
        for identity in (("client", 1), ("client", 6), ("client", 7), ("other", 3)):
            operation = TrainerOperation.capture(identity, "optim_step", {})
            await execute_operation(rank, operation)
            await execute_operation(rank, operation)
        assert len(calls) == 4

    asyncio.run(run())


def test_retiring_running_update_fences_retries_until_completion_is_dropped():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def optim_step():
            calls.append(1)
            entered.set()
            await release.wait()
            raise ValueError("failed after mutation")

        rank = SimpleNamespace(_rank=SimpleNamespace(), optim_step=optim_step)
        operation = TrainerOperation.capture(("client", 1), "optim_step", {})
        original = asyncio.create_task(execute_operation(rank, operation))
        await entered.wait()
        await execute_operation(
            rank, TrainerOperation.capture(operation.id, "acknowledge", ((), ()))
        )
        with pytest.raises(OperationResultReleasedError):
            await execute_operation(rank, operation)
        release.set()
        with pytest.raises(ValueError, match="failed after mutation"):
            await original
        assert len(calls) == 1 and not rank._rank._operation_outcomes.outcomes

    asyncio.run(run())


def test_batch_pulls_and_close_are_identified_without_advancing_twice():
    async def run():
        events = []
        rank = SimpleNamespace(
            _rank=SimpleNamespace(),
            open_forward_batches=lambda **kwargs: events.append("open") or "iterator",
            next_forward_batch=lambda **kwargs: events.append("next") or "batch",
            export_forward=lambda batch: SimpleNamespace(handle="packet", batch=batch),
            close_forward_batches=lambda handle: events.append(("close", handle)),
            release_forward=lambda handles: events.append(("release", tuple(handles))),
        )
        opening = TrainerOperation.capture(
            ("client", 1), "batches_open", {"inputs": []}
        )
        assert await execute_operation(rank, opening) == "iterator"
        assert await execute_operation(rank, opening) == "iterator"
        next_wave = TrainerOperation.capture(
            ("client", 2), "batches_next", {"handle": "iterator"}
        )
        first = await execute_operation(rank, next_wave)
        assert await execute_operation(rank, next_wave) is first
        # Other results can be acknowledged while this wave's reply is lost.
        await execute_operation(
            rank, TrainerOperation.capture(("client", 3), "acknowledge", ((2, 3), ()))
        )
        close = TrainerOperation.capture(
            ("client", 3),
            "batches_close",
            {"handle": "iterator", "pending_operation": next_wave.id},
        )
        await execute_operation(rank, close)
        await execute_operation(rank, close)
        assert events == [
            "open",
            "next",
            ("close", "iterator"),
            ("release", ("packet",)),
        ]
        await execute_operation(
            rank, TrainerOperation.capture(close.id, "acknowledge", ((), ()))
        )
        assert not rank._rank._operation_outcomes.outcomes
        with pytest.raises(OperationResultReleasedError):
            await execute_operation(rank, next_wave)

    asyncio.run(run())


@pytest.mark.parametrize("unserializable", [False, True])
def test_failed_outcome_releases_traceback_activations_and_replays_error(
    unserializable,
):
    from art.trainer_rank import TrainerRankMemoryError

    class UnserializableError(RuntimeError):
        def __reduce__(self):
            raise TypeError("cannot serialize this error")

    async def run():
        references = []

        def forward():
            activation = torch.ones(64, requires_grad=True).square()
            references.append(weakref.ref(activation))
            if unserializable:
                raise UnserializableError("failed forward")
            raise TrainerRankMemoryError(
                "failed forward", predicted_peak_bytes=123, usable_limit_bytes=100
            )

        rank = SimpleNamespace(
            _rank=SimpleNamespace(),
            forward=forward,
            export_forward=lambda output: output,
        )
        operation = TrainerOperation.capture(("client", 1), "forward", {})
        for _ in range(3):
            try:
                await execute_operation(rank, operation)
            except RuntimeError as error:
                assert "failed forward" in str(error)
                if not unserializable:
                    assert isinstance(error, TrainerRankMemoryError)
                    assert error.predicted_peak_bytes == 123
                    assert error.usable_limit_bytes == 100
            else:
                pytest.fail("failed operation unexpectedly succeeded")
            gc.collect()
            assert references[0]() is None
        assert len(references) == 1
        assert (
            rank._rank._operation_outcomes.outcomes[operation.id].completion.exception()
            is None
        )

    asyncio.run(run())


def test_concurrent_failed_retry_has_independent_error_without_retained_traceback():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        references = []

        async def optim_step():
            activation = torch.ones(64, requires_grad=True).square()
            references.append(weakref.ref(activation))
            entered.set()
            await release.wait()
            raise ValueError("update rejected")

        rank = SimpleNamespace(_rank=SimpleNamespace(), optim_step=optim_step)
        operation = TrainerOperation.capture(("client", 1), "optim_step", {})
        original = asyncio.create_task(execute_operation(rank, operation))
        await entered.wait()
        retry = asyncio.create_task(execute_operation(rank, operation))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(original, retry, return_exceptions=True)
        assert all(isinstance(error, ValueError) for error in results)
        assert results[0] is not results[1]
        assert len(references) == 1
        del original, retry, results
        await asyncio.sleep(0)
        gc.collect()
        assert references[0]() is None

    asyncio.run(run())


@pytest.mark.parametrize("retain_graph", [False, True])
@pytest.mark.parametrize("failure_stage", ["collect", "remote"])
def test_failed_exported_backward_replays_once_and_releases_only_consumed_graphs(
    retain_graph,
    failure_stage,
):
    from art.trainer_rank import TrainerRankZero
    from art.trainer_rank._tensors import CotangentCollector, detach_tree

    async def run():
        state = SimpleNamespace(collector=CotangentCollector(), sequence=0, exports={})
        owner = SimpleNamespace()
        view = TrainerRankZero(
            cast(Any, SimpleNamespace(rank=owner, state=state, dp_rank=0))
        )
        references, attempts = [], []

        def export(handle):
            graph = state.collector.attach(
                detach_tree(handle, torch.tensor(2.0, requires_grad=True))
            )
            references.append(weakref.ref(graph))
            return view.export_forward(graph)

        packet = export("used")
        unrelated = export("unrelated")
        client = CotangentCollector()
        loss = client.attach(packet).square()
        packets = client.backward(loss, retain_graph=retain_graph)

        def fail(*args, **kwargs):
            attempts.append("attempt")
            raise ValueError("remote gradient rejected")

        hook = None
        if failure_stage == "collect":
            hook = references[0]().grad_fn.register_hook(fail)
            setattr(view, "_submit_backward", lambda *args, **kwargs: None)
        else:
            setattr(view, "_submit_backward", fail)
        operation = TrainerOperation.capture(
            ("client", 1),
            "backward",
            {"packets": packets, "retain_graph": retain_graph},
        )
        for _ in range(2):
            try:
                await execute_operation(view, operation)
            except ValueError as error:
                assert str(error) == "remote gradient rejected"
            else:
                pytest.fail("failed backward unexpectedly succeeded")
            gc.collect()
        assert attempts == ["attempt"]
        assert (packet.handle in state.exports) is retain_graph
        assert (references[0]() is not None) is retain_graph
        assert unrelated.handle in state.exports and references[1]() is not None
        if hook is not None:
            hook.remove()
        if retain_graph:
            setattr(view, "_submit_backward", lambda *args, **kwargs: None)
            await execute_operation(
                view,
                TrainerOperation.capture(
                    ("client", 2), "backward", {"packets": packets}
                ),
            )
            gc.collect()
            assert packet.handle not in state.exports and references[0]() is None

    asyncio.run(run())


def test_malformed_nonretained_backward_preserves_unrelated_exports():
    from art.trainer_rank import TrainerRankZero
    from art.trainer_rank._tensors import CotangentCollector, CotangentPacket

    async def run():
        for handle, gradients in (("known", ()), ("missing", (torch.ones(1),))):
            state = SimpleNamespace(
                collector=CotangentCollector(),
                exports={"known": (torch.ones(1),), "unrelated": (torch.ones(1),)},
            )
            view = TrainerRankZero(
                cast(Any, SimpleNamespace(rank=SimpleNamespace(), state=state))
            )
            with pytest.raises((ValueError, KeyError)):
                await execute_operation(
                    view,
                    TrainerOperation.capture(
                        ("client", 1),
                        "backward",
                        {"packets": (CotangentPacket(handle, gradients),)},
                    ),
                )
            assert "unrelated" in state.exports
            assert ("known" in state.exports) is (handle == "missing")

    asyncio.run(run())
