from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from functools import partial
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.checkpoint import checkpoint

from art.trainer_rank import ForwardOutput, TrainerRank
from art.trainer_rank._graphs import GraphCache
from art.trainer_rank._impl import _CheckpointSlot
from art.trainer_rank._options import (
    ImportanceSamplingGradientCorrection,
    ResolvedForwardOptions,
)


def test_quantized_te_retained_backward_rejects_before_destructive_call():
    from art.megatron.compile_workarounds import _preserve_te_backward_metadata

    calls = []

    class QuantizedSavedState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            ctx.tensor_objects = [object()]
            return weight.square()

        @staticmethod
        def backward(ctx, *gradients):
            calls.append(True)
            return gradients[0]

    _preserve_te_backward_metadata(QuantizedSavedState)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    cache = GraphCache()
    handle, (output,) = cache.run(lambda _: (QuantizedSavedState.apply(parameter),), ())
    with pytest.raises(RuntimeError, match="quantized saved tensors"):
        cache.backward(handle, (torch.ones_like(output),), retain_graph=True)
    assert calls == []
    assert parameter.grad is None
    assert cache.handles() == ()


def test_retained_unpack_preserves_saved_view_when_consumer_clears_data():
    seen = []

    class ClearsSavedTensor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, x):
            ctx.save_for_backward(x.t()[1:, ::2])
            return weight * x.t()[1:, ::2].sum()

        @staticmethod
        def backward(ctx, *cotangents):
            (value,) = ctx.saved_tensors
            seen.append((value.stride(), value.storage_offset(), value.data_ptr()))
            result = cotangents[0] * value.sum()
            value.data = torch.empty(0)
            return result, None

    weight = torch.nn.Parameter(torch.tensor(2.0))
    inputs = torch.arange(24.0).reshape(4, 6)
    cache = GraphCache()
    handle, (output,) = cache.run(
        lambda x: (ClearsSavedTensor.apply(weight, x),), inputs
    )
    for retain in (True, True, False):
        weight.grad = None
        cache.backward(handle, (torch.ones_like(output),), retain_graph=retain)
        torch.testing.assert_close(weight.grad, inputs.t()[1:, ::2].sum())
    assert seen[0] == seen[1] == seen[2]
    assert seen[0][:2] == (inputs.t()[1:, ::2].stride(), 1)
    assert cache.handles() == ()


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
@pytest.mark.parametrize("recompute", [False, True])
def test_original_inputs_weights_and_dropout_survive_update(retention, recompute):
    torch.manual_seed(7)
    inputs = torch.randn(13, 5, dtype=torch.float64)
    weight = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64))
    historical = torch.nn.Parameter(weight.detach().clone())
    reference = torch.nn.Parameter(weight.detach().clone())
    executions = []

    def model(x, w):
        return torch.nn.functional.dropout(x @ w.square(), 0.3, training=True).sin()

    def execute(captured):
        executions.append(True)
        y = (
            checkpoint(lambda x: model(x, historical), captured, use_reentrant=False)
            if recompute
            else model(captured, historical)
        )
        return y, y.square(), torch.ones(2, dtype=torch.long)

    rng = torch.get_rng_state()
    cache = GraphCache()
    handle, (y, unused, tokens) = cache.run(execute, inputs, retention=retention)
    torch.set_rng_state(rng)
    expected = model(inputs.clone(), reference)
    torch.testing.assert_close(y, expected)
    # The caller receives leaves without a path back into the model graph.
    assert y.is_leaf and y.grad_fn is None
    assert unused.is_leaf and not tokens.requires_grad
    inputs.fill_(100)
    with torch.no_grad():
        weight.add_(9)
    ambient = torch.get_rng_state()
    (expected.cos().sum()).backward()
    cache.backward(handle, (-y.sin(), None, None))
    torch.testing.assert_close(historical.grad, reference.grad)
    assert torch.equal(torch.get_rng_state(), ambient)
    assert len(executions) == (2 if retention == "replay" else 1)
    assert cache.handles() == ()


def test_evict_frees_saved_state_while_caller_output_remains():
    cache = GraphCache()
    weight = torch.nn.Parameter(torch.tensor(2.0))

    def execute(x):
        activation = x.sin()
        return (activation * weight,)

    handle, (output,) = cache.run(execute, torch.arange(10.0))
    references = tuple(cache._records[handle].saved or ())
    assert any(reference() is not None for reference in references)
    cache.evict(handle)
    gc.collect()
    assert all(reference() is None for reference in references)
    assert output.shape == (10,)
    cache.backward(handle, (torch.ones_like(output),))
    torch.testing.assert_close(weight.grad, torch.arange(10.0).sin().sum())


def test_preflight_all_records_before_any_replay_or_gradient_mutation():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    executions = []

    def execute(x):
        executions.append(True)
        return (x * parameter,)

    def stale():
        raise RuntimeError("stale original version")

    first, _ = cache.run(execute, torch.tensor(3.0), retention="replay")
    second, _ = cache.run(execute, torch.tensor(4.0), validate_backward=stale)
    with pytest.raises(RuntimeError, match="stale original"):
        cache.backward_many(
            ((first, (torch.tensor(1.0),)), (second, (torch.tensor(1.0),)))
        )
    assert len(executions) == 2
    assert parameter.grad is None
    assert len(cache.handles()) == 2


def test_retain_graph_replay_preserves_origin_and_does_not_keep_replayed_graph():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    validated = []
    handle, _ = cache.run(
        lambda x: (x * parameter,),
        torch.tensor(3.0),
        retention="replay",
        validate_backward=lambda: validated.append(0),
    )
    cache.backward(handle, (torch.tensor(1.0),), retain_graph=True)
    assert cache.state(handle).retention == "replay"
    assert cache.state(handle).replay_count == 1
    cache.backward(handle, (torch.tensor(2.0),))
    assert parameter.grad is not None and parameter.grad.item() == 9
    assert validated == [0, 0]


def test_unused_graph_does_not_replay():
    cache = GraphCache()
    executions = []
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def execute(x):
        executions.append(True)
        return (x * parameter,)

    handle, _ = cache.run(execute, torch.tensor(3.0), retention="replay")
    cache.backward(handle, (None,))
    assert executions == [True]
    assert parameter.grad is None


@pytest.mark.parametrize(
    "retention,options",
    [
        ("cpu", SimpleNamespace(allow_cpu_offload=False)),
        ("replay", SimpleNamespace(allow_replay=False)),
    ],
)
def test_disabled_policy_rejects_before_execute(retention, options):
    cache = GraphCache()
    with pytest.raises(ValueError, match="disabled"):
        cache.run(
            lambda x: pytest.fail("should not execute"),
            None,
            retention=retention,
            options=options,
        )


@pytest.mark.parametrize("disabled_corrections", [False, True])
@pytest.mark.parametrize(
    "gradients,match",
    [
        ((), "count"),
        ((torch.ones(2),), "mismatch"),
        ((torch.ones((), dtype=torch.float64),), "mismatch"),
    ],
    ids=["count", "shape", "dtype"],
)
def test_bad_cotangent_rejects_before_replay(disabled_corrections, gradients, match):
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    handle, outputs = cache.run(
        lambda x: (x * parameter,), torch.tensor(3.0), retention="replay"
    )
    if disabled_corrections:
        cache.set_corrections(
            handle, _logprob_corrections(outputs, None), is_stale=lambda: True
        )
    with pytest.raises(ValueError, match=match):
        cache.backward(handle, gradients)
    assert cache.state(handle).replay_count == 0
    assert parameter.grad is None


def test_replay_backward_releases_each_child_before_replaying_next():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    handles = []
    replaying = False

    def execute(x):
        if replaying and x.item() == 4:
            assert handles[0] not in cache.handles()
        return (x * parameter,)

    for value in (3.0, 4.0):
        handle, _ = cache.run(execute, torch.tensor(value), retention="replay")
        handles.append(handle)
    replaying = True
    cache.backward_many([(handle, (torch.tensor(1.0),)) for handle in handles])
    assert parameter.grad is not None and parameter.grad.item() == 7


def test_tracker_rng_and_ambient_state_are_restored():
    tracker = SimpleNamespace(states={"stream": torch.tensor([17])})
    tracker.get_states = lambda: tracker.states
    tracker.set_states = lambda value: setattr(tracker, "states", value)
    seen = []
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def execute(x):
        seen.append(tracker.states["stream"].item())
        tracker.states["stream"].add_(1)
        return (x * parameter,)

    cache = GraphCache()
    handle, _ = cache.run(
        execute, torch.tensor(3.0), retention="replay", rng_tracker=tracker
    )
    tracker.states["stream"].fill_(99)
    cache.backward(handle, (torch.tensor(1.0),))
    assert seen == [17, 17]
    assert tracker.states["stream"].item() == 99


def test_backward_uses_creation_order_not_wire_handle_order():
    cache = GraphCache()
    seen, packets = [], []
    for index in range(3):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        parameter.register_hook(lambda gradient, index=index: seen.append(index))
        handle, _ = cache.run(
            lambda x, parameter=parameter: (parameter * x,), torch.tensor(3.0)
        )
        packets.append((handle, (torch.tensor(1.0),)))
    cache.backward_many(list(reversed(packets)))
    assert seen == [0, 1, 2]


def _logprob_corrections(outputs, policy):
    from art.trainer_rank._corrections import capture_forward_corrections

    return capture_forward_corrections(
        ForwardOutput(outputs[0], None, None, None),
        outputs,
        ResolvedForwardOptions(
            stale_gradient_corrections=()
            if policy is None
            else (ImportanceSamplingGradientCorrection(policy=policy),)
        ),
    )


def _corrected_cache(
    *, retention="gpu", policy="when_available", stale=True, cache=None
):
    cache = cache or GraphCache()
    original = torch.nn.Parameter(torch.tensor(-1.0))
    current = torch.nn.Parameter(torch.tensor(-0.5))
    selected = [original]
    executions = []

    @contextmanager
    def use(parameter):
        previous, selected[0] = selected[0], parameter
        try:
            yield
        finally:
            selected[0] = previous

    def execute(x):
        executions.append((selected[0] is current, torch.is_grad_enabled()))
        return (selected[0].square() * x,)

    handle, outputs = cache.run(execute, torch.tensor(-1.0), retention=retention)
    context = _logprob_corrections(outputs, policy)
    cache.set_corrections(
        handle,
        context,
        is_stale=lambda: stale,
        current_context_factory=lambda: use(current),
    )
    return cache, handle, original, current, executions, context


@pytest.mark.parametrize("retention", ["gpu", "replay"])
def test_opportunistic_exact_replay_never_adds_correction_forward(retention):
    cache, handle, original, current, executions, _ = _corrected_cache(
        retention=retention
    )
    cache.backward(handle, (torch.tensor(1.0),))
    assert len(executions) == (1 if retention == "gpu" else 2)
    assert original.grad is not None and original.grad.item() == 2
    assert current.grad is None


@pytest.mark.parametrize("evicted", [False, True], ids=["resident", "evicted"])
def test_always_correction_uses_no_grad_current_evaluation_and_old_jacobian(evicted):
    cache, handle, original, current, executions, _ = _corrected_cache(policy="always")
    expected_executions = [(False, True), (True, False)]
    if evicted:
        cache.evict(handle)
        expected_executions.append((False, True))
    cache.backward(handle, (torch.tensor(1.0),))
    torch.testing.assert_close(
        original.grad, torch.tensor(2.0) * torch.tensor(0.75).exp()
    )
    assert current.grad is None
    assert executions == expected_executions


def _observe_backward_storage(monkeypatch, request):
    from art.trainer_rank import _corrections as corrections
    from art.trainer_rank import _graphs as graphs

    if gc.isenabled():
        request.addfinalizer(gc.enable)
    gc.disable()
    storages, errors, borrowed = [], [], []

    def watch(value):
        if isinstance(value, torch.Tensor):
            storage = StorageWeakRef(value.untyped_storage())
            if storage not in borrowed:
                storages.append(storage)
        elif isinstance(value, (tuple, list)):
            for child in value:
                watch(child)

    def observe(function, *, inputs=False):
        def call(*args, **kwargs):
            try:
                if inputs:
                    watch(args[:2])
                result = function(*args, **kwargs)
                watch(result)
                return result
            except BaseException as error:
                errors.append(error)
                raise
            finally:
                args = kwargs = result = None

        return call

    monkeypatch.setattr(
        graphs._ForwardRecord, "run", observe(graphs._ForwardRecord.run)
    )
    monkeypatch.setattr(
        GraphCache,
        "_prepare_correction",
        staticmethod(observe(GraphCache._prepare_correction)),
    )
    monkeypatch.setattr(
        GraphCache,
        "_prepare_backward",
        staticmethod(observe(GraphCache._prepare_backward)),
    )
    monkeypatch.setattr(
        corrections,
        "importance_weights",
        observe(corrections.importance_weights, inputs=True),
    )
    return storages, errors, borrowed


@pytest.mark.parametrize(
    "phase", ["always", "coordinator", "correction_peer", "backward_peer"]
)
def test_correction_and_coordinator_failures_release_owned_storage(
    monkeypatch, request, phase
):
    from art.trainer_rank import _commands as commands

    cache = GraphCache()
    older_weight = torch.nn.Parameter(torch.tensor(3.0))
    older, (older_output,) = cache.run(lambda _: (older_weight.square(),), ())
    storages, errors, borrowed = _observe_backward_storage(monkeypatch, request)
    cache, handle, original, current, executions, _ = _corrected_cache(
        policy="always", cache=cache
    )
    gradient = torch.tensor(1.0)
    borrowed[:] = [
        StorageWeakRef(value.untyped_storage())
        for value in (original, current, gradient)
    ]
    primary, cause = RuntimeError("peer backward failed"), ValueError("peer cause")
    calls = 0
    peer_phase = phase in ("correction_peer", "backward_peer")
    fail_at = 1 if phase == "correction_peer" else 2

    def exchange(failures, local, *, group):
        nonlocal calls
        calls += 1
        failures[:] = [local, "peer prepare failed" if calls == fail_at else None]

    if peer_phase:
        monkeypatch.setattr(
            commands,
            "dist",
            SimpleNamespace(
                is_initialized=lambda: True,
                get_world_size=lambda group: 2,
                all_gather_object=exchange,
            ),
        )

    def coordinate(function):
        nonlocal calls
        if peer_phase:
            return commands._coordinate_call(function, group=None)
        calls += 1
        result = commands._coordinate_call(function, group=None)
        if phase == "coordinator" and calls == 3:
            raise primary from cause
        return result

    if phase == "always":
        with torch.no_grad():
            current.fill_(float("nan"))
    with pytest.raises(ValueError if phase == "always" else RuntimeError) as failure:
        cache.backward_many(((handle, (gradient,)),), coordinate=coordinate)
    if peer_phase:
        assert "Physical trainer preflight failed" in str(failure.value)
        assert calls == fail_at and original.grad is None
        assert failure.value.__cause__ is None
    elif phase != "coordinator":
        assert errors and all(error is failure.value for error in errors)
        assert "current logprobs" in str(failure.value)
        assert original.grad is None
    else:
        assert failure.value is primary and primary.__cause__ is cause and calls == 3
        torch.testing.assert_close(
            original.grad, torch.tensor(2.0) * torch.tensor(0.75).exp()
        )
    assert failure.value.__traceback__ is not None
    assert len(storages) >= 4 and all(storage.expired() for storage in storages)
    assert all(not storage.expired() for storage in borrowed)
    assert gradient.item() == 1 and original.item() == -1 and current.grad is None
    assert executions == [(False, True), (True, False)]
    assert cache.handles() == (older,)
    cache.backward(older, (torch.ones_like(older_output),))
    torch.testing.assert_close(older_weight.grad, torch.tensor(6.0))
    cache, retry, retry_weight, _, _, _ = _corrected_cache(policy="always", cache=cache)
    cache.backward_many(((retry, (gradient,)),), coordinate=coordinate)
    torch.testing.assert_close(
        retry_weight.grad, torch.tensor(2.0) * torch.tensor(0.75).exp()
    )
    assert all(storage.expired() for storage in storages) and not cache.handles()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_abandoned_release_frees_physical_record_without_autograd(checkpointing):
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.randn(4, 4))

    def execute(x):
        def compute(value):
            return (parameter @ value).sin()

        return (
            checkpoint(compute, x, use_reentrant=False)
            if checkpointing
            else compute(x),
        )

    handle, outputs = cache.run(execute, torch.ones(4, 4))
    record = weakref.ref(cache._records[handle])
    original = cache._records[handle].outputs
    assert original is not None
    physical = weakref.ref(original[0])
    del original
    cache.release(handle)
    gc.collect()
    assert record() is None and physical() is None
    assert outputs[0].requires_grad


def test_backward_releases_unused_differentiable_output_branches():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    handle, outputs = cache.run(
        lambda x: ((parameter @ x).sin().sum(), (parameter @ x).cos()),
        torch.ones(4, 4),
    )
    record = weakref.ref(cache._records[handle])
    cache.backward(handle, (torch.ones_like(outputs[0]), None))
    gc.collect()
    assert record() is None
    assert outputs[1].requires_grad


def test_restore_workspace_estimate_survives_eviction():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    handle, _ = cache.run(
        lambda x: (parameter * x,),
        torch.tensor(3.0),
        execution_peak_bytes=1024,
        checkpoint_versions=("original",),
    )
    cache.evict(handle)
    assert cache.state(handle).restore_workspace_bytes == 1024
    assert cache.state(handle).checkpoint_versions == ("original",)
    cache.release(handle)


def test_all_correction_availability_preflights_before_any_replay():
    cache, first, original, _, executions, context = _corrected_cache(
        retention="replay", policy="always"
    )
    second, _ = cache.run(
        lambda x: (original * x,), torch.tensor(-1.0), retention="replay"
    )
    cache.set_corrections(second, context, is_stale=lambda: True)
    with pytest.raises(RuntimeError, match="no current version context"):
        cache.backward_many(
            [(first, (torch.tensor(1.0),)), (second, (torch.tensor(1.0),))]
        )
    assert executions == [(False, True)]
    assert original.grad is None


def test_fresh_always_does_not_require_current_forward():
    cache, handle, original, current, executions, context = _corrected_cache(
        policy="always", stale=False
    )
    cache.set_corrections(handle, context, is_stale=lambda: False)
    cache.backward(handle, (torch.tensor(1.0),))
    assert original.grad is not None and original.grad.item() == 2
    assert current.grad is None and len(executions) == 1


def test_replay_restores_original_autocast_context():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.ones(3, 3))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        handle, (output,) = cache.run(
            lambda x: (x @ parameter,), torch.ones(2, 3), retention="replay"
        )
    assert output.dtype == torch.bfloat16
    cache.backward(handle, (torch.ones_like(output),))
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2.0))


@pytest.mark.parametrize("always_prepass", [False, True], ids=["uncorrected", "always"])
def test_replay_failure_discards_transaction_and_releases_participating_records(
    monkeypatch, request, always_prepass
):
    trainer = TrainerRank.__new__(TrainerRank)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    trainer._checkpoint_slots = {"student": _CheckpointSlot(params=(parameter,))}
    trainer.runtime = SimpleNamespace(model=[], optimizer=None)
    snapshot = trainer._snapshot_parameter(
        parameter, trainer._capture_checkpoint_version("student")
    )
    cache = GraphCache()
    older_weight = torch.nn.Parameter(torch.tensor(3.0))
    older, (older_output,) = cache.run(lambda _: (older_weight.square(),), ())
    observed, errors, borrowed = _observe_backward_storage(monkeypatch, request)
    gradient = torch.tensor(1.0)
    borrowed[:] = [
        StorageWeakRef(value.untyped_storage()) for value in (parameter, gradient)
    ]
    storages = []
    executions = [0]

    def failing(x):
        executions[0] += 1
        activation = snapshot * x + 1
        result = activation.square()
        storages.extend(
            StorageWeakRef(value.untyped_storage()) for value in (activation, result)
        )
        return (
            result
            if executions[0] == 1 or not torch.is_grad_enabled()
            else result.expand(2),
        )

    first, _ = cache.run(
        lambda x: (snapshot * x,), torch.tensor(3.0), retention="replay"
    )
    second, outputs = cache.run(failing, torch.tensor(4.0), retention="replay")
    if always_prepass:
        cache.set_corrections(
            second,
            _logprob_corrections(outputs, "always"),
            is_stale=lambda: True,
            current_context_factory=nullcontext,
        )
    parameter.grad = prior_gradient = torch.tensor(7.0)
    with pytest.raises(RuntimeError, match="metadata differs") as failure:
        with trainer._gradient_transaction():
            cache.backward_many([(first, (gradient,)), (second, (gradient,))])
    assert parameter.grad is prior_gradient and parameter.grad.item() == 7
    assert snapshot.grad is None
    assert not trainer._version_state()._origins
    assert failure.value.__traceback__ is not None and failure.value.__cause__ is None
    assert errors and all(error is failure.value for error in errors)
    assert len(storages) == 4 + 2 * always_prepass
    assert all(storage.expired() for storage in (*storages, *observed))
    assert all(not storage.expired() for storage in borrowed)
    assert parameter.item() == snapshot.item() == 2 and gradient.item() == 1
    assert cache.handles() == (older,)
    torch.testing.assert_close(older_output, torch.tensor(9.0))
    cache.backward(older, (torch.ones_like(older_output),))
    torch.testing.assert_close(older_weight.grad, torch.tensor(6.0))
    retry, (output,) = cache.run(
        lambda x: ((snapshot * x + 1).square(),),
        torch.arange(1.0, 4.0),
        retention="replay",
    )
    torch.testing.assert_close(output, torch.tensor([9.0, 25.0, 49.0]))
    with trainer._gradient_transaction():
        cache.backward(retry, (torch.ones_like(output),))
    assert parameter.grad.item() == 75 and snapshot.grad is None
    assert cache.handles() == ()


@pytest.mark.parametrize("failure_type", [MemoryError, asyncio.CancelledError])
@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
@pytest.mark.parametrize("copy_index", [pytest.param(0, id="forward"), 1, 2])
def test_initial_output_copy_failure_releases_only_failed_graph(
    monkeypatch, failure_type, retention, copy_index
):
    from art.trainer_rank._graphs import _ForwardRecord

    trainer = TrainerRank.__new__(TrainerRank)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    trainer._checkpoint_slots = {"student": _CheckpointSlot(params=(parameter,))}
    trainer.runtime = SimpleNamespace(model=[], optimizer=None)
    cache = GraphCache()
    older = torch.nn.Parameter(torch.tensor(3.0))
    old_handle, (old_output,) = cache.run(lambda _: (older.square(),), None)
    old_record = weakref.ref(cache._records[old_handle])
    records, physical, saved, snapshots, versions, copies = [], [], [], [], [], []
    primary = failure_type("initial output copy failed")
    primary.__cause__ = cause = RuntimeError("original cause")
    run, to = _ForwardRecord.run, torch.Tensor.to
    attempted = 0
    fail_forward = copy_index == 0
    input_snapshots = []

    def execute(snapshot, value):
        outputs = (snapshot * value, snapshot.square())
        if fail_forward:
            physical.extend(weakref.ref(output) for output in outputs)
            # Attribute ownership only to ART, not this injected model frame.
            del snapshot, value, outputs
            raise primary
        return outputs

    def arguments():
        version = trainer._capture_checkpoint_version("student")
        snapshot = trainer._snapshot_parameter(parameter, version)
        snapshots.append(weakref.ref(snapshot))
        versions.append(weakref.ref(version))
        return dict(
            execute=partial(execute, snapshot),
            inputs=torch.tensor(3.0),
            context_factory=lambda: nullcontext(snapshot),
            validate_backward=torch.nn.ParameterList([snapshot]).zero_grad
            if copy_index == 0
            else lambda: trainer._version_state().validate(version),
            checkpoint_versions=(version,),
            keep_on_device=lambda value: value is snapshot,
            retention=retention,
        )

    def observe(record):
        records.append(weakref.ref(record))
        input_snapshots.append(weakref.ref(record.inputs.value))
        try:
            outputs = run(record)
            physical.extend(weakref.ref(output) for output in outputs)
            saved.extend(record.saved or ())
            return outputs
        finally:
            del record

    def fail_copy(value, *args, **kwargs):
        nonlocal attempted
        if kwargs.get("copy") and "device" in kwargs:
            attempted += 1
            if attempted == copy_index:
                # The injected hook must not add its own tensor owner to the
                # retained traceback; only ART's failed attempt is under test.
                del value
                raise primary
            result = to(value, *args, **kwargs)
            copies.append(weakref.ref(result))
            return result
        return to(value, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(_ForwardRecord, "run", observe)
        failure.setattr(torch.Tensor, "to", fail_copy)
        with pytest.raises(failure_type) as caught:
            cache.run(**arguments())
    assert caught.value is primary and primary.__cause__ is cause
    assert primary.__traceback__ is not None and attempted == copy_index
    assert len(records) == len(snapshots) == len(versions) == 1
    assert len(physical) == 2 and len(input_snapshots) == 1
    if copy_index:
        assert saved and len(copies) == copy_index - 1
    else:
        assert not copies
    assert cache.handles() == (old_handle,)
    assert cache._records[old_handle] is old_record()
    assert all(
        reference() is None
        for reference in (
            *records,
            *physical,
            *saved,
            *snapshots,
            *versions,
            *copies,
            *input_snapshots,
        )
    )
    assert parameter.grad is None
    assert old_output.item() == 9 and old_output.requires_grad
    cache.backward(old_handle, (torch.tensor(1.0),))
    torch.testing.assert_close(older.grad, torch.tensor(6.0))

    fail_forward = False
    handle, outputs = cache.run(**arguments())
    assert tuple(output.item() for output in outputs) == (6, 4)
    with trainer._gradient_transaction():
        cache.backward(handle, (torch.tensor(1.0), torch.tensor(1.0)))
    torch.testing.assert_close(parameter.grad, torch.tensor(7.0))
    assert cache.handles() == ()
