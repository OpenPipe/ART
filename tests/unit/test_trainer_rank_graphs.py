from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from art.trainer_rank._graphs import GraphCache


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


def test_bad_cotangent_rejects_before_replay():
    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    handle, _ = cache.run(
        lambda x: (x * parameter,), torch.tensor(3.0), retention="replay"
    )
    with pytest.raises(ValueError, match="mismatch"):
        cache.backward(handle, (torch.ones(2),))
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


def _corrected_cache(*, retention="gpu", policy="when_available", stale=True):
    from contextlib import contextmanager

    from art.trainer_rank import ForwardOutput
    from art.trainer_rank._corrections import capture_forward_corrections
    from art.trainer_rank._options import (
        ImportanceSamplingGradientCorrection,
        ResolvedForwardOptions,
    )

    cache = GraphCache()
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
    context = capture_forward_corrections(
        ForwardOutput(outputs[0], None, None, None),
        outputs,
        ResolvedForwardOptions(
            stale_gradient_corrections=(
                ImportanceSamplingGradientCorrection(policy=policy),
            )
        ),
    )
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


def test_always_correction_uses_no_grad_current_evaluation_and_old_jacobian():
    cache, handle, original, current, executions, _ = _corrected_cache(policy="always")
    cache.backward(handle, (torch.tensor(1.0),))
    torch.testing.assert_close(
        original.grad, torch.tensor(2.0) * torch.tensor(0.75).exp()
    )
    assert current.grad is None
    assert executions == [(False, True), (True, False)]


def test_newer_replay_opportunistically_corrects_current_jacobian():
    cache, handle, original, current, executions, _ = _corrected_cache()
    cache.evict(handle, replay_with_current=True)
    cache.backward(handle, (torch.tensor(1.0),))
    assert original.grad is None
    torch.testing.assert_close(current.grad, torch.tensor(0.75).exp())
    assert executions == [(False, True), (True, True)]


@pytest.mark.parametrize("corrections", [False, True])
def test_current_replay_rejects_changed_selected_token_events(corrections):
    from contextlib import nullcontext

    from art.trainer_rank import ForwardOutput, TopK
    from art.trainer_rank._corrections import capture_forward_corrections
    from art.trainer_rank._options import (
        ImportanceSamplingGradientCorrection,
        ResolvedForwardOptions,
    )

    cache = GraphCache()
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    tokens = [torch.tensor([0, 1])]

    def execute(_):
        return parameter.log_softmax(-1)[tokens[0]], tokens[0]

    handle, outputs = cache.run(execute, None)
    context = capture_forward_corrections(
        ForwardOutput(None, TopK(outputs[0], outputs[1]), None, None),
        outputs,
        ResolvedForwardOptions(
            stale_gradient_corrections=(ImportanceSamplingGradientCorrection(),)
            if corrections
            else (),
        ),
    )
    cache.set_corrections(
        handle, context, is_stale=lambda: True, current_context_factory=nullcontext
    )
    cache.evict(handle, replay_with_current=True)
    tokens[0] = torch.tensor([1, 0])
    with pytest.raises(RuntimeError, match="token identit"):
        cache.backward(handle, (torch.ones(2), None))
    assert parameter.grad is None
    assert not cache.handles()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_abandoned_release_frees_physical_record_without_autograd(checkpointing):
    import gc
    import weakref

    from torch.utils.checkpoint import checkpoint

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
    import gc
    import weakref

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


def test_replay_failure_discards_transaction_and_releases_participating_records():
    from art.trainer_rank import TrainerRank
    from art.trainer_rank._impl import _CheckpointSlot

    trainer = TrainerRank.__new__(TrainerRank)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    trainer._checkpoint_slots = {"student": _CheckpointSlot(params=(parameter,))}
    trainer.runtime = SimpleNamespace(model=[], optimizer=None)
    snapshot = trainer._snapshot_parameter(
        parameter, trainer._capture_checkpoint_version("student")
    )
    cache = GraphCache()
    executions = [0]

    def failing(x):
        executions[0] += 1
        result = snapshot * x
        return (result if executions[0] == 1 else result.expand(2),)

    first, _ = cache.run(
        lambda x: (snapshot * x,), torch.tensor(3.0), retention="replay"
    )
    second, _ = cache.run(failing, torch.tensor(4.0), retention="replay")
    parameter.grad = torch.tensor(7.0)
    with pytest.raises(RuntimeError, match="metadata differs"):
        with trainer._gradient_transaction():
            cache.backward_many(
                [(first, (torch.tensor(1.0),)), (second, (torch.tensor(1.0),))]
            )
    assert parameter.grad.item() == 7
    assert snapshot.grad is None
    assert not trainer._version_state()._origins
    assert cache.handles() == ()
