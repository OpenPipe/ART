from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOptions,
    ImportanceSamplingGradientCorrection,
    TrainerRank,
    _impl,
)


@pytest.fixture
def rank(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.config = SimpleNamespace(
                hidden_size=8, num_layers=4, padded_vocab_size=32
            )
            self.decoder = object()

        def _preprocess(self, *args, **kwargs):
            return None

    result = TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[Model()],
                optimizer=None,
                provider=SimpleNamespace(
                    hidden_size=8, num_layers=4, recompute_granularity="full"
                ),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )
    monkeypatch.setattr(result, "_graph_memory_policy_enabled", lambda: True)
    monkeypatch.setattr(result, "_available_memory_bytes", lambda: 250)
    monkeypatch.setattr(result, "_available_cpu_memory_bytes", lambda: 1_000_000)
    monkeypatch.setattr(
        result,
        "_estimate_group_request_output_bytes",
        lambda requests: 10 * len(requests),
    )
    monkeypatch.setattr(
        result,
        "_plan_cost",
        lambda plan: _impl._SubforwardCost(
            100 * sum(len(g.items) for g in plan.groups),
            80 * sum(len(g.items) for g in plan.groups),
        ),
    )
    return result


def _requests(options=None):
    return [
        ForwardInput(
            input_tokens=torch.tensor([i, 10, 11]), hidden_states=True, options=options
        )
        for i in range(4)
    ]


def test_complete_root_can_split_and_replay_where_gpu_retention_refuses(rank):
    requests = _requests()
    plan, check = rank._plan_admissible_forward(
        requests, checkpoint=_impl.Unset, context="test"
    )
    assert isinstance(plan, _impl._SplitForwardPlan)
    assert plan.subforward_count == 2
    assert sorted(i for indices in plan.request_indices for i in indices) == list(
        range(4)
    )
    assert check.fits and check.estimated_required_bytes == 240
    assert all(g.memory_placement.backward_state == "replay" for g in plan.groups)
    assert all(g.memory_placement.output_device == "model" for g in plan.groups)
    with pytest.raises(_impl.TrainerRankMemoryError):
        rank._plan_admissible_forward(
            _requests(ForwardOptions(backward_state="gpu")),
            checkpoint=_impl.Unset,
            context="test",
        )


def test_auto_cpu_outputs_preserve_saved_state_without_replay(rank):
    plan, check = rank._plan_admissible_forward(
        _requests(ForwardOptions(output_device="auto")),
        checkpoint=_impl.Unset,
        context="test",
    )
    assert check.fits and check.estimated_required_bytes == 220
    assert all(g.memory_placement.backward_state == "cpu" for g in plan.groups)
    assert all(g.memory_placement.output_device == "cpu" for g in plan.groups)


@pytest.mark.parametrize(
    "transfer_seconds, samples, expected",
    [(0.3, 2, "replay"), (0.01, 2, "cpu"), (0.3, 1, "cpu"), (0.0001, 2, "cpu")],
)
def test_measured_fallback_costs_choose_only_after_gpu_refusal(
    rank, monkeypatch, transfer_seconds, samples, expected
):
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 150)
    requests = _requests(ForwardOptions(output_device="cpu"))[:2]
    children = tuple(rank._plan_flat_forward([request]) for request in requests)
    plan = _impl._SplitForwardPlan(children, ((0,), (1,)), 2)
    for _ in range(samples):
        rank._record_graph_forward_time(children[0], 0.1)
    rank._graph_cache = SimpleNamespace(
        handles=lambda: (),
        transfer_stats=SimpleNamespace(
            offload_bytes=140,
            restore_bytes=140,
            offload_seconds=transfer_seconds,
            restore_seconds=transfer_seconds,
        ),
    )
    selected, check = rank._admit_graph_memory(plan)
    assert check.fits
    assert {g.memory_placement.backward_state for g in selected.groups} == {expected}
    assert check.fallback_costs["preferred"] == expected
    assert check.fallback_costs["source"] == (
        "measured_forward_and_transfers"
        if samples >= 2 and transfer_seconds >= 0.001
        else "insufficient_samples"
    )


def test_forward_timing_requires_matching_shape_and_gpu_retention(rank):
    plan = rank._plan_flat_forward(_requests()[:1])
    for seconds in (100.0, 0.1, 0.2, 0.3):
        rank._record_graph_forward_time(plan, seconds)
    assert next(rank._graph_memory_units(plan))[2].replay_seconds == 0.3
    other = replace(plan, packed_tokens=plan.packed_tokens + 1)
    assert next(rank._graph_memory_units(other))[2].replay_seconds is None
    group = plan.groups[0]
    segment = group.packed.segments[0]
    different_tree = replace(
        group.packed,
        segments=(replace(segment, parent_id=segment.parent_id + 1),),
    )
    other = replace(plan, groups=(replace(group, packed=different_tree),))
    assert next(rank._graph_memory_units(other))[2].replay_seconds is None
    replay, _ = rank._admit_graph_memory(
        rank._plan_flat_forward(_requests(ForwardOptions(backward_state="replay"))[:1])
    )
    rank._record_graph_forward_time(replay, 99.0)
    assert next(rank._graph_memory_units(plan))[2].replay_seconds == 0.3


def test_gpu_headroom_does_not_consult_transfer_costs(rank):
    class Cache:
        @staticmethod
        def handles():
            return ()

        @property
        def transfer_stats(self):
            raise AssertionError("GPU headroom path consulted fallback costs")

    rank._graph_cache = Cache()
    selected, check = rank._admit_graph_memory(rank._plan_flat_forward(_requests()[:1]))
    assert check.fits and check.fallback_costs is None
    assert selected.groups[0].memory_placement.backward_state == "gpu"


@pytest.mark.parametrize(
    "policy, expected",
    [
        (ForwardOptions(backward_state="cpu"), "cpu"),
        (ForwardOptions(backward_state="replay"), "replay"),
        (ForwardOptions(allow_replay=False), "cpu"),
        (ForwardOptions(allow_cpu_offload=False), "replay"),
    ],
)
def test_fallback_costs_preserve_forced_and_disabled_policies(
    rank, monkeypatch, policy, expected
):
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 150)
    requests = _requests(replace(policy, output_device="cpu"))[:2]
    plan = _impl._SplitForwardPlan(
        tuple(rank._plan_flat_forward([request]) for request in requests),
        ((0,), (1,)),
        2,
    )
    selected, check = rank._admit_graph_memory(plan)
    assert check.fits
    assert {g.memory_placement.backward_state for g in selected.groups} == {expected}


def test_mixed_policies_split_physical_groups_and_share_estimator_keys(rank):
    requests = _requests()[:2]
    requests[1] = replace(
        requests[1],
        options=ForwardOptions(backward_state="replay", output_device="cpu"),
    )
    plan = rank._plan_flat_forward(requests)
    assert len(plan.groups) == 2
    estimated = rank._estimate_flat_forward(requests, exact=True)
    assert estimated[2] == plan.signature
    selected, check = rank._admit_graph_memory(plan)
    assert check.fits
    assert [g.memory_placement.backward_state for g in selected.groups] == [
        "gpu",
        "replay",
    ]
    assert [g.memory_placement.output_device for g in selected.groups] == [
        "model",
        "cpu",
    ]
    assert selected.signature.memory_placement == (("gpu", "model"), ("replay", "cpu"))
    assert selected.signature != plan.signature


def test_equal_resolved_policies_still_pack_together(rank):
    requests = _requests()[:2]
    requests[1] = replace(requests[1], options=ForwardOptions(max_gradient_staleness=2))
    assert len(rank._plan_flat_forward(requests).groups) == 1


def test_cpu_shortage_is_not_overwritten_by_fresh_gpu_check(rank, monkeypatch):
    monkeypatch.setattr(rank, "_available_cpu_memory_bytes", lambda: 0)
    plan = rank._plan_flat_forward(_requests()[:1])
    selected, check = rank._admit_graph_memory(plan)
    assert not check.fits and not check.cpu_fits
    refusal = _impl._ForwardRefusal(selected, check, "test")
    with pytest.raises(_impl.TrainerRankMemoryError, match="per-rank CPU headroom=0"):
        rank._recover_admission(
            lambda: (selected, check),
            lambda value: value,
            lambda value, check: (value[0], check),
            context="test",
            sync_across_dp=True,
        )
    assert "CPU retained" in str(refusal.error("test"))


def test_retained_weight_versions_stay_in_gpu_budget_even_for_replay(rank, monkeypatch):
    monkeypatch.setattr(
        rank, "_lora_version_capture_bytes", lambda *args: 200, raising=False
    )
    request = _requests(ForwardOptions(backward_state="replay", output_device="cpu"))[
        :1
    ]
    _, check = rank._admit_graph_memory(rank._plan_flat_forward(request))
    assert check.estimated_required_bytes == 300
    assert not check.fits


def test_gradient_transaction_reserves_one_aggregate_per_slot_across_children(
    rank, monkeypatch
):
    monkeypatch.setattr(rank, "_lora_gradient_staging_bytes", lambda _ref: 60)
    requests = _requests(ForwardOptions(backward_state="replay", output_device="cpu"))
    split = _impl._SplitForwardPlan(
        tuple(rank._plan_flat_forward([request]) for request in requests),
        tuple((i,) for i in range(4)),
        4,
    )
    _, check = rank._admit_graph_memory(split)
    assert check.estimated_required_bytes == 160
    assert check.fits


def test_prior_larger_replay_workspace_survives_new_small_root_admission(rank):
    rank._graph_cache = SimpleNamespace(
        handles=lambda: ("old",),
        state=lambda _handle: SimpleNamespace(restore_workspace_bytes=300),
    )
    requests = _requests(ForwardOptions(backward_state="replay", output_device="cpu"))[
        :1
    ]
    _, check = rank._admit_graph_memory(rank._plan_flat_forward(requests))
    assert not check.fits
    assert check.estimated_required_bytes == 300


def test_prior_checkpoint_staging_deduplicates_old_and_new_graph_targets(
    rank, monkeypatch
):
    rank._checkpoint_slots["old"] = _impl._CheckpointSlot()
    old = rank._slot_ref("old")
    monkeypatch.setattr(
        rank, "_lora_gradient_staging_bytes", lambda ref: 40 if ref == old else 0
    )
    monkeypatch.setattr(rank, "_lora_version_capture_bytes", lambda *_args: 0)
    rank._graph_cache = SimpleNamespace(
        handles=lambda: ("old1", "old2"),
        state=lambda _handle: SimpleNamespace(
            restore_workspace_bytes=150,
            checkpoint_versions=(SimpleNamespace(checkpoint="old"),),
        ),
    )
    requests = _requests(ForwardOptions(backward_state="replay", output_device="cpu"))[
        :1
    ]
    plan = rank._plan_flat_forward(requests)
    _, check = rank._admit_graph_memory(plan)
    assert check.estimated_required_bytes == 190
    plan = replace(plan, groups=(replace(plan.groups[0], slot_ref=old),))
    _, check = rank._admit_graph_memory(plan)
    assert check.estimated_required_bytes == 190


@pytest.mark.parametrize("existing_gradient", [False, True])
def test_outstanding_forwards_reserve_sequential_atomic_gradient_publication(
    rank, monkeypatch, existing_gradient
):
    parameter = torch.nn.Parameter(torch.ones(16))
    if existing_gradient:
        parameter.grad = torch.zeros_like(parameter)
    size = parameter.numel() * parameter.element_size()
    baseline_grad_bytes = size if existing_gradient else 0
    rank._checkpoint_slots["student"] = _impl._CheckpointSlot(params=(parameter,))
    ref = rank._slot_ref("student")
    monkeypatch.setattr(rank, "_iter_slot_parameters", lambda _ref: iter((parameter,)))
    monkeypatch.setattr(rank, "_lora_version_capture_bytes", lambda *_args: 0)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 100 + 3 * size)
    options = ForwardOptions(backward_state="replay", output_device="cpu")
    plan = rank._plan_flat_forward(_requests(options)[:1])
    plan = replace(plan, groups=(replace(plan.groups[0], slot_ref=ref),))
    # Both forwards are admitted before either one publishes a gradient. Replay
    # and CPU outputs need not release GPU storage between their backwards.
    checks = [rank._admit_graph_memory(plan)[1] for _ in range(2)]
    reservation = min(check.estimated_required_bytes for check in checks) - 100
    snapshot = rank._snapshot_parameter(
        parameter, rank._capture_checkpoint_version("student")
    )
    losses = [(snapshot * factor).sum() for factor in (2, 3)]
    state = rank._version_state()
    publish = state._publish
    observed = []

    def measure(prepared):
        batch = state._transaction
        assert batch is not None
        tensors = [gradient for _, gradient in batch.gradients.values()]
        tensors.extend(
            tensor
            for _, combined, previous in prepared.parameters
            for tensor in (combined, previous)
            if tensor is not None
        )
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors
        }
        live = sum(storages.values())
        observed.append(live)
        assert live - baseline_grad_bytes <= reservation
        publish(prepared)

    monkeypatch.setattr(state, "_publish", measure)
    for loss in losses:
        with rank._gradient_transaction():
            loss.backward()
    assert observed == ([3 * size, 3 * size] if existing_gradient else [size, 3 * size])
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 5))


def test_empty_output_does_not_pin_hidden_storage_and_keeps_autograd():
    hidden = torch.randn(1024, 8, requires_grad=True)
    empty = _impl._select_positions(hidden, torch.empty(0, dtype=torch.long))
    assert empty.untyped_storage().nbytes() == 0
    empty.sum().backward()
    assert hidden.grad is not None and hidden.grad.count_nonzero() == 0


def test_correction_metadata_and_explicit_prepass_are_budgeted(rank):
    request = ForwardInput(
        input_tokens=torch.arange(3),
        target_tokens=torch.arange(3),
        top_k=2,
    )
    group = rank._plan_flat_forward([request]).groups[0]
    default = _impl._resolved_request_policy(None)
    always = _impl._resolved_request_policy(
        ForwardOptions(
            stale_gradient_corrections=(
                ImportanceSamplingGradientCorrection(policy="always"),
            )
        )
    )
    assert _impl._correction_state_bytes(group, default) == 3 * 4 + 6 * 12
    assert _impl._correction_state_bytes(group, always) == 3 * 8 + 6 * 16
    assert (
        _impl._correction_state_bytes(
            group,
            _impl._resolved_request_policy(
                ForwardOptions(stale_gradient_corrections=())
            ),
        )
        == 6 * 12
    )


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("top_k", [0, 4])
@pytest.mark.parametrize("label_columns", [0, 1, 2])
@pytest.mark.parametrize(
    "corrections",
    [None, (), (ImportanceSamplingGradientCorrection(policy="always"),)],
    ids=["default", "disabled", "always"],
)
def test_correction_budget_covers_captured_storage_and_aliases(
    rank, grad_enabled, top_k, label_columns, corrections
):
    from art.trainer_rank._corrections import capture_forward_corrections
    from art.trainer_rank._tensors import flatten_tensors

    labels = (
        torch.arange(3 * label_columns).reshape(
            (3,) if label_columns == 1 else (3, label_columns)
        )
        if label_columns
        else None
    )
    options = _impl._resolved_request_policy(
        ForwardOptions(
            stale_gradient_corrections=_impl.Unset
            if corrections is None
            else corrections
        )
    )
    request = ForwardInput(
        input_tokens=torch.arange(3),
        target_tokens=labels,
        top_k=top_k or None,
        hidden_states=True,
        no_grad=not grad_enabled,
    )
    plan = rank._plan_flat_forward([request])
    group = plan.groups[0]
    output = _impl.ForwardOutput(
        target_logprobs=None
        if labels is None
        else torch.zeros_like(labels, dtype=torch.float32, requires_grad=grad_enabled),
        top_k=_impl.TopK(
            torch.zeros(3, top_k, dtype=torch.float32, requires_grad=grad_enabled),
            torch.arange(3 * top_k).reshape(3, top_k),
        )
        if top_k
        else None,
        logits=None,
        hidden_states=torch.zeros(3, 1, requires_grad=grad_enabled),
    )
    tree = {"output": output, "alias": [output]}
    tensors, _ = flatten_tensors(tree)
    context = capture_forward_corrections(tree, tensors, options)
    retained = sum(tensor.untyped_storage().nbytes() for tensor in context.tensors)
    gradients = tuple(
        torch.ones_like(tensor) if tensor.requires_grad else None for tensor in tensors
    )
    staged = 0
    if corrections and corrections[0].policy == "always":
        corrected = context.correct(gradients, tensors)
        staged = sum(
            after.untyped_storage().nbytes()
            for before, after in zip(gradients, corrected, strict=True)
            if after is not None and after is not before
        )
    assert _impl._correction_state_bytes(group, options) == retained + staged
    if not grad_enabled:
        assert next(rank._graph_memory_units(plan))[2].replay_bytes == 0
    elif output.top_k is not None:
        current = list(tensors)
        index = next(
            i for i, tensor in enumerate(tensors) if tensor is output.top_k.tokens
        )
        current[index] = current[index].flip(-1)
        with pytest.raises(RuntimeError, match="changed active top-k token identities"):
            context.validate_replay(gradients, current)


@pytest.mark.parametrize(
    "corrections,expected",
    [
        ((), 144),
        (None, 144),
        ((ImportanceSamplingGradientCorrection(policy="always"),), 192),
    ],
    ids=["disabled", "default", "always"],
)
def test_top_k_context_must_fit_host_budget_before_forward(
    rank, monkeypatch, corrections, expected
):
    options = ForwardOptions(
        backward_state="gpu",
        stale_gradient_corrections=_impl.Unset if corrections is None else corrections,
    )
    request = ForwardInput(input_tokens=torch.arange(3), top_k=4, options=options)
    plan = rank._plan_flat_forward([request])
    required = _impl._snapshot_tensor_bytes(plan.groups[0]) + 64 * 1024 + expected
    monkeypatch.setattr(rank, "_available_cpu_memory_bytes", lambda: required - 1)
    _, check = rank._admit_graph_memory(plan)
    assert not check.fits and not check.cpu_fits
    assert check.cpu_required_bytes == required
    monkeypatch.setattr(rank, "_available_cpu_memory_bytes", lambda: required)
    _, check = rank._admit_graph_memory(plan)
    assert check.fits and check.cpu_fits


def test_reclamation_respects_captured_forced_policy_and_cpu_capacity(
    rank, monkeypatch
):
    decisions = []
    states = {
        "forced": SimpleNamespace(
            retention="gpu", offloadable=False, replayable=False, offload_bytes=100
        ),
        "auto": SimpleNamespace(
            retention="gpu", offloadable=True, replayable=True, offload_bytes=100
        ),
    }
    rank._graph_cache = SimpleNamespace(
        handles=lambda: tuple(states),
        state=states.__getitem__,
        offload=lambda handle: decisions.append(("cpu", handle)),
        evict=lambda handle: decisions.append(("replay", handle)),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    check = _impl._MemoryCheck(1000, 250, False, cpu_required_bytes=50)
    assert rank._reclaim_graph_memory(check, sync_across_dp=False)
    assert decisions == [("cpu", "auto")]
    decisions.clear()
    monkeypatch.setattr(rank, "_available_cpu_memory_bytes", lambda: 100)
    assert rank._reclaim_graph_memory(check, sync_across_dp=False)
    assert decisions == [("replay", "auto")]


def test_reclamation_transfer_error_is_exchanged_before_propagating(rank, monkeypatch):
    error = RuntimeError("CPU offload allocation failed")
    exchanges = []

    def fail(_handle):
        raise error

    def exchange(values, **_kwargs):
        exchanges.append(values)
        return values

    rank._graph_cache = SimpleNamespace(
        handles=lambda: ("graph",),
        state=lambda _: SimpleNamespace(
            retention="gpu", offloadable=True, replayable=True, offload_bytes=100
        ),
        offload=fail,
        evict=lambda _: None,
    )
    monkeypatch.setattr(rank, "_recovery_reduce", exchange)
    with pytest.raises(RuntimeError) as caught:
        rank._reclaim_graph_memory(
            _impl._MemoryCheck(1000, 250, False), sync_across_dp=False
        )
    assert caught.value is error
    assert exchanges[-1] == [0.0]
