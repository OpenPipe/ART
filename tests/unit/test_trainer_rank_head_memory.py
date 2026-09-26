"""Standard BF16 head capacity: CPU/source pricing, not a native peak bound."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank._impl import (
    _PACKED_PRICED_LOGICAL_ROW_BYTES,
    Unset,
    _MemoryProfile,
)


def rank():
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    spec = importlib.util.spec_from_file_location(
        "checkpoint_memory_tests",
        Path(__file__).with_name("test_trainer_rank_checkpoint_memory.py"),
    )
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    r = source.rank()
    model = r.runtime.model[0]
    head = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(
        torch.empty(248320, 2048, device="meta", dtype=torch.bfloat16)
    )
    head.input_size = 2048
    head.output_size = head.output_size_per_partition = 248320
    model.output_layer = head
    model.share_embeddings_and_output_weights = False
    from types import MethodType

    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    model._scale_logits = MethodType(LanguageModule._scale_logits, model)
    model.config.use_mup = False
    model.config.padded_vocab_size = 248320
    r._padded_vocab_size = 248320
    return r


def request(rows=512, *, grad=False, hidden=False, ignored=False):
    return ForwardInput(
        input_tokens=torch.arange(rows),
        target_tokens=torch.full((rows,), -100) if ignored else torch.arange(rows),
        no_grad=not grad,
        hidden_states=hidden,
    )


@pytest.mark.parametrize("grad,budget", [(False, 128 * 1024**2), (True, 220 * 1024**2)])
def test_actual_admission_rejects_below_dense_head_tensor(grad, budget):
    r = rank()
    plan = r._plan_flat_forward([request(grad=grad)])
    r._available_memory_bytes = lambda: budget
    assert not r._memory_check(plan).fits


@pytest.mark.parametrize("fits_after", (False, True))
def test_mixed_checkpoint_head_demand_survives_recovery(monkeypatch, fits_after):
    from test_trainer_rank_cache_recovery import _check_component_demand_recovery

    r = rank()
    requests = [request(8, grad=True), request(16, grad=False, hidden=True)]
    values = r._estimate_flat_forward(requests, exact=True)
    assert values[3] == ((8, True), (16, False))
    assert r._checkpoint_memory_floor(values[3])[0] == 8 * 40 * 2048 * 2
    assert values[4] == 3 * 8 * 248320 * 2
    _check_component_demand_recovery(monkeypatch, r, requests, fits_after=fits_after)


def test_recovery_keeps_profile_demand_above_cold_head_floor(monkeypatch):
    from test_trainer_rank_cache_recovery import _check_component_demand_recovery

    r = rank()
    requests = [request(1, grad=True)]  # One row cannot be split smaller.
    plan = r._plan_flat_forward(requests)
    cold = r._memory_check(plan).estimated_required_bytes
    r._update_memory_profile(plan, 4 * cold, retained_bytes=plan.output_bytes)
    assert r._memory_check(plan).estimated_required_bytes > cold
    _check_component_demand_recovery(
        monkeypatch, r, requests, fits_after=False, after_available=cold
    )


def test_real_head_fitting_split_after_cache_recovery_denial(monkeypatch):
    from test_trainer_rank_split import _recording_executor

    from art.trainer_rank import TrainerRank, _impl

    r = rank()
    monkeypatch.setattr(r, "_dp_rank_and_size", lambda: (0, 1))
    requests = [
        replace(request(8), input_tokens=torch.arange(8) + 100 * i) for i in range(4)
    ]
    whole = r._plan_flat_forward(requests)
    r._update_memory_profile(
        whole, r._plan_cost(whole).required, retained_bytes=whole.output_bytes
    )
    children = [r._plan_flat_forward(requests[i : i + 2]) for i in (0, 2)]
    left, right = [r._plan_cost(child) for child in children]
    budget = max(left.required, left.retained + right.required)
    assert 0 < left.retained and budget < r._plan_cost(whole).required
    total = 10 * r._plan_cost(whole).required
    free = budget + int(total * _impl._MEMORY_RESERVE_FRACTION)
    probe = TrainerRank.__new__(TrainerRank)
    probe.device = torch.device("cuda")
    monkeypatch.delenv(_impl._TEST_HOOKS_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: total)
    monkeypatch.setattr(
        r, "_available_memory_bytes", lambda: TrainerRank._available_memory_bytes(probe)
    )
    denied_check = r._memory_check(whole)
    assert not denied_check.fits
    recovery_attempts = []

    def deny_recovery(check, *, require_unused_cache=False, **kwargs):
        recovery_attempts.append((check, require_unused_cache))
        return False

    monkeypatch.setattr(r, "_try_cache_recovery", deny_recovery)
    executed = _recording_executor(monkeypatch, r)
    batches = list(r.forward_micro_batches([requests]))
    assert recovery_attempts == [(denied_check, True)]
    assert len(batches) == 1 and batches[0].stats.subforward_count == 2
    assert batches[0].stats.global_count == 1 and len(executed) == 2
    assert [
        [int(output.target_logprobs.item()) for output in group]
        for group in batches[0].outputs
    ] == [[7, 107, 207, 307]]
    assert r.last_forward_telemetry()["subforward_request_indices"] == ((0, 1), (2, 3))
    assert not torch.cuda.is_initialized()


def test_outputs_retention_and_empirical_peak_are_counted_once():
    r = rank()
    plan = r._plan_flat_forward([request(grad=True)])
    retained = 512 * 40 * 2048 * 2
    gradient = 512 * 40 * 2048 * 2
    head = 3 * 512 * 248320 * 2
    cost = r._plan_cost(plan)
    assert cost.retained == int((plan.output_bytes + retained + head) * 1.1)
    assert cost.required == int((plan.output_bytes + retained + gradient + head) * 1.1)
    r._memory_profiles[plan.signature] = _MemoryProfile(
        bytes_per_token=2_000_000,
        packed_tokens=512,
        logical_per_packed=1,
        retained_compute_bytes_per_token=1,
    )
    cost = r._plan_cost(plan)
    # Packed pricing adds head and caller memory for every logical row.
    rows = _PACKED_PRICED_LOGICAL_ROW_BYTES * 512
    assert cost.required == int((plan.output_bytes + 512 * 2_000_000 + rows) * 1.1)
    assert cost.retained == int((plan.output_bytes + retained) * 1.1)


def test_ignored_targets_hidden_only_and_tiny_target_group():
    r = rank()
    ignored = request(ignored=True)
    hidden = ForwardInput(
        input_tokens=torch.arange(10000), hidden_states=True, no_grad=True
    )
    target = request(1, grad=True)
    assert r._head_projection_rows([ignored, hidden]) == 0
    assert r._plan_head_workspace_bytes(r._plan_flat_forward([ignored, hidden])) == 0
    plan = r._plan_flat_forward([hidden, target])
    assert r._plan_head_workspace_bytes(plan) == 3 * 248320 * 2
    assert r._estimate_flat_forward([hidden, target])[-1] == 3 * 248320 * 2
    assert r._estimate_flat_forward([hidden, target], exact=True)[-1] == 3 * 248320 * 2


def test_multilabel_row_validity_matches_projection():
    r = rank()
    item = replace(
        request(4),
        target_tokens=torch.tensor([[-100, -100], [-100, 2], [3, -100], [-100, -100]]),
    )
    assert r._head_projection_rows([item]) == 2
    assert r._plan_head_workspace_bytes(r._plan_flat_forward([item])) == 2 * 248320 * 2


def test_shared_rows_use_lower_upper_and_exact_layout_union():
    r = rank()
    a = replace(request(2), target_tokens=torch.tensor([1, -100]))
    b = replace(a, target_tokens=torch.tensor([-100, 1]))
    req = [a, a, b, b]
    assert r._head_projection_rows(req, lower_bound=True) == 1
    assert r._head_projection_rows(req) == 4
    exact = r._estimate_flat_forward(req, exact=True, memory_minimal=True)
    plan = r._plan_flat_forward(req, memory_minimal=True)
    assert exact[-1] == r._plan_head_workspace_bytes(plan) == 2 * 248320 * 2
    lower = r._split_chunk_lower_cost(
        req, tuple(x.input_tokens for x in req), checkpoint=Unset
    )
    assert lower.required <= r._plan_cost(plan).required
    assert r._estimate_flat_forward(req, memory_minimal=True)[-1] == 248320 * 2


def test_exact_selector_estimate_matches_executed_layout():
    r = rank()
    req = [replace(request(2), target_tokens=torch.tensor([1, -100])) for _ in range(4)]
    for minimal in (False, True):
        values = r._estimate_flat_forward(req, exact=True, memory_minimal=minimal)
        plan = r._plan_flat_forward(req, memory_minimal=minimal)
        assert values[-1] == r._plan_head_workspace_bytes(plan)
        n, out, sig, groups, head = values
        assert (
            r._estimate_required_memory_bytes_from_values(
                packed_tokens=n,
                output_bytes=out,
                signature=sig,
                logical_tokens=plan.active_logical_tokens,
                group_rows=groups,
                head_workspace_bytes=head,
            )
            == r._memory_check(plan).estimated_required_bytes
        )


def test_device_labels_use_capacity_without_reading_values():
    r = rank()
    item = replace(
        request(128), target_tokens=torch.empty(128, device="meta", dtype=torch.long)
    )
    assert r._head_projection_rows([item]) == 128
    assert r._head_projection_rows([item], lower_bound=True) == 0
    assert r._head_projection_rows([item], positions=(torch.arange(128),)) == 128


def test_topk_and_logits_project_ignored_rows_and_chunk_cap():
    r = rank()
    ignored = request(2048, ignored=True)
    assert r._head_projection_rows([replace(ignored, top_k=2)]) == 512
    assert r._head_projection_rows([replace(ignored, logits=True)]) == 512
    assert r._head_workspace_bytes(4096) == 512 * 248320 * 2


@pytest.mark.parametrize(
    "mutation",
    [
        "dtype",
        "tp",
        "head_hook",
        "head_override",
        "head_dispatch_override",
        "vocab_shape",
        "quantized",
        "missing_weight",
        "unknown_vocab",
    ],
)
def test_unsupported_head_scope_does_not_claim_dense_bf16_component(mutation):
    r = rank()
    head = r.runtime.model[0].output_layer
    assert r._head_workspace_bytes(512) > 0
    if mutation == "dtype":
        head.weight = torch.nn.Parameter(head.weight.float())
    elif mutation == "tp":
        r._topology_key = lambda: (1, 2, 1, 1)
    elif mutation == "head_hook":
        head.register_forward_hook(lambda *args: None)
    elif mutation == "head_override":
        head.forward = lambda *args, **kwargs: None
    elif mutation == "head_dispatch_override":
        head._forward_impl = lambda *args, **kwargs: None
    elif mutation == "vocab_shape":
        head.output_size_per_partition -= 1
    elif mutation == "missing_weight":
        head.weight = None
    elif mutation == "unknown_vocab":
        r._padded_vocab_size = None
    else:
        r.runtime.model[0].config.fp8 = "hybrid"
    assert r._head_workspace_bytes(512) == 0


def test_device_positions_preserve_capacity_without_read():
    r = rank()
    item = request(128)
    assert (
        r._head_projection_rows(
            [item], positions=(torch.empty(128, device="meta", dtype=torch.long),)
        )
        == 128
    )


def test_tied_standard_head_weight_uses_the_same_capacity():
    r = rank()
    model = r.runtime.model[0]
    head = model.output_layer
    weight = head.weight
    head.weight = None
    model.share_embeddings_and_output_weights = True
    model.embedding = torch.nn.Module()
    model.embedding.word_embeddings = torch.nn.Module()
    model.embedding.word_embeddings.weight = weight
    assert r._head_workspace_bytes(512) == 512 * 248320 * 2


@pytest.mark.parametrize("rows", [128, 512])
def test_target_backward_refuses_budget_below_logits_and_both_gradients(rows):
    r = rank()
    plan = r._plan_flat_forward([request(rows, grad=True)])
    retained, _ = r._checkpoint_memory_floor(r._plan_group_rows(plan))
    gradient = rows * 40 * 2048 * 2
    dense = min(rows, 512) * 248320 * 2
    before = int((plan.output_bytes + retained + gradient + 2 * dense) * 1.1)
    expected = int((plan.output_bytes + retained + gradient + 3 * dense) * 1.1)
    r._available_memory_bytes = lambda: (before + expected) // 2
    check = r._memory_check(plan)
    assert check.estimated_required_bytes == expected
    assert not check.fits


@pytest.mark.parametrize("gradient_rows,reference_rows", [(1, 512), (512, 1)])
def test_group_head_workspace_keeps_gradient_mode_with_its_rows(
    gradient_rows, reference_rows
):
    r = rank()
    requests = [request(gradient_rows, grad=True), request(reference_rows)]
    expected = max(3 * gradient_rows, reference_rows) * 248320 * 2
    plan = r._plan_flat_forward(requests)
    assert r._plan_head_workspace_bytes(plan) == expected
    for exact in (False, True):
        for minimal in (False, True):
            assert (
                r._estimate_flat_forward(requests, exact=exact, memory_minimal=minimal)[
                    -1
                ]
                == expected
            )


@pytest.mark.parametrize(
    "mutation", ["custom", "other_model", "forged", "mup", "missing"]
)
def test_gradient_statistics_floor_requires_exact_effective_scaling(mutation):
    from types import MethodType, SimpleNamespace

    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    r = rank()
    model = r.runtime.model[0]
    req = [request(128, grad=True)]
    assert (
        r._plan_head_workspace_bytes(r._plan_flat_forward(req)) == 3 * 128 * 248320 * 2
    )
    if mutation == "custom":
        model._scale_logits = lambda logits: logits
    elif mutation == "other_model":
        model._scale_logits = MethodType(
            LanguageModule._scale_logits,
            SimpleNamespace(config=SimpleNamespace(use_mup=True, mup_output_mult=2)),
        )
    elif mutation == "forged":

        class Forged:
            __self__ = model
            __func__ = LanguageModule._scale_logits

            def __call__(self, logits):
                return logits[..., :1]

        model._scale_logits = Forged()
    elif mutation == "mup":
        model.config.use_mup = True
    else:
        del model._scale_logits
    # The standard head still allocates its original one-buffer component.
    assert r._plan_head_workspace_bytes(r._plan_flat_forward(req)) == 128 * 248320 * 2


@pytest.mark.parametrize("extra", [{"logits": True}, {"top_k": 2}])
def test_gradient_statistics_floor_survives_additional_output_modes(extra):
    r = rank()
    req = [replace(request(128, grad=True), **extra)]
    assert (
        r._plan_head_workspace_bytes(r._plan_flat_forward(req)) == 3 * 128 * 248320 * 2
    )


def test_gradient_shared_rows_price_same_union_in_exact_and_split_lower_cost():
    r = rank()
    a = replace(request(2, grad=True), target_tokens=torch.tensor([1, -100]))
    b = replace(a, target_tokens=torch.tensor([-100, 1]))
    requests = [a, a, b, b]
    plan = r._plan_flat_forward(requests, memory_minimal=True)
    expected = 3 * 2 * 248320 * 2
    exact = r._estimate_flat_forward(requests, exact=True, memory_minimal=True)
    assert exact[-1] == r._plan_head_workspace_bytes(plan) == expected
    assert r._estimate_flat_forward(requests, memory_minimal=True)[-1] == expected // 2
    lower = r._split_chunk_lower_cost(
        requests, tuple(x.input_tokens for x in requests), checkpoint=Unset
    )
    assert lower.required <= r._plan_cost(plan).required


def test_later_sparse_loss_does_not_reduce_6330_projected_targets():
    r = rank()
    item = request(6330, grad=True)
    plan = r._plan_flat_forward([item])
    assert item.target_tokens.numel() == 6330
    assert r._plan_head_workspace_bytes(plan) == 3 * 512 * 248320 * 2
