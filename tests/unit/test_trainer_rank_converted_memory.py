"""Source-derived affine routed-expert stages; no complete backward/compiled bound."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_moe_memory import _enclosing_moe, _rank
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import module, rank_with_moe
import torch

from art.trainer_rank import ForwardInput, _gdn_memory
from art.trainer_rank._impl import _expert_lora_weight_storage


def weights(layer: Any, rank: int, *, fc1: bool = True, dtype=torch.bfloat16):
    from art.megatron.lora import LoRA, TEColumnParallelGroupedLinear

    _enclosing_moe(layer)
    for fc, inputs, outputs in (
        (layer.experts.linear_fc2, 512, 2048),
        *(([(layer.experts.linear_fc1, 2048, 1024)]) if fc1 else []),
    ):
        fc.lora = module(LoRA)
        fc.lora.A_T = torch.nn.Parameter(torch.empty(256, inputs, rank, dtype=dtype))
        fc.lora.B_T = torch.nn.Parameter(torch.empty(256, rank, outputs, dtype=dtype))
    if fc1:
        layer.experts.linear_fc1.linear_fc1 = module(TEColumnParallelGroupedLinear)
    return layer


def expected(rows, rank, grad, *, fc1=True, shared=0):
    effective = max(8, rank)
    t2 = 256 * effective * (512 + 2048) * 2
    p2 = t2 if rank < 8 else 0
    p1 = 256 * effective * (2048 + 1024) * 2 if rank < 8 and fc1 and grad else 0
    r1 = effective if fc1 and grad else 0
    inner = rows * (188416 + 16 * (effective + r1 - 2048)) + p2 + t2 + p1
    summed = rows * (188416 + 16 * (effective + r1)) + p2 + p1
    first_stage = (
        rows * 16 * (2 * 2048 + 2 * 1024 + effective)
        + (2 if rank < 8 else 1) * 256 * effective * (2048 + 1024) * 2
    )
    first_sum = rows * 16 * (2 * 2048 + 3 * 1024 + (effective if grad else 0)) + p1
    backward_return = (
        rows * 16 * (2 * 512 + 2 * 2048 + 3 * effective)
        + p1
        + p2
        + t2
        + 256 * rank * 2560 * 2
        + 2 * 257 * 4
    )
    return max(
        backward_return if grad and fc1 and rank < 8 else 0,
        rows * 188416 + shared,
        inner + shared,
        summed + shared if grad else 0,
        first_stage + shared if fc1 else 0,
        first_sum + shared if fc1 else 0,
    )


@pytest.mark.parametrize("rank_value", [1, 7, 8, 16])
@pytest.mark.parametrize("grad", [False, True])
def test_same_stage_crossover_and_constructor(layer, rank_value, grad):
    rank, _ = rank_with_moe(weights(layer, rank_value))
    for rows in (1, 8, 64, 128, 512, 50640):
        assert rank._moe_workspace_bytes(rows, checkpoint_grad=grad) == expected(
            rows, rank_value, grad
        )
    assert rank._moe_workspace_bytes(0, checkpoint_grad=grad) == 0
    assert rank._moe_workspace_bytes(1, checkpoint_grad=grad) > 188416
    if not grad:
        assert rank._moe_workspace_bytes(50640) == 50640 * 188416
    # Metadata was cached before the original dispatcher partial is installed.
    assert "dispatch_preprocess" in vars(layer.token_dispatcher)
    assert rank._moe_workspace_bytes(1, checkpoint_grad=grad) == expected(
        1, rank_value, grad
    )


@pytest.mark.parametrize("rank_value", [1, 7, 8, 16])
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("output", ["hidden", "logprob", "both"])
def test_actual_plan_cost_and_admission(layer, rank_value, grad, output):
    rank, _ = rank_with_moe(weights(layer, rank_value))
    request = ForwardInput(
        input_tokens=torch.arange(8),
        no_grad=not grad,
        hidden_states=output != "logprob",
        target_tokens=torch.arange(8) if output != "hidden" else None,
    )
    plan = rank._plan_flat_forward([request])
    required = rank._memory_check(plan).estimated_required_bytes
    assert required == rank._plan_cost(plan).required
    retained, workspace = rank._checkpoint_memory_floor(rank._plan_group_rows(plan))
    pending = _gdn_memory.plan_floor(rank, plan)
    if grad:
        assert workspace == expected(8, rank_value, True)
        assert pending[0] == retained == 8 * 40 * 2048 * 2
        assert pending[1] >= workspace
    else:
        assert (retained, workspace) == pending == (0, 0)
    assert required >= int((plan.output_bytes + expected(8, rank_value, grad)) * 1.1)
    rank._available_memory_bytes = lambda: required - 1
    assert not rank._memory_check(plan).fits


@pytest.mark.parametrize("order", [False, True])
@pytest.mark.parametrize("rank_value", [1, 7])
def test_reference_and_gradient_keep_distinct_stage_modes(layer, order, rank_value):
    rank, _ = rank_with_moe(weights(layer, rank_value))
    requests = [
        ForwardInput(input_tokens=torch.arange(3), hidden_states=True),
        ForwardInput(
            input_tokens=torch.arange(9) + 100, hidden_states=True, no_grad=True
        ),
    ]
    if order:
        requests.reverse()
    plan = rank._plan_flat_forward(requests)
    groups = rank._plan_group_rows(plan)
    assert set(groups) == {(3, True), (9, False)}
    retained, workspace = rank._checkpoint_memory_floor(groups)
    assert retained == 3 * 40 * 2048 * 2
    assert workspace == max(
        expected(3, rank_value, True), expected(9, rank_value, False)
    )
    assert (
        rank._memory_check(plan).estimated_required_bytes
        == rank._plan_cost(plan).required
    )


@pytest.mark.parametrize("rank_value", [1, 7, 8, 16])
def test_alias_original_parameters_and_dtype(layer, rank_value):
    weights(layer, rank_value, dtype=torch.float16)
    storage = _expert_lora_weight_storage(layer.experts.linear_fc2.lora)
    assert storage is not None
    padding, transposes, effective = storage
    assert effective == max(8, rank_value)
    assert transposes == 256 * effective * 2560 * 2
    assert padding == (transposes if rank_value < 8 else 0)
    assert _rank(layer)._moe_workspace_bytes(1) == expected(1, rank_value, False)


@pytest.mark.parametrize(
    "mutation", ["rank9", "noncontiguous", "owner", "hook", "missing"]
)
def test_unsupported_conversion_keeps_prior_component(layer, mutation):
    weights(layer, 1, fc1=False)
    lora = layer.experts.linear_fc2.lora
    if mutation == "rank9":
        weights(layer, 9, fc1=False)
    elif mutation == "noncontiguous":
        lora.A_T = torch.nn.Parameter(
            torch.empty(256, 1, 512, dtype=torch.bfloat16).transpose(1, 2)
        )
        # Rank-one transpose is contiguous; use a genuine noncontiguous slice.
        lora.A_T = torch.nn.Parameter(
            torch.empty(256, 512, 2, dtype=torch.bfloat16)[..., :1]
        )
    elif mutation == "owner":
        lora.forward = lambda *args: None
    elif mutation == "hook":
        lora.register_forward_hook(lambda *args: None)
    else:
        lora.A_T = None
    rank = _rank(layer)
    assert rank._moe_forward_stages == rank._moe_gradient_stages == ()
    assert rank._moe_workspace_bytes(1) == rank._moe_output_bytes_per_token


@pytest.mark.parametrize("bad", [None, [], ((True, 1),), ((1, -1),), ((1, 2, 3),)])
def test_corrupted_cache_refuses_before_memory_reduction(layer, bad, monkeypatch):
    rank, _ = rank_with_moe(weights(layer, 1))
    plan = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(1), hidden_states=True)]
    )
    rank._moe_forward_stages = bad

    def reduce(*args, **kwargs):
        raise AssertionError("entered memory reduction before local planning failed")

    monkeypatch.setattr(rank, "_memory_check_required", reduce)
    with pytest.raises(ValueError, match="converted-weight"):
        rank._memory_check(plan)


def test_missing_fc1_metadata_does_not_invent_saved_bank(layer):
    rank, _ = rank_with_moe(weights(layer, 1, fc1=False))
    assert rank._moe_workspace_bytes(1, checkpoint_grad=True) == expected(
        1, 1, True, fc1=False
    )


def test_heterogeneous_joint_stage_max_not_separate_maxima(layer):
    from test_trainer_rank_moe_memory import layer as factory

    first = weights(layer, 1)
    second = weights(cast(Any, factory).__wrapped__(), 16)
    second.config.moe_router_topk = second.router.topk = 1
    model = torch.nn.ModuleList([first, second])
    rank = _rank(model)
    one = _rank(weights(cast(Any, factory).__wrapped__(), 1))
    other = weights(cast(Any, factory).__wrapped__(), 16)
    other.config.moe_router_topk = other.router.topk = 1
    two = _rank(other)
    for n in (1, 64, 50640):
        assert rank._moe_workspace_bytes(n) == max(
            one._moe_workspace_bytes(n), two._moe_workspace_bytes(n)
        )


@pytest.mark.parametrize(
    "mutation", ["base owner", "base hook", "adapter override", "shape"]
)
def test_unqualified_fc1_saves_are_not_invented(layer, mutation):
    weights(layer, 1)
    fc1 = layer.experts.linear_fc1
    if mutation == "base owner":
        fc1.linear_fc1 = torch.nn.Identity()
    elif mutation == "base hook":
        fc1.linear_fc1.register_forward_pre_hook(lambda *args: None)
    elif mutation == "adapter override":
        fc1.lora.active_lora_tensors = lambda: None
    else:
        fc1.lora.A_T = torch.nn.Parameter(
            torch.empty(256, 2047, 1, dtype=torch.bfloat16)
        )
    rank, _ = rank_with_moe(layer)
    assert rank._moe_workspace_bytes(1, checkpoint_grad=True) == expected(
        1, 1, True, fc1=False
    )


def test_other_checkpoint_modes_remain_partial_forward_only(layer):
    rank, _ = rank_with_moe(weights(layer, 1))
    rank.runtime.model[0].decoder.config.recompute_granularity = None
    p = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(1), hidden_states=True)]
    )
    assert rank._checkpoint_memory_floor(rank._plan_group_rows(p)) == (0, 0)
    assert _gdn_memory.plan_floor(rank, p) == (0, 0)
    assert rank._memory_check(p).estimated_required_bytes == int(
        (p.output_bytes + expected(1, 1, False)) * 1.1
    )


@pytest.mark.parametrize("topk", [1, 4, 8])
def test_fc1_fixed_weights_do_not_scale_with_topk(layer, topk):
    weights(layer, 1)
    layer.config.moe_router_topk = layer.router.topk = topk
    rank = _rank(layer)
    for rows in (1, 64, 1024):
        first_inner = rows * topk * (2 * 2048 + 2 * 1024 + 8) * 2 + 25165824
        second_inner = rows * topk * (4 * 2048 + 3 * 512 + 8) * 2 + 20971520
        original = rows * topk * (5 * 2048 + 3 * 512) * 2
        assert rank._moe_workspace_bytes(rows) == max(
            first_inner, second_inner, original
        )


def test_wide_fc1_sum_is_a_separate_stage(layer):
    weights(layer, 8)
    experts = layer.experts
    experts.linear_fc1.out_features = 32768
    layer.config.moe_ffn_hidden_size = 16384
    layer.token_dispatcher.num_local_experts = 16
    layer.config.num_moe_experts = 16
    for fc, inputs, outputs in (
        (experts.linear_fc1, 2048, 32768),
        (experts.linear_fc2, 16384, 2048),
    ):
        fc.lora.A_T = torch.nn.Parameter(
            torch.empty(16, inputs, 8, dtype=torch.bfloat16)
        )
        fc.lora.B_T = torch.nn.Parameter(
            torch.empty(16, 8, outputs, dtype=torch.bfloat16)
        )
    rank = _rank(layer)
    # Large N crosses from fixed conversion weights to the three simultaneous
    # 2F outputs at the original eager FC1 sum; this is not an FC2 coefficient.
    for grad in (False, True):
        expected_sum = 1024 * 8 * (2 * 2048 + 3 * 32768 + (8 if grad else 0)) * 2
        assert rank._moe_workspace_bytes(1024, checkpoint_grad=grad) == expected_sum


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("grad", [False, True])
def test_fc1_stage_keeps_same_layer_shared_output(layer, gated, grad):
    from test_trainer_rank_shared_memory import shared_layer

    rank, _ = rank_with_moe(weights(shared_layer(layer, gated), 1))
    for rows in (1, 64, 50640):
        assert rank._moe_workspace_bytes(rows, checkpoint_grad=grad) == expected(
            rows, 1, grad, shared=rows * 4096 * (2 if gated and grad else 1)
        )


@pytest.mark.parametrize("rows,known", [(1, 42813832), (8, 43389960)])
def test_rank7_original_return_storage_fits_total_admission(layer, rows, known):
    # Actual-source CPU return witness counts distinct storage, excluding
    # parameters and all speculative GDN/boundary/compiled terms.
    rank, _ = rank_with_moe(weights(layer, 7))
    plan = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(rows), hidden_states=True)]
    )
    assert rank._moe_workspace_bytes(rows, checkpoint_grad=True) == known
    assert rank._memory_check(plan).estimated_required_bytes >= known
    assert rank._plan_cost(plan).required >= known


@pytest.mark.parametrize("rank_value", [1, 7, 8, 16])
def test_backward_return_stage_is_checkpoint_only_and_not_shared_max(layer, rank_value):
    from test_trainer_rank_shared_memory import shared_layer

    rank, _ = rank_with_moe(weights(shared_layer(layer, True), rank_value))
    assert len(rank._moe_gradient_stages) == (5 if rank_value < 8 else 4)
    assert len(rank._moe_forward_stages) == 3
    if rank_value < 8:
        # The witnessed backward stage excludes unproved shared-output lifetime.
        assert (
            82304,
            33554432 + 256 * rank_value * 2560 * 2 + 2056,
        ) in rank._moe_gradient_stages
    for rows in (1, 8, 50640):
        assert rank._moe_workspace_bytes(rows, checkpoint_grad=True) == expected(
            rows, rank_value, True, shared=rows * 8192
        )
        assert rank._moe_workspace_bytes(rows) == expected(
            rows, rank_value, False, shared=rows * 4096
        )
