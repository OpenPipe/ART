"""One supported shared return held across routed compute; not all backward saves."""

from types import SimpleNamespace

import pytest
from test_trainer_rank_moe_memory import _enclosing_moe, _rank
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import full_requests, module, rank_with_moe
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank import _gdn_memory as g
from art.trainer_rank._impl import (
    _moe_output_bytes_per_token,
    _shared_expert_output_bytes_per_token,
)
from art.trainer_rank._planner_cost import ParallelShape


def shared_layer(layer, gated=True):
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    from art.megatron.lora import (
        LoRA,
        SelfAttentionLinearProjLoRA,
        SharedExpertsLinearFC1LoRA,
        SharedExpertsLinearFC2LoRA,
    )

    _enclosing_moe(layer)
    values = dict(
        params_dtype=torch.bfloat16,
        moe_shared_expert_overlap=False,
        sequence_parallel=False,
        fp32_residual_connection=False,
        add_bias_linear=False,
        use_te_activation_func=False,
        bias_activation_fusion=False,
        gated_linear_unit=True,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        moe_shared_expert_intermediate_size=512,
        activation_func=torch.nn.functional.silu,
    )
    vars(layer.config).update(values)
    layer.use_shared_expert = True
    layer.shared_expert_overlap = False
    layer.shared_experts_recompute = False
    layer.moe_layer_recompute = False
    layer.fwd_execution_map = ["route", "expert_compute", "postprocess"]
    shared = module(SharedExpertMLP)
    layer.shared_experts = shared
    shared.config = SimpleNamespace(**{**vars(layer.config), "ffn_hidden_size": 512})
    shared.activation_func = torch.nn.functional.silu
    shared.use_shared_expert_gate = gated
    shared.gate_weight = torch.nn.Parameter(
        torch.empty(1, 2048, dtype=torch.bfloat16), requires_grad=False
    )
    fc1 = module(SharedExpertsLinearFC1LoRA)
    shared.linear_fc1 = fc1
    fc1.non_gated = False
    fc1.out_features = 1024
    fc1.linear_fc1 = module(TEColumnParallelLinear)
    fc1.linear_fc1.weight = torch.nn.Parameter(
        torch.empty(1024, 2048, dtype=torch.bfloat16), requires_grad=False
    )

    def adapter(inputs, outputs):
        value = module(LoRA)
        value.A_T = torch.nn.Parameter(torch.empty(inputs, 8, dtype=torch.bfloat16))
        value.B_T = torch.nn.Parameter(torch.empty(8, outputs, dtype=torch.bfloat16))
        return value

    fc1.gate_lora = adapter(2048, 512)
    fc1.up_lora = adapter(2048, 512)
    fc2 = module(SharedExpertsLinearFC2LoRA)
    shared.linear_fc2 = fc2
    row = module(SelfAttentionLinearProjLoRA)
    fc2.row_parallel_lora = row
    row.provider = SimpleNamespace(
        tensor_model_parallel_size=1, sequence_parallel=False
    )
    row.lora = adapter(512, 2048)
    row.linear_proj = module(TERowParallelLinear)
    row.linear_proj.weight = torch.nn.Parameter(
        torch.empty(2048, 512, dtype=torch.bfloat16), requires_grad=False
    )
    return layer


def coefficient(layer):
    return _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1))


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("no_grad", [False, True])
def test_shared_return_in_actual_constructor_and_plan(layer, gate, no_grad):
    rank, _ = rank_with_moe(shared_layer(layer, gate))
    assert rank._moe_output_bytes_per_token == 192512
    checkpoint_coefficient = 196608 if gate else 192512
    assert rank._moe_checkpoint_grad_bytes_per_token == checkpoint_coefficient
    shapes = g.model_shapes(rank)
    assert shapes is not None and shapes[1][0].moe_bytes_per_row == 192512
    requests = full_requests(no_grad)
    plan = rank._plan_flat_forward(requests)
    assert (
        rank._memory_check(plan).estimated_required_bytes
        == rank._plan_cost(plan).required
    )
    if no_grad:
        assert g.plan_floor(rank, plan) == (0, 0)
        assert rank._checkpoint_memory_floor(rank._plan_group_rows(plan)) == (
            0,
            50640 * (192512 + 4 * 2048 * 2),
        )
        assert rank._plan_cost(plan).required == 11636565600
    else:
        assert g.plan_floor(rank, plan) == (
            8296857600,
            50640 * (checkpoint_coefficient + 128) + 3157761952,
        )
        assert rank._plan_cost(plan).required == (23559286467 if gate else 23331122883)
    selected = rank._select_next_micro_batch(requests, 0)
    assert (
        selected.check.estimated_required_bytes
        == rank._memory_check(selected.plan).estimated_required_bytes
    )


@pytest.mark.parametrize("gated", [False, True])
def test_original_norm_installation_preserves_shared_return(layer, gated):
    layer = shared_layer(layer, gated)
    rank, _ = rank_with_moe(layer, install_hooks=True)
    assert _shared_expert_output_bytes_per_token(layer) == 4096
    assert rank._moe_output_bytes_per_token == 192512
    plan = rank._plan_flat_forward(full_requests())
    checkpoint_coefficient = 196608 if gated else 192512
    assert rank._moe_checkpoint_grad_bytes_per_token == checkpoint_coefficient
    assert g.plan_floor(rank, plan) == (
        8296857600,
        50640 * (checkpoint_coefficient + 128) + 3157761952,
    )
    expected = 23559286467 if gated else 23331122883
    assert rank._memory_check(plan).estimated_required_bytes == expected
    assert rank._plan_cost(plan).required == expected


mutations = {
    "no shared": lambda x: delattr(x, "shared_experts"),
    "disabled shared": lambda x: setattr(x, "use_shared_expert", False),
    "overlap": lambda x: setattr(x, "shared_expert_overlap", True),
    "config overlap": lambda x: setattr(x.config, "moe_shared_expert_overlap", True),
    "shared unknown owner": lambda x: setattr(x, "shared_experts", torch.nn.Identity()),
    "shared forward replaced": lambda x: setattr(
        x.shared_experts, "forward", lambda *a: None
    ),
    "shared forward hook": lambda x: x.shared_experts.register_forward_hook(
        lambda *a: None
    ),
    "fc2 unknown owner": lambda x: setattr(
        x.shared_experts, "linear_fc2", torch.nn.Identity()
    ),
    "fc1 missing": lambda x: delattr(x.shared_experts, "linear_fc1"),
    "adapter missing": lambda x: delattr(x.shared_experts.linear_fc1.gate_lora, "A_T"),
    "adapter shape": lambda x: setattr(
        x.shared_experts.linear_fc1.gate_lora,
        "A_T",
        torch.nn.Parameter(torch.empty(2047, 8, dtype=torch.bfloat16)),
    ),
    "base dtype": lambda x: (
        x.shared_experts.linear_fc2.row_parallel_lora.linear_proj.float()
    ),
    "gate dtype": lambda x: setattr(
        x.shared_experts, "gate_weight", torch.nn.Parameter(torch.empty(1, 2048))
    ),
    "gate shape": lambda x: setattr(
        x.shared_experts,
        "gate_weight",
        torch.nn.Parameter(torch.empty(2, 2048, dtype=torch.bfloat16)),
    ),
    "gate bool": lambda x: setattr(x.shared_experts, "use_shared_expert_gate", 1),
    "sequence parallel": lambda x: setattr(
        x.shared_experts.config, "sequence_parallel", True
    ),
    "activation": lambda x: setattr(
        x.shared_experts, "activation_func", torch.nn.functional.relu
    ),
    "partial layer execution": lambda x: setattr(
        x, "fwd_execution_map", ["expert_compute", "postprocess"]
    ),
    "recomputed shared": lambda x: setattr(x, "shared_experts_recompute", True),
    "recomputed MoE": lambda x: setattr(x, "moe_layer_recompute", True),
    "shared compute replaced": lambda x: setattr(
        x, "shared_experts_compute", lambda *a: None
    ),
    "topology bool": lambda x: setattr(x.config, "tensor_model_parallel_size", True),
    "topology two": lambda x: setattr(x.config, "context_parallel_size", 2),
    "missing config": lambda x: delattr(x.shared_experts, "config"),
}


@pytest.mark.parametrize("name", list(mutations))
def test_unsupported_shared_branch_keeps_prior_routed_component(layer, name):
    shared_layer(layer)
    mutations[name](layer)
    assert _shared_expert_output_bytes_per_token(layer) == 0
    assert coefficient(layer) == 188416
    assert (
        _moe_output_bytes_per_token(
            [layer], ParallelShape(tp=1, cp=1), checkpoint_grad=True
        )
        == 188416
    )
    rank = _rank(layer)
    assert rank._moe_output_bytes_per_token == 188416
    assert rank._moe_checkpoint_grad_bytes_per_token == 188416


def test_shared_return_has_no_topk_or_layer_multiplier(layer):
    shared_layer(layer)
    assert coefficient(layer) == 192512
    assert (
        _moe_output_bytes_per_token([layer, layer], ParallelShape(tp=1, cp=1)) == 192512
    )
    layer.config.moe_router_topk = layer.router.topk = 4
    assert coefficient(layer) == 4 * (3 * 512 + 5 * 2048) * 2 + 4096
    assert _shared_expert_output_bytes_per_token(layer) == 4096


def test_shared_component_is_joint_layer_max_and_survives_fc1_fallback(layer):
    shared_layer(layer)
    # One layer has the larger shared return; the other has the larger routed
    # stage. Duplicate only tiny metadata; CPU tensors are never executed.
    import copy

    other = copy.deepcopy(layer)
    del other.shared_experts
    other.config.moe_router_topk = other.router.topk = 9
    assert (
        _moe_output_bytes_per_token([layer, other], ParallelShape(tp=1, cp=1))
        == 9 * (3 * 512 + 5 * 2048) * 2
    )
    del layer.experts.linear_fc1
    assert coefficient(layer) == 8 * (512 + 3 * 2048) * 2 + 4096


def test_reference_group_has_shared_stage_but_no_pending_save(layer):
    rank, _ = rank_with_moe(shared_layer(layer))
    grad = ForwardInput(
        input_tokens=torch.arange(67), hidden_states=True, no_grad=False
    )
    reference = ForwardInput(
        input_tokens=torch.arange(4096), hidden_states=True, no_grad=True
    )
    one = rank._plan_flat_forward([grad])
    mixed = rank._plan_flat_forward([grad, reference])
    retained, workspace = g.plan_floor(rank, one)
    assert g.plan_floor(rank, mixed) == (retained, max(workspace, 4096 * 192512))
    assert g.plan_floor(rank, rank._plan_flat_forward([reference])) == (0, 0)


def test_pre_gate_is_same_layer_stage_not_sum_of_separate_maxima(layer):
    import copy

    shared_layer(layer)
    other = copy.deepcopy(layer)
    del other.shared_experts
    other.experts.linear_fc2.lora.A_T = torch.nn.Parameter(
        torch.empty(256, 640, 8, dtype=torch.bfloat16)
    )
    other.experts.linear_fc1.out_features = 1280
    shape = ParallelShape(tp=1, cp=1)
    # Routed-only width640 wins forward; gated width512 wins recomputation.
    assert _moe_output_bytes_per_token([layer, other], shape) == 194560
    assert (
        _moe_output_bytes_per_token([layer, other], shape, checkpoint_grad=True)
        == 196608
    )
    assert (
        _moe_output_bytes_per_token([layer, layer], shape, checkpoint_grad=True)
        == 196608
    )
    layer.config.moe_router_topk = layer.router.topk = 4
    assert (
        _moe_output_bytes_per_token([layer], shape, checkpoint_grad=True)
        == 4 * (3 * 512 + 5 * 2048) * 2 + 2 * 4096
    )


def test_pre_gate_cache_precedes_owned_dispatcher_and_is_checkpoint_only(layer):
    rank, _ = rank_with_moe(shared_layer(layer))
    assert (
        _moe_output_bytes_per_token(
            rank.runtime.model, rank._parallel_shape, checkpoint_grad=True
        )
        == 0
    )  # Installed dispatcher partials must not be repriced.
    assert rank._moe_checkpoint_grad_bytes_per_token == 196608
    groups = ((19, True), (23, False))
    assert rank._checkpoint_memory_floor(groups) == (
        19 * 40 * 4096,
        max(
            19 * (196608 + 128),
            23 * 192512,
            19 * (196608 - 32768 + 128) + 10485760,
            23 * (192512 - 32768 + 128) + 10485760,
        ),
    )
    for mode in (None, "selective"):
        rank.runtime.model[0].decoder.config.recompute_granularity = mode
        assert rank._checkpoint_memory_floor(groups) == (0, 0)
        assert g.plan_floor(rank, rank._plan_flat_forward(full_requests())) == (0, 0)


@pytest.mark.parametrize("gradient_first", [False, True])
def test_pre_gate_mixed_reference_and_exact_cost_mode_selection(layer, gradient_first):
    rank, _ = rank_with_moe(shared_layer(layer))
    grad = ForwardInput(
        input_tokens=torch.arange(67), hidden_states=True, no_grad=False
    )
    reference = ForwardInput(
        input_tokens=torch.arange(4096), hidden_states=True, no_grad=True
    )
    single = rank._plan_flat_forward([grad])
    requests = [grad, reference] if gradient_first else [reference, grad]
    mixed = rank._plan_flat_forward(requests)
    retained, workspace = g.plan_floor(rank, single)
    assert g.plan_floor(rank, mixed) == (retained, max(workspace, 4096 * 192512))
    assert (
        rank._memory_check(mixed).estimated_required_bytes
        == rank._plan_cost(mixed).required
    )
    assert rank._checkpoint_memory_floor(rank._plan_group_rows(mixed)) == (
        67 * 40 * 4096,
        4096 * 192512,
    )
    # A reference-only path must not read or validate the unused gradient cache.
    rank._moe_checkpoint_grad_bytes_per_token = None
    rank._moe_gradient_stages = None
    reference_plan = rank._plan_flat_forward([reference])
    assert g.plan_floor(rank, reference_plan) == (0, 0)
    assert rank._checkpoint_memory_floor(rank._plan_group_rows(reference_plan)) == (
        0,
        4096 * (192512 + 4 * 2048 * 2),
    )
    assert (
        rank._memory_check(reference_plan).estimated_required_bytes
        == rank._plan_cost(reference_plan).required
    )


@pytest.mark.parametrize("bad", [None, True, -1, 1.5, 192511])
def test_invalid_pre_gate_cache_stays_inside_planning_status(layer, bad):
    rank, _ = rank_with_moe(shared_layer(layer))
    requests = full_requests()
    plan = rank._plan_flat_forward(requests)
    rank._moe_checkpoint_grad_bytes_per_token = bad
    statuses = []
    rank._all_ranks_true = lambda value: (statuses.append(value), value)[1]
    rank._memory_check_required = lambda *a, **kw: pytest.fail(
        "memory reduction entered"
    )
    for call in (
        lambda: rank._memory_check(
            plan, sync_planning_errors=True, sync_across_dp=True
        ),
        lambda: rank._estimate_flat_forward(requests, sync_planning_errors=True),
    ):
        with pytest.raises(
            ValueError, match="Invalid constructor checkpoint MoE coefficient"
        ):
            call()
        assert statuses == [False]
        statuses.clear()
    with pytest.raises(
        ValueError, match="Invalid constructor checkpoint MoE coefficient"
    ):
        rank._plan_cost(plan)
