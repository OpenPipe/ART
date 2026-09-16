"""Conditional CP1 save accounting: original geometry and real CPU owner metadata."""

from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_moe_memory import _enclosing_moe
from test_trainer_rank_moe_memory import layer as layer
import torch

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import ForwardInput, TrainerRank
from art.trainer_rank import _gdn_memory as g
from art.trainer_rank._impl import Unset, _MemoryProfile


def module(cls):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    return obj


def rank_with_moe(moe_layer):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.transformer_block import TransformerBlock
    from transformer_engine.pytorch import RMSNorm

    from art.megatron.gdn.operator import _prefix_tree_forward
    from art.megatron.lora import LoRA, SelfAttentionLinearProjLoRA

    decoder = module(TransformerBlock)
    decoder.config = SimpleNamespace(
        hidden_size=2048,
        num_layers=40,
        padded_vocab_size=32,
        params_dtype=torch.bfloat16,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        sequence_parallel=False,
        fp32_residual_connection=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
    )
    decoder.layers = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1).bfloat16() for _ in range(40)]
    )
    decoder.num_layers_per_pipeline_rank = 40
    layer = torch.nn.Module()
    layer.mlp = moe_layer
    gd = module(GatedDeltaNet)
    gd.num_key_heads = 16
    gd.num_value_heads = 32
    gd.key_head_dim = 128
    gd.value_head_dim = 128
    gd.conv_kernel_dim = 4
    gd.use_qk_l2norm = True
    gd.tp_size = gd.sp_size = 1
    gd.forward = MethodType(_prefix_tree_forward, gd)
    gd.conv1d = torch.nn.Conv1d(
        8192, 8192, 4, groups=8192, bias=False, dtype=torch.bfloat16
    )
    gd.out_norm = module(RMSNorm)
    gd.out_norm.weight = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
    gd.out_proj = module(SelfAttentionLinearProjLoRA)
    gd.out_proj.lora = module(LoRA)
    gd.out_proj.lora.A_T = torch.nn.Parameter(
        torch.empty(4096, 1, dtype=torch.bfloat16)
    )
    gd.out_proj.lora.B_T = torch.nn.Parameter(
        torch.empty(1, 2048, dtype=torch.bfloat16)
    )
    layer.self_attention = gd
    decoder.layers[38] = layer
    model: Any = torch.nn.Module()
    model.config = decoder.config
    model.decoder = decoder
    model._preprocess = lambda: None
    r: Any = TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[model],
                optimizer=None,
                provider=SimpleNamespace(hidden_size=2048, num_layers=40),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )
    r._dp_rank_and_size = lambda: (0, 1)  # Uninitialized MCore has no CPU DP group.
    return r, gd


@pytest.fixture
def pending_rank(layer):
    return rank_with_moe(_enclosing_moe(layer))[0]


def full_requests(no_grad=False):
    return [
        ForwardInput(
            input_tokens=torch.arange(6330) + i * 10000,
            target_tokens=torch.arange(6330),
            no_grad=no_grad,
        )
        for i in range(8)
    ]


def test_actual_constructor_cache_and_full_plan(pending_rank):
    rank = pending_rank
    assert rank._moe_output_bytes_per_token == 188416
    shapes = g.model_shapes(rank)
    assert shapes is not None and shapes[1][0].moe_bytes_per_row == 188416
    requests = full_requests()
    plan = rank._plan_flat_forward(requests)
    assert rank._estimate_flat_forward(requests) is None
    assert (
        plan.packed_tokens == plan.logical_tokens == 50640 and plan.request_count == 8
    )
    assert g.plan_floor(rank, plan) == (8296857600, 9541386240 + 3157761952)
    assert (
        rank._memory_check(plan).estimated_required_bytes
        == rank._plan_cost(plan).required
        == 23095829187
    )
    selected = rank._select_next_micro_batch(requests, 0)
    assert (
        selected.check.estimated_required_bytes
        == rank._memory_check(selected.plan).estimated_required_bytes
    )
    lower = rank._split_chunk_lower_cost(
        requests, tuple(r.input_tokens for r in requests), checkpoint=Unset
    )
    assert lower.required <= rank._memory_check(plan).estimated_required_bytes


def test_pending_no_grad_empty_mixed_and_learned_max(pending_rank):
    rank = pending_rank
    grad = ForwardInput(
        input_tokens=torch.arange(67), hidden_states=True, no_grad=False
    )
    reference = replace(grad, input_tokens=torch.arange(4096), no_grad=True)
    plan = rank._plan_flat_forward([grad])
    retained, workspace = g.plan_floor(rank, plan)
    assert g.plan_floor(rank, rank._plan_flat_forward([])) == (0, 0)
    assert g.plan_floor(rank, rank._plan_flat_forward([reference])) == (0, 0)
    mixed = rank._plan_flat_forward([grad, reference])
    mr, mw = g.plan_floor(rank, mixed)
    assert mr == retained and mw == max(workspace, 4096 * 188416)
    assert rank._estimate_flat_forward([reference]) is not None
    rank._memory_profiles[plan.signature] = _MemoryProfile(
        bytes_per_token=1,
        packed_tokens=plan.packed_tokens,
        logical_per_packed=1,
        retained_compute_bytes_per_token=1,
    )
    cost = rank._plan_cost(plan)
    assert cost.retained == int((plan.output_bytes + retained) * 1.1)
    rank._memory_profiles[plan.signature] = replace(
        rank._memory_profiles[plan.signature], bytes_per_token=10**9
    )
    assert rank._plan_cost(plan).required == int(
        (plan.output_bytes + plan.packed_tokens * 10**9) * 1.1
    )


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, None])
def test_invalid_cached_coefficient_stops_before_memory_reduction(pending_rank, bad):
    rank = pending_rank
    plan = rank._plan_flat_forward(full_requests())
    rank._moe_output_bytes_per_token = bad
    statuses = []
    rank._all_ranks_true = lambda v: (statuses.append(v), v)[1]
    rank._memory_check_required = lambda *a, **kw: pytest.fail(
        "memory reduction entered"
    )
    with pytest.raises(ValueError, match="Invalid constructor MoE coefficient"):
        rank._memory_check(plan, sync_planning_errors=True, sync_across_dp=True)
    assert statuses == [False]
    statuses.clear()
    with pytest.raises(ValueError, match="Invalid constructor MoE coefficient"):
        rank._estimate_flat_forward(full_requests(), sync_planning_errors=True)
    assert statuses == [False]


@pytest.mark.parametrize(
    "field,value",
    [
        ("recompute_granularity", None),
        ("recompute_num_layers", 2),
        ("sequence_parallel", True),
        ("params_dtype", torch.float32),
    ],
)
def test_unsupported_pending_modes_are_not_qualified(pending_rank, field, value):
    setattr(pending_rank.runtime.model[0].decoder.config, field, value)
    assert g.model_shapes(pending_rank) is None


@pytest.mark.parametrize("root_length", [1, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("depth", [0, 1, 2, 8])
def test_cp1_bucket_geometry_matches_original_builder(root_length, depth):
    from art.megatron.gdn import gdn_prefix_tree as original

    prefix = list(range(root_length))
    pack = prefix_tree_pack(
        tuple(
            torch.tensor(x)
            for x in [
                prefix + [10000, 10001, 10002],
                prefix + [10000, 10001, 10003],
                prefix + [20000, 20001],
                [30000, 30001],
            ]
        ),
        max_depth=depth,
    )
    spec = original.parse_gdn_prefix_tree_segments(pack.group_ids, pack.parent_ids)
    has_children = tuple(
        i in spec.tree_parent_indices for i in range(len(spec.tree_segments))
    )
    actual = original._build_chunk_aligned_cp1_tree_buckets(
        spec, has_children, device="cpu", planner_config=original.GdnPlannerConfig()
    )
    actual_rows = tuple(
        (
            tuple(
                zip(
                    cast(torch.Tensor, b.family_indices_cpu).tolist(),
                    cast(torch.Tensor, b.parent_indices_cpu).tolist(),
                    b.lengths_cpu.tolist(),
                )
            ),
            b.needs_final_state,
        )
        for level in actual
        for b in level
    )
    assert actual_rows == tuple(
        (b.columns, b.final) for b in g.cp1_buckets(pack.segments)
    )


@pytest.mark.parametrize(
    "change", [{"parent_id": 999}, {"packed_start": 1}, {"end": 0}, {"group_id": True}]
)
def test_invalid_cp1_geometry_refused(change):
    pack = prefix_tree_pack((torch.arange(65),), max_depth=0)
    with pytest.raises(ValueError):
        g.cp1_buckets((replace(pack.segments[0], **change),))


def test_pending_counts_bucket_backing_and_output_rank_separately():
    shape = g.Shape(16, 32, 128, 128, 4, 1, 188416)
    pack = prefix_tree_pack(tuple(torch.arange(6330) for _ in range(8)), max_depth=0)
    buckets = g.cp1_buckets(pack.segments)
    assert shape.pending(50640, buckets) == 3157267456 + 8 * 8192 * 3 * 2 + 50640 * 2
    assert replace(shape, output_lora_rank=0).pending(50640, buckets) == shape.pending(
        50640, buckets
    ) - 50640 * (8192 + 2)
