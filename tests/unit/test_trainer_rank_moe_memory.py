"""CPU contracts for one known MoE component, not a whole-model memory bound."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank
from art.trainer_rank._impl import (
    _MemoryProfile,
    _MemorySignature,
    _moe_output_bytes_per_token,
)
from art.trainer_rank._planner_cost import ParallelShape


@pytest.fixture
def layer() -> Any:
    lora_module = pytest.importorskip("art.megatron.lora")
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    def module(cls):
        value = cls.__new__(cls)
        torch.nn.Module.__init__(value)
        return value

    # Actual supported types, without CUDA/process-group initialization. No
    # model forward is faked here: these tests inspect constructor metadata only.
    config = SimpleNamespace(
        hidden_size=2048,
        num_layers=40,
        num_moe_experts=256,
        moe_router_topk=8,
        moe_ffn_hidden_size=512,
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_router_padding_for_quantization=False,
        fp8=None,
        fp4=None,
        moe_latent_size=None,
        cuda_graph_impl="none",
    )
    value = module(MoELayer)
    value.config = config
    value.router = module(TopKRouter)
    value.router.topk = 8
    value.token_dispatcher = object.__new__(MoEAlltoAllTokenDispatcher)
    value.token_dispatcher.config = config
    value.experts = module(lora_module.TEGroupedMLP)
    value.experts.linear_fc2 = module(lora_module.MLPExpertsLinearFC2LoRA)
    value.experts.linear_fc2.out_features = 2048
    value.experts.linear_fc2.linear_fc2 = module(lora_module.TERowParallelGroupedLinear)
    value.experts.linear_fc2.lora = module(lora_module.LoRA)
    value.experts.linear_fc2.lora.B_T = torch.nn.Parameter(
        torch.empty(256, 8, 2048, dtype=torch.bfloat16)
    )
    return value


def _rank(layer=None):
    model = layer if layer is not None else torch.nn.Linear(1, 1).bfloat16()
    return TrainerRank(
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


def _signature():
    return _MemorySignature((1, 1, 1, 1), (1, None), 1, (), True, (True,))


def test_supported_constructor_and_original_shape(layer):
    rank = _rank(layer)
    assert rank._moe_output_bytes_per_token == 2 * 8 * 2048 * 2
    estimate = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=106432,
        output_bytes=0,
        signature=_signature(),
    )
    # This independent storage arithmetic exceeds the former 6,713,560,268.
    base = torch.empty(106432 * 8, 2048, dtype=torch.bfloat16, device="meta")
    adapter = torch.empty_like(base)
    assert base.untyped_storage()._cdata != adapter.untyped_storage()._cdata
    pair = base.untyped_storage().nbytes() + adapter.untyped_storage().nbytes()
    assert pair == 6_975_127_552
    assert estimate == int(pair * 1.1)
    assert rank._memory_check_required(6_800_000_000).fits  # CPU default budget
    rank._available_memory_bytes = lambda: 6_800_000_000
    assert not rank._memory_check_required(estimate).fits


@pytest.mark.parametrize("field", ["tp", "cp", "ep", "etp"])
def test_sharded_path_unchanged(layer, field):
    assert (
        _moe_output_bytes_per_token(
            [layer], replace(ParallelShape(tp=1, cp=1), **{field: 2})
        )
        == 0
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("moe_expert_capacity_factor", 1.0),
        ("moe_pad_expert_input_to_capacity", True),
        ("moe_router_padding_for_quantization", True),
        ("fp8", "hybrid"),
        ("fp4", "e2m1"),
        ("moe_latent_size", 1024),
        ("cuda_graph_impl", "local"),
    ],
)
def test_padded_or_alternate_config_unchanged(layer, field, value):
    setattr(layer.config, field, value)
    assert _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1)) == 0


@pytest.mark.parametrize(
    "site",
    [
        "",
        "experts",
        "experts.linear_fc2",
        "experts.linear_fc2.lora",
        "experts.linear_fc2.linear_fc2",
        "router",
        "token_dispatcher",
    ],
)
def test_custom_subclasses_unchanged(layer, site):
    value = layer
    for name in site.split(".") if site else ():
        value = getattr(value, name)
    value.__class__ = type("Custom", (type(value),), {})
    assert _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1)) == 0


@pytest.mark.parametrize(
    "site,method",
    [
        (
            "experts.linear_fc2",
            "forward",
        ),  # Includes an instance residual-fusion patch.
        ("experts.linear_fc2.lora", "forward"),
        ("token_dispatcher", "preprocess"),
        ("token_dispatcher", "dispatch_preprocess"),
        ("token_dispatcher", "dispatch_postprocess"),
        ("router", "routing"),
    ],
)
def test_instance_execution_overrides_unchanged(layer, site, method):
    value = layer
    for name in site.split("."):
        value = getattr(value, name)
    setattr(value, method, lambda *args: None)
    assert _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1)) == 0


def test_hooks_dtype_and_geometry_exclusions(layer):
    shape = ParallelShape(tp=1, cp=1)
    hook = layer.experts.linear_fc2.register_forward_pre_hook(lambda *args: None)
    assert _moe_output_bytes_per_token([layer], shape) == 0
    hook.remove()
    lora = layer.experts.linear_fc2.lora
    lora.B_T = torch.nn.Parameter(lora.B_T.float())
    assert _moe_output_bytes_per_token([layer], shape) == 0
    lora.B_T = torch.nn.Parameter(lora.B_T.bfloat16())
    layer.experts.linear_fc2.out_features = 1024
    assert _moe_output_bytes_per_token([layer], shape) == 0


def test_topk_not_expert_count_controls_envelope(layer):
    shape = ParallelShape(tp=1, cp=1)
    old = _moe_output_bytes_per_token([layer], shape)
    layer.config.num_moe_experts = 512
    assert _moe_output_bytes_per_token([layer], shape) == old
    layer.config.moe_router_topk = layer.router.topk = 4
    assert _moe_output_bytes_per_token([layer], shape) == old // 2


def test_profiles_outputs_trust_and_empty_plan_unchanged():
    rank = _rank()
    signature = _signature()
    assert rank._moe_output_bytes_per_token == 0
    rank._moe_output_bytes_per_token = 65536
    estimate = rank._estimate_required_memory_bytes_from_values
    assert estimate(packed_tokens=0, output_bytes=123, signature=signature) == 123
    assert estimate(packed_tokens=100, output_bytes=123, signature=signature) == int(
        (100 * 65536 + 123) * 1.1
    )
    rank._memory_profiles[signature] = _MemoryProfile(
        bytes_per_token=100000,
        packed_tokens=100,
        logical_per_packed=1,
    )
    for tokens in (100, 800):
        assert estimate(
            packed_tokens=tokens, output_bytes=123, signature=signature
        ) == int((tokens * 100000 + 123) * 1.1)
    assert estimate(packed_tokens=801, output_bytes=123, signature=signature) == int(
        (801 * 65536 + 123) * 1.1
    )
    assert estimate(
        packed_tokens=100, logical_tokens=200, output_bytes=123, signature=signature
    ) == int((100 * 200000 + 123) * 1.1)


def test_summed_group_envelope_and_retained_profile_unchanged():
    rank = _rank()
    rank._moe_output_bytes_per_token = 65536
    tokens = torch.arange(8)
    plan = rank._plan_flat_forward(
        [
            ForwardInput(input_tokens=tokens, target_tokens=tokens),
            ForwardInput(input_tokens=tokens, target_tokens=tokens, no_grad=True),
        ]
    )
    assert len(plan.groups) == 2
    assert plan.packed_tokens == 16
    assert plan.output_bytes == 16 * 4
    assert rank._plan_cost(plan).required == int((16 * 65536 + 16 * 4) * 1.1)
    plan = replace(plan, packed_tokens=200, logical_tokens=200, output_bytes=4000)
    required = rank._plan_cost(plan).required
    assert required == int((200 * 65536 + 4000) * 1.1)
    # Existing cold fallback retains the full estimate, not a newly derived rate.
    assert rank._plan_cost(plan).retained == required
    rank._memory_profiles[plan.signature] = _MemoryProfile(
        bytes_per_token=100000,
        packed_tokens=200,
        retained_compute_bytes_per_token=100,
    )
    cost = rank._plan_cost(plan)
    assert cost.retained == int((4000 + 200 * 100) * 1.1)
    assert cost.ephemeral == cost.required - cost.retained
    # Two sequential groups use a conservative sum envelope, not a simultaneous
    # output-pair lower bound. No per-group coefficients or profiles are changed.
    doubled = replace(plan, packed_tokens=400, logical_tokens=400)
    assert rank._plan_cost(doubled).required == int((400 * 100000 + 4000) * 1.1)


def test_fc2_base_remains_live_during_native_lora_output_allocation(layer, monkeypatch):
    from art.megatron import lora as lora_module
    from art.megatron.kernels import cute_grouped_lora_quack as kernel

    fc2 = layer.experts.linear_fc2
    rows = 16
    x = torch.empty(rows, 512, dtype=torch.bfloat16)
    counts = torch.zeros(256, dtype=torch.int64)
    counts[0] = rows  # Imbalance does not reduce the total routed-row allocation.
    a = torch.empty(256, 512, 1, dtype=x.dtype)
    b = torch.empty(256, 1, 2048, dtype=x.dtype)
    seen = {}

    def base(value, _counts):
        output = value.new_empty(rows, 2048)
        seen["base"] = weakref.ref(output)
        return output, None

    def gemm(value, weights, output, residual, bias, **kwargs):
        if output.shape[-1] == 2048:
            base_output = seen["base"]()
            assert base_output is not None
            assert residual is None
            assert (
                base_output.untyped_storage()._cdata != output.untyped_storage()._cdata
            )
            seen["pair_bytes"] = base_output.nbytes + output.nbytes
            assert value.shape == (rows, 8)  # Actual rank-1 padding path.

    def adapter(_lora, value, tokens_per_expert, _width):
        # Original Python allocation path; only CUDA validation/compute and
        # distributed base-layer initialization are bypassed in this CPU witness.
        return kernel._QuackGroupedLoraFn.forward(
            SimpleNamespace(save_for_backward=lambda *args: None),
            value,
            a,
            b,
            tokens_per_expert,
            1.0,
            None,
        )

    monkeypatch.setattr(fc2.linear_fc2, "forward", base)
    monkeypatch.setattr(lora_module, "_expert_grouped_lora_forward", adapter)
    monkeypatch.setattr(kernel, "_quack_gemm", gemm)
    output, bias = fc2(x, counts)
    assert seen["pair_bytes"] == 2 * rows * 2048 * 2
    assert output.shape == (rows, 2048) and bias is None
    # Residual/output views are intentionally not covered by the capability.
    monkeypatch.setattr(kernel, "_quack_gemm", lambda *args, **kwargs: None)
    result = kernel._varlen_quack_gemm(
        x,
        b,
        out_features=2048,
        expert_offsets=torch.zeros(257, dtype=torch.int32),
        tile_m=64,
        tile_n=128,
        residual=output,
    )
    assert result is output


def test_memory_check_preserves_collective_order(monkeypatch):
    import art.trainer_rank._impl as impl

    rank = _rank()
    rank._moe_output_bytes_per_token = 65536
    rank._available_memory_bytes = lambda: 6_800_000_000
    group = object()
    rank._forward_memory_group = lambda: group
    monkeypatch.setattr(impl.dist, "is_available", lambda: True)
    monkeypatch.setattr(impl.dist, "is_initialized", lambda: True)
    calls = []

    def reduce(value, op, group):
        calls.append((value.item(), op, group))

    monkeypatch.setattr(impl.dist, "all_reduce", reduce)
    estimate = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=106432,
        output_bytes=0,
        signature=_signature(),
    )
    assert calls == []
    assert not rank._memory_check_required(estimate).fits
    assert calls == [
        (float(estimate), impl.dist.ReduceOp.MAX, group),
        (6_800_000_000.0, impl.dist.ReduceOp.MIN, group),
    ]
