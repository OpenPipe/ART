"""CPU contracts for one known MoE component, not a whole-model memory bound."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank
from art.trainer_rank._impl import (
    _PACKED_PRICED_LOGICAL_ROW_BYTES,
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
    value.experts.linear_fc2.lora.A_T = torch.nn.Parameter(
        torch.empty(256, 512, 8, dtype=torch.bfloat16)
    )
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
                provider=SimpleNamespace(
                    hidden_size=2048,
                    num_layers=40,
                    recompute_granularity="full",
                    recompute_method="uniform",
                    recompute_num_layers=1,
                ),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )


def _signature():
    return _MemorySignature((1, 1, 1, 1), (1, None), 1, (), True, (True,))


def test_cp_memory_charges_local_and_gathered_outputs():
    rank = _rank()
    signature = replace(_signature(), topology=(1, 1, 4, 1))
    output_bytes = 1 << 30
    estimate = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=16, output_bytes=output_bytes, signature=signature
    )
    compute = 16 * 2048 * 2 * 14
    assert estimate == int((compute + 2 * output_bytes) * 1.1)
    # A warm profile includes gather workspace already; do not add it twice.
    rank._update_memory_profile(
        SimpleNamespace(
            signature=signature,
            packed_tokens=16,
            output_bytes=output_bytes,
            active_logical_tokens=16,
        ),
        compute + 2 * output_bytes,
        retained_bytes=None,
    )
    assert (
        rank._estimate_required_memory_bytes_from_values(
            packed_tokens=16, output_bytes=output_bytes, signature=signature
        )
        == estimate
    )


def test_supported_constructor_and_original_shape(layer):
    rank = _rank(layer)
    assert rank._moe_output_bytes_per_token == (512 + 3 * 2048) * 8 * 2
    estimate = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=106432,
        output_bytes=0,
        signature=_signature(),
    )
    # Eager FC2 arguments and outputs; compiler reuse can reduce this component.
    inputs = torch.empty(106432 * 8, 512, dtype=torch.bfloat16, device="meta")
    base = torch.empty(106432 * 8, 2048, dtype=torch.bfloat16, device="meta")
    adapter = torch.empty_like(base)
    combined = base + adapter
    tensors = (inputs, base, adapter, combined)
    assert len({x.untyped_storage()._cdata for x in tensors}) == len(tensors)
    pair = base.untyped_storage().nbytes() + adapter.untyped_storage().nbytes()
    assert pair == 6_975_127_552
    assert estimate == int(sum(x.untyped_storage().nbytes() for x in tensors) * 1.1)
    assert rank._memory_check_required(6_800_000_000).fits  # CPU default budget
    rank._available_memory_bytes = lambda: 6_800_000_000
    assert not rank._memory_check_required(estimate).fits


def test_cold_fixed_pressure_budget_and_partial_envelope_limit(layer):
    rank = _rank(layer)
    signature = replace(_signature(), grad_enabled=False, grad_modes=(False,))
    required = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=45981, output_bytes=707325952, signature=signature
    )
    assert required == 6_164_530_380
    rank._available_memory_bytes = lambda: 4_652_784_333
    assert rank._memory_check_required(4_092_810_444).fits
    assert not rank._memory_check_required(required).fits
    # This source component remains below the retained whole-forward increment.
    # No general physical bound follows from rejecting this recorded budget.
    assert required < 11_445_665_280


@pytest.mark.parametrize("mode", ["missing", "dtype", "rank", "experts"])
def test_unknown_fc2_input_keeps_previous_pair(layer, mode):
    lora = layer.experts.linear_fc2.lora
    if mode == "missing":
        lora.A_T = None
    else:
        shape = (
            (128, 512, 8)
            if mode == "experts"
            else (256, 512, 4)
            if mode == "rank"
            else (256, 512, 8)
        )
        dtype = torch.float32 if mode == "dtype" else torch.bfloat16
        lora.A_T = torch.nn.Parameter(torch.empty(shape, dtype=dtype))
    assert _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1)) == 65536


@pytest.mark.parametrize("field", ["tp", "cp", "ep", "etp"])
def test_sharded_path_unchanged(layer, field):
    # CP shards rows, not the per-token working set; TP/EP/ETP are unmodeled.
    single = _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1))
    sharded = replace(ParallelShape(tp=1, cp=1), **{field: 2})
    assert single > 0
    assert _moe_output_bytes_per_token([layer], sharded) == (
        single if field == "cp" else 0
    )


def _hybridep(layer, ep: int, manager: str = "hybridep"):
    from megatron.core.transformer.moe import token_dispatcher

    dispatcher = object.__new__(token_dispatcher.MoEFlexTokenDispatcher)
    dispatcher.config = layer.config
    dispatcher.ep_size, dispatcher.tp_size = ep, 1
    managers = {
        "hybridep": token_dispatcher._HybridEPManager,
        "deepep": token_dispatcher._DeepepManager,
    }
    dispatcher._comm_manager = object.__new__(managers[manager])
    layer.token_dispatcher = dispatcher
    return layer


@pytest.mark.parametrize("ep", [2, 4])
def test_hybridep_prices_routed_rows_with_imbalance_allowance(layer, ep):
    # HybridEP gives each rank the pairs routed to its local experts: balanced
    # routing matches EP1's top-k rows per local token, with a 1.5x allowance.
    single = _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1))
    sharded = ParallelShape(tp=1, cp=ep, ep=ep)
    expert = _moe_output_bytes_per_token([_hybridep(layer, ep)], sharded)
    # Top-k 8 routed rows per local token become 12; no shared experts here.
    assert single > 0 and expert == single // 8 * 12


@pytest.mark.parametrize(
    "manager,ep,shape_ep", [("deepep", 2, 2), ("hybridep", 2, 4), ("hybridep", 1, 1)]
)
def test_other_flex_dispatchers_stay_unmodeled(layer, manager, ep, shape_ep):
    dispatcher = _hybridep(layer, ep, manager)
    shape = ParallelShape(tp=1, cp=1, ep=shape_ep)
    assert _moe_output_bytes_per_token([dispatcher], shape) == 0


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


def test_profiles_outputs_and_empty_plan_preserve_empirical_floor():
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
    for tokens in (100, 800, 801):
        assert estimate(
            packed_tokens=tokens, output_bytes=123, signature=signature
        ) == int((tokens * 100000 + 123) * 1.1)
    assert not rank._all_ranks_have_memory_profile(
        packed_tokens=801, signature=signature
    )
    assert estimate(
        packed_tokens=100, logical_tokens=200, output_bytes=123, signature=signature
    ) == int((100 * 100000 + _PACKED_PRICED_LOGICAL_ROW_BYTES * 200 + 123) * 1.1)


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


def _enclosing_moe(layer):
    from art.megatron.lora import MLPExpertsLinearFC1LoRA

    fc1 = MLPExpertsLinearFC1LoRA.__new__(MLPExpertsLinearFC1LoRA)
    torch.nn.Module.__init__(fc1)
    fc1.fused_gate_up = True
    fc1.non_gated = False
    fc1.out_features = 1024
    layer.experts.linear_fc1 = fc1
    layer.experts.offload_expert_fc1 = False
    layer.experts.offload_moe_act = False
    layer.experts.activation_recompute = False
    layer.token_dispatcher.ep_size = 1
    layer.token_dispatcher.tp_size = 1
    layer.token_dispatcher.num_local_experts = 256
    layer.config.moe_permute_fusion = True
    return layer


def test_compiled_moe_retained_inputs_cold_floor(layer):
    # Three complete intervals in a retained native trace have these seven
    # distinct storages live together. Their sum is not the whole-model peak.
    rank = _rank(_enclosing_moe(layer))
    rows, hidden, intermediate = 45_981 * 8, 2048, 512
    widths = (hidden, hidden, 2 * intermediate, intermediate, hidden, hidden, hidden)
    storages = [
        torch.empty(rows, width, dtype=torch.bfloat16, device="meta")
        for width in widths
    ]
    assert len({tensor.untyped_storage()._cdata for tensor in storages}) == 7
    observed_component_bytes = sum(
        tensor.untyped_storage().nbytes() for tensor in storages
    )
    assert observed_component_bytes == 8_663_556_096
    assert rank._moe_output_bytes_per_token * 45_981 == observed_component_bytes
    for no_grad in (False, True):
        signature = replace(
            _signature(), grad_enabled=not no_grad, grad_modes=(not no_grad,)
        )
        assert rank._estimate_required_memory_bytes_from_values(
            packed_tokens=45_981, output_bytes=0, signature=signature
        ) == int(observed_component_bytes * 1.1)


@pytest.mark.parametrize(
    "site,attribute,value",
    [
        ("dispatcher", "ep_size", 2),
        ("dispatcher", "tp_size", 2),
        ("dispatcher", "num_local_experts", 1),
        ("config", "moe_permute_fusion", False),
        ("experts", "offload_expert_fc1", True),
        ("experts", "offload_moe_act", True),
        ("experts", "activation_recompute", True),
        ("fc1", "fused_gate_up", False),
        ("fc1", "non_gated", True),
        ("fc1", "out_features", 2048),
        ("fc1", "forward", lambda *args: None),
    ],
)
def test_unknown_enclosing_lifetimes_keep_previous_fc2_floor(
    layer, site, attribute, value
):
    _enclosing_moe(layer)
    sites = {
        "dispatcher": layer.token_dispatcher,
        "config": layer.config,
        "experts": layer.experts,
        "fc1": layer.experts.linear_fc1,
    }
    setattr(sites[site], attribute, value)
    assert _moe_output_bytes_per_token([layer], ParallelShape(tp=1, cp=1)) == 106_496


def test_hybridep_buffer_growth_is_charged_before_forward(monkeypatch):
    from megatron.core.transformer.moe import fused_a2a

    from art.trainer_rank._impl import _hybridep_buffer_bytes

    # Two ranks' tokens to one rank: BF16 H, FP32 probs and a routing-map byte
    # per expert (256), and FP32 scaling factors per H/128.
    assert _hybridep_buffer_bytes(1000, 2, 2048, 256) == 2000 * (4096 + 1280 + 64)
    assert _hybridep_buffer_bytes(0, 2, 2048, 256) == 0
    rank = _rank()
    rank.runtime.provider.expert_model_parallel_size = 2
    rank.runtime.provider.num_moe_experts = 256
    plan = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(64), target_tokens=torch.arange(64))]
    )
    monkeypatch.setattr(rank, "_topology", lambda: SimpleNamespace(tp=1, cp=2))
    monkeypatch.setattr(rank, "_plan_group_rows", lambda plan: ((600, True),))
    monkeypatch.setattr(
        "art.megatron.train._hybridep_token_capacity", lambda sequence, cp: 1000
    )

    def held(capacity):
        config = SimpleNamespace(max_num_of_tokens_per_rank=capacity)
        return SimpleNamespace(configurer=SimpleNamespace(buffer_config=config))

    full = _hybridep_buffer_bytes(1000, 2, 2048, 256)
    for current, growth in (
        (None, full),
        (held(400), full - _hybridep_buffer_bytes(400, 2, 2048, 256)),
        (held(1000), 0),
    ):
        monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", current)
        assert rank._plan_hybridep_growth_bytes(plan) == growth
    # The growth enters required memory, not forward retention.
    rank._update_memory_profile(plan, 10**9, retained_bytes=10**8)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    grown = rank._plan_cost(plan)
    monkeypatch.setattr(rank, "_plan_hybridep_growth_bytes", lambda plan: 0)
    base = rank._plan_cost(plan)
    assert grown.required - base.required == pytest.approx(full * 1.1, abs=2)
    assert grown.retained == base.retained < base.required
    rank.runtime.provider.expert_model_parallel_size = 1
    monkeypatch.undo()
    assert rank._plan_hybridep_growth_bytes(plan) == 0
