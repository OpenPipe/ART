"""CPU Dynamo/AOTAutograd coverage without importing CUDA-only Megatron/TE.

Load complete production classes unchanged; only TE, parallel state and grouped
CUDA GEMMs are replaced. Real ContextVars retain production graph boundaries.
"""

import ast
from collections.abc import Callable, Sequence
import contextvars
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

SOURCE = Path(__file__).resolve().parents[2] / "src/art/megatron/lora.py"


def grouped_lora(x, a, b, counts, *, scale=1.0):
    # CPU substitute for QuACK only; exercise the real expert forward dispatch.
    ids = torch.repeat_interleave(
        torch.arange(counts.numel()), counts, output_size=x.shape[0]
    )
    return (x.unsqueeze(1).bmm(a[ids]).bmm(b[ids])).squeeze(1) * scale


@pytest.fixture
def lm(monkeypatch):
    names = {
        "LoraLayout",
        "LoRAParallelSpec",
        "LoRASlotRef",
        "LoRASlot",
        "LoRA",
        "lora_factor_view",
        "_distributed_initialized",
        "_get_shard_world_size",
        "_get_shard_rank",
        "_normalize_axis",
        "_set_lora_parallel_metadata",
        "_set_lora_shard_strategy_metadata",
        "_copy_lora_param_metadata",
        "_parallel_lora",
        "_parallel_lora_pair",
        "_linear_disables_tensor_parallel_comm",
        "_column_parallel_lora_input",
        "_targets_include",
        "SelfAttentionLinearProjLoRA",
        "SelfAttentionLinearQKVLoRA",
        "GatedDeltaNetInProjLoRA",
        "ComponentwiseColumnParallelLinearLoRA",
        "SharedExpertsLinearFC1LoRA",
    }
    tree = ast.parse(SOURCE.read_text())
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    module = ModuleType("_art_lora_compile_test")
    # Dataclasses and Dynamo resolve types through their defining module.
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        torch=torch,
        math=math,
        asdict=asdict,
        dataclass=dataclass,
        replace=replace,
        cast=cast,
        Any=Any,
        Callable=Callable,
        Literal=Literal,
        Sequence=Sequence,
        TP_DEFAULT_GRAD_SYNC_DOMAIN="tp_default",
        EXPERT_TP_GRAD_SYNC_DOMAIN="expert_tp",
        GRAD_SYNC_OP_NONE="none",
        GRAD_SYNC_OP_SUM="sum",
        GRAD_SYNC_OP_AVG="avg",
        LORA_ALPHA=8.0,
        ps=SimpleNamespace(
            get_expert_model_parallel_rank=lambda: 0,
            get_expert_data_parallel_rank=lambda: 0,
            get_data_parallel_rank=lambda **_: 0,
            get_expert_tensor_and_model_parallel_group=lambda **_: None,
            get_tensor_model_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
        ),
        _CURRENT_LORA_SLOT=contextvars.ContextVar("slot", default=None),
        _CURRENT_LORA_SEGMENTS=contextvars.ContextVar("segments", default=None),
        _CURRENT_EXPERT_ADAPTER_PLAN=contextvars.ContextVar("experts", default=None),
        quack_grouped_lora=grouped_lora,
        TELayerNormColumnParallelLinear=NormLinear,
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        *nodes,
                    ],
                    type_ignores=[],
                )
            ),
            str(SOURCE),
            "exec",
        ),
        module.__dict__,
    )
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield module
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


class Linear(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_features, self.out_features = 4, width
        self.weight = torch.nn.Parameter(torch.randn(width, 4, dtype=torch.float64))
        self.tp_size = 1

    def forward(self, x):
        return x @ self.weight.T, None


class NormLinear(Linear):
    def forward(self, x):
        return (x @ self.weight.T, x), None


def make_lora(lm, index, kind):
    expert = kind != "dense"
    prefix = f"base_model.model.model.layers.{index}.mlp"
    prefix += ".experts.{expert}" if expert else ""
    prefix += ".down_proj" if kind == "shared_b" else ".up_proj"
    layout = lm.LoraLayout(
        expert="shared_outer" if kind.startswith("shared") else "per_expert",
        shared_outer_kernel=False,
    )
    lora = lm.LoRA(
        prefix,
        4,
        4,
        2,
        1.5,
        torch.float64,
        torch.device("cpu"),
        num_local_experts=3 if expert else 1,
        lora_layout=layout,
    )
    if expert:
        # Export identities differ as well as prefixes; neither belongs in guards.
        ids = tuple(range(index * 3, index * 3 + 3))
        lora.bind_expert_layout(ids, ids)
    return lora


def make_wrapper(lm, index, family):
    prefix = f"base_model.model.model.layers.{index}"
    args = dict(adapter_model_prefix=prefix, rank=2, alpha=1.5)
    if family == "qkv":
        return lm.SelfAttentionLinearQKVLoRA(
            **args,
            linear_qkv=NormLinear(12),
            provider=SimpleNamespace(
                kv_channels=2, num_query_groups=2, num_attention_heads=2
            ),
            target_modules={"q_proj", "k_proj", "v_proj"},
        )
    if family.startswith("gdn"):
        return lm.GatedDeltaNetInProjLoRA(
            **args,
            in_proj=NormLinear(20),
            gated_delta_net=SimpleNamespace(num_value_heads=2, qk_dim=4, v_dim=4),
            lora_layout=lm.LoraLayout(
                qwen_parameterization="tinker_v1"
                if family == "gdn_split"
                else "tied_v0"
            ),
        )
    if family.startswith("component"):
        return lm.ComponentwiseColumnParallelLinearLoRA(
            **args,
            in_proj=(NormLinear if family.endswith("norm") else Linear)(8),
            component_sizes=(4, 4),
        )
    if family == "row":
        return lm.SelfAttentionLinearProjLoRA(
            **args,
            linear_proj=Linear(4),
            provider=SimpleNamespace(tensor_model_parallel_size=1),
        )
    return lm.SharedExpertsLinearFC1LoRA(
        **args, linear_fc1=NormLinear(8), non_gated=family == "shared_non_gated"
    )


def assert_vjp_equal(compiled, module, x, *args):
    eager = deepcopy(module)
    eager_x = x.detach().clone().requires_grad_()
    actual = compiled(module, x, *args)
    expected = eager(eager_x, *args)
    if isinstance(actual, tuple):
        assert actual[1] is expected[1] is None
        actual, expected = actual[0], expected[0]
    assert torch.equal(actual, expected)
    cotangent = torch.arange(actual.numel(), dtype=x.dtype).reshape(actual.shape) / 16
    params = [p for p in module.parameters() if p.requires_grad]
    eager_params = [p for p in eager.parameters() if p.requires_grad]
    actual_grads = torch.autograd.grad(actual, (x, *params), cotangent)
    expected_grads = torch.autograd.grad(expected, (eager_x, *eager_params), cotangent)
    assert all(
        torch.equal(a, b) for a, b in zip(actual_grads, expected_grads, strict=True)
    )


@pytest.mark.parametrize("kind", ["dense", "expert", "shared_a", "shared_b"])
@pytest.mark.parametrize("slot", [False, True])
def test_forward_graphs_ignore_prefix_and_expert_ids(lm, kind, slot, record_property):
    torch.manual_seed(19)
    backend = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(
        lambda module, x, counts: module(x, counts), backend=backend
    )
    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    counts = None if kind == "dense" else torch.tensor([2, 1, 3])
    ref = lm.LoRASlotRef("checkpoint", "policy") if slot else None
    token = lm._CURRENT_LORA_SLOT.set(ref)
    try:
        for index in range(32):
            module = make_lora(lm, index, kind)
            with torch.no_grad():
                module.B_T.normal_()  # Nonzero factors exercise every gradient edge.
            if ref is not None:
                assert module.load_lora_slot(
                    ref, module.sharded_lora_state_dict(), alpha=1.5, requires_grad=True
                )
                # Only the selected slot participates in this forward's VJP.
                module.A_T.requires_grad_(False)
                module.B_T.requires_grad_(False)
            if index == 0:
                assert_vjp_equal(compiled, module, x, counts)
                graphs = backend.frame_count
                record_property("compiled_graphs", graphs)
                assert 1 <= graphs <= 4
            else:
                with torch._dynamo.config.patch(error_on_recompile=True):
                    assert_vjp_equal(compiled, module, x, counts)
                assert backend.frame_count == graphs
    finally:
        lm._CURRENT_LORA_SLOT.reset(token)


@pytest.mark.parametrize(
    "family",
    [
        "qkv",
        "gdn",
        "gdn_split",
        "component",
        "component_norm",
        "shared",
        "shared_non_gated",
        "row",
    ],
)
def test_wrapper_graphs_ignore_layer_prefix(lm, family, record_property):
    torch.manual_seed(23)
    backend = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(lambda module, x: module(x), backend=backend)
    x = torch.randn(3, 1, 4, dtype=torch.float64, requires_grad=True)
    for index in range(24):
        module = make_wrapper(lm, index, family)
        with torch.no_grad():
            for child in module.modules():
                if isinstance(child, lm.LoRA):
                    child.B_T.normal_()
        if index == 0:
            assert_vjp_equal(compiled, module, x)
            graphs = backend.frame_count
            record_property("compiled_graphs", graphs)
            assert 1 <= graphs <= 12
        else:
            with torch._dynamo.config.patch(error_on_recompile=True):
                assert_vjp_equal(compiled, module, x)
            assert backend.frame_count == graphs


@pytest.mark.parametrize("kind", ["dense", "expert", "shared_a", "shared_b"])
def test_state_dict_and_export_names(lm, kind):
    module = make_lora(lm, 7, kind)
    assert set(module.state_dict()) == {"A_T", "B_T"}
    exports = module.sharded_lora_state_dict()
    experts = module.expert_ids if module.is_expert else (None,)
    expected = {
        f"{module.adapter_model_prefix.format(expert=expert)}.{suffix}.weight": lm.lora_factor_view(
            param, local if module.is_expert else None
        ).T
        for local, expert in enumerate(experts)
        for suffix, param in (("lora_A", module.A_T), ("lora_B", module.B_T))
    }
    assert exports.keys() == expected.keys()
    assert all(torch.equal(exports[key], value) for key, value in expected.items())
    restored = make_lora(lm, 7, kind)
    restored.load_state_dict(module.state_dict(), strict=True)
    restored.load_lora(exports)
    assert all(
        torch.equal(module.state_dict()[key], value)
        for key, value in restored.state_dict().items()
    )
    ref = lm.LoRASlotRef("checkpoint", "policy")
    assert restored.load_lora_slot(ref, exports, alpha=1.5, requires_grad=True)
    assert set(restored.state_dict()) == {
        "A_T",
        "B_T",
        "_slot_modules.slot_0.A_T",
        "_slot_modules.slot_0.B_T",
    }
    slot_exports = restored.sharded_lora_state_dict(ref)
    assert slot_exports.keys() == exports.keys()
    assert all(torch.equal(slot_exports[key], value) for key, value in exports.items())
