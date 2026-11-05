"""FP64 CPU VJPs of production wrappers with explicit TE/TP contracts.

AST loading avoids importing CUDA-only TE. Collectives model SUM adjoints;
TE's gathered returned-norm output drops its cotangent (TE 2.11/2.14.1).
These tests validate ART wiring, not native TE kernels or compiled autograd.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "src/art/megatron/lora.py"
FAMILIES = (
    "SelfAttentionLinearQKVLoRA",
    "GatedDeltaNetInProjLoRA",
    "ComponentwiseColumnParallelLinearLoRA",
    "SharedExpertsLinearFC1LoRA",
)


def load_wrappers(copy, gather):
    tree = ast.parse(SOURCE.read_text())
    names = {
        "_linear_disables_tensor_parallel_comm",
        "_compile_disabled_collective",
        "_column_parallel_lora_input",
    }
    aliases = {
        "_gather_lora_sequence_parallel_region",
        "_copy_lora_tensor_model_parallel_region",
    }
    nodes: list[ast.stmt] = []
    flags = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in aliases for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (*FAMILIES, "LoRA"):
            forward = deepcopy(
                next(
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "forward"
                )
            )
            forward.name = node.name
            nodes.append(forward)
            if node.name in FAMILIES:
                flags[node.name] = next(
                    n.value.value
                    for n in ast.walk(node)
                    if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Constant)
                    and any(
                        isinstance(t, ast.Attribute)
                        and t.attr == "return_layernorm_output_gathered"
                        for t in n.targets
                    )
                )
    ns = dict(
        torch=torch,
        cast=cast,
        _F=Any,
        _CURRENT_LORA_SEGMENTS=SimpleNamespace(get=lambda: None),
        copy_to_tensor_model_parallel_region=copy,
        gather_from_sequence_parallel_region=gather,
    )
    exec(
        compile(
            "from __future__ import annotations\n"
            + ast.unparse(ast.Module(body=nodes, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        ns,
    )
    return ns, flags


def summed_copy(x, peers):
    # Identity forward with a SUM backward to each rank's replicated input.
    return x + sum((p - p.detach() for p in peers if p is not x), torch.zeros_like(x))


def rms(x):
    return x * (x.square().mean(-1, keepdim=True) + 1e-5).rsqrt()


@pytest.mark.parametrize(
    "family,fused",
    [(family, True) for family in FAMILIES]
    + [(family, False) for family in FAMILIES[2:]],
)
@pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("owner", ["inner", "outer_mode", "outer_expert"])
@pytest.mark.parametrize("arm", ["dense", "adapter", "combined"])
def test_wrapper_input_vjp(family, fused, tp, sp, owner, arm):
    torch.manual_seed(19)
    rows = torch.randn(4, 1, 4, dtype=torch.float64)
    inputs = [
        p.clone().requires_grad_() for p in (rows.chunk(tp) if sp else [rows] * tp)
    ]
    outer = owner != "inner"
    feed = (
        [torch.cat(inputs) if sp else summed_copy(x, inputs) for x in inputs]
        if outer
        else inputs
    )
    norms = [rms(x) for x in feed] if fused else feed
    peers = norms if fused else feed
    group = SimpleNamespace(size=lambda: tp)
    calls = []

    def copy(x, *, group=None):
        calls.append(("copy", group))
        return summed_copy(x, peers)

    def gather(x, *, group=None, tensor_parallel_output_grad=True):
        assert tensor_parallel_output_grad
        calls.append(("gather", group))
        return torch.cat(peers)

    ns, flags = load_wrappers(copy, gather)
    width = 12 if family == FAMILIES[1] else 6
    widths = {
        FAMILIES[0]: (2, 2, 2),
        FAMILIES[1]: (6, 2),
        FAMILIES[2]: (6,),
        FAMILIES[3]: (3, 3),
    }[family]
    weights = [
        torch.randn(4, width, dtype=rows.dtype) * (arm != "adapter") for _ in inputs
    ]
    factors = [
        [
            (
                torch.randn(4, 2, dtype=rows.dtype).requires_grad_(),
                (
                    torch.randn(2, w, dtype=rows.dtype) * (arm != "dense")
                ).requires_grad_(),
            )
            for w in widths
        ]
        for _ in inputs
    ]
    cotangents = [torch.randn(4, 1, width, dtype=rows.dtype) for _ in inputs]
    outputs = []

    def adapter(a, b):
        obj = SimpleNamespace(is_expert=False, active_lora_tensors=lambda: (a, b, 0.75))
        return lambda x: ns["LoRA"](obj, x)

    for rank, x in enumerate(feed):
        # A plain TE column reduces only its own base input cotangent.
        # A fused TE layer does that before adding the local norm cotangent.
        base_input = norms[rank]
        if not outer and tp > 1:
            base_input = torch.cat(peers) if sp else summed_copy(base_input, peers)
        base = base_input @ weights[rank]
        returned = norms[rank]
        if fused and flags[family]:
            # Gathered norm gradients are ignored by TE 2.11, including TP1.
            returned = (torch.cat(norms) if sp and not outer else returned).detach()

        class Inner:
            def __init__(self):
                self.tp_size, self.tp_group, self.sequence_parallel = tp, group, sp
                self.parallel_mode = None if owner == "outer_mode" else "column"
                self.explicit_expert_comm = owner == "outer_expert"

            def __call__(self, _x):
                return ((base, returned) if fused else base), None

        inner = Inner()
        loras = [adapter(a, b) for a, b in factors[rank]]
        obj = SimpleNamespace(
            linear_qkv=inner,
            in_proj=inner,
            linear_fc1=inner,
            non_gated=False,
            split_qkv=False,
            num_value_heads_per_partition=2,
            num_query_groups_per_partition=1,
            num_attention_heads_per_group=1,
            attention_output_gate=False,
            hidden_size_per_attention_head=2,
            replicated_qkv=False,
            q_and_gate_out_features_per_rank=2,
            kv_out_features_per_rank=2,
        )
        obj._qkv_lora_output = lambda lora, x, width: lora(x)
        if family == FAMILIES[0]:
            obj.q_proj_lora, obj.k_proj_lora, obj.v_proj_lora = loras
        elif family == FAMILIES[1]:
            obj.qkv_lora, obj.z_lora = loras
        elif family == FAMILIES[2]:
            obj.lora = loras[0]
        else:
            obj.gate_lora, obj.up_lora = loras
        output, bias = ns[family](obj, x)
        assert bias is None
        outputs.append(output)
    loss = sum((y * c).sum() for y, c in zip(outputs, cotangents, strict=True))
    params = [p for fs in factors for ab in fs for p in ab]
    actual = torch.autograd.grad(loss, [*inputs, *params])

    # Independent ordinary, unsharded autograd reference; no mock collectives.
    ref_x = rows.clone().requires_grad_()
    ref_norm = rms(ref_x) if fused else ref_x
    ref_params = [p.detach().clone().requires_grad_() for p in params]
    cursor, ref_outputs = 0, []
    for rank in range(tp):
        chunks = []
        for _ in widths:
            a, b = ref_params[cursor : cursor + 2]
            cursor += 2
            chunks.append((ref_norm @ a) @ b * 0.75)
        if family == FAMILIES[1]:
            chunks += [ref_norm.new_zeros(4, 1, 4)]
        ref_outputs.append(ref_norm @ weights[rank] + torch.cat(chunks, dim=-1))
    ref_loss = sum((y * c).sum() for y, c in zip(ref_outputs, cotangents, strict=True))
    expected = torch.autograd.grad(ref_loss, [ref_x, *ref_params])
    for output, ref in zip(outputs, ref_outputs, strict=True):
        torch.testing.assert_close(output, ref, atol=1e-12, rtol=1e-12)
    for rank in range(tp):
        dx = expected[0].chunk(tp)[rank] if sp else expected[0]
        torch.testing.assert_close(actual[rank], dx, atol=1e-12, rtol=1e-12)
    for grad, ref in zip(actual[tp:], expected[1:], strict=True):
        torch.testing.assert_close(grad, ref, atol=1e-12, rtol=1e-12)
    expected_call = "gather" if sp else "copy"
    assert calls == (
        [] if tp == 1 or outer else [(expected_call, group if fused else None)] * tp
    )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("bad_group", [None, SimpleNamespace(size=lambda: 3)])
def test_returned_norm_requires_matching_group(sp, bad_group):
    ns, _ = load_wrappers(
        lambda *a, **k: pytest.fail("unexpected copy"),
        lambda *a, **k: pytest.fail("unexpected gather"),
    )
    with pytest.raises(RuntimeError, match="initialized TP group"):
        ns["_column_parallel_lora_input"](
            torch.ones(2, 1, 4, dtype=torch.float64),
            SimpleNamespace(
                tp_size=2,
                sequence_parallel=sp,
                parallel_mode="column",
                tp_group=bad_group,
            ),
            returned_norm=True,
        )
