"""CPU contracts and FP64 gradient oracles; no native GPU imports."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron.batch_invariant.modules import _te_grouped, _te_linear
from art.megatron.batch_invariant.ops import (
    fixed_sum,
    grouped_lora,
    linear,
    matmul,
    rms_norm,
)

LENGTHS = (1, 4, 5, 63, 64, 65, 129, 1000)


@pytest.mark.parametrize("length", LENGTHS)
def test_rows_do_not_depend_on_prefix_batch_or_tile_offset(length):
    torch.manual_seed(905)
    x = torch.randn(1000, 35)
    w = torch.randn(35, 17)
    gamma = torch.randn(35)
    reference = matmul(rms_norm(x, gamma, 1e-6), w)
    actual = matmul(rms_norm(x[:length], gamma, 1e-6), w)
    assert torch.equal(actual, reference[:length])
    # Move a row across a tile boundary and fill companions with adversarial data.
    companions = torch.randn(131, 35) * 100
    companions[128] = x[0]
    actual = matmul(rms_norm(companions, gamma, 1e-6), w)
    assert torch.equal(reference[0], actual[128])


def test_reduction_tree_is_adjacent_and_zero_padded():
    # Reassociation changes this result, so a native mean is not a substitute.
    x = torch.tensor([[1e20, -1e20, 1.0], [1.0, 1e20, -1e20]])
    assert torch.equal(fixed_sum(x), torch.tensor([[1.0], [0.0]]))
    for n in LENGTHS:
        assert torch.equal(fixed_sum(x.repeat(n, 1))[:2], fixed_sum(x))


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_norm_gradients_against_fp64(zero_centered, gated):
    torch.manual_seed(20)
    x, w, gate = [
        t.requires_grad_()
        for t in (
            torch.randn(3, 7, dtype=torch.float64),
            torch.randn(7, dtype=torch.float64),
            torch.randn(3, 7, dtype=torch.float64),
        )
    ]
    actual = rms_norm(x, w, 1e-6, zero_centered=zero_centered)
    expected = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        * (w + int(zero_centered))
    )
    if gated:
        actual, expected = [
            out * torch.nn.functional.silu(gate) for out in (actual, expected)
        ]
    cotangent = torch.randn_like(actual)
    inputs = (x, w, gate) if gated else (x, w)
    ga = torch.autograd.grad(actual, inputs, cotangent, retain_graph=True)
    ge = torch.autograd.grad(expected, inputs, cotangent)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-13, rtol=1e-13)


def test_dense_backward_fp64_and_zero_cotangent_suffix():
    torch.manual_seed(8)
    x = torch.randn(5, 35, dtype=torch.float64, requires_grad=True)
    w = torch.randn(13, 35, dtype=torch.float64, requires_grad=True)
    b = torch.randn(13, dtype=torch.float64, requires_grad=True)
    dy = torch.randn(5, 13, dtype=torch.float64)
    actual = linear(x, w, b)
    reference = torch.nn.functional.linear(x, w, b)
    ga = torch.autograd.grad(actual, (x, w, b), dy)
    ge = torch.autograd.grad(reference, (x, w, b), dy)
    torch.testing.assert_close(actual, reference, atol=1e-13, rtol=1e-13)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-13, rtol=1e-13)
    long = torch.cat((x.detach(), torch.randn_like(x))).requires_grad_()
    padded = torch.cat((dy, torch.zeros_like(dy)))
    gl = torch.autograd.grad(linear(long, w, b), (long, w, b), padded)
    assert torch.equal(ga[0], gl[0][:5])
    torch.testing.assert_close(ga[1], gl[1], atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("shared", [None, "A", "B"])
def test_grouped_experts_fp64_and_composition(shared):
    torch.manual_seed(17)
    x = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    a = torch.randn(
        1 if shared == "A" else 3, 11, 3, dtype=torch.float64, requires_grad=True
    )
    b = torch.randn(
        1 if shared == "B" else 3, 3, 13, dtype=torch.float64, requires_grad=True
    )
    actual = grouped_lora(x, a, b, (2, 0, 5), scale=0.3)
    expected = (
        torch.cat(
            [
                (part @ a[0 if shared == "A" else e]) @ b[0 if shared == "B" else e]
                for e, part in enumerate(x.split((2, 0, 5)))
            ]
        )
        * 0.3
    )
    dy = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (x, a, b), dy, retain_graph=True)
    ge = torch.autograd.grad(expected, (x, a, b), dy)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    for aa, ee in zip(ga, ge, strict=True):
        torch.testing.assert_close(aa, ee, atol=1e-13, rtol=1e-13)
    alone = grouped_lora(x[:2], a[:1], b[:1], (2,), scale=0.3)
    assert torch.equal(alone, actual[:2])


def test_te_fused_output_and_returned_norm_both_retain_autograd():
    torch.manual_seed(21)
    module = SimpleNamespace(
        weight=torch.nn.Parameter(torch.randn(9, 7, dtype=torch.float64)),
        layer_norm_weight=torch.nn.Parameter(torch.randn(7, dtype=torch.float64)),
        eps=1e-6,
        zero_centered_gamma=True,
        parallel_mode=None,
        tp_size=1,
        use_bias=False,
        return_bias=False,
        return_layernorm_output=True,
        return_layernorm_output_gathered=True,
        te_return_bias=False,
    )
    x = torch.randn(5, 1, 7, dtype=torch.float64, requires_grad=True)
    (out, norm), bias = _te_linear(module, x)
    assert bias is None
    expected_norm = rms_norm(x, module.layer_norm_weight, 1e-6, zero_centered=True)
    assert torch.equal(norm, expected_norm)
    assert torch.equal(out, linear(expected_norm, module.weight))
    (out.square().sum() + norm.square().sum()).backward()
    assert x.grad is not None and module.layer_norm_weight.grad is not None


def test_grouped_te_empty_expert_keeps_parameter_gradient():
    module = SimpleNamespace(
        num_gemms=2,
        use_bias=False,
        apply_bias=False,
        return_bias=False,
        te_return_bias=False,
        weight0=torch.nn.Parameter(torch.randn(5, 3)),
        weight1=torch.nn.Parameter(torch.randn(5, 3)),
    )
    out, bias = _te_grouped(module, torch.randn(4, 3, requires_grad=True), [4, 0])
    assert bias is None
    out.sum().backward()
    assert module.weight1.grad is not None and module.weight1.grad.count_nonzero() == 0


def test_install_preserves_parameters_hooks_and_rejects_before_mutation(monkeypatch):
    import sys
    from types import ModuleType

    from art.megatron.batch_invariant.modules import install

    class Native(torch.nn.Module):
        pass

    # Only optional type discovery is replaced; execution uses the real CPU ops.
    layers = ModuleType("megatron.core.tensor_parallel.layers")
    setattr(layers, "ColumnParallelLinear", Native)
    router = ModuleType("megatron.core.transformer.moe.router")
    setattr(router, "TopKRouter", Native)
    shared = ModuleType("megatron.core.transformer.moe.shared_experts")
    setattr(shared, "SharedExpertMLP", type("SharedExpertMLP", (torch.nn.Module,), {}))
    te = ModuleType("transformer_engine.pytorch")
    for name in ("LayerNormLinear", "Linear", "GroupedLinear", "RMSNorm"):
        setattr(te, name, type(name, (torch.nn.Module,), {}))
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.tensor_parallel",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "transformer_engine",
    ):
        parent = ModuleType(name)
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    for name, module in (
        (layers.__name__, layers),
        (router.__name__, router),
        (shared.__name__, shared),
        (te.__name__, te),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    model = torch.nn.Sequential(
        torch.nn.RMSNorm(7, dtype=torch.float64),
        torch.nn.Linear(7, 3, dtype=torch.float64),
    )
    parameters = tuple(model.parameters())
    calls = []
    model[1].register_forward_hook(lambda *_: calls.append(True))
    bad = torch.nn.Linear(7, 3)
    setattr(bad, "config", SimpleNamespace(delay_wgrad_compute=True))
    original = model[0].forward
    with pytest.raises(ValueError, match="delayed"):
        install([model, bad])
    assert model[0].forward == original
    assert install([model]) == [model]
    forward = model[1].forward
    install([model])
    assert model[1].forward == forward
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    x = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert calls == [True]
    assert all(p.grad is not None for p in parameters)


def test_gdn_cp_chunks_have_segment_absolute_64_token_boundaries():
    import ast
    from pathlib import Path

    # Execute the actual pure CPU predicate without importing CUDA/FLA packages.
    source = Path(__file__).parents[2] / "src/art/megatron/gdn/fla_cp.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fla_chunk_boundaries_aligned_cpu"
    )
    namespace: dict[str, Any] = {"torch": torch, "Tensor": torch.Tensor}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    aligned = namespace[function.name]
    for length in LENGTHS:
        # Every packed segment has its own origin. Final partial chunks are legal.
        lengths = torch.tensor([[64, 0], [length, 65]])
        assert aligned(lengths)
        lengths[0, 0] = 63
        assert not aligned(lengths)
