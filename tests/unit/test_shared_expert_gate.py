from types import SimpleNamespace
from typing import Any

from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
import pytest
import torch

from art.megatron.batch_invariant.modules import install


def shared_expert(monkeypatch, *, overlap=False):
    module: Any = SharedExpertMLP.__new__(SharedExpertMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(moe_shared_expert_overlap=overlap)
    module.use_shared_expert_gate = True
    module.gate_weight = torch.nn.Parameter(torch.linspace(-0.3, 0.4, 7)[None].double())
    monkeypatch.setattr(MLP, "forward", lambda self, x: (x.square(), None))
    return module


def test_shared_expert_gate_uses_fixed_gemm_and_preserves_fp64_vjp(monkeypatch):
    module = shared_expert(monkeypatch)
    parameter = module.gate_weight
    old_keys = tuple(module.state_dict())
    functional = torch.nn.functional.linear
    calls = []

    def shape_dependent_gate(x, weight, bias=None):
        calls.append(len(x))
        return functional(x, weight, bias) + len(x) * 0.01

    monkeypatch.setattr(torch.nn.functional, "linear", shape_dependent_gate)
    install([module])
    x = torch.linspace(-0.5, 0.5, 4 * 7).reshape(4, 7).double().requires_grad_()
    short, long = module(x[:3]), module(x)
    assert not calls  # The previously uncovered functional gate GEMM is bypassed.
    assert torch.equal(short, long[:3])
    cotangent = torch.cat((torch.ones(3, 7), torch.zeros(1, 7)))
    actual = torch.autograd.grad(long, (x, parameter), cotangent, retain_graph=True)
    small = torch.autograd.grad(short, (x, parameter), torch.ones_like(short))
    assert all(torch.equal(a, b) for a, b in zip(actual, small, strict=True))
    rx, rw = (
        x.detach().clone().requires_grad_(),
        parameter.detach().clone().requires_grad_(),
    )
    reference = rx.square() * torch.sigmoid(rx @ rw.T)
    expected = torch.autograd.grad(reference, (rx, rw), cotangent)
    torch.testing.assert_close(long, reference, atol=1e-12, rtol=1e-12)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert module.gate_weight is parameter and tuple(module.state_dict()) == old_keys
    assert not torch.cuda.is_initialized()


def test_unqualified_shared_expert_overlap_fails_before_mutation(monkeypatch):
    module = shared_expert(monkeypatch, overlap=True)
    with pytest.raises(ValueError, match="shared-expert overlap"):
        install([module])
    assert "forward" not in vars(module)
