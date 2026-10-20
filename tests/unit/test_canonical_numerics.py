from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from art.megatron.canonical import (
    RootLayout,
    current_layout,
    execution,
    install,
    target_logprobs,
)
from art.megatron.canonical.operators import _attention_forward, _row_forward


def test_layout_pads_roots_independently_and_masks_labels():
    layout = RootLayout((3, 129))
    roots = [torch.arange(n) for n in layout.lengths]
    packed = layout.pad(roots, value=-100)
    assert layout.padded_lengths == (128, 256)
    assert packed.shape == (384,)
    assert (packed[3:128] == -100).all()
    assert (packed[257:] == -100).all()
    for expected, actual in zip(roots, layout.unpad(packed), strict=True):
        assert torch.equal(expected, actual)
    with pytest.raises(ValueError):
        layout.pad(roots[::-1])


@pytest.mark.parametrize("lengths", [(), (0,), (-1,), (1.5,), (True,)])
def test_invalid_root_lengths(lengths):
    with pytest.raises(ValueError):
        RootLayout(lengths)


def test_policy_restores_context_and_process_flags_after_error():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = getattr(torch.utils.deterministic, "fill_uninitialized_memory")
    outer, inner = RootLayout((5,)), RootLayout((9,))
    with pytest.raises(RuntimeError, match="failure"):
        with execution(outer):
            assert current_layout() is outer
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert not getattr(torch.utils.deterministic, "fill_uninitialized_memory")
            with execution(inner):
                assert current_layout() is inner
            with execution(None, numerical_policy="native"):
                assert current_layout() is None
            assert current_layout() is outer
            raise RuntimeError("failure")
    assert current_layout() is None
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert getattr(torch.utils.deterministic, "fill_uninitialized_memory") == fill


def test_root_rows_keep_forward_and_target_gradient_under_repacking():
    torch.manual_seed(4)
    roots = [torch.randn(n, 7) for n in (3, 129)]
    weight = torch.randn(7, 5, requires_grad=True)
    bias = torch.randn(5)
    calls = []

    def original(x):
        calls.append(tuple(x.shape))
        return x @ weight, bias

    forward = _row_forward(original)
    layout = RootLayout((3, 129))
    packed = layout.pad(roots).requires_grad_()
    with execution(layout):
        out, returned_bias = forward(packed)
        out[:3].square().sum().backward()
    reference = out[:3].detach().clone()
    assert weight.grad is not None
    gradient = weight.grad.clone()
    assert packed.grad is not None
    dx = packed.grad[:3].clone()
    assert returned_bias is bias
    assert calls == [(128, 7), (256, 7)]
    weight.grad = None
    singleton = RootLayout((3,))
    x = singleton.pad(roots[:1]).requires_grad_()
    with execution(singleton):
        out, _ = forward(x)
        out[:3].square().sum().backward()
    assert torch.equal(reference, out[:3])
    assert weight.grad is not None
    assert torch.equal(gradient, weight.grad)
    assert x.grad is not None
    assert torch.equal(dx, x.grad[:3])
    # Native calls retain their original call geometry.
    forward(torch.randn(3, 7))
    assert calls[-1] == (3, 7)


def test_checkpoint_recomputation_observes_same_layout():
    layout = RootLayout((3, 5))
    seen = []

    def function(x):
        seen.append(current_layout())
        return x.sin()

    with execution(layout):
        x = torch.randn(8, requires_grad=True)
        checkpoint(function, x, use_reentrant=True).sum().backward()
    assert seen == [layout, layout]


def test_fp32_head_is_exact_across_root_packing_and_masks_padding():
    torch.manual_seed(9)
    layout = RootLayout((3, 129))
    roots = [torch.randn(n, 8, dtype=torch.bfloat16) for n in layout.lengths]
    targets = [torch.randint(0, 19, (n,)) for n in layout.lengths]
    targets[0][1] = -100
    weight = torch.randn(19, 8, dtype=torch.bfloat16)
    hidden = layout.pad(roots).requires_grad_()
    packed = target_logprobs(
        hidden, weight, layout.pad(targets, value=-100), layout=layout
    )
    packed[0].sum().backward()
    assert hidden.grad is not None
    expected_dx = hidden.grad[:3].clone()
    solo_layout = RootLayout((3,))
    solo_hidden = solo_layout.pad(roots[:1]).requires_grad_()
    solo = target_logprobs(
        solo_hidden,
        weight,
        solo_layout.pad(targets[:1], value=-100),
        layout=solo_layout,
    )
    solo[0].sum().backward()
    assert packed[0].dtype == torch.float32
    assert packed[0][1] == 0
    assert torch.equal(packed[0], solo[0])
    assert solo_hidden.grad is not None
    assert torch.equal(expected_dx, solo_hidden.grad[:3])
    expected = -F.cross_entropy(
        F.linear(roots[0].float(), weight.float()), targets[0], reduction="none"
    )
    torch.testing.assert_close(packed[0], expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_moe_experts", 8),
        ("context_parallel_size", 2),
        ("tensor_model_parallel_size", 2),
        ("hidden_dropout", 0.1),
    ],
)
def test_install_rejects_unqualified_models_but_native_is_untouched(field, value):
    model = torch.nn.Linear(2, 2)
    setattr(model, "config", SimpleNamespace(**{field: value}))
    original = model.forward
    install(model, numerical_policy="native")
    assert model.forward == original
    with pytest.raises(ValueError):
        install(model)


def test_attention_rejects_partial_root_before_launching_kernel():
    module = SimpleNamespace(softmax_scale=0.5)
    forward = _attention_forward(module, lambda *args: args[0])
    q = torch.zeros(4, 1, 2, 4)
    assert forward(q, q, q, None) is q
    with execution(RootLayout((4,))):
        with pytest.raises(ValueError, match="complete roots"):
            forward(q, q, q, None)


def test_canonical_gdn_never_selects_uniform_bucket_algorithm():
    from art.megatron.gdn.operator import _bucket_has_uniform_lengths

    bucket: Any = SimpleNamespace(length=128, lengths_cpu=torch.tensor([128, 128]))
    assert _bucket_has_uniform_lengths(bucket)
    with execution(RootLayout((128, 128))):
        assert not _bucket_has_uniform_lengths(bucket)
    assert _bucket_has_uniform_lengths(bucket)
