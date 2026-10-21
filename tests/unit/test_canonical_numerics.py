from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from art.megatron.canonical import (
    RootLayout,
    current_layout,
    execution,
    install,
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


def test_deterministic_native_scope_keeps_layout_native_and_restores_flags():
    initial = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = getattr(torch.utils.deterministic, "fill_uninitialized_memory")
    with pytest.raises(RuntimeError, match="abort"):
        with execution(None, numerical_policy="native", deterministic_native=True):
            assert current_layout() is None
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            with execution(None, numerical_policy="native"):
                assert current_layout() is None
                assert torch.are_deterministic_algorithms_enabled()
            raise RuntimeError("abort")
    assert torch.are_deterministic_algorithms_enabled() == initial
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warning
    assert getattr(torch.utils.deterministic, "fill_uninitialized_memory") == fill


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

    bucket = cast(
        Any, SimpleNamespace(length=128, lengths_cpu=torch.tensor([128, 128]))
    )
    assert _bucket_has_uniform_lengths(bucket)
    with execution(RootLayout((128, 128))):
        assert not _bucket_has_uniform_lengths(bucket)
    assert _bucket_has_uniform_lengths(bucket)


def test_checkpoint_capture_carries_layout_to_autograd_worker_thread():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import Context

    from art.megatron.canonical.layout import capture

    layout = RootLayout((3, 5))
    # Caladan intentionally propagates ContextVars through thread pools. Force
    # the empty context used by CUDA autograd independently of pool patches.
    empty = Context()
    with ThreadPoolExecutor(max_workers=1) as pool, execution(layout):
        assert pool.submit(empty.run, current_layout).result() is None
        captured = capture(current_layout)
        assert pool.submit(empty.run, captured).result() is layout
        assert pool.submit(empty.run, current_layout).result() is None


@pytest.fixture
def operator_types(monkeypatch):
    # Test dispatch by identity with neutral names. The runtime integration test
    # exercises the production registry against real TE/MCore/ART modules.
    class Projection(torch.nn.Linear):
        pass

    monkeypatch.setattr(
        "art.megatron.canonical.operators._module_types",
        lambda: ((Projection,), (), (torch.nn.Embedding,)),
    )
    return Projection


def test_install_reports_types_and_is_idempotent(operator_types):
    linear = operator_types(2, 2)
    model = torch.nn.Sequential(linear)
    setattr(model, "config", SimpleNamespace())
    weight = linear.weight
    assert install(model) == {"0": "row"}
    wrapper = linear.forward
    assert install(model) == {"0": "row"}
    assert linear.forward is wrapper
    assert linear.weight is weight
    layout = RootLayout((3, 129))
    x = torch.randn(layout.rows, 2)
    with execution(layout):
        actual = model(x)
    torch.testing.assert_close(actual, F.linear(x, linear.weight, linear.bias))


@pytest.mark.parametrize("name", ["TECanonicalLinear", "UnexpectedNorm"])
def test_unknown_parameter_types_are_rejected_before_wrapping(operator_types, name):
    impostor = type(name, (torch.nn.Linear,), {})(2, 2)
    model = torch.nn.Sequential(operator_types(2, 2), impostor)
    setattr(model, "config", SimpleNamespace())
    original = model[0].forward
    with pytest.raises(ValueError, match=rf"1 \({name}\)"):
        install(model)
    assert model[0].forward == original
    assert install(model, numerical_policy="native") == {}


def test_output_layer_stays_native_and_is_not_rejected(operator_types):
    model = torch.nn.Sequential(operator_types(2, 2))
    setattr(model, "config", SimpleNamespace())
    model.output_layer = torch.nn.Linear(2, 4)  # e.g. MCore ColumnParallelLinear
    head_forward = model.output_layer.forward
    assert install(model) == {"0": "row"}
    assert model.output_layer.forward == head_forward


def test_empty_operator_layout_is_rejected(operator_types):
    model = torch.nn.Identity()
    setattr(model, "config", SimpleNamespace())
    with pytest.raises(ValueError, match="no supported operators"):
        install(model)


def test_non_row_input_and_shared_output_are_not_split():
    calls = []
    shared = torch.randn(2, 2)

    def original(x):
        calls.append(x)
        return shared

    forward = _row_forward(original)
    layout = RootLayout((3, 5))
    non_row = torch.randn(7, 2)
    with execution(layout):
        assert forward(non_row) is shared
        assert forward(torch.randn(layout.rows, 2)) is shared
    assert calls[0] is non_row
    assert len(calls) == 3


def test_process_flags_are_visible_to_other_threads_and_restored():
    from concurrent.futures import ThreadPoolExecutor

    previous = torch.are_deterministic_algorithms_enabled()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with execution(RootLayout((3,))):
            assert pool.submit(torch.are_deterministic_algorithms_enabled).result()
        assert (
            pool.submit(torch.are_deterministic_algorithms_enabled).result() == previous
        )


def test_native_scope_does_not_write_process_flags(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("native scope modified deterministic settings")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", unexpected)
    with execution(None, numerical_policy="native"):
        assert current_layout() is None


def test_production_registry_matches_imported_te_type():
    pytest.importorskip("megatron.core")
    import transformer_engine.pytorch as te

    class Projection(te.RMSNorm):
        pass

    projection = Projection(8, device="cpu")
    model = torch.nn.Sequential(projection)
    setattr(model, "config", SimpleNamespace())
    weight = projection.weight
    assert install(model) == {"0": "row"}
    assert projection.weight is weight
    wrapper = projection.forward
    assert install(model) == {"0": "row"}
    assert projection.forward is wrapper


def test_segmented_slots_follow_canonical_root_slices_through_backward():
    from art.megatron.segmented_lora import (
        _CURRENT_LORA_SEGMENTS,
        LoRASegments,
        segmented_lora_factors,
        use_lora_segments,
    )

    factors = [
        (
            torch.randn(3, 2, requires_grad=True),
            torch.randn(2, 4, requires_grad=True),
            1.0,
        )
        for _ in range(2)
    ]
    segments = LoRASegments((None, cast(Any, "second")), (128, 256))
    seen = []

    def original(x):
        current = _CURRENT_LORA_SEGMENTS.get()
        assert current is not None
        seen.append(current.counts)
        return segmented_lora_factors(
            x, factors, current, template_a=factors[0][0], template_b=factors[0][1]
        )

    x = torch.randn(384, 3)
    with use_lora_segments(segments), execution(RootLayout((3, 129))):
        actual = _row_forward(original)(x)
        actual[:3].sum().backward()
    assert seen == [(128, 0), (0, 256)]
    assert torch.equal(actual[:128], x[:128] @ factors[0][0] @ factors[0][1])
    assert torch.equal(actual[128:], x[128:] @ factors[1][0] @ factors[1][1])
    assert not cast(torch.Tensor, factors[1][0].grad).any()
    assert not cast(torch.Tensor, factors[1][1].grad).any()
    assert _CURRENT_LORA_SEGMENTS.get() is None
