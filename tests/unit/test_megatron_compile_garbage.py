import gc
import weakref

import pytest
import torch
import torch.utils.checkpoint

lora = pytest.importorskip("art.megatron.lora")


class _Cycle:
    pass


def _garbage_holding_tensor() -> weakref.ref:
    # A tensor reachable only through a reference cycle, as dynamo's traced
    # frames hold a recompiled call's activations.
    first, second = _Cycle(), _Cycle()
    first.other, second.other = second, first
    first.tensor = torch.empty(1024)
    return weakref.ref(first)


@pytest.fixture
def no_automatic_gc(monkeypatch):
    monkeypatch.delenv("ART_COLLECT_COMPILE_GARBAGE", raising=False)
    monkeypatch.setattr(lora, "_COMPILE_GARBAGE", False)
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def test_next_checkpointed_call_collects_compile_garbage(no_automatic_gc):
    calls = []
    wrapped = lora._with_captured_lora_slot(lambda: calls.append(1))
    held = _garbage_holding_tensor()
    wrapped()
    assert held() is not None  # nothing compiled, nothing collected
    lora._mark_compile_garbage(None)
    wrapped()
    assert held() is None and calls == [1, 1]
    assert lora._COMPILE_GARBAGE is False
    # Only one collection per compile: later calls do not collect again.
    held = _garbage_holding_tensor()
    wrapped()
    assert held() is not None


def test_compile_inside_a_call_is_collected_before_it_returns(no_automatic_gc):
    # The compiling call's own garbage is freed before its backward runs.
    held = []

    def compiling_call():
        held.append(_garbage_holding_tensor())
        lora._mark_compile_garbage(None)

    lora._with_captured_lora_slot(compiling_call)()
    assert held[0]() is None and lora._COMPILE_GARBAGE is False


def test_failed_call_raises_its_own_error_and_keeps_the_mark(no_automatic_gc):
    held = []

    def failing_call():
        held.append(_garbage_holding_tensor())
        lora._mark_compile_garbage(None)
        raise ValueError("model error")

    with pytest.raises(ValueError, match="model error"):
        lora._with_captured_lora_slot(failing_call)()
    assert held[0]() is not None and lora._COMPILE_GARBAGE is True
    lora._with_captured_lora_slot(lambda: None)()
    assert held[0]() is None


@pytest.mark.parametrize(
    "deferring",
    [
        {(torch.compiler, "is_compiling"): True},
        {
            (torch.cuda, "is_initialized"): True,
            (torch.cuda, "is_current_stream_capturing"): True,
        },
    ],
    ids=["tracing", "cuda-graph-capture"],
)
def test_deferred_collection_keeps_the_mark_for_a_later_call(
    no_automatic_gc, monkeypatch, deferring
):
    wrapped = lora._with_captured_lora_slot(lambda: None)
    held = _garbage_holding_tensor()
    lora._mark_compile_garbage(None)
    with monkeypatch.context() as patch:
        for (module, name), value in deferring.items():
            patch.setattr(module, name, lambda value=value: value)
        wrapped()
    assert held() is not None and lora._COMPILE_GARBAGE is True
    lora._with_captured_lora_slot(lambda: None)()
    assert held() is None


def test_registration_survives_dynamo_reset_and_can_be_disabled(monkeypatch):
    handler = torch._dynamo.callback_handler
    callbacks = list(handler.end_callbacks)
    monkeypatch.setattr(handler, "end_callbacks", [])  # as torch._dynamo.reset()
    try:
        monkeypatch.setenv("ART_COLLECT_COMPILE_GARBAGE", "0")
        assert lora.install_compile_garbage_collection() is False
        lora._with_captured_lora_slot(lambda: None)()
        assert handler.end_callbacks == []
        monkeypatch.delenv("ART_COLLECT_COMPILE_GARBAGE")
        lora._with_captured_lora_slot(lambda: None)()
        lora._with_captured_lora_slot(lambda: None)()
        assert handler.end_callbacks == [lora._mark_compile_garbage]
    finally:
        monkeypatch.setattr(handler, "end_callbacks", callbacks)


def test_real_compile_marks_garbage(no_automatic_gc):
    torch._dynamo.reset()
    lora.install_compile_garbage_collection()

    def double(x):
        return x * 2 + 1

    torch.compile(double, backend="eager")(torch.ones(3))
    assert lora._COMPILE_GARBAGE is True


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_backward_with_compile_keeps_gradients(
    no_automatic_gc, use_reentrant
):
    torch._dynamo.reset()
    lora.install_compile_garbage_collection()
    torch.manual_seed(0)
    weight = torch.randn(8, 8)

    def layer(x, w):
        return torch.tanh(x @ w).square()

    compiled = torch.compile(layer, backend="eager")
    x = torch.randn(4, 8, requires_grad=True)
    w = weight.clone().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_w = weight.clone().requires_grad_()
    layer(reference_x, reference_w).sum().backward()
    # The patched checkpoint wraps the function, so recompute runs the
    # collection hook, and compiles (forward and recompute) set the mark.
    out = torch.utils.checkpoint.checkpoint(compiled, x, w, use_reentrant=use_reentrant)
    out.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(w.grad, reference_w.grad)
    assert lora._COMPILE_GARBAGE is False
