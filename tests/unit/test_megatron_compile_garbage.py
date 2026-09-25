import gc
import weakref

import pytest
import torch

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


def test_collection_is_registered_once_and_can_be_disabled(monkeypatch):
    handler = torch._dynamo.callback_handler
    callbacks = list(handler.end_callbacks)
    monkeypatch.setattr(handler, "end_callbacks", [])
    try:
        monkeypatch.setenv("ART_COLLECT_COMPILE_GARBAGE", "0")
        lora.install_compile_garbage_collection()
        assert handler.end_callbacks == []
        monkeypatch.delenv("ART_COLLECT_COMPILE_GARBAGE")
        lora.install_compile_garbage_collection()
        lora.install_compile_garbage_collection()
        assert handler.end_callbacks == [lora._mark_compile_garbage]
    finally:
        monkeypatch.setattr(handler, "end_callbacks", callbacks)
    assert lora._mark_compile_garbage in callbacks
