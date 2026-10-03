"""Logical head registration enforces the native object kind contract."""

import asyncio

import pytest
from test_trainer_rank_custom_tensors import _trainer
import torch

from art.trainer_rank import ModuleHandle, run_rank_callback


@pytest.mark.parametrize("mode", ("rank", "zero"))
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("kind", ("buffer", "parameter", "module"))
def test_logical_registration_checks_kind_before_cached_or_native_lookup(
    mode, cached, kind
):
    native, api = _trainer("student")
    factory = torch.nn.Identity if kind == "module" else lambda: torch.tensor(2.0)
    original = getattr(api, kind)("head", factory, checkpoint="student")
    calls = []

    def unexpected_factory():
        calls.append(True)
        raise AssertionError("existing head must not call its factory")

    def callback(view):
        if cached:
            getattr(view, kind)("head", unexpected_factory, checkpoint="student")
        for wrong in {"buffer", "parameter", "module"} - {kind}:
            with pytest.raises(ValueError, match=f"already a {kind}"):
                getattr(view, wrong)("head", unexpected_factory, checkpoint="student")
        reopened = getattr(view, kind)("head", unexpected_factory, checkpoint="student")
        assert isinstance(reopened, ModuleHandle if kind == "module" else torch.Tensor)
        if kind != "module":
            assert reopened.item() == original.item() == 2
            assert reopened.requires_grad == (kind == "parameter")
        assert (
            getattr(view, kind)("head", unexpected_factory, checkpoint="student")
            is reopened
        )

    asyncio.run(run_rank_callback(native, callback, mode=mode))
    assert not calls


@pytest.mark.parametrize("selection", ("explicit", "default", "pushed"))
def test_loaded_logical_lookup_resolves_locally(monkeypatch, selection):
    native, api = _trainer("student", "teacher")
    for name, value in (("student", 2.0), ("teacher", 3.0)):
        api.buffer("head", lambda: torch.tensor(value), checkpoint=name)
    native._default_slot_ref = native._slot_ref("student")
    if selection == "pushed":
        native._slot_stack.append(native._slot_ref("teacher"))

    def unexpected_load(_names):
        raise AssertionError(
            "lookup of a loaded slot must not coordinate global loading"
        )

    monkeypatch.setattr(native, "_ensure_checkpoint_slots", unexpected_load)

    def callback(view):
        options = {"checkpoint": "student"} if selection == "explicit" else {}
        head = view.buffer("head", lambda: None, **options)
        assert head.item() == (3 if selection == "pushed" else 2)

    asyncio.run(run_rank_callback(native, callback, mode="rank"))


@pytest.mark.parametrize("mode", ("rank", "zero"))
@pytest.mark.parametrize("prefetched", (False, True))
def test_unloaded_logical_lookup_requires_global_loading(monkeypatch, mode, prefetched):
    native, api = _trainer("pending")
    api.buffer("head", lambda: torch.tensor(2.0), checkpoint="pending")
    pending = native._checkpoint_slots.pop("pending")
    loaded = []
    if prefetched:
        native._checkpoint_prefetch_sources["pending"] = "/test/prefetched"

    def load(name):
        loaded.append(name)
        native._checkpoint_slots[name] = pending

    monkeypatch.setattr(native, "_load_registered_checkpoint", load)

    def callback(view):
        if mode == "zero" and prefetched:
            assert view.buffer("head", lambda: None, checkpoint="pending").item() == 2
        else:
            message = (
                "Load checkpoint .* across all ranks"
                if mode == "rank"
                else "unloaded checkpoint"
            )
            with pytest.raises(RuntimeError, match=message):
                view.buffer("head", lambda: None, checkpoint="pending")
        with pytest.raises(RuntimeError, match="require a loaded named checkpoint"):
            view.buffer("head", lambda: None, checkpoint=None)

    asyncio.run(run_rank_callback(native, callback, mode=mode))
    assert loaded == (["pending"] if mode == "zero" and prefetched else [])


@pytest.mark.parametrize("mode", ("rank", "zero"))
@pytest.mark.parametrize("dp_size", (1, 2))
@pytest.mark.parametrize("kind", (None, "parameter", "module", "buffer"))
def test_only_multi_dp_callbacks_with_buffers_add_reconciliation(
    monkeypatch, mode, dp_size, kind
):
    native, api = _trainer("student")
    monkeypatch.setattr(native, "_dp_rank_and_size", lambda: (0, dp_size))
    if kind is not None:
        factory = torch.nn.Identity if kind == "module" else lambda: torch.tensor(2.0)
        getattr(api, kind)("head", factory, checkpoint="student")
    synchronized = []
    monkeypatch.setattr(
        "art.trainer_rank._heads.synchronize_head_buffers", synchronized.append
    )
    asyncio.run(run_rank_callback(native, lambda _: None, mode=mode))
    assert synchronized == (
        [native] if mode == "rank" and dp_size > 1 and kind == "buffer" else []
    )
