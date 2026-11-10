"""CPU contracts for the pinned FA4 preprocess dependency workaround."""

import importlib.util
from pathlib import Path
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patch():
    return _load(
        Path(__file__).parents[2] / "src/art/megatron/flex_attn/flash_pdl_patch.py",
        "pdl_patch_test",
    )


@pytest.fixture
def interface(tmp_path):
    path = tmp_path / "fake_fa4.py"
    path.write_text("""
class FlashAttentionBackwardPreprocess:
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.use_pdl = True

class FlashAttentionBackwardSm90:
    def __init__(self, sparse):
        self.use_block_sparsity = sparse
    def kernel(self):
        return self
    def launch(self, **kwargs):
        return kwargs["use_pdl"]
    def __call__(self):
        return self.kernel().launch(use_pdl=True)

def _compile_bwd_preprocess(head_dim):
    return FlashAttentionBackwardPreprocess(head_dim)

def _bwd_preprocess(head_dim):
    if head_dim not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[head_dim] = _compile_bwd_preprocess(head_dim)
    return _bwd_preprocess.compile_cache[head_dim].use_pdl

_bwd_preprocess.compile_cache = {}

def _flash_attn_bwd(arch, block_sparse_tensors=None, head_dim=256):
    use_block_sparsity = block_sparse_tensors is not None
    key = (arch, use_block_sparsity, head_dim)
    if key not in _flash_attn_bwd.compile_cache:
        _flash_attn_bwd.compile_cache[key] = object()
    return _bwd_preprocess(head_dim)

_flash_attn_bwd.compile_cache = {}

_caches = {}
def get_jit_cache(name):
    return _caches.setdefault(name, {})
""")
    return _load(path, "fake_fa4")


@pytest.mark.parametrize("arch", [80, 90, 91, 100, 103, 110, 120])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_preprocess_pdl_selection(patch, interface, arch, sparse, head_dim):
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    expected = not (arch // 10 == 9 and sparse)
    assert (
        interface._flash_attn_bwd(arch, object() if sparse else None, head_dim)
        is expected
    )
    assert (
        interface._flash_attn_bwd(
            arch=arch,
            block_sparse_tensors=object() if sparse else None,
            head_dim=head_dim,
        )
        is expected
    )


def test_warm_cache_isolation_and_no_shared_class_mutation(patch, interface):
    backward = interface._flash_attn_bwd
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    assert backward(90) is True
    warm = interface._bwd_preprocess.compile_cache[256]
    for _ in range(3):
        assert backward(90, object()) is False
        assert backward(90) is True
        assert backward(100, object()) is True
    assert warm.use_pdl is True
    assert interface.FlashAttentionBackwardPreprocess(256).use_pdl is True
    assert interface._bwd_preprocess.compile_cache[256] is warm
    assert len(interface._caches["bwd_pre_art_no_pdl_v1"]) == 1


def test_idempotent_across_module_reload_and_import_order(patch, interface):
    imported_before = interface._flash_attn_bwd
    old_cache = imported_before.compile_cache
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    cache = imported_before.compile_cache
    assert cache is old_cache
    installed_code = imported_before.__code__
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    fresh_patch = _load(Path(patch.__file__), "pdl_patch_reloaded")
    fresh_patch.apply_flash_sm90_sparse_pdl_patch(interface)
    imported_after = interface._flash_attn_bwd
    assert imported_after is imported_before
    assert imported_after.__code__ is installed_code
    assert imported_after.compile_cache is cache
    assert imported_before(90, object()) is False
    assert imported_after(90, object()) is False


@pytest.mark.parametrize("sparse", [False, True])
def test_consumer_launch_keeps_pdl_only_for_nonsparse(patch, interface, sparse):
    imported_before = interface.FlashAttentionBackwardSm90.__call__
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    assert interface.FlashAttentionBackwardSm90.__call__ is imported_before
    assert imported_before(interface.FlashAttentionBackwardSm90(sparse)) is (not sparse)


def test_missing_install_is_retryable(patch, interface, monkeypatch):
    def absent(name):
        raise patch.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(patch.importlib.metadata, "version", absent)
    assert patch.apply_flash_sm90_sparse_pdl_patch() is False
    assert patch.apply_flash_sm90_sparse_pdl_patch(interface) is True


def test_unreviewed_version_fails_explicitly_only_on_affected_use(patch, interface):
    launch = interface.FlashAttentionBackwardSm90.__call__.__code__
    patch.apply_flash_sm90_sparse_pdl_patch(interface, version="4.0.0b26")
    assert interface.FlashAttentionBackwardSm90.__call__.__code__ is launch
    assert interface._flash_attn_bwd(100, object()) is True
    assert interface._flash_attn_bwd(90) is True
    before = dict(interface._flash_attn_bwd.compile_cache)
    with pytest.raises(RuntimeError, match="before any launch"):
        interface._flash_attn_bwd(90, object())
    assert interface._flash_attn_bwd.compile_cache == before


def test_unrecognized_call_site_does_not_partially_install(patch, interface):
    interface._flash_attn_bwd = interface.get_jit_cache
    original = interface._flash_attn_bwd.__code__
    with pytest.raises(RuntimeError, match="call site"):
        patch.apply_flash_sm90_sparse_pdl_patch(interface)
    assert interface._flash_attn_bwd.__code__ is original


@pytest.mark.parametrize("component", ["_flash_attn_bwd", "_bwd_preprocess"])
def test_compile_then_install_is_rejected_before_mutation(patch, interface, component):
    getattr(interface, component).compile_cache["old"] = object()
    code = interface._flash_attn_bwd.__code__
    launch = interface.FlashAttentionBackwardSm90.__call__.__code__
    with pytest.raises(RuntimeError, match="before any FA4 backward compilation"):
        patch.apply_flash_sm90_sparse_pdl_patch(interface)
    assert interface._flash_attn_bwd.__code__ is code
    assert interface.FlashAttentionBackwardSm90.__call__.__code__ is launch


def test_install_after_dsl_preprocessing_is_rejected(patch, interface, monkeypatch):
    launch = interface.FlashAttentionBackwardSm90.__call__
    monkeypatch.setattr(launch, "_preprocessed", True, raising=False)
    with pytest.raises(RuntimeError, match="SM90 DSL preprocessing started"):
        patch.apply_flash_sm90_sparse_pdl_patch(interface)


def test_install_then_compile_populates_only_sparse_private_cache(patch, interface):
    baseline = interface._flash_attn_bwd.compile_cache
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    assert interface._flash_attn_bwd(90, object()) is False
    assert not baseline
    private = interface._caches["bwd_art_sm90_sparse_no_pdl_v2"]
    assert list(private) == [(90, True, 256)]


def test_concurrent_installers_are_serialized(patch, interface, monkeypatch):
    # Deterministic two-thread interleaving: hold the first rewrite while the
    # second installer attempts it. Before locking, the second changes the
    # source underneath the first, which then raises on the patched call site.
    ready, resume, fast_started, fast_done = Event(), Event(), Event(), Event()
    original = patch._patched_backward_code
    errors = []

    def interleave(function):
        if current_thread().name == "slow":
            ready.set()
            assert resume.wait(10)
        return original(function)

    monkeypatch.setattr(patch, "_patched_backward_code", interleave)

    def install(fast=False):
        if fast:
            fast_started.set()
        try:
            patch.apply_flash_sm90_sparse_pdl_patch(interface)
        except Exception as error:
            errors.append(error)
        finally:
            if fast:
                fast_done.set()

    slow = Thread(target=install, name="slow")
    fast = Thread(target=install, args=(True,), name="fast")
    slow.start()
    try:
        assert ready.wait(10)
        fast.start()
        assert fast_started.wait(10)
        fast_done.wait(0.5)
    finally:
        resume.set()
        slow.join(10)
        if fast.ident is not None:
            fast.join(10)
    assert not slow.is_alive() and not fast.is_alive()
    assert not errors


def test_install_during_backward_requires_quiescence(patch, interface, monkeypatch):
    ready, resume = Event(), Event()
    original = interface._bwd_preprocess

    def paused(*args):
        ready.set()
        assert resume.wait(10)
        return original(*args)

    paused.compile_cache = original.compile_cache
    # Unaffected calls retain a live reference to this module global.
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
    monkeypatch.setattr(interface, "_bwd_preprocess", paused)
    worker = Thread(target=lambda: interface._flash_attn_bwd(100))
    worker.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RuntimeError, match="backward call active"):
            patch.apply_flash_sm90_sparse_pdl_patch(interface)
    finally:
        resume.set()
        worker.join(10)
    assert not worker.is_alive()


def test_install_during_graph_capture_is_rejected(patch, interface, monkeypatch):
    import sys

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True, is_current_stream_capturing=lambda: True
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="CUDA graph capture active"):
        patch.apply_flash_sm90_sparse_pdl_patch(interface)
