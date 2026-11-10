"""CPU contracts for the pinned FA4 preprocess dependency workaround."""

import importlib.util
from pathlib import Path

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


def test_warm_cache_isolation_and_no_shared_class_mutation(patch, interface):
    backward = interface._flash_attn_bwd
    assert backward(90, object()) is True
    warm = interface._bwd_preprocess.compile_cache[256]
    patch.apply_flash_sm90_sparse_pdl_patch(interface)
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
    assert cache is not old_cache
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


def test_unreviewed_version_fails_explicitly(patch, monkeypatch):
    monkeypatch.setattr(patch.importlib.metadata, "version", lambda name: "4.0.0b26")
    with pytest.raises(RuntimeError, match="Recheck FA4"):
        patch.apply_flash_sm90_sparse_pdl_patch()


def test_unrecognized_call_site_does_not_partially_install(patch, interface):
    interface._flash_attn_bwd = interface.get_jit_cache
    original = interface._flash_attn_bwd.__code__
    with pytest.raises(RuntimeError, match="call site"):
        patch.apply_flash_sm90_sparse_pdl_patch(interface)
    assert interface._flash_attn_bwd.__code__ is original
