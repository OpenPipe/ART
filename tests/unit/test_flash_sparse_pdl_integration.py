"""Isolated CPU probes of FA4 imports, reloads and persistent cache routing."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _probe(tmp_path, mode, version="4.0.0b5"):
    try:
        installed = importlib.metadata.version("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Requires the optional FA4 b5 CPU compiler environment")
    if installed != "4.0.0b5":
        pytest.skip("This integration fixture uses the exact b5 wheel")
    result = subprocess.run(
        [sys.executable, __file__, mode, str(tmp_path), version],
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_90a",
            "CUTE_DSL_DISABLE_FILE_CACHING": "1",
            "CUTE_DSL_CACHE_DIR": str(tmp_path / "dsl-cache"),
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(tmp_path / "cache"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("version", ["4.0.0b5", "4.0.0b26", "absent"])
def test_package_and_compiled_import_install_first_without_version_failure(
    tmp_path, version
):
    assert _probe(tmp_path, "imports", version)["ok"]


@pytest.mark.parametrize("order", ["reload-forward", "reload-reverse"])
def test_real_module_reload_orders_are_idempotent(tmp_path, order):
    assert _probe(tmp_path, order)["ok"]


def test_real_unaffected_keys_domains_disk_paths_and_warm_entries_are_unchanged(
    tmp_path,
):
    baseline = _probe(tmp_path, "cache-baseline")
    patched = _probe(tmp_path, "cache-patched")
    assert baseline["caches"] == patched["caches"]
    assert baseline["compilations"] > 0
    assert patched["compilations"] == 0


def test_package_import_rejects_preexisting_backward_compilation(tmp_path):
    assert _probe(tmp_path, "late-import")["ok"]


@pytest.mark.parametrize("mode", ["history-reload", "retained-before-bootstrap"])
def test_backward_history_survives_reload(tmp_path, mode):
    assert _probe(tmp_path, mode)["ok"]


def _run_probe(mode, root, version):
    import importlib
    import inspect
    import pickle
    from types import ModuleType, SimpleNamespace

    source = Path(__file__).parents[2] / "src/art"

    def import_art():
        # Execute the real flex package/compiled module without requiring ART's
        # unrelated service dependencies in this small CPU compiler fixture.
        for name, directory in [("art", source), ("art.megatron", source / "megatron")]:
            package = ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
        return importlib.import_module("art.megatron.flex_attn")

    if mode == "imports":
        original_version = importlib.metadata.version

        def metadata_version(name):
            if name != "flash-attn-4":
                return original_version(name)
            if version == "absent":
                raise importlib.metadata.PackageNotFoundError(name)
            return version

        importlib.metadata.version = metadata_version
        package = import_art()
        if version == "absent":
            assert "flash_attn.cute.interface" not in sys.modules
        else:
            from flash_attn.cute import interface

            assert interface._flash_attn_bwd._art_sm90_sparse_pdl
        import torch

        torch.cuda.get_device_capability = lambda *args: (10, 0)
        compiled = importlib.import_module(package.__name__ + ".compiled")
        assert (
            compiled.flex_backend_for_head_dims(
                head_dim=256, head_dim_v=256, device=torch.device("cuda")
            )
            == "TRITON"
        )
        if version == "4.0.0b5":
            assert interface._tile_size_fwd_sm90.__name__ == "tile_size_fwd_sm90_art"
            assert (
                interface._bwd_postprocess_convert.__name__
                == "bwd_postprocess_convert_art"
            )
        return {"ok": True}

    from flash_attn.cute import interface

    if mode == "retained-before-bootstrap":
        from cutlass import BFloat16
        from flash_attn.cute import flash_bwd_preprocess, flash_bwd_sm90

        retained = interface._compile_bwd_preprocess(
            BFloat16, 128, 128, 64, False, False, False
        )
        assert "flash_bwd" in retained.function_name
        for module in (flash_bwd_preprocess, flash_bwd_sm90, interface):
            importlib.reload(module)
        try:
            import_art()
        except RuntimeError as error:
            assert "retained FA4 backward handle" in str(error)
        else:
            raise AssertionError("Retained pre-bootstrap compiled handle was accepted")
        assert retained is not None
        return {"ok": True}

    if mode == "late-import":
        interface._flash_attn_bwd.compile_cache.cache["already compiled"] = object()
        try:
            import_art()
        except RuntimeError as error:
            assert "before any FA4 backward compilation" in str(error)
        else:
            raise AssertionError("Package import accepted a compiled backward")
        return {"ok": True}

    if mode.startswith("reload-"):
        package = import_art()
        patch = importlib.import_module(package.__name__ + ".flash_pdl_patch")
        from flash_attn.cute import flash_bwd_preprocess, flash_bwd_sm90

        modules = [interface, flash_bwd_sm90, flash_bwd_preprocess, patch]
        if mode == "reload-reverse":
            modules.reverse()
        for module in modules * 2:
            importlib.reload(module)
            patch.apply_flash_sm90_sparse_pdl_patch()
            patch.apply_flash_sm90_sparse_pdl_patch()
            assert interface._flash_attn_bwd._art_sm90_sparse_pdl
            for cls in (
                interface.FlashAttentionBackwardSm90,
                flash_bwd_sm90.FlashAttentionBackwardSm90,
            ):
                assert inspect.unwrap(cls.__call__)._art_sm90_sparse_pdl_launch
        return {"ok": True}

    from flash_attn.cute import cache_utils
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    compilations = []

    class Compiled:
        def export_to_c(self, object_file_path, function_name):
            Path(object_file_path).write_bytes(b"CPU fixture: no GPU binary")

    def compile_kernel(kernel, *args, **kwargs):
        compilations.append(type(kernel).__name__)
        return Compiled()

    # Keep FA4's real persistent cache hash/path/load/export implementation;
    # stub only binary generation/loading, which would otherwise need kernels.
    interface.cute.compile = compile_kernel
    cache_utils.cute.runtime.load_module = lambda *a, **kw: SimpleNamespace(
        func=Compiled()
    )
    baseline_cache = interface._flash_attn_bwd.compile_cache
    if mode in {"cache-patched", "history-reload"}:
        import_art()
        assert interface._flash_attn_bwd.compile_cache is baseline_cache
    with FakeTensorMode():
        q = torch.empty(1, 256, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
        v, o, dout = torch.empty_like(k), torch.empty_like(q), torch.empty_like(q)
        lse = torch.empty(1, 16, 256, device="cuda", dtype=torch.float32)
        for arch in [90, 100]:
            interface._get_device_arch = lambda arch=arch: arch
            interface._flash_attn_bwd(q, k, v, o, dout, lse)
    if mode == "history-reload":
        from flash_attn.cute import flash_bwd_preprocess, flash_bwd_sm90

        state = interface._art_sm90_pdl_state
        assert state["used"] and state["compiled"]
        retained = list(baseline_cache.cache.values())
        for cache in state["caches"]:
            cache.cache.clear()
        interface._bwd_preprocess.compile_cache.cache.clear()
        interface._bwd_postprocess_convert.compile_cache.cache.clear()
        for module in (flash_bwd_preprocess, flash_bwd_sm90, interface):
            importlib.reload(module)
        assert interface._art_sm90_pdl_state is state
        patch = importlib.import_module("art.megatron.flex_attn.flash_pdl_patch")
        try:
            patch.apply_flash_sm90_sparse_pdl_patch()
        except RuntimeError as error:
            assert "historical FA4 backward use" in str(error)
        else:
            raise AssertionError("Reload erased compilation history")
        assert retained
        return {"ok": True}
    records = []
    for function in (
        interface._flash_attn_bwd,
        interface._bwd_preprocess,
        interface._bwd_postprocess_convert,
    ):
        cache = function.compile_cache
        records.append(
            {
                "domain": str(cache.cache_path),
                "keys": sorted(pickle.dumps(key).hex() for key in cache.cache),
                "paths": sorted(
                    str(cache.cache_path / (cache._key_to_hash(key) + ".o"))
                    for key in cache.cache
                ),
            }
        )
    return {"caches": records, "compilations": len(compilations)}


if __name__ == "__main__":
    print(json.dumps(_run_probe(sys.argv[1], Path(sys.argv[2]), sys.argv[3])))
