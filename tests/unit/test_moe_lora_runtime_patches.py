import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def patches():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_runtime/src/art_vllm_runtime/moe_lora_patches.py"
    )
    spec = importlib.util.spec_from_file_location("_art_moe_lora_patches", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vllm_modules(monkeypatch):
    triton_utils = ModuleType("vllm.triton_utils")
    setattr(triton_utils, "HAS_TRITON", True)
    triton_ops = ModuleType("vllm.lora.ops.triton_ops")
    fused_op = SimpleNamespace()
    setattr(triton_ops, "fused_moe_lora_op", fused_op)
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    monkeypatch.setitem(sys.modules, "vllm.lora.ops.triton_ops", triton_ops)
    return triton_utils, fused_op


def test_small_batch_patch_skips_triton_placeholder(patches, vllm_modules):
    triton_utils, fused_op = vllm_modules
    triton_utils.HAS_TRITON = False

    # vLLM's placeholder decorators leave an ordinary function without .fn.
    def placeholder():
        pass

    fused_op._fused_moe_lora_small_batch_kernel = placeholder

    patches.patch_small_batch_moe_lora_intermediate_dtype()

    assert fused_op._fused_moe_lora_small_batch_kernel is placeholder


def test_small_batch_patch_casts_intermediate_once(patches, vllm_modules):
    _, fused_op = vllm_modules
    anchor = (
        "            # EXPAND: walk n_tiles_per_program consecutive output-N tiles\n"
    )
    source = "            rank_vec = tl.sum(acc, axis=0)\n" + anchor
    updates = []
    kernel = SimpleNamespace(src=source)

    def update_source(updated):
        updates.append(updated)
        kernel.src = updated

    kernel._unsafe_update_src = update_source
    fused_op._fused_moe_lora_small_batch_kernel = SimpleNamespace(fn=kernel)

    patches.patch_small_batch_moe_lora_intermediate_dtype()
    patches.patch_small_batch_moe_lora_intermediate_dtype()

    cast = "            rank_vec = rank_vec.to(out_ptr.dtype.element_ty)\n"
    assert updates == [source.replace(anchor, cast + "\n" + anchor)]


def test_small_batch_patch_rejects_changed_kernel_source(patches, vllm_modules):
    _, fused_op = vllm_modules
    fused_op._fused_moe_lora_small_batch_kernel = SimpleNamespace(
        fn=SimpleNamespace(src="unexpected kernel source")
    )

    with pytest.raises(RuntimeError, match="Unsupported vLLM small-batch"):
        patches.patch_small_batch_moe_lora_intermediate_dtype()


def test_small_batch_patch_does_not_hide_active_triton_api_changes(
    patches, vllm_modules
):
    _, fused_op = vllm_modules
    fused_op._fused_moe_lora_small_batch_kernel = lambda: None

    with pytest.raises(AttributeError, match="fn"):
        patches.patch_small_batch_moe_lora_intermediate_dtype()
