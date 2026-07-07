from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch


def _load_dsv4_patches_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_runtime/src/art_vllm_runtime/dsv4_patches.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_art_vllm_runtime_dsv4_patches",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLoraLinear:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lora_a_stacked = ()
        self.lora_b_stacked = ()
        self.punica_wrapper = SimpleNamespace(no_lora=False, indices_len=[2])


def test_dsv4_compressor_patch_uses_fp32_compressor_helper(monkeypatch) -> None:
    patches = _load_dsv4_patches_module()
    calls: list[tuple[str, object, object]] = []

    class FakeWrapper:
        compressor = SimpleNamespace(fused_wkv_wgate=_FakeLoraLinear("compressor"))
        indexer = SimpleNamespace(
            compressor=SimpleNamespace(fused_wkv_wgate=_FakeLoraLinear("indexer"))
        )

        def attn_gemm_parallel_execute(self, hidden):
            return "qr", "kv", "indexer_kv", "weights"

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            raise AssertionError("not used")

    fake_dsv4_attn = SimpleNamespace(
        DeepseekV4MultiHeadLatentAttentionWrapper=FakeWrapper
    )
    monkeypatch.setattr(
        patches.importlib,
        "import_module",
        lambda name: (
            fake_dsv4_attn
            if name == "vllm.model_executor.layers.deepseek_v4_attention"
            else __import__(name)
        ),
    )
    monkeypatch.setattr(patches, "_register_dsv4_inv_rope_lora_input_op", lambda: None)
    monkeypatch.setattr(
        patches, "_register_dsv4_lora_expand_fp32_output_op", lambda: None
    )

    def apply_compressor_lora(module, hidden, output):
        calls.append((module.name, hidden, output))
        return f"{output}_fp32_lora"

    monkeypatch.setattr(
        patches,
        "_apply_dsv4_compressor_lora_to_existing_output",
        apply_compressor_lora,
    )

    patches.patch_dsv4_fast_path_lora()
    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        FakeWrapper().attn_gemm_parallel_execute("hidden")
    )

    assert (qr_kv, kv_score, indexer_kv_score, indexer_weights) == (
        "qr",
        "kv_fp32_lora",
        "indexer_kv_fp32_lora",
        "weights",
    )
    assert calls == [
        ("compressor", "hidden", "kv"),
        ("indexer", "hidden", "indexer_kv"),
    ]


def test_dsv4_current_vllm_attention_patch_targets_split_modules(monkeypatch) -> None:
    patches = _load_dsv4_patches_module()
    calls: list[tuple[str, object, object]] = []
    o_proj_calls: list[tuple[object, object, object]] = []

    class FakeAttention:
        compressor = SimpleNamespace(fused_wkv_wgate=_FakeLoraLinear("compressor"))
        indexer = SimpleNamespace(
            compressor=SimpleNamespace(fused_wkv_wgate=_FakeLoraLinear("indexer"))
        )

        def attn_gemm_parallel_execute(self, hidden):
            return "qr", "kv", "indexer_kv", "weights"

    class FakeFlashMLA:
        rotary_emb = SimpleNamespace(cos_sin_cache="cache")
        wo_a = "wo_a"
        wo_b = "wo_b"
        n_local_groups = 2
        n_local_heads = 4
        nope_head_dim = 448
        rope_head_dim = 64
        o_lora_rank = 512
        _einsum_recipe = (1, 128, 128)
        _tma_aligned_scales = False

        def _o_proj(self, o, positions):
            raise AssertionError("unpatched")

    class FakeFlashInfer(FakeFlashMLA): ...

    fake_attention_mod = types.ModuleType("vllm.models.deepseek_v4.attention")
    setattr(fake_attention_mod, "DeepseekV4Attention", FakeAttention)
    fake_o_proj_mod = types.ModuleType("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    fake_flashmla_mod = types.ModuleType("vllm.models.deepseek_v4.nvidia.flashmla")
    setattr(fake_flashmla_mod, "DeepseekV4FlashMLAAttention", FakeFlashMLA)
    fake_flashinfer_mod = types.ModuleType(
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse"
    )
    setattr(fake_flashinfer_mod, "DeepseekV4FlashInferMLAAttention", FakeFlashInfer)
    for name, module in {
        "vllm": types.ModuleType("vllm"),
        "vllm.models": types.ModuleType("vllm.models"),
        "vllm.models.deepseek_v4": types.ModuleType("vllm.models.deepseek_v4"),
        "vllm.models.deepseek_v4.nvidia": types.ModuleType(
            "vllm.models.deepseek_v4.nvidia"
        ),
        "vllm.models.deepseek_v4.nvidia.ops": types.ModuleType(
            "vllm.models.deepseek_v4.nvidia.ops"
        ),
        "vllm.models.deepseek_v4.attention": fake_attention_mod,
        "vllm.models.deepseek_v4.nvidia.ops.o_proj": fake_o_proj_mod,
        "vllm.models.deepseek_v4.nvidia.flashmla": fake_flashmla_mod,
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse": fake_flashinfer_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(patches, "_register_dsv4_inv_rope_lora_input_op", lambda: None)
    monkeypatch.setattr(
        patches, "_register_dsv4_lora_expand_fp32_output_op", lambda: None
    )

    def apply_compressor_lora(module, hidden, output):
        calls.append((module.name, hidden, output))
        return f"{output}_fp32_lora"

    def apply_o_proj(o_proj_mod, o, positions, cos_sin_cache, *args, **kwargs):
        o_proj_calls.append((o_proj_mod, o, positions))
        assert cos_sin_cache == "cache"
        return "o_proj_out"

    monkeypatch.setattr(
        patches,
        "_apply_dsv4_compressor_lora_to_existing_output",
        apply_compressor_lora,
    )
    monkeypatch.setattr(
        patches,
        "_dsv4_deep_gemm_fp8_o_proj_with_lora",
        apply_o_proj,
    )

    patches.patch_dsv4_fast_path_lora()

    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        FakeAttention().attn_gemm_parallel_execute("hidden")
    )

    assert (qr_kv, kv_score, indexer_kv_score, indexer_weights) == (
        "qr",
        "kv_fp32_lora",
        "indexer_kv_fp32_lora",
        "weights",
    )
    assert calls == [
        ("compressor", "hidden", "kv"),
        ("indexer", "hidden", "indexer_kv"),
    ]
    assert FakeFlashMLA()._o_proj("o", "pos") == "o_proj_out"
    assert FakeFlashInfer()._o_proj("o", "pos") == "o_proj_out"
    assert o_proj_calls == [(fake_o_proj_mod, "o", "pos")] * 2


def test_dsv4_compressor_helper_uses_punica_metadata_without_full_batch_lora(
    monkeypatch,
) -> None:
    patches = _load_dsv4_patches_module()
    fake_vllm = types.ModuleType("vllm")
    fake_platforms = types.ModuleType("vllm.platforms")
    setattr(
        fake_platforms,
        "current_platform",
        SimpleNamespace(can_update_inplace=lambda: True),
    )
    setattr(fake_vllm, "platforms", fake_platforms)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        fake_platforms,
    )
    monkeypatch.setattr(
        patches, "_register_dsv4_lora_expand_fp32_output_op", lambda: None
    )

    expand_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_expand(inputs, lora_b, output, *args):
        offset = args[-1]
        width = lora_b.shape[2]
        expand_calls.append((tuple(lora_b.shape), offset))
        output[:, offset : offset + width].add_(inputs.sum(dim=-1, keepdim=True))

    monkeypatch.setattr(
        torch.ops.vllm,
        "art_dsv4_lora_expand_fp32_output",
        fake_expand,
        raising=False,
    )

    class FakeTokenMappingMeta:
        def meta_args(self, token_count, specialize_active_lora):
            assert token_count == 4
            assert specialize_active_lora is False
            return (
                torch.tensor([0, 0, 1, 1], dtype=torch.int32),
                torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                torch.tensor([2, 2, 0], dtype=torch.int32),
                torch.tensor([0, 2, 4, 4], dtype=torch.int32),
                torch.tensor([0, 1, -1], dtype=torch.int32),
                torch.tensor([False]),
                torch.tensor([2], dtype=torch.int32),
            )

    class FakeWrapper:
        no_lora = False
        indices_len = [4]
        lora_config = SimpleNamespace(specialize_active_lora=False)
        token_mapping_meta = FakeTokenMappingMeta()

        def add_shrink(self, buffers, x, lora_a_stacked, scale):
            assert buffers.shape == (2, 4, 2)
            assert scale == 1.0
            buffers.copy_(
                torch.arange(buffers.numel(), dtype=torch.float32).view_as(buffers)
            )
            return None

    module = SimpleNamespace(
        lora_a_stacked=(
            torch.zeros(2, 1, 2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 1, 2, 4, dtype=torch.bfloat16),
        ),
        lora_b_stacked=(
            torch.zeros(2, 1, 3, 2, dtype=torch.bfloat16),
            torch.zeros(2, 1, 5, 2, dtype=torch.bfloat16),
        ),
        output_slices=(3, 5),
        punica_wrapper=FakeWrapper(),
        tp_size=1,
    )
    output = torch.zeros(4, 8, dtype=torch.float32)

    result = patches._apply_dsv4_compressor_lora_to_existing_output(
        module, torch.zeros(4, 4, dtype=torch.bfloat16), output
    )

    assert result is output
    assert expand_calls == [((2, 1, 3, 2), 0), ((2, 1, 5, 2), 3)]
    assert output.abs().sum() > 0
