from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch


def _load_patches_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_runtime/src/art_vllm_runtime/patches.py"
    )
    spec = importlib.util.spec_from_file_location("_art_vllm_runtime_patches", path)
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
    patches = _load_patches_module()
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


def test_dsv4_compressor_helper_uses_punica_metadata_without_full_batch_lora(
    monkeypatch,
) -> None:
    patches = _load_patches_module()
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
