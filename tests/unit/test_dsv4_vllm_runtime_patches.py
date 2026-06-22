from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
    def __init__(self, name: str, calls: list[tuple[str, object, object]]) -> None:
        self.name = name
        self.calls = calls
        self.lora_a_stacked = ()
        self.lora_b_stacked = ()
        self.punica_wrapper = SimpleNamespace(no_lora=False, indices_len=[2])

    def _apply_lora_to_output(self, hidden, output):
        self.calls.append((self.name, hidden, output))
        return f"{output}_lora"


def test_dsv4_compressor_patch_uses_standard_lora_wrapper(monkeypatch) -> None:
    patches = _load_patches_module()
    calls: list[tuple[str, object, object]] = []

    class FakeWrapper:
        compressor = SimpleNamespace(
            fused_wkv_wgate=_FakeLoraLinear("compressor", calls)
        )
        indexer = SimpleNamespace(
            compressor=SimpleNamespace(
                fused_wkv_wgate=_FakeLoraLinear("indexer", calls)
            )
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

    patches.patch_dsv4_fast_path_lora()
    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        FakeWrapper().attn_gemm_parallel_execute("hidden")
    )

    assert (qr_kv, kv_score, indexer_kv_score, indexer_weights) == (
        "qr",
        "kv_lora",
        "indexer_kv_lora",
        "weights",
    )
    assert calls == [
        ("compressor", "hidden", "kv"),
        ("indexer", "hidden", "indexer_kv"),
    ]
