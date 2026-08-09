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


def test_dsv4_lora_support_declares_vllm_025_manager_protocol(monkeypatch) -> None:
    patches = _load_dsv4_patches_module()

    class FakeDeepseekV4ForCausalLM:
        pass

    manager_patches: list[type] = []
    monkeypatch.setattr(
        patches.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(DeepseekV4ForCausalLM=FakeDeepseekV4ForCausalLM)
            if name == "vllm.models.deepseek_v4.nvidia.model"
            else None
        ),
    )
    monkeypatch.setattr(
        patches,
        "_patch_dsv4_lora_manager_indexer_skip",
        manager_patches.append,
    )

    patches.patch_dsv4_lora_support()

    assert getattr(FakeDeepseekV4ForCausalLM, "supports_lora") is True
    assert getattr(FakeDeepseekV4ForCausalLM, "lora_manager") is None
    assert manager_patches == [FakeDeepseekV4ForCausalLM]


def test_dsv4_fp8_o_proj_normalizes_rope_cache_once() -> None:
    patches = _load_dsv4_patches_module()
    rotary_emb = SimpleNamespace(cos_sin_cache=torch.ones(4, 8, dtype=torch.bfloat16))

    cache = patches._dsv4_fp32_cos_sin_cache(rotary_emb)

    assert cache.dtype == torch.float32
    assert rotary_emb.cos_sin_cache is cache
    assert patches._dsv4_fp32_cos_sin_cache(rotary_emb) is cache


def test_dsv4_layerwise_reload_restores_merged_column_metadata() -> None:
    patches = _load_dsv4_patches_module()
    param = torch.nn.Parameter(torch.empty(3, 4), requires_grad=False)

    patches._restore_merged_column_output_dim(param)

    assert getattr(param, "output_dim") == 0


def test_dsv4_layerwise_reload_restores_direct_linear_shard_metadata() -> None:
    patches = _load_dsv4_patches_module()
    param = torch.nn.Parameter(torch.empty(3, 4), requires_grad=False)

    patches._restore_linear_shard_dim(param, torch.empty(3, 8))

    assert getattr(param, "input_dim") == 1
    assert getattr(param, "output_dim") == 1


def test_dsv4_bmm_weight_selects_tp_rows_before_grouping() -> None:
    patches = _load_dsv4_patches_module()
    param = torch.nn.Parameter(torch.empty(2, 3, 4), requires_grad=False)
    loaded = torch.arange(48).view(12, 4)

    reshaped = patches._reshape_dsv4_bmm_weight(
        "layers.0.attn.wo_a.weight", param, loaded, tp_rank=1, tp_size=2
    )

    assert reshaped.shape == param.shape
    assert torch.equal(reshaped.flatten(0, 1), loaded[6:])


def test_dsv4_expert_delta_names_use_checkpoint_projection_names() -> None:
    patches = _load_dsv4_patches_module()

    assert (
        patches._dsv4_expert_checkpoint_name("layers.0.ffn.experts.3.down_proj.weight")
        == "layers.0.ffn.experts.3.w2.weight"
    )
    assert (
        patches._dsv4_expert_checkpoint_name(
            "layers.0.ffn.shared_experts.gate_proj.weight"
        )
        == "layers.0.ffn.shared_experts.w1.weight"
    )
    assert (
        patches._dsv4_expert_checkpoint_name(
            "layers.0.ffn.shared_experts.down_proj.weight"
        )
        == "layers.0.ffn.shared_experts.down_proj.weight"
    )


def test_dsv4_fp8_weight_is_linked_to_its_block_scale() -> None:
    patches = _load_dsv4_patches_module()
    param = torch.nn.Parameter(
        torch.empty(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    scale = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)

    patches._attach_block_fp8_scale(
        param,
        "layers.0.attn.q_proj.weight",
        {"layers.0.attn.q_proj.weight_scale_inv": scale},
        (2, 2),
    )

    assert getattr(param, "_art_block_fp8_scale") is scale
    assert getattr(param, "_art_block_fp8_size") == (2, 2)


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
