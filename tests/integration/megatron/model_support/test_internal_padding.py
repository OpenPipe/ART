from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from art.megatron.model_support.handlers.gemma4 import (
    _GEMMA4_LOGICAL_MOE_FFN_ATTR,
    GEMMA4_MOE_HANDLER,
    _configure_gemma4_moe_internal_padding,
)
from art.megatron.model_support.handlers.gpt_oss import (
    _GPT_OSS_INTERNAL_HIDDEN_ATTR,
    _GPT_OSS_INTERNAL_MOE_FFN_ATTR,
    _GPT_OSS_LOGICAL_HIDDEN_ATTR,
    _GPT_OSS_LOGICAL_MOE_FFN_ATTR,
    GPT_OSS_MOE_HANDLER,
    _configure_gpt_oss_moe_internal_padding,
)


def _adapter_config(tmp_path: Path, *, logical: int) -> dict[str, Any]:
    base_model = tmp_path / "gemma4_moe"
    base_model.mkdir()
    (base_model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "enable_moe_block": True,
                    "moe_intermediate_size": logical,
                }
            }
        ),
        encoding="utf-8",
    )
    return {"base_model_name_or_path": str(base_model), "r": 2}


def _gpt_oss_adapter_config(
    tmp_path: Path,
    *,
    hidden: int,
    ffn: int,
) -> dict[str, Any]:
    base_model = tmp_path / "gpt_oss_moe"
    base_model.mkdir()
    (base_model / "config.json").write_text(
        json.dumps({"hidden_size": hidden, "intermediate_size": ffn}),
        encoding="utf-8",
    )
    return {"base_model_name_or_path": str(base_model), "r": 2}


def _set_unsharded(param: torch.nn.Parameter) -> None:
    param.lora_tp_sharded = False  # type: ignore[attr-defined]
    param.lora_shard_domain = "expert_tp"  # type: ignore[attr-defined]


def test_gemma4_handler_pads_provider_moe_ffn_size() -> None:
    provider = SimpleNamespace(num_moe_experts=8, moe_ffn_hidden_size=4)

    _configure_gemma4_moe_internal_padding(provider)

    assert getattr(provider, _GEMMA4_LOGICAL_MOE_FFN_ATTR) == 4
    assert provider.moe_ffn_hidden_size == 128


def test_gpt_oss_handler_marks_internal_moe_padding_sizes() -> None:
    provider = SimpleNamespace(num_moe_experts=8, hidden_size=4, moe_ffn_hidden_size=6)

    _configure_gpt_oss_moe_internal_padding(provider)

    assert getattr(provider, _GPT_OSS_LOGICAL_HIDDEN_ATTR) == 4
    assert getattr(provider, _GPT_OSS_INTERNAL_HIDDEN_ATTR) == 128
    assert getattr(provider, _GPT_OSS_LOGICAL_MOE_FFN_ATTR) == 6
    assert getattr(provider, _GPT_OSS_INTERNAL_MOE_FFN_ATTR) == 128
    assert provider.hidden_size == 4
    assert provider.moe_ffn_hidden_size == 6


def test_gemma4_handler_trims_and_restores_external_moe_lora_padding(
    tmp_path: Path,
) -> None:
    logical = 4
    internal = 128
    rank = 2
    experts = 2
    prefix = "base_model.model.model.layers.0.mlp.experts"
    adapter_config = _adapter_config(tmp_path, logical=logical)
    tensors: dict[str, torch.Tensor] = {}

    for expert in range(experts):
        gate_up_b = torch.arange(2 * internal * rank, dtype=torch.float32).reshape(
            2 * internal, rank
        )
        gate_up_b[logical:internal] = -1000 - expert
        gate_up_b[internal + logical :] = -2000 - expert
        down_a = torch.arange(rank * internal, dtype=torch.float32).reshape(
            rank, internal
        )
        down_a[:, logical:] = -3000 - expert
        tensors.update(
            {
                f"{prefix}.{expert}.gate_up_proj.lora_A.weight": torch.full(
                    (rank, 3),
                    float(expert + 1),
                ),
                f"{prefix}.{expert}.gate_up_proj.lora_B.weight": gate_up_b,
                f"{prefix}.{expert}.down_proj.lora_A.weight": down_a,
                f"{prefix}.{expert}.down_proj.lora_B.weight": torch.full(
                    (5, rank),
                    float(expert + 10),
                ),
            }
        )

    vllm_tensors, _config = GEMMA4_MOE_HANDLER.to_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )

    vllm_prefix = "base_model.model.model.layers.0.moe.experts"
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"].shape == (
        2 * logical,
        rank * experts,
    )
    assert vllm_tensors[f"{vllm_prefix}.lora_A.weight"].shape == (
        rank * experts,
        logical,
    )
    assert not torch.any(vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"] < 0)
    assert not torch.any(vllm_tensors[f"{vllm_prefix}.lora_A.weight"] < 0)

    restored = GEMMA4_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=adapter_config,
    )

    for expert in range(experts):
        original_gate_up_b = tensors[f"{prefix}.{expert}.gate_up_proj.lora_B.weight"]
        restored_gate_up_b = restored[f"{prefix}.{expert}.gate_up_proj.lora_B.weight"]
        assert restored_gate_up_b.shape == (2 * internal, rank)
        assert torch.equal(restored_gate_up_b[:logical], original_gate_up_b[:logical])
        assert torch.equal(
            restored_gate_up_b[internal : internal + logical],
            original_gate_up_b[internal : internal + logical],
        )
        assert torch.count_nonzero(restored_gate_up_b[logical:internal]) == 0
        assert torch.count_nonzero(restored_gate_up_b[internal + logical :]) == 0

        original_down_a = tensors[f"{prefix}.{expert}.down_proj.lora_A.weight"]
        restored_down_a = restored[f"{prefix}.{expert}.down_proj.lora_A.weight"]
        assert restored_down_a.shape == (rank, internal)
        assert torch.equal(restored_down_a[:, :logical], original_down_a[:, :logical])
        assert torch.count_nonzero(restored_down_a[:, logical:]) == 0


def test_gpt_oss_handler_trims_and_restores_external_moe_lora_padding(
    tmp_path: Path,
) -> None:
    logical_hidden = 4
    internal_hidden = 128
    logical_ffn = 6
    internal_ffn = 128
    rank = 2
    experts = 2
    prefix = "base_model.model.model.layers.0.mlp.experts"
    adapter_config = _gpt_oss_adapter_config(
        tmp_path,
        hidden=logical_hidden,
        ffn=logical_ffn,
    )
    tensors: dict[str, torch.Tensor] = {}

    for expert in range(experts):
        gate_up_a = torch.arange(rank * internal_hidden, dtype=torch.float32).reshape(
            rank,
            internal_hidden,
        )
        gate_up_a[:, logical_hidden:] = -1000 - expert
        gate_up_b = torch.arange(2 * internal_ffn * rank, dtype=torch.float32).reshape(
            2 * internal_ffn,
            rank,
        )
        gate_up_b[logical_ffn:internal_ffn] = -2000 - expert
        gate_up_b[internal_ffn + logical_ffn :] = -3000 - expert
        down_a = torch.arange(rank * internal_ffn, dtype=torch.float32).reshape(
            rank,
            internal_ffn,
        )
        down_a[:, logical_ffn:] = -4000 - expert
        down_b = torch.arange(internal_hidden * rank, dtype=torch.float32).reshape(
            internal_hidden,
            rank,
        )
        down_b[logical_hidden:] = -5000 - expert
        tensors.update(
            {
                f"{prefix}.{expert}.gate_up_proj.lora_A.weight": gate_up_a,
                f"{prefix}.{expert}.gate_up_proj.lora_B.weight": gate_up_b,
                f"{prefix}.{expert}.down_proj.lora_A.weight": down_a,
                f"{prefix}.{expert}.down_proj.lora_B.weight": down_b,
            }
        )

    vllm_tensors, _config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )

    vllm_prefix = "base_model.model.model.layers.0.mlp.experts"
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_A.weight"].shape == (
        experts * rank,
        logical_hidden,
    )
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"].shape == (
        2 * logical_ffn,
        experts * rank,
    )
    assert vllm_tensors[f"{vllm_prefix}.lora_A.weight"].shape == (
        experts * rank,
        logical_ffn,
    )
    assert vllm_tensors[f"{vllm_prefix}.lora_B.weight"].shape == (
        logical_hidden,
        experts * rank,
    )
    assert all(not torch.any(tensor < 0) for tensor in vllm_tensors.values())

    restored = GPT_OSS_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=adapter_config,
    )

    for expert in range(experts):
        restored_gate_up_a = restored[f"{prefix}.{expert}.gate_up_proj.lora_A.weight"]
        restored_gate_up_b = restored[f"{prefix}.{expert}.gate_up_proj.lora_B.weight"]
        restored_down_a = restored[f"{prefix}.{expert}.down_proj.lora_A.weight"]
        restored_down_b = restored[f"{prefix}.{expert}.down_proj.lora_B.weight"]
        assert restored_gate_up_a.shape == (rank, internal_hidden)
        assert restored_gate_up_b.shape == (2 * internal_ffn, rank)
        assert restored_down_a.shape == (rank, internal_ffn)
        assert restored_down_b.shape == (internal_hidden, rank)
        assert torch.count_nonzero(restored_gate_up_a[:, logical_hidden:]) == 0
        assert torch.count_nonzero(restored_gate_up_b[logical_ffn:internal_ffn]) == 0
        assert (
            torch.count_nonzero(restored_gate_up_b[internal_ffn + logical_ffn :]) == 0
        )
        assert torch.count_nonzero(restored_down_a[:, logical_ffn:]) == 0
        assert torch.count_nonzero(restored_down_b[logical_hidden:]) == 0


def test_gpt_oss_handler_preserves_logical_fused_vllm_moe_lora(
    tmp_path: Path,
) -> None:
    logical_hidden = 4
    logical_ffn = 6
    rank = 2
    experts = 3
    adapter_config = _gpt_oss_adapter_config(
        tmp_path,
        hidden=logical_hidden,
        ffn=logical_ffn,
    )
    prefix = "base_model.model.model.layers.0.mlp.experts"
    tensors = {
        f"{prefix}.base_layer.lora_A.weight": torch.arange(
            experts * rank * logical_hidden,
            dtype=torch.float32,
        ).reshape(experts * rank, logical_hidden),
        f"{prefix}.base_layer.lora_B.weight": torch.arange(
            2 * logical_ffn * experts * rank,
            dtype=torch.float32,
        ).reshape(2 * logical_ffn, experts * rank),
        f"{prefix}.lora_A.weight": torch.arange(
            experts * rank * logical_ffn,
            dtype=torch.float32,
        ).reshape(experts * rank, logical_ffn),
        f"{prefix}.lora_B.weight": torch.arange(
            logical_hidden * experts * rank,
            dtype=torch.float32,
        ).reshape(logical_hidden, experts * rank),
    }

    vllm_tensors, _config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )

    assert set(vllm_tensors) == set(tensors)
    for key, tensor in tensors.items():
        assert torch.equal(vllm_tensors[key], tensor)


def test_gemma4_handler_preserves_logical_fused_vllm_moe_lora(
    tmp_path: Path,
) -> None:
    logical = 4
    rank = 2
    experts = 3
    adapter_config = _adapter_config(tmp_path, logical=logical)
    prefix = "base_model.model.model.layers.0.experts"
    tensors = {
        f"{prefix}.base_layer.lora_A.weight": torch.arange(
            experts * rank * 5,
            dtype=torch.float32,
        ).reshape(experts * rank, 5),
        f"{prefix}.base_layer.lora_B.weight": torch.arange(
            2 * logical * experts * rank,
            dtype=torch.float32,
        ).reshape(2 * logical, experts * rank),
        f"{prefix}.lora_A.weight": torch.arange(
            experts * rank * logical,
            dtype=torch.float32,
        ).reshape(experts * rank, logical),
        f"{prefix}.lora_B.weight": torch.arange(
            5 * experts * rank,
            dtype=torch.float32,
        ).reshape(5, experts * rank),
    }

    vllm_tensors, _config = GEMMA4_MOE_HANDLER.to_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )

    vllm_prefix = "base_model.model.model.layers.0.moe.experts"
    assert set(vllm_tensors) == {
        f"{vllm_prefix}.base_layer.lora_A.weight",
        f"{vllm_prefix}.base_layer.lora_B.weight",
        f"{vllm_prefix}.lora_A.weight",
        f"{vllm_prefix}.lora_B.weight",
    }
    for key, tensor in tensors.items():
        converted_key = key.replace(".experts", ".moe.experts")
        assert torch.equal(vllm_tensors[converted_key], tensor)


def test_gemma4_handler_trims_art_packed_fused_moe_lora(
    tmp_path: Path,
) -> None:
    logical = 4
    internal = 128
    rank = 2
    experts = 3
    adapter_config = _adapter_config(tmp_path, logical=logical)
    prefix = "base_model.model.model.layers.0.mlp.experts"
    gate_up_b = torch.arange(
        2 * internal * experts * rank,
        dtype=torch.float32,
    ).reshape(2 * internal, experts * rank)
    gate_up_b[logical:internal] = -1
    gate_up_b[internal + logical :] = -2
    down_a = torch.arange(
        experts * rank * internal,
        dtype=torch.float32,
    ).reshape(experts * rank, internal)
    down_a[:, logical:] = -3
    tensors = {
        f"{prefix}.base_layer.lora_A.weight": torch.ones(experts * rank, 5),
        f"{prefix}.base_layer.lora_B.weight": gate_up_b,
        f"{prefix}.lora_A.weight": down_a,
        f"{prefix}.lora_B.weight": torch.ones(5, experts * rank),
    }

    vllm_tensors, _config = GEMMA4_MOE_HANDLER.to_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )

    vllm_prefix = "base_model.model.model.layers.0.moe.experts"
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"].shape == (
        2 * logical,
        experts * rank,
    )
    assert vllm_tensors[f"{vllm_prefix}.lora_A.weight"].shape == (
        experts * rank,
        logical,
    )
    assert not torch.any(vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"] < 0)
    assert not torch.any(vllm_tensors[f"{vllm_prefix}.lora_A.weight"] < 0)


class _FakeLora(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        *,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.adapter_model_prefix = adapter_model_prefix
        self.A_T = torch.nn.Parameter(torch.ones(a_shape))
        self.B_T = torch.nn.Parameter(torch.ones(b_shape))
        _set_unsharded(self.A_T)
        _set_unsharded(self.B_T)
        self.A_T.grad = torch.ones_like(self.A_T)
        self.B_T.grad = torch.ones_like(self.B_T)
        self.A_T.main_grad = torch.ones_like(self.A_T)  # type: ignore[attr-defined]
        self.B_T.main_grad = torch.ones_like(self.B_T)  # type: ignore[attr-defined]


class _FakeChunk(torch.nn.Module):
    def __init__(self, *, logical: int, internal: int, rank: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_moe_experts=2,
            moe_ffn_hidden_size=internal,
            **{_GEMMA4_LOGICAL_MOE_FFN_ATTR: logical},
        )
        self.gate_up = _FakeLora(
            "base_model.model.model.layers.0.mlp.experts.{expert}.gate_up_proj",
            a_shape=(2, 3, rank),
            b_shape=(2, rank, 2 * internal),
        )
        self.down = _FakeLora(
            "base_model.model.model.layers.0.mlp.experts.{expert}.down_proj",
            a_shape=(2, internal, rank),
            b_shape=(2, rank, 5),
        )


class _FakeGptOssChunk(torch.nn.Module):
    def __init__(
        self,
        *,
        logical_hidden: int,
        internal_hidden: int,
        logical_ffn: int,
        internal_ffn: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_moe_experts=2,
            hidden_size=logical_hidden,
            moe_ffn_hidden_size=logical_ffn,
            **{
                _GPT_OSS_LOGICAL_HIDDEN_ATTR: logical_hidden,
                _GPT_OSS_INTERNAL_HIDDEN_ATTR: internal_hidden,
                _GPT_OSS_LOGICAL_MOE_FFN_ATTR: logical_ffn,
                _GPT_OSS_INTERNAL_MOE_FFN_ATTR: internal_ffn,
            },
        )
        self.gate_up = _FakeLora(
            "base_model.model.model.layers.0.mlp.experts.{expert}.gate_up_proj",
            a_shape=(2, internal_hidden, rank),
            b_shape=(2, rank, 2 * internal_ffn),
        )
        self.down = _FakeLora(
            "base_model.model.model.layers.0.mlp.experts.{expert}.down_proj",
            a_shape=(2, internal_ffn, rank),
            b_shape=(2, rank, internal_hidden),
        )


def test_gemma4_handler_masks_internal_padding_params_and_grads() -> None:
    logical = 4
    internal = 128
    chunk = _FakeChunk(logical=logical, internal=internal, rank=2)

    GEMMA4_MOE_HANDLER.zero_internal_padding_grads([chunk])

    gate_up_b_grad = cast(torch.Tensor, chunk.gate_up.B_T.grad)
    down_a_grad = cast(torch.Tensor, chunk.down.A_T.grad)
    down_a_main_grad = cast(torch.Tensor, getattr(chunk.down.A_T, "main_grad"))
    assert torch.all(gate_up_b_grad[..., :logical] == 1)
    assert torch.all(gate_up_b_grad[..., internal : internal + logical] == 1)
    assert torch.count_nonzero(gate_up_b_grad[..., logical:internal]) == 0
    assert torch.count_nonzero(gate_up_b_grad[..., internal + logical :]) == 0
    assert torch.count_nonzero(down_a_grad[:, logical:, :]) == 0
    assert torch.count_nonzero(down_a_main_grad[:, logical:, :]) == 0
    assert torch.count_nonzero(chunk.gate_up.B_T) > 0
    assert torch.count_nonzero(chunk.down.A_T) > 0

    GEMMA4_MOE_HANDLER.zero_internal_padding_params([chunk])

    assert torch.count_nonzero(chunk.gate_up.B_T[..., logical:internal]) == 0
    assert torch.count_nonzero(chunk.gate_up.B_T[..., internal + logical :]) == 0
    assert torch.count_nonzero(chunk.down.A_T[:, logical:, :]) == 0
    assert torch.all(chunk.gate_up.B_T[..., :logical] == 1)
    assert torch.all(chunk.down.A_T[:, :logical, :] == 1)


def test_gpt_oss_handler_masks_internal_padding_params_and_grads() -> None:
    logical_hidden = 4
    internal_hidden = 128
    logical_ffn = 6
    internal_ffn = 128
    chunk = _FakeGptOssChunk(
        logical_hidden=logical_hidden,
        internal_hidden=internal_hidden,
        logical_ffn=logical_ffn,
        internal_ffn=internal_ffn,
        rank=2,
    )

    GPT_OSS_MOE_HANDLER.zero_internal_padding_grads([chunk])

    gate_up_a_grad = cast(torch.Tensor, chunk.gate_up.A_T.grad)
    gate_up_b_grad = cast(torch.Tensor, chunk.gate_up.B_T.grad)
    down_a_grad = cast(torch.Tensor, chunk.down.A_T.grad)
    down_b_grad = cast(torch.Tensor, chunk.down.B_T.grad)
    assert torch.count_nonzero(gate_up_a_grad[:, logical_hidden:, :]) == 0
    assert torch.count_nonzero(gate_up_b_grad[..., logical_ffn:internal_ffn]) == 0
    assert torch.count_nonzero(gate_up_b_grad[..., internal_ffn + logical_ffn :]) == 0
    assert torch.count_nonzero(down_a_grad[:, logical_ffn:, :]) == 0
    assert torch.count_nonzero(down_b_grad[..., logical_hidden:]) == 0
    assert torch.count_nonzero(chunk.gate_up.A_T) > 0
    assert torch.count_nonzero(chunk.gate_up.B_T) > 0
    assert torch.count_nonzero(chunk.down.A_T) > 0
    assert torch.count_nonzero(chunk.down.B_T) > 0

    GPT_OSS_MOE_HANDLER.zero_internal_padding_params([chunk])

    assert torch.count_nonzero(chunk.gate_up.A_T[:, logical_hidden:, :]) == 0
    assert torch.count_nonzero(chunk.gate_up.B_T[..., logical_ffn:internal_ffn]) == 0
    assert (
        torch.count_nonzero(chunk.gate_up.B_T[..., internal_ffn + logical_ffn :]) == 0
    )
    assert torch.count_nonzero(chunk.down.A_T[:, logical_ffn:, :]) == 0
    assert torch.count_nonzero(chunk.down.B_T[..., logical_hidden:]) == 0
    assert torch.all(chunk.gate_up.A_T[:, :logical_hidden, :] == 1)
    assert torch.all(chunk.gate_up.B_T[..., :logical_ffn] == 1)
    assert torch.all(
        chunk.gate_up.B_T[..., internal_ffn : internal_ffn + logical_ffn] == 1
    )
    assert torch.all(chunk.down.A_T[:, :logical_ffn, :] == 1)
    assert torch.all(chunk.down.B_T[..., :logical_hidden] == 1)


def test_gemma4_handler_canonicalizes_loaded_lora_state_padding() -> None:
    logical = 4
    internal = 128
    chunk = _FakeChunk(logical=logical, internal=internal, rank=2)
    prefix = "base_model.model.model.layers.0.mlp.experts"
    state = {
        f"{prefix}.0.gate_up_proj.lora_B.weight": torch.ones(2 * internal, 2),
        f"{prefix}.0.down_proj.lora_A.weight": torch.ones(2, internal),
        f"{prefix}.base_layer.lora_B.weight": torch.ones(2 * internal, 2),
        f"{prefix}.lora_A.weight": torch.ones(2, internal),
        f"{prefix}.0.gate_up_proj.lora_A.weight": torch.ones(2, 3),
    }

    canonical = GEMMA4_MOE_HANDLER.canonicalize_loaded_lora_state(state, [chunk])

    gate_up = canonical[f"{prefix}.0.gate_up_proj.lora_B.weight"]
    down = canonical[f"{prefix}.0.down_proj.lora_A.weight"]
    packed_gate_up = canonical[f"{prefix}.base_layer.lora_B.weight"]
    packed_down = canonical[f"{prefix}.lora_A.weight"]
    assert torch.count_nonzero(gate_up[logical:internal]) == 0
    assert torch.count_nonzero(gate_up[internal + logical :]) == 0
    assert torch.count_nonzero(down[:, logical:]) == 0
    assert torch.count_nonzero(packed_gate_up[logical:internal]) == 0
    assert torch.count_nonzero(packed_gate_up[internal + logical :]) == 0
    assert torch.count_nonzero(packed_down[:, logical:]) == 0
    assert torch.all(gate_up[:logical] == 1)
    assert torch.all(gate_up[internal : internal + logical] == 1)
    assert torch.all(down[:, :logical] == 1)
    assert torch.equal(
        canonical[f"{prefix}.0.gate_up_proj.lora_A.weight"],
        state[f"{prefix}.0.gate_up_proj.lora_A.weight"],
    )


def test_gpt_oss_handler_canonicalizes_loaded_lora_state_padding() -> None:
    logical_hidden = 4
    internal_hidden = 128
    logical_ffn = 6
    internal_ffn = 128
    chunk = _FakeGptOssChunk(
        logical_hidden=logical_hidden,
        internal_hidden=internal_hidden,
        logical_ffn=logical_ffn,
        internal_ffn=internal_ffn,
        rank=2,
    )
    prefix = "base_model.model.model.layers.0.mlp.experts"
    state = {
        f"{prefix}.0.gate_up_proj.lora_A.weight": torch.ones(2, internal_hidden),
        f"{prefix}.0.gate_up_proj.lora_B.weight": torch.ones(2 * internal_ffn, 2),
        f"{prefix}.0.down_proj.lora_A.weight": torch.ones(2, internal_ffn),
        f"{prefix}.0.down_proj.lora_B.weight": torch.ones(internal_hidden, 2),
        f"{prefix}.base_layer.lora_A.weight": torch.ones(2, internal_hidden),
        f"{prefix}.base_layer.lora_B.weight": torch.ones(2 * internal_ffn, 2),
        f"{prefix}.lora_A.weight": torch.ones(2, internal_ffn),
        f"{prefix}.lora_B.weight": torch.ones(internal_hidden, 2),
    }

    canonical = GPT_OSS_MOE_HANDLER.canonicalize_loaded_lora_state(state, [chunk])

    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.0.gate_up_proj.lora_A.weight"][:, logical_hidden:]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.0.gate_up_proj.lora_B.weight"][
                logical_ffn:internal_ffn
            ]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.0.gate_up_proj.lora_B.weight"][
                internal_ffn + logical_ffn :
            ]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.0.down_proj.lora_A.weight"][:, logical_ffn:]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.0.down_proj.lora_B.weight"][logical_hidden:]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.base_layer.lora_A.weight"][:, logical_hidden:]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.base_layer.lora_B.weight"][logical_ffn:internal_ffn]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            canonical[f"{prefix}.base_layer.lora_B.weight"][
                internal_ffn + logical_ffn :
            ]
        )
        == 0
    )
    assert (
        torch.count_nonzero(canonical[f"{prefix}.lora_A.weight"][:, logical_ffn:]) == 0
    )
    assert (
        torch.count_nonzero(canonical[f"{prefix}.lora_B.weight"][logical_hidden:]) == 0
    )
