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


def _set_unsharded(param: torch.nn.Parameter) -> None:
    param.lora_tp_sharded = False  # type: ignore[attr-defined]
    param.lora_shard_domain = "expert_tp"  # type: ignore[attr-defined]


def test_gemma4_handler_pads_provider_moe_ffn_size() -> None:
    provider = SimpleNamespace(num_moe_experts=8, moe_ffn_hidden_size=4)

    _configure_gemma4_moe_internal_padding(provider)

    assert getattr(provider, _GEMMA4_LOGICAL_MOE_FFN_ATTR) == 4
    assert provider.moe_ffn_hidden_size == 128


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
