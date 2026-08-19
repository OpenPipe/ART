from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, cast

import pytest
from safetensors.torch import load_file, save_file
import torch

import art.megatron.lora as lora_module
from art.megatron.lora import LoRA
from art.megatron.model_support.handlers import QWEN3_5_MOE_HANDLER
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
    load_lora_tensors_for_megatron,
    normalize_lora_checkpoint_to_vllm,
)
from art.megatron.weights.lora_publish import save_vllm_lora_from_model

REPO_ROOT = Path(__file__).parents[1]
VLLM_PYTHON = REPO_ROOT / "vllm_runtime/.venv/bin/python"
GROUP_PREFIX = "base_model.model.model.layers.0.mlp.experts"
EXPERTS = 2
HIDDEN = 3
INTERMEDIATE = 4
RANK = 2


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        lora_module.ps,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: 0,
    )


def _config() -> dict[str, object]:
    return {
        "base_model_name_or_path": "Qwen/Qwen3.5-35B-A3B",
        "r": RANK,
        "lora_alpha": RANK,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 3,
        "moe_parameterization": "shared_outer",
    }


def _canonical_compact() -> dict[str, torch.Tensor]:
    gate_a = torch.arange(RANK * HIDDEN, dtype=torch.float32).reshape(RANK, HIDDEN)
    down_b = (
        torch.arange(HIDDEN * RANK, dtype=torch.float32).reshape(HIDDEN, RANK) + 100
    )
    tensors = {
        f"{GROUP_PREFIX}.shared.gate_up_proj.lora_A.weight": gate_a,
        f"{GROUP_PREFIX}.shared.down_proj.lora_B.weight": down_b,
    }
    for expert in range(EXPERTS):
        offset = 1000 * (expert + 1)
        tensors[f"{GROUP_PREFIX}.{expert}.gate_up_proj.lora_B.weight"] = (
            torch.arange(2 * INTERMEDIATE * RANK, dtype=torch.float32).reshape(
                2 * INTERMEDIATE, RANK
            )
            + offset
        )
        tensors[f"{GROUP_PREFIX}.{expert}.down_proj.lora_A.weight"] = (
            torch.arange(RANK * INTERMEDIATE, dtype=torch.float32).reshape(
                RANK, INTERMEDIATE
            )
            + offset
            + 100
        )
    return tensors


def _canonical_expanded(
    compact: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expanded = {key: value for key, value in compact.items() if ".shared." not in key}
    gate_a = compact[f"{GROUP_PREFIX}.shared.gate_up_proj.lora_A.weight"]
    down_b = compact[f"{GROUP_PREFIX}.shared.down_proj.lora_B.weight"]
    for expert in range(EXPERTS):
        expanded[f"{GROUP_PREFIX}.{expert}.gate_up_proj.lora_A.weight"] = gate_a
        expanded[f"{GROUP_PREFIX}.{expert}.down_proj.lora_B.weight"] = down_b
    return expanded


def _assert_tensors_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


def _expected_vllm(
    compact: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    return QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        _canonical_expanded(compact),
        adapter_config=_config(),
    )


def test_compact_handler_transform_normalize_and_stock_vllm(
    tmp_path: Path,
) -> None:
    compact = _canonical_compact()
    expected, expected_config = _expected_vllm(compact)
    transformed, transformed_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        compact,
        adapter_config=_config(),
    )
    _assert_tensors_equal(transformed, expected)
    assert transformed_config == expected_config
    assert not any(".shared." in key for key in transformed)
    _assert_tensors_equal(
        QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
            transformed,
            adapter_config=transformed_config,
        ),
        compact,
    )

    save_file(compact, tmp_path / "adapter_model.safetensors")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(_config()),
        encoding="utf-8",
    )
    normalize_lora_checkpoint_to_vllm(
        tmp_path,
        handler=QWEN3_5_MOE_HANDLER,
    )
    _assert_tensors_equal(load_file(tmp_path / "adapter_model.safetensors"), expected)
    normalized_config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert normalized_config[ART_LORA_FORMAT_CONFIG_KEY] == ART_LORA_FORMAT_VLLM
    _assert_tensors_equal(
        load_lora_tensors_for_megatron(
            tmp_path,
            handler=QWEN3_5_MOE_HANDLER,
        ),
        compact,
    )

    script = r"""
import json
import sys
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

path = sys.argv[1]
peft = PEFTHelper.from_local_dir(path, max_position_embeddings=None)
lora = LoRAModel.from_local_checkpoint(
    path,
    {"experts"},
    peft,
    lora_model_id=1,
    device="cpu",
    weights_mapper=Qwen3VLForConditionalGeneration.hf_to_vllm_mapper,
)
print(json.dumps(sorted(lora.loras)))
"""
    result = subprocess.run(
        [str(VLLM_PYTHON), "-c", script, str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    loaded_modules = json.loads(result.stdout.strip().splitlines()[-1])
    assert "language_model.model.layers.0.mlp.experts" in loaded_modules
    assert "language_model.model.layers.0.mlp.experts.base_layer" in loaded_modules


def test_compact_publication_expands_only_at_serving_boundary(tmp_path: Path) -> None:
    compact = _canonical_compact()
    expected, _expected_config = _expected_vllm(compact)
    gate_up = LoRA(
        f"{GROUP_PREFIX}.{{expert}}.gate_up_proj",
        HIDDEN,
        2 * INTERMEDIATE,
        RANK,
        RANK,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=EXPERTS,
        moe_parameterization="shared_outer",
        shared_factor="A",
    )
    down = LoRA(
        f"{GROUP_PREFIX}.{{expert}}.down_proj",
        INTERMEDIATE,
        HIDDEN,
        RANK,
        RANK,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=EXPERTS,
        moe_parameterization="shared_outer",
        shared_factor="B",
    )
    gate_up.load_lora(compact)
    down.load_lora(compact)
    canonical_before = {
        **gate_up.sharded_lora_state_dict(),
        **down.sharded_lora_state_dict(),
    }
    _assert_tensors_equal(canonical_before, compact)

    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(gate_up, down)]),
        adapter_dtypes={key: value.dtype for key, value in compact.items()},
        handler=QWEN3_5_MOE_HANDLER,
        adapter_config=_config(),
        output_dir=str(tmp_path),
        rank=0,
        world_size=1,
    )
    _assert_tensors_equal(load_file(tmp_path / "adapter_model.safetensors"), expected)
    _assert_tensors_equal(
        {
            **gate_up.sharded_lora_state_dict(),
            **down.sharded_lora_state_dict(),
        },
        compact,
    )
