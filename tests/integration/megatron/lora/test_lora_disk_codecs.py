import asyncio
from collections.abc import Sequence
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from safetensors.torch import load_file, save_file
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

pytest.importorskip("megatron.bridge.models.gpt_provider")

from art.megatron import lora as lora_module
from art.megatron.lora import (
    LoRA,
    LoRAParallelSpec,
    LoraShardManifest,
    LoRASlotRef,
)
from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    GPT_OSS_MOE_HANDLER,
    QWEN3_5_MOE_HANDLER,
    QWEN3_MOE_HANDLER,
)
from art.megatron.model_support.handlers.dsv4 import DSV4_HANDLER
from art.megatron.model_support.handlers.gemma4 import GEMMA4_MOE_HANDLER
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_MEGATRON,
    ART_LORA_FORMAT_VLLM,
    load_lora_tensors_for_megatron,
    normalize_lora_checkpoint_to_vllm,
    save_adapter_config,
    save_vllm_lora_tensors,
)
from art.megatron.model_support.spec import ModelSupportHandler
from art.megatron.weights import lora_publish
from art.megatron.weights.lora_publish import (
    LoraShardMeta,
    merge_sharded_adapter_entries,
    save_vllm_lora_from_model,
)
from art.trainer_rank import AdamParams, TrainerRank
from art.trainer_rank._checkpoint import (
    _PreparedSave,
    materialize_lora,
    prepare_checkpoint,
    validate_checkpoint,
)
from art.trainer_rank._checkpoint import (
    load_checkpoint as load_trainer_checkpoint,
)
from art.trainer_rank._impl import _AdapterConfig
from art.utils.convert_moe_lora import convert_checkpoint_if_needed

REPO_ROOT = Path(__file__).parents[4]
VLLM_PYTHON = REPO_ROOT / "vllm_runtime/.venv/bin/python"
VLLM_RUNTIME_SRC = REPO_ROOT / "vllm_runtime/src"
_VLLM_RUNTIME_UNAVAILABLE_REASON: str | None | object = object()


def _vllm_python_cmd() -> list[str]:
    override = os.environ.get("ART_TEST_VLLM_PYTHON")
    if override:
        return [override]
    if VLLM_PYTHON.exists():
        return [str(VLLM_PYTHON)]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            f"{VLLM_PYTHON} does not exist and uv is not available to run "
            "the locked vLLM runtime project"
        )
    return [
        uv,
        "run",
        "--project",
        str(REPO_ROOT / "vllm_runtime"),
        "--frozen",
        "--no-dev",
        "python",
    ]


def _vllm_runtime_unavailable_reason() -> str | None:
    global _VLLM_RUNTIME_UNAVAILABLE_REASON
    if isinstance(_VLLM_RUNTIME_UNAVAILABLE_REASON, str):
        return _VLLM_RUNTIME_UNAVAILABLE_REASON
    if _VLLM_RUNTIME_UNAVAILABLE_REASON is None:
        return None
    try:
        subprocess.run(
            [
                *_vllm_python_cmd(),
                "-c",
                "import vllm; from vllm.lora.lora_model import LoRAModel",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:
        _VLLM_RUNTIME_UNAVAILABLE_REASON = (
            "Stock vLLM loader runtime is unavailable. Run "
            "`uv sync --project vllm_runtime --frozen --no-dev`, or set "
            "`ART_TEST_VLLM_PYTHON` to a Python environment with vLLM installed. "
            f"Original error: {exc}"
        )
        return _VLLM_RUNTIME_UNAVAILABLE_REASON
    _VLLM_RUNTIME_UNAVAILABLE_REASON = None
    return None


def test_stock_vllm_loader_runtime_is_available() -> None:
    reason = _vllm_runtime_unavailable_reason()
    if reason is not None:
        pytest.fail(reason)


def _config(base_model: str, rank: int = 2, alpha: int = 4) -> dict:
    return {
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
    }


def _manifest(**values: object) -> LoraShardManifest:
    return cast(LoraShardManifest, values)


def _qwen35_config(base_model: str, rank: int = 2, alpha: int = 4) -> dict:
    config = _config(base_model, rank=rank, alpha=alpha)
    config.update(
        {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 3,
        }
    )
    return config


def _assert_tensors_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert set(actual) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(actual[key], tensor), key


def _save_adapter(path: Path, tensors: dict[str, torch.Tensor], config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _old_merge_shard_files_to_vllm(
    lora_path: Path,
    *,
    handler: ModelSupportHandler,
    adapter_config: dict,
) -> None:
    entries_by_key: dict[str, list[tuple[LoraShardManifest, torch.Tensor]]] = {}
    shard_paths = sorted(lora_path.glob("adapter_model-*-of-*.safetensors"))
    manifest_paths = sorted(lora_path.glob("adapter_manifest-*-of-*.json"))
    for shard_path in shard_paths:
        suffix = shard_path.name.removeprefix("adapter_model-").removesuffix(
            ".safetensors"
        )
        manifest = json.loads(
            (lora_path / f"adapter_manifest-{suffix}.json").read_text()
        )
        shard_tensors = load_file(shard_path)
        assert set(shard_tensors) == set(manifest)
        for key, tensor in shard_tensors.items():
            entries_by_key.setdefault(key, []).append(
                (cast(LoraShardManifest, manifest[key]), tensor)
            )

    merged = merge_sharded_adapter_entries(entries_by_key)
    vllm_tensors, adapter_config = handler.to_vllm_lora_tensors(
        merged,
        adapter_config=adapter_config,
    )
    save_vllm_lora_tensors(lora_path, vllm_tensors, adapter_config)
    for path in [*shard_paths, *manifest_paths]:
        path.unlink()


def _assert_stock_vllm_loads(
    path: Path,
    *,
    expected_modules: set[str],
    mapper: str = "none",
) -> list[str]:
    if reason := _vllm_runtime_unavailable_reason():
        pytest.skip(reason)
    script = r"""
import json
import sys
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

path = sys.argv[1]
expected = set(json.loads(sys.argv[2]))
mapper_name = sys.argv[3]
weights_mapper = None
if mapper_name == "qwen35":
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    weights_mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
peft = PEFTHelper.from_local_dir(path, max_position_embeddings=None)
lora = LoRAModel.from_local_checkpoint(
    path,
    expected,
    peft,
    lora_model_id=1,
    device="cpu",
    weights_mapper=weights_mapper,
)
print(json.dumps(sorted(lora.loras)))
"""
    result = subprocess.run(
        [
            *_vllm_python_cmd(),
            "-c",
            script,
            str(path),
            json.dumps(sorted(expected_modules)),
            mapper,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _qwen35_moe_art_tensors(prefix: str, *, rank: int = 2) -> dict[str, torch.Tensor]:
    hidden = 3
    q_out = 12
    intermediate = 4
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.self_attn.q_proj.lora_A.weight": torch.arange(
            rank * hidden,
            dtype=torch.float32,
        ).reshape(rank, hidden),
        f"{prefix}.self_attn.q_proj.lora_B.weight": torch.arange(
            q_out * rank,
            dtype=torch.float32,
        ).reshape(q_out, rank)
        + 100,
    }
    offset = 200
    for expert in range(2):
        for module in ("gate_up_proj", "down_proj"):
            out_dim = hidden if module == "down_proj" else 2 * intermediate
            in_dim = intermediate if module == "down_proj" else hidden
            tensors[f"{prefix}.mlp.experts.{expert}.{module}.lora_A.weight"] = (
                torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
                + offset
            )
            offset += 100
            tensors[f"{prefix}.mlp.experts.{expert}.{module}.lora_B.weight"] = (
                torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank)
                + offset
            )
            offset += 100
    return tensors


def _qwen35_shared_expert_art_tensors(
    prefix: str,
    *,
    rank: int = 2,
) -> dict[str, torch.Tensor]:
    hidden = 3
    intermediate = 4
    tensors: dict[str, torch.Tensor] = {}
    offset = 1000
    for module, in_dim, out_dim in (
        ("gate_proj", hidden, intermediate),
        ("up_proj", hidden, intermediate),
        ("down_proj", intermediate, hidden),
    ):
        module_prefix = f"{prefix}.mlp.shared_expert.{module}"
        tensors[f"{module_prefix}.lora_A.weight"] = (
            torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
            + offset
        )
        offset += 100
        tensors[f"{module_prefix}.lora_B.weight"] = (
            torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank)
            + offset
        )
        offset += 100
    return tensors


def _pack_qwen35_vllm_lora_b(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def _qwen35_fused_expert_vllm_tensors(
    original: dict[str, torch.Tensor],
    art_prefix: str,
) -> dict[str, torch.Tensor]:
    vllm_prefix = art_prefix.replace(
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
        1,
    )
    expert_prefix = f"{vllm_prefix}.mlp.experts"
    art_expert_prefix = f"{art_prefix}.mlp.experts"
    gate_up_a: list[torch.Tensor] = []
    gate_up_b: list[torch.Tensor] = []
    down_a: list[torch.Tensor] = []
    down_b: list[torch.Tensor] = []
    for expert in range(2):
        prefix = f"{art_expert_prefix}.{expert}"
        gate_up_a.append(original[f"{prefix}.gate_up_proj.lora_A.weight"])
        gate_up_b.append(original[f"{prefix}.gate_up_proj.lora_B.weight"])
        down_a.append(original[f"{prefix}.down_proj.lora_A.weight"])
        down_b.append(original[f"{prefix}.down_proj.lora_B.weight"])
    return {
        f"{expert_prefix}.base_layer.lora_A.weight": torch.cat(
            gate_up_a,
            dim=0,
        ).contiguous(),
        f"{expert_prefix}.base_layer.lora_B.weight": _pack_qwen35_vllm_lora_b(
            gate_up_b
        ),
        f"{expert_prefix}.lora_A.weight": torch.cat(down_a, dim=0).contiguous(),
        f"{expert_prefix}.lora_B.weight": _pack_qwen35_vllm_lora_b(down_b),
    }


def _gpt_oss_config(base_model: str, rank: int = 2, alpha: int = 4) -> dict:
    config = _config(base_model, rank=rank, alpha=alpha)
    config["target_modules"] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return config


def _gpt_oss_model_dir(tmp_path: Path) -> str:
    model_dir = tmp_path / "gpt_oss_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": 128, "intermediate_size": 128}),
        encoding="utf-8",
    )
    return str(model_dir)


def _gpt_oss_moe_art_tensors(prefix: str, *, rank: int = 2) -> dict[str, torch.Tensor]:
    hidden = 128
    intermediate = 128
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.self_attn.q_proj.lora_A.weight": torch.arange(
            rank * hidden,
            dtype=torch.float32,
        ).reshape(rank, hidden),
        f"{prefix}.self_attn.q_proj.lora_B.weight": torch.arange(
            hidden * rank,
            dtype=torch.float32,
        ).reshape(hidden, rank)
        + 100,
    }
    offset = 200
    for expert in range(2):
        for module in ("gate_up_proj", "down_proj"):
            out_dim = hidden if module == "down_proj" else 2 * intermediate
            in_dim = intermediate if module == "down_proj" else hidden
            tensors[f"{prefix}.mlp.experts.{expert}.{module}.lora_A.weight"] = (
                torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
                + offset
            )
            offset += 100
            tensors[f"{prefix}.mlp.experts.{expert}.{module}.lora_B.weight"] = (
                torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank)
                + offset
            )
            offset += 100
    return tensors


def _gpt_oss_gate_up_lora_b_to_vllm(tensor: torch.Tensor) -> torch.Tensor:
    gate, up = tensor.split(tensor.shape[0] // 2, dim=0)
    return torch.stack((gate, up), dim=1).flatten(0, 1).contiguous()


def _gpt_oss_fused_expert_vllm_tensors(
    original: dict[str, torch.Tensor],
    art_prefix: str,
) -> dict[str, torch.Tensor]:
    expert_prefix = f"{art_prefix}.mlp.experts"
    gate_up_a: list[torch.Tensor] = []
    gate_up_b: list[torch.Tensor] = []
    down_a: list[torch.Tensor] = []
    down_b: list[torch.Tensor] = []
    for expert in range(2):
        prefix = f"{expert_prefix}.{expert}"
        gate_up_a.append(original[f"{prefix}.gate_up_proj.lora_A.weight"])
        gate_up_b.append(
            _gpt_oss_gate_up_lora_b_to_vllm(
                original[f"{prefix}.gate_up_proj.lora_B.weight"]
            )
        )
        down_a.append(original[f"{prefix}.down_proj.lora_A.weight"])
        down_b.append(original[f"{prefix}.down_proj.lora_B.weight"])
    return {
        f"{expert_prefix}.base_layer.lora_A.weight": torch.cat(
            gate_up_a,
            dim=0,
        ).contiguous(),
        f"{expert_prefix}.base_layer.lora_B.weight": _pack_qwen35_vllm_lora_b(
            gate_up_b
        ),
        f"{expert_prefix}.lora_A.weight": torch.cat(down_a, dim=0).contiguous(),
        f"{expert_prefix}.lora_B.weight": _pack_qwen35_vllm_lora_b(down_b),
    }


def _qwen3_dense_lora_tensors(prefix: str, *, rank: int = 2) -> dict[str, torch.Tensor]:
    module_dims = {
        "self_attn.q_proj": (rank, 3, 3),
        "self_attn.k_proj": (rank, 3, 3),
        "self_attn.v_proj": (rank, 3, 3),
        "self_attn.o_proj": (rank, 3, 3),
        "mlp.gate_proj": (rank, 3, 4),
        "mlp.up_proj": (rank, 3, 4),
        "mlp.down_proj": (rank, 4, 3),
    }
    tensors: dict[str, torch.Tensor] = {}
    offset = 0
    for module, (rank_dim, in_dim, out_dim) in module_dims.items():
        tensors[f"{prefix}.{module}.lora_A.weight"] = (
            torch.arange(rank_dim * in_dim, dtype=torch.float32).reshape(
                rank_dim,
                in_dim,
            )
            + offset
        )
        offset += 100
        tensors[f"{prefix}.{module}.lora_B.weight"] = (
            torch.arange(out_dim * rank_dim, dtype=torch.float32).reshape(
                out_dim,
                rank_dim,
            )
            + offset
        )
        offset += 100
    return tensors


def _qwen3_moe_lora_tensors(prefix: str, *, rank: int = 2) -> dict[str, torch.Tensor]:
    tensors = {
        key: value
        for key, value in _qwen3_dense_lora_tensors(prefix, rank=rank).items()
        if ".mlp." not in key
    }
    offset = 1000
    for expert in range(2):
        for module, in_dim, out_dim in (
            ("gate_proj", 3, 4),
            ("up_proj", 3, 4),
            ("down_proj", 4, 3),
        ):
            expert_prefix = f"{prefix}.mlp.experts.{expert}.{module}"
            tensors[f"{expert_prefix}.lora_A.weight"] = (
                torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
                + offset
            )
            offset += 100
            tensors[f"{expert_prefix}.lora_B.weight"] = (
                torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank)
                + offset
            )
            offset += 100
    return tensors


def _pack_lora_b_by_expert(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def _qwen3_fused_moe_fixture(
    prefix: str,
    *,
    rank: int = 2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hidden = 3
    intermediate = 4
    num_experts = 2
    gate_up_a = torch.arange(
        num_experts * rank * hidden,
        dtype=torch.float32,
    ).reshape(num_experts * rank, hidden)
    down_a = (
        torch.arange(
            num_experts * rank * intermediate,
            dtype=torch.float32,
        ).reshape(num_experts * rank, intermediate)
        + 100
    )
    gate_up_b_blocks = [
        torch.arange(
            2 * intermediate * rank,
            dtype=torch.float32,
        ).reshape(2 * intermediate, rank)
        + 200
        + expert * 100
        for expert in range(num_experts)
    ]
    down_b_blocks = [
        torch.arange(hidden * rank, dtype=torch.float32).reshape(hidden, rank)
        + 500
        + expert * 100
        for expert in range(num_experts)
    ]
    fused = {
        f"{prefix}.base_layer.lora_A.weight": gate_up_a,
        f"{prefix}.base_layer.lora_B.weight": _pack_lora_b_by_expert(gate_up_b_blocks),
        f"{prefix}.lora_A.weight": down_a,
        f"{prefix}.lora_B.weight": _pack_lora_b_by_expert(down_b_blocks),
    }
    expected: dict[str, torch.Tensor] = {}
    for expert in range(num_experts):
        rows = slice(expert * rank, (expert + 1) * rank)
        gate_b, up_b = gate_up_b_blocks[expert].split(intermediate, dim=0)
        expert_prefix = f"{prefix}.{expert}"
        expected[f"{expert_prefix}.gate_proj.lora_A.weight"] = gate_up_a[rows].clone()
        expected[f"{expert_prefix}.gate_proj.lora_B.weight"] = gate_b
        expected[f"{expert_prefix}.up_proj.lora_A.weight"] = gate_up_a[rows].clone()
        expected[f"{expert_prefix}.up_proj.lora_B.weight"] = up_b
        expected[f"{expert_prefix}.down_proj.lora_A.weight"] = down_a[rows].clone()
        expected[f"{expert_prefix}.down_proj.lora_B.weight"] = down_b_blocks[expert]
    return fused, expected


def test_peft_fused_moe_checkpoint_converts_to_vllm_3d_layout(tmp_path: Path) -> None:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    peft_tensors = {
        f"{prefix}.base_layer.lora_A.weight": torch.arange(
            2 * 8,
            dtype=torch.float32,
        ).reshape(2, 8),
        f"{prefix}.base_layer.lora_B.weight": torch.arange(
            3 * 2,
            dtype=torch.float32,
        ).reshape(3, 2)
        + 100,
        f"{prefix}.lora_A.weight": torch.arange(
            2 * 3,
            dtype=torch.float32,
        ).reshape(2, 3)
        + 200,
        f"{prefix}.lora_B.weight": torch.arange(
            4 * 2,
            dtype=torch.float32,
        ).reshape(4, 2)
        + 300,
    }
    _save_adapter(
        tmp_path,
        peft_tensors,
        {
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "target_parameters": [
                "model.layers.0.mlp.experts.gate_up_proj",
                "model.layers.0.mlp.experts.down_proj",
            ],
        },
    )

    convert_checkpoint_if_needed(str(tmp_path))

    converted = load_file(tmp_path / "adapter_model.safetensors")
    _assert_tensors_equal(
        converted,
        {
            f"{prefix}.base_layer.lora_A.weight": peft_tensors[
                f"{prefix}.base_layer.lora_B.weight"
            ].T.contiguous(),
            f"{prefix}.base_layer.lora_B.weight": peft_tensors[
                f"{prefix}.base_layer.lora_A.weight"
            ].T.contiguous(),
            f"{prefix}.lora_A.weight": peft_tensors[
                f"{prefix}.lora_B.weight"
            ].T.contiguous(),
            f"{prefix}.lora_B.weight": peft_tensors[
                f"{prefix}.lora_A.weight"
            ].T.contiguous(),
        },
    )
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert adapter_config["target_modules"] == ["q_proj", "experts"]
    assert "target_parameters" not in adapter_config


def test_qwen3_fused_identity_normalizes_to_per_expert_vllm_layout(
    tmp_path: Path,
) -> None:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    rank = 2
    fused, expected = _qwen3_fused_moe_fixture(prefix, rank=rank)
    _save_adapter(
        tmp_path,
        {
            f"{prefix}.base_layer.lora_A.weight": fused[
                f"{prefix}.base_layer.lora_B.weight"
            ].T.contiguous(),
            f"{prefix}.base_layer.lora_B.weight": fused[
                f"{prefix}.base_layer.lora_A.weight"
            ].T.contiguous(),
            f"{prefix}.lora_A.weight": fused[f"{prefix}.lora_B.weight"].T.contiguous(),
            f"{prefix}.lora_B.weight": fused[f"{prefix}.lora_A.weight"].T.contiguous(),
        },
        {
            "r": rank,
            "lora_alpha": 4,
            "target_modules": ["q_proj"],
            "target_parameters": [
                "model.layers.0.mlp.experts.gate_up_proj",
                "model.layers.0.mlp.experts.down_proj",
            ],
        },
    )

    convert_checkpoint_if_needed(str(tmp_path))
    normalize_lora_checkpoint_to_vllm(
        tmp_path,
        handler=QWEN3_MOE_HANDLER,
        adapter_config=_config("Qwen/Qwen3-30B-A3B", rank=rank),
    )

    converted = load_file(tmp_path / "adapter_model.safetensors")
    _assert_tensors_equal(converted, expected)
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert "experts" in adapter_config["target_modules"]


def test_qwen3_target_parameter_identity_normalizes_to_per_expert_vllm_layout(
    tmp_path: Path,
) -> None:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    rank = 2
    hidden = 3
    intermediate = 4
    num_experts = 2
    gate_up_a = torch.arange(
        num_experts * rank * 2 * intermediate,
        dtype=torch.float32,
    ).reshape(num_experts * rank, 2 * intermediate)
    gate_up_b = (
        torch.arange(hidden * num_experts * rank, dtype=torch.float32).reshape(
            hidden, num_experts * rank
        )
        + 100
    )
    down_a = (
        torch.arange(num_experts * rank * hidden, dtype=torch.float32).reshape(
            num_experts * rank, hidden
        )
        + 200
    )
    down_b = (
        torch.arange(intermediate * num_experts * rank, dtype=torch.float32).reshape(
            intermediate, num_experts * rank
        )
        + 300
    )
    _save_adapter(
        tmp_path,
        {
            f"{prefix}.base_layer.lora_A.weight": gate_up_a,
            f"{prefix}.base_layer.lora_B.weight": gate_up_b,
            f"{prefix}.lora_A.weight": down_a,
            f"{prefix}.lora_B.weight": down_b,
        },
        _config("Qwen/Qwen3-30B-A3B", rank=rank),
    )

    normalize_lora_checkpoint_to_vllm(
        tmp_path,
        handler=QWEN3_MOE_HANDLER,
        adapter_config=_config("Qwen/Qwen3-30B-A3B", rank=rank),
    )

    expected: dict[str, torch.Tensor] = {}
    for expert in range(num_experts):
        rows = slice(expert * rank, (expert + 1) * rank)
        gate_a, up_a = gate_up_a[rows].split(intermediate, dim=1)
        expert_prefix = f"{prefix}.{expert}"
        expected[f"{expert_prefix}.gate_proj.lora_A.weight"] = gate_up_b[
            :, rows
        ].T.contiguous()
        expected[f"{expert_prefix}.gate_proj.lora_B.weight"] = gate_a.T.contiguous()
        expected[f"{expert_prefix}.up_proj.lora_A.weight"] = gate_up_b[
            :, rows
        ].T.contiguous()
        expected[f"{expert_prefix}.up_proj.lora_B.weight"] = up_a.T.contiguous()
        expected[f"{expert_prefix}.down_proj.lora_A.weight"] = down_b[
            :, rows
        ].T.contiguous()
        expected[f"{expert_prefix}.down_proj.lora_B.weight"] = down_a[
            rows
        ].T.contiguous()
    _assert_tensors_equal(load_file(tmp_path / "adapter_model.safetensors"), expected)
    loaded_modules = _assert_stock_vllm_loads(
        tmp_path,
        expected_modules={
            "experts.0.gate_proj",
            "experts.0.up_proj",
            "experts.0.down_proj",
            "experts.1.gate_proj",
            "experts.1.up_proj",
            "experts.1.down_proj",
        },
    )
    assert loaded_modules == [
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.1.down_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.1.up_proj",
    ]


def test_qwen35_and_qwen36_vllm_canonical_roundtrip_and_stock_loader(tmp_path: Path):
    art_prefix = "base_model.model.model.layers.0"
    original = _qwen35_moe_art_tensors(art_prefix)
    expected_experts = _qwen35_fused_expert_vllm_tensors(original, art_prefix)
    for base_model in ("Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.6-35B-A3B"):
        vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
            original,
            adapter_config=_qwen35_config(base_model),
        )
        assert vllm_config["r"] == 2
        assert vllm_config["lora_alpha"] == 4
        assert vllm_config["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "experts",
        ]
        assert all("language_model.layers" in key for key in vllm_tensors)
        assert not any(".mlp.experts.0." in key for key in vllm_tensors)
        for key, tensor in expected_experts.items():
            assert torch.equal(vllm_tensors[key], tensor), key
        roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
            vllm_tensors,
            adapter_config=vllm_config,
        )
        _assert_tensors_equal(roundtrip, original)
        adapter_dir = tmp_path / base_model.replace("/", "_")
        _save_adapter(adapter_dir, vllm_tensors, vllm_config)
        loaded_modules = _assert_stock_vllm_loads(
            adapter_dir,
            expected_modules={"q_proj", "experts"},
            mapper="qwen35",
        )
        assert "language_model.model.layers.0.mlp.experts" in loaded_modules
        assert "language_model.model.layers.0.mlp.experts.base_layer" in loaded_modules


def test_qwen35_vllm_config_preserves_shared_expert_targets_when_present():
    art_prefix = "base_model.model.model.layers.0"
    original = {
        **_qwen35_moe_art_tensors(art_prefix),
        **_qwen35_shared_expert_art_tensors(art_prefix),
    }
    adapter_config = _qwen35_config("Qwen/Qwen3.6-35B-A3B")
    adapter_config["target_modules"] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "experts",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        original,
        adapter_config=adapter_config,
    )
    assert vllm_config["target_modules"] == adapter_config["target_modules"]
    assert any(".mlp.shared_expert.gate_proj." in key for key in vllm_tensors)
    assert any(".mlp.shared_expert.up_proj." in key for key in vllm_tensors)
    assert any(".mlp.shared_expert.down_proj." in key for key in vllm_tensors)
    roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=vllm_config,
    )
    _assert_tensors_equal(roundtrip, original)


def test_dsv4_vllm_canonical_moe_roundtrip(tmp_path: Path) -> None:
    prefix = "base_model.model.model.layers.4.mlp.experts"
    vllm_prefix = "base_model.model.model.layers.4.ffn.experts"
    original: dict[str, torch.Tensor] = {}
    for expert in range(2):
        offset = expert * 100
        original.update(
            {
                f"{prefix}.{expert}.gate_up_proj.lora_A.weight": torch.arange(
                    offset, offset + 6, dtype=torch.float32
                ).reshape(2, 3),
                f"{prefix}.{expert}.gate_up_proj.lora_B.weight": torch.arange(
                    offset, offset + 16, dtype=torch.float32
                ).reshape(8, 2),
                f"{prefix}.{expert}.down_proj.lora_A.weight": torch.arange(
                    offset, offset + 8, dtype=torch.float32
                ).reshape(2, 4),
                f"{prefix}.{expert}.down_proj.lora_B.weight": torch.arange(
                    offset, offset + 6, dtype=torch.float32
                ).reshape(3, 2),
            }
        )
    attention_prefix = "base_model.model.model.layers.4.self_attn.compressor"
    original.update(
        {
            f"{attention_prefix}.kv_proj.lora_A.weight": torch.arange(
                6, dtype=torch.float32
            ).reshape(2, 3),
            f"{attention_prefix}.kv_proj.lora_B.weight": torch.arange(
                10, dtype=torch.float32
            ).reshape(5, 2),
            f"{attention_prefix}.gate_proj.lora_A.weight": torch.arange(
                6, dtype=torch.float32
            ).reshape(2, 3),
            f"{attention_prefix}.gate_proj.lora_B.weight": torch.arange(
                4, dtype=torch.float32
            ).reshape(2, 2),
        }
    )
    config = _config("deepseek-ai/DeepSeek-V4-Flash")

    vllm_tensors, vllm_config = DSV4_HANDLER.to_vllm_lora_tensors(
        original,
        adapter_config=config,
    )

    assert set(vllm_tensors) == {
        f"{vllm_prefix}.base_layer.lora_A.weight",
        f"{vllm_prefix}.base_layer.lora_B.weight",
        f"{vllm_prefix}.lora_A.weight",
        f"{vllm_prefix}.lora_B.weight",
        "base_model.model.model.layers.4.attn.compressor.wkv.lora_A.weight",
        "base_model.model.model.layers.4.attn.compressor.wkv.lora_B.weight",
        "base_model.model.model.layers.4.attn.compressor.wgate.lora_A.weight",
        "base_model.model.model.layers.4.attn.compressor.wgate.lora_B.weight",
    }
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_A.weight"].shape == (4, 3)
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"].shape == (8, 4)
    assert vllm_tensors[f"{vllm_prefix}.lora_A.weight"].shape == (4, 4)
    assert vllm_tensors[f"{vllm_prefix}.lora_B.weight"].shape == (3, 4)
    assert "experts" in vllm_config["target_modules"]
    _assert_tensors_equal(
        DSV4_HANDLER.from_vllm_lora_tensors(
            vllm_tensors,
            adapter_config=vllm_config,
        ),
        original,
    )
    adapter_dir = tmp_path / "dsv4"
    _save_adapter(adapter_dir, vllm_tensors, vllm_config)
    loaded_modules = _assert_stock_vllm_loads(
        adapter_dir,
        expected_modules={"experts", "wgate", "wkv"},
    )
    assert f"model.layers.4.ffn.experts" in loaded_modules
    assert f"model.layers.4.ffn.experts.base_layer" in loaded_modules
    assert "model.layers.4.attn.compressor.wgate" in loaded_modules
    assert "model.layers.4.attn.compressor.wkv" in loaded_modules


def test_gemma4_shared_experts_plural_keys_map_to_vllm_dense_mlp(tmp_path: Path):
    art_prefix = "base_model.model.model.layers.0"
    hidden_size = 3
    model_dir = tmp_path / "gemma4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"num_hidden_layers": 1}),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.0.pre_feedforward_layernorm.weight": torch.tensor(
                [2.0, 4.0, 8.0]
            ),
            "model.layers.0.pre_feedforward_layernorm_2.weight": torch.tensor(
                [1.0, 2.0, 4.0]
            ),
        },
        model_dir / "model-00001-of-00001.safetensors",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.pre_feedforward_layernorm.weight": (
                        "model-00001-of-00001.safetensors"
                    ),
                    "model.layers.0.pre_feedforward_layernorm_2.weight": (
                        "model-00001-of-00001.safetensors"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    original = {
        f"{art_prefix}.mlp.shared_experts.gate_proj.lora_A.weight": torch.ones(
            2,
            hidden_size,
        ),
        f"{art_prefix}.mlp.shared_experts.gate_proj.lora_B.weight": torch.ones(4, 2),
        f"{art_prefix}.mlp.shared_experts.up_proj.lora_A.weight": torch.ones(
            2,
            hidden_size,
        ),
        f"{art_prefix}.mlp.shared_experts.up_proj.lora_B.weight": torch.ones(4, 2),
        f"{art_prefix}.mlp.shared_experts.down_proj.lora_A.weight": torch.ones(2, 4),
        f"{art_prefix}.mlp.shared_experts.down_proj.lora_B.weight": torch.ones(
            hidden_size,
            2,
        ),
    }
    adapter_config = _config(str(model_dir))
    vllm_tensors, _ = GEMMA4_MOE_HANDLER.to_vllm_lora_tensors(
        original,
        adapter_config=adapter_config,
    )

    assert set(vllm_tensors) == {
        f"{art_prefix}.mlp.gate_proj.lora_A.weight",
        f"{art_prefix}.mlp.gate_proj.lora_B.weight",
        f"{art_prefix}.mlp.up_proj.lora_A.weight",
        f"{art_prefix}.mlp.up_proj.lora_B.weight",
        f"{art_prefix}.mlp.down_proj.lora_A.weight",
        f"{art_prefix}.mlp.down_proj.lora_B.weight",
    }
    assert not any("shared_expert" in key for key in vllm_tensors)
    assert torch.equal(
        vllm_tensors[f"{art_prefix}.mlp.gate_proj.lora_A.weight"],
        torch.full((2, hidden_size), 0.5),
    )
    roundtrip = GEMMA4_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=adapter_config,
    )
    _assert_tensors_equal(roundtrip, original)


def test_gpt_oss_vllm_canonical_roundtrip_and_stock_loader(tmp_path: Path):
    art_prefix = "base_model.model.model.layers.0"
    original = _gpt_oss_moe_art_tensors(art_prefix)
    expected_experts = _gpt_oss_fused_expert_vllm_tensors(original, art_prefix)
    vllm_tensors, vllm_config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        original,
        adapter_config=_gpt_oss_config(_gpt_oss_model_dir(tmp_path)),
    )

    assert vllm_config["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "experts",
    ]
    assert "base_model.model.model.layers.0.attn.q_proj.lora_A.weight" in vllm_tensors
    assert not any(".self_attn." in key for key in vllm_tensors)
    assert not any(".mlp.experts.0." in key for key in vllm_tensors)
    for key, tensor in expected_experts.items():
        assert torch.equal(vllm_tensors[key], tensor), key

    internal = GPT_OSS_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=vllm_config,
    )
    gate_up_b = internal[f"{art_prefix}.mlp.experts.0.gate_up_proj.lora_B.weight"]
    assert gate_up_b.shape == (2048, 2)
    assert not torch.count_nonzero(gate_up_b[128:1024])
    assert not torch.count_nonzero(gate_up_b[1152:])
    reexported, reexported_config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        internal,
        adapter_config=vllm_config,
    )
    _assert_tensors_equal(reexported, vllm_tensors)
    assert reexported_config == vllm_config

    adapter_dir = tmp_path / "gpt_oss"
    _save_adapter(adapter_dir, vllm_tensors, vllm_config)
    loaded_modules = _assert_stock_vllm_loads(
        adapter_dir,
        expected_modules={"q_proj", "experts"},
    )
    assert "model.layers.0.attn.q_proj" in loaded_modules
    assert "model.layers.0.mlp.experts" in loaded_modules
    assert "model.layers.0.mlp.experts.base_layer" in loaded_modules


def test_qwen35_target_parameter_identity_normalizes_to_fused_vllm_layout(
    tmp_path: Path,
) -> None:
    art_prefix = "base_model.model.model.layers.0"
    original = _qwen35_moe_art_tensors(art_prefix)
    expected = _qwen35_fused_expert_vllm_tensors(original, art_prefix)
    raw = {
        key.replace(
            "base_model.model.model.language_model.layers.",
            "base_model.model.model.layers.",
            1,
        ): tensor
        for key, tensor in expected.items()
    }
    _save_adapter(
        tmp_path,
        raw,
        {
            **_qwen35_config("Qwen/Qwen3.5-35B-A3B"),
            "target_parameters": [
                "model.layers.0.mlp.experts.gate_up_proj",
                "model.layers.0.mlp.experts.down_proj",
            ],
        },
    )

    normalize_lora_checkpoint_to_vllm(
        tmp_path,
        handler=QWEN3_5_MOE_HANDLER,
        adapter_config=_qwen35_config("Qwen/Qwen3.5-35B-A3B"),
    )

    _assert_tensors_equal(load_file(tmp_path / "adapter_model.safetensors"), expected)


def test_qwen35_and_qwen36_dense_prefix_roundtrip_and_stock_loader(tmp_path: Path):
    original = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
            2,
            3,
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
            12,
            2,
        ),
    }
    for base_model in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-4B"):
        vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
            original,
            adapter_config=_qwen35_config(base_model),
        )
        assert set(vllm_tensors) == {
            key.replace(
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            )
            for key in original
        }
        roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
            vllm_tensors,
            adapter_config=vllm_config,
        )
        _assert_tensors_equal(roundtrip, original)
        adapter_dir = tmp_path / base_model.replace("/", "_")
        _save_adapter(adapter_dir, vllm_tensors, vllm_config)
        loaded_modules = _assert_stock_vllm_loads(
            adapter_dir,
            expected_modules={"q_proj"},
            mapper="qwen35",
        )
        assert loaded_modules == ["language_model.model.layers.0.self_attn.q_proj"]


def test_qwen3_dense_and_moe_are_already_vllm_canonical(tmp_path: Path):
    dense = _qwen3_dense_lora_tensors("base_model.model.model.layers.0")
    assert (
        DEFAULT_DENSE_HANDLER.to_vllm_lora_tensors(
            dense,
            adapter_config=_config("Qwen/Qwen3-0.6B"),
        )[0]
        == dense
    )
    dense_dir = tmp_path / "qwen3_dense"
    _save_adapter(dense_dir, dense, _config("Qwen/Qwen3-0.6B"))
    assert _assert_stock_vllm_loads(
        dense_dir,
        expected_modules={
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        },
    ) == [
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]

    moe = _qwen3_moe_lora_tensors("base_model.model.model.layers.0")
    assert (
        QWEN3_MOE_HANDLER.to_vllm_lora_tensors(
            moe,
            adapter_config=_config("Qwen/Qwen3-30B-A3B"),
        )[0]
        == moe
    )
    moe_dir = tmp_path / "qwen3_moe"
    _save_adapter(moe_dir, moe, _config("Qwen/Qwen3-30B-A3B"))
    assert _assert_stock_vllm_loads(
        moe_dir,
        expected_modules={
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "experts.0.gate_proj",
            "experts.0.up_proj",
            "experts.0.down_proj",
            "experts.1.gate_proj",
            "experts.1.up_proj",
            "experts.1.down_proj",
        },
    ) == [
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.1.down_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.1.up_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]


def test_qwen35_megatron_shards_merge_to_vllm_checkpoint_and_roundtrip(
    tmp_path: Path,
):
    prefix = "base_model.model.model.layers.0.mlp.experts.0"
    rank = 1
    hidden = 2
    intermediate = 4
    full = {
        f"{prefix}.gate_up_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        f"{prefix}.gate_up_proj.lora_B.weight": torch.arange(
            2 * intermediate * rank,
            dtype=torch.float32,
        ).reshape(2 * intermediate, rank),
        f"{prefix}.down_proj.lora_A.weight": torch.arange(
            rank * intermediate,
            dtype=torch.float32,
        ).reshape(rank, intermediate)
        + 20,
        f"{prefix}.down_proj.lora_B.weight": torch.arange(
            hidden * rank,
            dtype=torch.float32,
        ).reshape(hidden, rank)
        + 30,
    }

    def unsharded() -> LoraShardManifest:
        return _manifest(sharded=False, shard_world_size=1, shard_rank=0)

    def sharded(rank_id: int, dim: int) -> LoraShardManifest:
        return _manifest(
            sharded=True,
            shard_world_size=2,
            shard_rank=rank_id,
            export_shard_dim=dim,
            export_shard_strategy="uniform",
        )

    shard0 = {
        f"{prefix}.gate_up_proj.lora_A.weight": full[
            f"{prefix}.gate_up_proj.lora_A.weight"
        ],
        f"{prefix}.down_proj.lora_B.weight": full[f"{prefix}.down_proj.lora_B.weight"],
        f"{prefix}.gate_up_proj.lora_B.weight": full[
            f"{prefix}.gate_up_proj.lora_B.weight"
        ][:4],
        f"{prefix}.down_proj.lora_A.weight": full[f"{prefix}.down_proj.lora_A.weight"][
            :, :2
        ],
    }
    manifest0 = {
        f"{prefix}.gate_up_proj.lora_A.weight": unsharded(),
        f"{prefix}.down_proj.lora_B.weight": unsharded(),
        f"{prefix}.gate_up_proj.lora_B.weight": sharded(0, 0),
        f"{prefix}.down_proj.lora_A.weight": sharded(0, 1),
    }
    shard1 = {
        f"{prefix}.gate_up_proj.lora_B.weight": full[
            f"{prefix}.gate_up_proj.lora_B.weight"
        ][4:],
        f"{prefix}.down_proj.lora_A.weight": full[f"{prefix}.down_proj.lora_A.weight"][
            :, 2:
        ],
    }
    manifest1 = {
        f"{prefix}.gate_up_proj.lora_B.weight": sharded(1, 0),
        f"{prefix}.down_proj.lora_A.weight": sharded(1, 1),
    }
    adapter_dir = tmp_path / "qwen35_megatron_shards"
    adapter_config = _config("Qwen/Qwen3.5-35B-A3B", rank=rank, alpha=rank)
    entries_by_key = {key: [(manifest0[key], tensor)] for key, tensor in shard0.items()}
    for key, tensor in shard1.items():
        entries_by_key.setdefault(key, []).append((manifest1[key], tensor))
    merged = merge_sharded_adapter_entries(entries_by_key)
    vllm_tensors, adapter_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        merged,
        adapter_config=adapter_config,
    )
    save_vllm_lora_tensors(adapter_dir, vllm_tensors, adapter_config)

    roundtrip = load_lora_tensors_for_megatron(
        str(adapter_dir),
        handler=QWEN3_5_MOE_HANDLER,
    )
    _assert_tensors_equal(roundtrip, full)
    final_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    loaded_modules = _assert_stock_vllm_loads(
        adapter_dir,
        expected_modules={"experts"},
        mapper="qwen35",
    )
    assert "language_model.model.layers.0.mlp.experts" in loaded_modules
    assert "language_model.model.layers.0.mlp.experts.base_layer" in loaded_modules


def test_lora_publish_keeps_same_key_shards_separate():
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    manifest = {
        "sharded": True,
        "shard_world_size": 2,
        "export_shard_dim": 0,
        "export_shard_strategy": "uniform",
    }
    shard0 = torch.tensor([[1.0], [2.0]])
    shard1 = torch.tensor([[3.0], [4.0]])
    metadata = [
        LoraShardMeta(
            key=key,
            owner_rank=0,
            shape=tuple(shard0.shape),
            dtype_name="float32",
            manifest=_manifest(**manifest, shard_rank=0),
            block="base_model.model.model.layers.0",
        ),
        LoraShardMeta(
            key=key,
            owner_rank=1,
            shape=tuple(shard1.shape),
            dtype_name="float32",
            manifest=_manifest(**manifest, shard_rank=1),
            block="base_model.model.model.layers.0",
        ),
    ]
    entries = lora_publish._entries_by_key(
        metadata,
        {
            (0, key): shard0,
            (1, key): shard1,
        },
    )

    merged = merge_sharded_adapter_entries(entries)

    assert torch.equal(merged[key], torch.tensor([[1.0], [2.0], [3.0], [4.0]]))


def test_batched_lora_publish_matches_old_shard_merge_exactly(tmp_path: Path):
    uniform_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    componentwise_key = (
        "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_B.weight"
    )
    unsharded_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    full_uniform = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    full_componentwise = torch.tensor(
        [[0.0], [1.0], [10.0], [11.0], [2.0], [3.0], [12.0], [13.0]]
    )
    shard0 = {
        unsharded_key: torch.arange(4, dtype=torch.float32).reshape(2, 2) + 100,
        uniform_key: full_uniform[:2],
        componentwise_key: torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
    }
    shard1 = {
        uniform_key: full_uniform[2:],
        componentwise_key: torch.tensor([[10.0], [11.0], [12.0], [13.0]]),
    }
    unsharded_manifest = _manifest(sharded=False, shard_world_size=1, shard_rank=0)
    uniform_manifest = {
        "sharded": True,
        "shard_world_size": 2,
        "export_shard_dim": 0,
        "export_shard_strategy": "uniform",
    }
    componentwise_manifest = {
        "sharded": True,
        "shard_world_size": 2,
        "export_shard_dim": 0,
        "export_shard_strategy": "componentwise",
        "component_sizes": [4, 4],
    }
    manifest0 = {
        unsharded_key: unsharded_manifest,
        uniform_key: _manifest(**uniform_manifest, shard_rank=0),
        componentwise_key: _manifest(**componentwise_manifest, shard_rank=0),
    }
    manifest1 = {
        uniform_key: _manifest(**uniform_manifest, shard_rank=1),
        componentwise_key: _manifest(**componentwise_manifest, shard_rank=1),
    }

    class IdentityHandler:
        def to_vllm_lora_tensors(self, tensors, *, adapter_config):
            return dict(tensors), dict(adapter_config)

    old_dir = tmp_path / "old"
    current_dir = tmp_path / "current"
    old_dir.mkdir()
    save_file(shard0, old_dir / "adapter_model-01-of-02.safetensors")
    save_file(shard1, old_dir / "adapter_model-02-of-02.safetensors")
    (old_dir / "adapter_manifest-01-of-02.json").write_text(
        json.dumps(manifest0, sort_keys=True)
    )
    (old_dir / "adapter_manifest-02-of-02.json").write_text(
        json.dumps(manifest1, sort_keys=True)
    )
    adapter_config = _config("Qwen/Qwen3-30B-A3B")
    handler = IdentityHandler()
    _old_merge_shard_files_to_vllm(
        old_dir,
        handler=cast(ModelSupportHandler, handler),
        adapter_config=adapter_config,
    )

    metadata = [
        LoraShardMeta(
            key=key,
            owner_rank=0,
            shape=tuple(tensor.shape),
            dtype_name=str(tensor.dtype).removeprefix("torch."),
            manifest=manifest0[key],
            block="base_model.model.model.layers.0",
        )
        for key, tensor in shard0.items()
    ] + [
        LoraShardMeta(
            key=key,
            owner_rank=1,
            shape=tuple(tensor.shape),
            dtype_name=str(tensor.dtype).removeprefix("torch."),
            manifest=manifest1[key],
            block="base_model.model.model.layers.0",
        )
        for key, tensor in shard1.items()
    ]
    current_tensors, current_config = lora_publish._rank0_vllm_lora_tensors(
        metadata=metadata,
        tensors_by_owner_key={
            **{(0, key): tensor for key, tensor in shard0.items()},
            **{(1, key): tensor for key, tensor in shard1.items()},
        },
        handler=cast(ModelSupportHandler, handler),
        adapter_config=adapter_config,
    )
    save_vllm_lora_tensors(current_dir, current_tensors, current_config)

    old_tensors = load_file(old_dir / "adapter_model.safetensors")
    current_tensors = load_file(current_dir / "adapter_model.safetensors")
    _assert_tensors_equal(current_tensors, old_tensors)
    assert torch.equal(current_tensors[uniform_key], full_uniform)
    assert torch.equal(current_tensors[componentwise_key], full_componentwise)
    assert (current_dir / "adapter_model.safetensors").read_bytes() == (
        old_dir / "adapter_model.safetensors"
    ).read_bytes()
    assert json.loads((current_dir / "adapter_config.json").read_text()) == json.loads(
        (old_dir / "adapter_config.json").read_text()
    )


def test_save_vllm_lora_from_model_writes_single_vllm_checkpoint(tmp_path: Path):
    prefix = "base_model.model.model.layers.0.mlp.experts.0"
    full = {
        f"{prefix}.gate_up_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        f"{prefix}.gate_up_proj.lora_B.weight": torch.arange(
            8,
            dtype=torch.float32,
        ).reshape(8, 1),
        f"{prefix}.down_proj.lora_A.weight": torch.arange(
            4,
            dtype=torch.float32,
        ).reshape(1, 4),
        f"{prefix}.down_proj.lora_B.weight": torch.arange(
            2,
            dtype=torch.float32,
        ).reshape(2, 1),
    }

    gate_up_lora = LoRA(
        adapter_model_prefix=f"{prefix}.gate_up_proj",
        in_features=2,
        out_features=8,
        rank=1,
        alpha=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    gate_up_lora.A_T.data.copy_(full[f"{prefix}.gate_up_proj.lora_A.weight"].T)
    gate_up_lora.B_T.data.copy_(full[f"{prefix}.gate_up_proj.lora_B.weight"].T)
    down_lora = LoRA(
        adapter_model_prefix=f"{prefix}.down_proj",
        in_features=4,
        out_features=2,
        rank=1,
        alpha=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    down_lora.A_T.data.copy_(full[f"{prefix}.down_proj.lora_A.weight"].T)
    down_lora.B_T.data.copy_(full[f"{prefix}.down_proj.lora_B.weight"].T)

    publish_dir = tmp_path / "published_from_model"
    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(gate_up_lora, down_lora)]),
        adapter_dtypes={key: tensor.dtype for key, tensor in full.items()},
        handler=QWEN3_5_MOE_HANDLER,
        adapter_config=_config("Qwen/Qwen3.5-35B-A3B", rank=1, alpha=1),
        output_dir=str(publish_dir),
        rank=0,
        world_size=1,
    )

    assert not list(publish_dir.glob("adapter_model-*-of-*.safetensors"))
    roundtrip = load_lora_tensors_for_megatron(
        str(publish_dir),
        handler=QWEN3_5_MOE_HANDLER,
    )
    _assert_tensors_equal(roundtrip, full)


def test_trainer_rank_publishes_named_checkpoint_slot_without_mutating_base(
    tmp_path: Path,
):
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    lora = LoRA(prefix, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    baseline = (lora.A_T.detach().clone(), lora.B_T.detach().clone())
    adapter = {
        f"{prefix}.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        f"{prefix}.lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        model=[lora],
        model_support_handler=DEFAULT_DENSE_HANDLER,
        rank=0,
        world_size=1,
    )
    trainer._slot_stack = []
    trainer._pending_slot_graphs = {}
    trainer._dynamic_optimizers = {}
    trainer._checkpoint_slot_params_by_name = {}
    trainer._checkpoint_slot_adapter_configs = {}
    trainer._checkpoint_revisions = {}
    config = _config("Qwen/Qwen3-8B", rank=2, alpha=2)
    assert trainer._load_checkpoint_slot("student", adapter, alpha=2) == 1
    trainer._checkpoint_slot_params_by_name["student"] = tuple(
        trainer._iter_slot_parameters(trainer._slot_ref("student"))
    )
    trainer._checkpoint_slot_adapter_configs["student"] = config
    trainer._checkpoint_revisions["student"] = 0
    output_dir = tmp_path / "checkpoint"

    assert trainer.export_lora(str(output_dir), "student") == 0

    _assert_tensors_equal(load_file(output_dir / "adapter_model.safetensors"), adapter)
    assert json.loads((output_dir / "adapter_config.json").read_text()) == {
        **config,
        ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM,
    }
    assert torch.equal(lora.A_T, baseline[0])
    assert torch.equal(lora.B_T, baseline[1])
    with pytest.raises(RuntimeError, match="canonical optimizer state"):
        validate_checkpoint(output_dir, require_optimizer=True)


def test_model_lora_disk_export_streams_one_layer_block_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefixes = [
        f"base_model.model.model.layers.{index}.self_attn.q_proj" for index in range(2)
    ]
    loras = [
        LoRA(prefix, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
        for prefix in prefixes
    ]
    expected: dict[str, torch.Tensor] = {}
    for index, lora in enumerate(loras):
        lora.A_T.data.fill_(index + 1)
        lora.B_T.data.fill_(index + 2)
        expected.update(lora.sharded_lora_state_dict())

    exchanged_blocks: list[set[str]] = []
    original_exchange = lora_publish._exchange_tensors

    def record_exchange(metadata, **kwargs):
        if metadata:
            exchanged_blocks.append({meta.block for meta in metadata})
        return original_exchange(metadata, **kwargs)

    monkeypatch.setattr(lora_publish, "_exchange_tensors", record_exchange)
    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(*loras)]),
        adapter_dtypes={},
        handler=DEFAULT_DENSE_HANDLER,
        adapter_config=_config("Qwen/Qwen3-8B", rank=2, alpha=2),
        output_dir=str(tmp_path),
        rank=0,
        world_size=1,
    )

    assert exchanged_blocks == [
        {prefixes[0].rsplit(".", 2)[0]},
        {prefixes[1].rsplit(".", 2)[0]},
    ]
    _assert_tensors_equal(load_file(tmp_path / "adapter_model.safetensors"), expected)
    assert not list(tmp_path.glob(".adapter_model-*.safetensors"))


def test_checkpoint_load_fetches_only_local_pipeline_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    remote_prefix = "base_model.model.model.layers.1.self_attn.q_proj"
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.lora_{side}.weight": torch.ones(shape)
        for prefix in (local_prefix, remote_prefix)
        for side, shape in (("A", (2, 3)), ("B", (4, 2)))
    }
    source = tmp_path / "two-layers"
    save_vllm_lora_tensors(source, tensors, _config("Qwen/Qwen3-8B", rank=2, alpha=2))
    prepared = prepare_checkpoint(str(source))
    trainer = _portable_trainer(
        LoRA(local_prefix, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )

    lora_disk = importlib.import_module("art.megatron.model_support.lora_disk")
    original_safe_open = lora_disk.safe_open
    fetched: list[str] = []

    class TrackedSafeOpen:
        def __init__(self, *args, **kwargs) -> None:
            self._context = original_safe_open(*args, **kwargs)
            self._file = None

        def __enter__(self):
            self._file = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def get_tensor(self, key: str) -> torch.Tensor:
            fetched.append(key)
            assert self._file is not None
            return self._file.get_tensor(key)

    monkeypatch.setattr(lora_disk, "safe_open", TrackedSafeOpen)
    load_trainer_checkpoint(trainer, prepared, "student")

    assert set(fetched) == {
        f"{local_prefix}.lora_A.weight",
        f"{local_prefix}.lora_B.weight",
    }
    assert not any(remote_prefix in key for key in fetched)


def test_native_megatron_lora_load_bypasses_vllm_decoder(tmp_path: Path) -> None:
    source = tmp_path / "native"
    source.mkdir()
    save_file(_PORTABLE_ADAPTER, source / "adapter_model.safetensors")
    save_adapter_config(
        source,
        {
            **_PORTABLE_CONFIG,
            ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_MEGATRON,
        },
    )
    handler = SimpleNamespace(
        from_vllm_lora_tensors=lambda *_args, **_kwargs: pytest.fail(
            "native checkpoint must not be decoded as vLLM"
        )
    )

    _assert_tensors_equal(
        load_lora_tensors_for_megatron(source, handler=cast(Any, handler)),
        _PORTABLE_ADAPTER,
    )


def test_native_checkpoint_reader_materializes_only_local_shards() -> None:
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    tensor = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    requests: list[tuple[slice, ...]] = []

    class TrackedSlice:
        def get_shape(self) -> list[int]:
            return list(tensor.shape)

        def __getitem__(self, slices: tuple[slice, ...]) -> torch.Tensor:
            requests.append(slices)
            return tensor[slices]

    class SliceOnlyFile:
        def keys(self) -> list[str]:
            return ["weight"]

        def get_slice(self, key: str) -> TrackedSlice:
            assert key == "weight"
            return TrackedSlice()

        def get_tensor(self, key: str) -> torch.Tensor:
            del key
            raise AssertionError("full canonical tensors must not be materialized")

    uniform = checkpoint_module._read_local_slice(
        SliceOnlyFile(),
        "weight",
        cast(
            LoraShardManifest,
            {
                "sharded": True,
                "shard_world_size": 2,
                "shard_rank": 1,
                "export_shard_dim": 0,
                "export_shard_strategy": "uniform",
            },
        ),
        (6, 2),
    )
    torch.testing.assert_close(uniform, tensor[6:12])
    assert requests == [(slice(6, 12), slice(None))]

    requests.clear()
    componentwise = checkpoint_module._read_local_slice(
        SliceOnlyFile(),
        "weight",
        cast(
            LoraShardManifest,
            {
                "sharded": True,
                "shard_world_size": 2,
                "shard_rank": 1,
                "export_shard_dim": 0,
                "export_shard_strategy": "componentwise",
                "component_sizes": (4, 8),
            },
        ),
        (6, 2),
    )
    torch.testing.assert_close(componentwise, torch.cat((tensor[2:4], tensor[8:12])))
    assert requests == [
        (slice(2, 4), slice(None)),
        (slice(8, 12), slice(None)),
    ]


_PORTABLE_PREFIX = "base_model.model.model.layers.0.self_attn.q_proj"
_PORTABLE_CONFIG = _config("Qwen/Qwen3-8B", rank=2, alpha=2)
_PORTABLE_ADAPTER = {
    f"{_PORTABLE_PREFIX}.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(
        2, 3
    ),
    f"{_PORTABLE_PREFIX}.lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(
        4, 2
    ),
}


_PIPELINE_PREFIXES = tuple(
    f"base_model.model.model.layers.{index}.self_attn.q_proj" for index in range(2)
)
_PIPELINE_ADAPTER = {
    f"{prefix}.lora_{side}.weight": tensor + 100 * layer
    for layer, prefix in enumerate(_PIPELINE_PREFIXES)
    for side, tensor in (
        ("A", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("B", torch.arange(8, dtype=torch.float32).reshape(4, 2)),
    )
}
_MOE_PREFIX = "base_model.model.model.layers.0.mlp.experts.{expert}.gate_up_proj"
_MOE_CONFIG = _config("Qwen/Qwen3.5-35B-A3B", rank=2, alpha=2)
_MOE_ADAPTER = {
    f"{_MOE_PREFIX.format(expert=expert)}.lora_{side}.weight": (tensor + 100 * expert)
    for expert in range(2)
    for side, tensor in (
        ("A", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("B", torch.arange(16, dtype=torch.float32).reshape(8, 2)),
    )
}


def _pipeline_loras(*layers: int) -> list[LoRA]:
    return [
        LoRA(
            _PIPELINE_PREFIXES[layer],
            3,
            4,
            2,
            2,
            torch.float32,
            torch.device("cpu"),
        )
        for layer in layers
    ]


def _moe_lora(*, num_local_experts: int, expert_tp: bool = False) -> LoRA:
    a_parallel_spec = LoRAParallelSpec(shard_domain="expert_tp" if expert_tp else "tp")
    b_parallel_spec = LoRAParallelSpec(
        shard_domain="expert_tp" if expert_tp else "tp",
        sharded=expert_tp,
        shard_dim=-1 if expert_tp else None,
    )
    return LoRA(
        _MOE_PREFIX,
        3,
        4 if expert_tp else 8,
        2,
        2,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=num_local_experts,
        a_parallel_spec=a_parallel_spec,
        b_parallel_spec=b_parallel_spec,
    )


def _portable_trainer(
    lora: torch.nn.Module,
    *,
    rank: int = 0,
    world_size: int = 1,
    model_identifier: str = str(_PORTABLE_CONFIG["base_model_name_or_path"]),
    model_names: tuple[str, ...] = (),
) -> TrainerRank:
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        model=[lora],
        model_support_handler=DEFAULT_DENSE_HANDLER,
        rank=rank,
        world_size=world_size,
        model_identifier=model_identifier,
        model_support_spec=SimpleNamespace(model_names=model_names),
    )
    trainer.device = torch.device("cpu")
    trainer._slot_stack = []
    trainer._default_slot_ref = None
    trainer._pending_slot_graphs = {}
    trainer._dynamic_optimizers = {}
    trainer._checkpoint_slot_params_by_name = {}
    trainer._checkpoint_slot_adapter_configs = {}
    trainer._checkpoint_revisions = {}
    return trainer


def _install_checkpoint(
    trainer: TrainerRank,
    adapter: dict[str, torch.Tensor],
    config: dict[str, object],
) -> None:
    alpha = config["lora_alpha"]
    assert isinstance(alpha, int | float)
    assert trainer._load_checkpoint_slot("student", adapter, alpha=float(alpha)) > 0
    trainer._checkpoint_slot_params_by_name["student"] = tuple(
        trainer._iter_slot_parameters(trainer._slot_ref("student"))
    )
    trainer._checkpoint_slot_adapter_configs["student"] = cast(_AdapterConfig, config)
    trainer._checkpoint_revisions["student"] = 0


def _install_portable_checkpoint(trainer: TrainerRank) -> None:
    _install_checkpoint(trainer, _PORTABLE_ADAPTER, _PORTABLE_CONFIG)


def _step_portable_checkpoint(trainer: TrainerRank, gradient: float) -> None:
    dynamic = trainer._dynamic_optimizers["student"]
    for master in dynamic.master_params:
        master.grad = torch.full_like(master, gradient)
    dynamic.optimizer.step()
    dynamic.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for model, master in zip(
            trainer._checkpoint_slot_params_by_name["student"],
            dynamic.master_params,
            strict=True,
        ):
            model.copy_(master)
            model.grad = None


def _assert_checkpoint_state_equal(expected: TrainerRank, actual: TrainerRank) -> None:
    expected_params = expected._checkpoint_slot_params_by_name["student"]
    actual_params = actual._checkpoint_slot_params_by_name["student"]
    for expected_param, actual_param in zip(
        expected_params, actual_params, strict=True
    ):
        torch.testing.assert_close(actual_param, expected_param, atol=0, rtol=0)
    expected_optimizer = expected._dynamic_optimizers["student"]
    actual_optimizer = actual._dynamic_optimizers["student"]
    for expected_master, actual_master in zip(
        expected_optimizer.master_params,
        actual_optimizer.master_params,
        strict=True,
    ):
        torch.testing.assert_close(actual_master, expected_master, atol=0, rtol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual_optimizer.optimizer.state[actual_master][key],
                expected_optimizer.optimizer.state[expected_master][key],
                atol=0,
                rtol=0,
            )


def _save_before_next_step(trainer: TrainerRank, path: Path) -> None:
    trainer._dynamic_optimizers["student"] = trainer._new_dynamic_optimizer(
        "student",
        AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95, weight_decay=0.1),
    )
    _step_portable_checkpoint(trainer, 0.25)
    trainer.save_checkpoint(str(path), "student")
    _step_portable_checkpoint(trainer, -0.125)


def test_single_local_expert_uses_global_expert_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)
    module = _moe_lora(num_local_experts=1)
    ref = LoRASlotRef("student")
    expected = {
        key: tensor for key, tensor in _MOE_ADAPTER.items() if ".experts.1." in key
    }

    assert module._expected_weight_keys("lora_A") == [
        f"{_MOE_PREFIX.format(expert=1)}.lora_A.weight"
    ]
    assert module.load_lora_slot(ref, expected, alpha=2)
    _assert_tensors_equal(module.sharded_lora_state_dict(ref), expected)
    assert set(module.sharded_lora_manifest(ref)) == set(expected)


def test_checkpoint_load_coordinates_rank_local_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    save_vllm_lora_tensors(source, _PORTABLE_ADAPTER, _PORTABLE_CONFIG)
    prepared = prepare_checkpoint(str(source))
    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    monkeypatch.setattr(
        checkpoint_module,
        "_load_stage_lora",
        lambda *_args: (_ for _ in ()).throw(OSError("rank-local read failed")),
    )

    def coordinated(error: BaseException | None, phase: str) -> None:
        assert phase == "read checkpoint"
        assert isinstance(error, OSError)
        raise RuntimeError("coordinated read failure") from error

    monkeypatch.setattr(checkpoint_module, "_raise_distributed", coordinated)
    with pytest.raises(RuntimeError, match="coordinated read failure"):
        load_trainer_checkpoint(trainer, prepared, "student")


def test_checkpoint_load_rejects_same_keys_with_different_rank_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / "first", tmp_path / "second"]
    save_vllm_lora_tensors(paths[0], _PORTABLE_ADAPTER, _PORTABLE_CONFIG)
    save_vllm_lora_tensors(
        paths[1],
        {key: value + 1 for key, value in _PORTABLE_ADAPTER.items()},
        _PORTABLE_CONFIG,
    )
    first, second = (prepare_checkpoint(str(path)) for path in paths)
    assert first.artifact_keys == second.artifact_keys
    assert first.digest != second.digest
    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    monkeypatch.setattr(
        checkpoint_module,
        "_gather_objects",
        lambda value: (value, second.digest),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_load_stage_lora",
        lambda *_args: pytest.fail("mismatched content must fail before reading"),
    )

    with pytest.raises(RuntimeError, match="content differs across ranks"):
        load_trainer_checkpoint(trainer, first, "student")


def test_checkpoint_payload_validation_elects_one_rank_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    monkeypatch.setattr(checkpoint_module, "_distributed", lambda: True)
    monkeypatch.setattr(checkpoint_module, "_rank", lambda: 7)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert checkpoint_module._is_node_validator()
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert not checkpoint_module._is_node_validator()
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert checkpoint_module._is_node_validator()


def test_prepared_checkpoint_pins_a_symlink_target(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_vllm_lora_tensors(first, _PORTABLE_ADAPTER, _PORTABLE_CONFIG)
    save_vllm_lora_tensors(
        second,
        {key: tensor + 100 for key, tensor in _PORTABLE_ADAPTER.items()},
        _PORTABLE_CONFIG,
    )
    latest = tmp_path / "latest"
    latest.symlink_to(first, target_is_directory=True)
    prepared = prepare_checkpoint(str(latest))
    latest.unlink()
    latest.symlink_to(second, target_is_directory=True)

    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    assert prepared.path == first.resolve()
    _assert_tensors_equal(
        checkpoint_module._load_stage_lora(trainer, prepared), _PORTABLE_ADAPTER
    )


def test_native_checkpoint_rejects_unmapped_optimizer_tensor(
    tmp_path: Path,
) -> None:
    original = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(original)
    original._dynamic_optimizers["student"] = original._new_dynamic_optimizer(
        "student", AdamParams(learning_rate=3e-4)
    )
    _step_portable_checkpoint(original, 0.25)
    output = tmp_path / "exact-with-extra"
    original.save_checkpoint(str(output), "student")

    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    manifest = validate_checkpoint(output, require_optimizer=True)
    assert manifest is not None and manifest.optimizer is not None
    source_key = next(iter(manifest.parameters))
    extra_key = f"{source_key}.unmapped"
    adapter = load_file(output / "adapter_model.safetensors")
    adapter[extra_key] = adapter[source_key].clone()
    save_file(adapter, output / "adapter_model.safetensors")

    source_record = manifest.parameters[source_key]
    component_updates = {}
    for component in ("master", "exp_avg", "exp_avg_sq"):
        record = getattr(source_record, component)
        tensors = load_file(output / record.file)
        tensors[extra_key] = tensors[source_key].clone()
        save_file(tensors, output / record.file)
        component_updates[component] = record.model_copy(update={"tensor": extra_key})
    parameters = {
        **manifest.parameters,
        extra_key: source_record.model_copy(update=component_updates),
    }
    unsigned = manifest.model_copy(
        update={
            "parameters": parameters,
            "steps": {**manifest.steps, extra_key: manifest.steps[source_key]},
            "digest": "",
        }
    )
    checkpoint_module._write_manifest(
        output,
        unsigned.model_copy(
            update={"digest": checkpoint_module._digest(output, unsigned)}
        ),
    )

    prepared = prepare_checkpoint(str(output))
    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    with pytest.raises(RuntimeError, match="coverage differs from the target runtime"):
        load_trainer_checkpoint(restored, prepared, "student")
    assert not restored._checkpoint_slot_params_by_name


@pytest.mark.parametrize("with_optimizer", [False, True])
def test_canonical_checkpoint_rejects_different_model_in_same_support_spec(
    tmp_path: Path,
    with_optimizer: bool,
) -> None:
    original = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(original)
    if with_optimizer:
        original._dynamic_optimizers["student"] = original._new_dynamic_optimizer(
            "student", AdamParams(learning_rate=3e-4)
        )
    output = tmp_path / f"canonical-model-{with_optimizer}"
    original.save_checkpoint(str(output), "student")

    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu")),
        model_identifier="Qwen/Qwen3-8B-Base",
        model_names=("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"),
    )
    with pytest.raises(RuntimeError, match="Exact checkpoint base model"):
        load_trainer_checkpoint(restored, prepare_checkpoint(str(output)), "student")
    assert not restored._checkpoint_slot_params_by_name


def test_lora_only_checkpoint_rejects_different_model_in_same_support_spec(
    tmp_path: Path,
) -> None:
    source = tmp_path / "lora-only-different-model"
    save_vllm_lora_tensors(source, _PORTABLE_ADAPTER, _PORTABLE_CONFIG)
    prepared = prepare_checkpoint(str(source))
    assert prepared.manifest is None

    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu")),
        model_identifier="Qwen/Qwen3-8B-Base",
        model_names=("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"),
    )
    with pytest.raises(RuntimeError, match="Checkpoint base model"):
        load_trainer_checkpoint(restored, prepared, "student")
    assert not restored._checkpoint_slot_params_by_name


def test_checkpoint_prepare_captures_immutable_state_and_finish_is_idempotent(
    tmp_path: Path,
) -> None:
    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(trainer)
    trainer._dynamic_optimizers["student"] = trainer._new_dynamic_optimizer(
        "student",
        AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95),
    )
    _step_portable_checkpoint(trainer, 0.25)
    expected_params = tuple(
        value.detach().clone()
        for value in trainer._checkpoint_slot_params_by_name["student"]
    )
    dynamic = trainer._dynamic_optimizers["student"]
    expected_masters = tuple(value.detach().clone() for value in dynamic.master_params)
    expected_states = tuple(
        {
            key: value.detach().clone()
            for key, value in dynamic.optimizer.state[master].items()
            if isinstance(value, torch.Tensor)
        }
        for master in dynamic.master_params
    )

    output = tmp_path / "immutable"
    trainer._prepare_checkpoint_save(str(output), "student")
    with torch.no_grad():
        for value in trainer._checkpoint_slot_params_by_name["student"]:
            value.add_(100)
        for master in dynamic.master_params:
            master.add_(200)
            for value in dynamic.optimizer.state[master].values():
                if isinstance(value, torch.Tensor):
                    value.add_(300)
    trainer._finish_checkpoint_save(str(output))
    trainer._finish_checkpoint_save(str(output))

    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    load_trainer_checkpoint(
        restored,
        prepare_checkpoint(str(output)),
        "student",
    )
    for expected, actual in zip(
        expected_params,
        restored._checkpoint_slot_params_by_name["student"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    restored_dynamic = restored._dynamic_optimizers["student"]
    for expected, expected_state, actual in zip(
        expected_masters,
        expected_states,
        restored_dynamic.master_params,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for key, value in expected_state.items():
            torch.testing.assert_close(
                restored_dynamic.optimizer.state[actual][key],
                value,
                atol=0,
                rtol=0,
            )
    assert not list(tmp_path.glob(".immutable.snapshot-*"))


def test_checkpoint_finalizers_run_fifo_and_clean_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(trainer)
    trainer._dynamic_optimizers["student"] = trainer._new_dynamic_optimizer(
        "student", AdamParams(learning_rate=3e-4)
    )
    _step_portable_checkpoint(trainer, 0.25)
    first = tmp_path / "fifo-first"
    second = tmp_path / "fifo-second"
    trainer._prepare_checkpoint_save(str(first), "student")
    trainer._prepare_checkpoint_save(str(second), "student")

    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    original_finish = checkpoint_module._finish_prepared_save
    order: list[str] = []

    def record_finish(trainer_rank: TrainerRank, prepared: _PreparedSave) -> None:
        order.append(prepared.destination.name)
        original_finish(trainer_rank, prepared)

    monkeypatch.setattr(checkpoint_module, "_finish_prepared_save", record_finish)
    second_started = threading.Event()
    second_done = threading.Event()

    def finish_second() -> None:
        second_started.set()
        trainer._finish_checkpoint_save(str(second))
        second_done.set()

    second_thread = threading.Thread(target=finish_second)
    second_thread.start()
    assert second_started.wait(1)
    assert not second_done.wait(0.1)
    first_thread = threading.Thread(
        target=trainer._finish_checkpoint_save,
        args=(str(first),),
    )
    first_thread.start()
    first_thread.join(10)
    second_thread.join(10)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert order == ["fifo-first", "fifo-second"]
    assert first.is_dir() and second.is_dir()
    assert not list(tmp_path.glob(".*.snapshot-*"))


def test_checkpoint_prepare_failure_does_not_advance_or_wedge_fifo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(trainer)
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    original_save_file = checkpoint_module._save_file

    def fail_snapshot(*_args: object) -> None:
        raise OSError("injected checkpoint snapshot failure")

    monkeypatch.setattr(checkpoint_module, "_save_file", fail_snapshot)
    failed = tmp_path / "prepare-failure"
    with pytest.raises(OSError, match="injected checkpoint snapshot failure"):
        trainer._prepare_checkpoint_save(str(failed), "student")
    assert trainer._checkpoint_save_sequence == 0
    assert trainer._checkpoint_finish_sequence == 0
    assert not trainer._prepared_checkpoint_saves
    assert not list(tmp_path.glob(".prepare-failure.snapshot-*"))

    monkeypatch.setattr(checkpoint_module, "_save_file", original_save_file)
    recovered = tmp_path / "prepare-recovered"
    trainer.save_checkpoint(str(recovered), "student")
    assert recovered.is_dir()
    assert trainer._checkpoint_save_sequence == 1
    assert trainer._checkpoint_finish_sequence == 1


def test_checkpoint_finalizer_failure_cleans_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(trainer)
    output = tmp_path / "retry"
    trainer._prepare_checkpoint_save(str(output), "student")
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    original_finish = checkpoint_module._finish_prepared_save

    def fail_finish(*_args: object) -> None:
        raise OSError("injected checkpoint finalization failure")

    monkeypatch.setattr(checkpoint_module, "_finish_prepared_save", fail_finish)
    with pytest.raises(OSError, match="injected checkpoint finalization failure"):
        trainer._finish_checkpoint_save(str(output))
    assert not output.exists()
    assert not list(tmp_path.glob(".retry.snapshot-*"))

    monkeypatch.setattr(checkpoint_module, "_finish_prepared_save", original_finish)
    trainer._prepare_checkpoint_save(str(output), "student")
    trainer._finish_checkpoint_save(str(output))
    assert output.is_dir()
    assert not list(tmp_path.glob(".retry.snapshot-*"))


def test_trainer_rank_checkpoint_roundtrip_preserves_next_adam_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    original = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(original)
    original._dynamic_optimizers["student"] = original._new_dynamic_optimizer(
        "student",
        AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95, weight_decay=0.1),
    )
    unstepped_output = tmp_path / "unstepped"
    original.save_checkpoint(str(unstepped_output), "student")
    unstepped = prepare_checkpoint(str(unstepped_output))
    assert unstepped.manifest is not None
    assert set(unstepped.manifest.steps.values()) == {0.0}
    _step_portable_checkpoint(original, 0.25)
    original_optimizer = original._dynamic_optimizers["student"]
    for step, master in enumerate(original_optimizer.master_params, start=2):
        original_optimizer.optimizer.state[master]["step"].fill_(step)
    output = tmp_path / "exact"

    original.save_checkpoint(str(output), "student")
    original.save_checkpoint(str(output), "student")
    manifest = validate_checkpoint(output, require_optimizer=True)
    assert manifest is not None and manifest.optimizer is not None
    assert set(manifest.steps.values()) == {2.0, 3.0}
    checkpoint_module = importlib.import_module("art.trainer_rank._checkpoint")
    parameter_key, parameter = next(iter(manifest.parameters.items()))
    for invalid_path in (
        "",
        ".",
        "../escape.safetensors",
        "/tmp/escape.safetensors",
        "optimizer/../escape.safetensors",
        "optimizer\\escape.safetensors",
        "C:/escape.safetensors",
        "optimizer//escape.safetensors",
        "./optimizer/escape.safetensors",
    ):
        invalid_parameter = parameter.model_copy(
            update={
                "master": parameter.master.model_copy(update={"file": invalid_path})
            }
        )
        invalid_manifest = manifest.model_copy(
            update={
                "parameters": {
                    **manifest.parameters,
                    parameter_key: invalid_parameter,
                }
            }
        )
        with pytest.raises(RuntimeError, match="Unsafe checkpoint tensor path"):
            checkpoint_module._validate_manifest(
                invalid_manifest, tuple(manifest.parameters)
            )
    source_adapter = (output / "adapter_model.safetensors").read_bytes()
    source_config = (output / "adapter_config.json").read_bytes()
    assert json.loads(source_config)[ART_LORA_FORMAT_CONFIG_KEY] == (
        ART_LORA_FORMAT_MEGATRON
    )
    _assert_tensors_equal(
        load_lora_tensors_for_megatron(output),
        load_file(output / "adapter_model.safetensors"),
    )
    assert len(list((output / "optimizer").glob("*.safetensors"))) == 3
    malformed = tmp_path / "malformed"
    shutil.copytree(output, malformed)
    malformed_manifest = json.loads((malformed / "checkpoint.json").read_text())
    first = next(iter(malformed_manifest["parameters"].values()))
    first["master"]["shape"][0] += 1
    (malformed / "checkpoint.json").write_text(json.dumps(malformed_manifest))
    with pytest.raises(RuntimeError, match="tensor metadata mismatch"):
        validate_checkpoint(malformed, require_optimizer=True)
    inference = tmp_path / "inference"
    materialize_lora(output, inference, require_optimizer=True)
    assert (output / "adapter_model.safetensors").read_bytes() == source_adapter
    assert (output / "adapter_config.json").read_bytes() == source_config
    assert (
        json.loads((inference / "adapter_config.json").read_text())[
            ART_LORA_FORMAT_CONFIG_KEY
        ]
        == ART_LORA_FORMAT_VLLM
    )
    assert {item.name for item in inference.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    entries = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file()
    }
    staged = tmp_path / "selective-stage"
    staged.mkdir()
    for relative in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint.json",
    ):
        shutil.copy2(output / relative, staged / relative)
    selective = tmp_path / "selective-inference"
    materialize_lora(
        staged,
        selective,
        require_optimizer=True,
        artifact_entries=entries,
        expected_digest=manifest.digest,
    )
    assert {item.name for item in selective.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    with pytest.raises(RuntimeError, match="missing referenced files"):
        materialize_lora(
            staged,
            tmp_path / "missing-entry",
            require_optimizer=True,
            artifact_entries=entries
            - {next(iter(manifest.parameters.values())).master.file},
            expected_digest=manifest.digest,
        )
    with pytest.raises(RuntimeError, match="durable artifact digest"):
        materialize_lora(
            staged,
            tmp_path / "wrong-digest",
            require_optimizer=True,
            artifact_entries=entries,
            expected_digest="wrong",
        )
    with pytest.raises(FileExistsError, match="not empty"):
        materialize_lora(output, inference, require_optimizer=True)

    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    source = prepare_checkpoint(str(output))
    load_trainer_checkpoint(restored, source, "student")
    assert source.manifest is not None and source.manifest.optimizer is not None
    assert restored._checkpoint_revisions["student"] == 0

    _step_portable_checkpoint(original, -0.125)
    _step_portable_checkpoint(restored, -0.125)
    for expected, actual in zip(
        original._checkpoint_slot_params_by_name["student"],
        restored._checkpoint_slot_params_by_name["student"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    with pytest.raises(FileExistsError, match="not empty"):
        original.save_checkpoint(str(output), "student")

    failed = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(failed)
    before = tuple(
        parameter.detach().clone()
        for parameter in failed._checkpoint_slot_params_by_name["student"]
    )

    tampered_dtype = tmp_path / "tampered-dtype"
    shutil.copytree(output, tampered_dtype)
    tampered_manifest = json.loads((tampered_dtype / "checkpoint.json").read_text())
    tampered_parameter = next(iter(tampered_manifest["parameters"].values()))
    tampered_parameter["exp_avg"]["dtype"] = "float16"
    (tampered_dtype / "checkpoint.json").write_text(json.dumps(tampered_manifest))

    async def load_tampered_dtype() -> None:
        await failed.load_checkpoint(str(tampered_dtype))

    with pytest.raises(RuntimeError, match="must use float32"):
        asyncio.run(load_tampered_dtype())
    assert set(failed._checkpoint_slot_params_by_name) == {"student"}
    assert failed._default_slot_ref is None
    for expected, actual in zip(
        before,
        failed._checkpoint_slot_params_by_name["student"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def reject_optimizer(*_args: object) -> None:
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(failed, "_restore_canonical_optimizer", reject_optimizer)
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        load_trainer_checkpoint(failed, source, "student")
    assert set(failed._checkpoint_slot_params_by_name) == {"student"}
    for expected, actual in zip(
        before,
        failed._checkpoint_slot_params_by_name["student"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _portable_topology_worker(
    rank: int,
    world_size: int,
    init_method: str,
    source_path: str,
    output_path: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        lora_module.ps.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        import art.trainer_rank._checkpoint as checkpoint_module

        setattr(checkpoint_module, "_device", lambda: torch.device("cpu"))
        setattr(
            lora_publish,
            "_rank_and_device",
            lambda: (rank, torch.device("cpu")),
        )
        lora = LoRA(
            _PORTABLE_PREFIX,
            3,
            4 // world_size,
            2,
            2,
            torch.float32,
            torch.device("cpu"),
            b_parallel_spec=LoRAParallelSpec(sharded=True, shard_dim=-1),
        )
        trainer = _portable_trainer(lora, rank=rank, world_size=world_size)
        load_trainer_checkpoint(
            trainer,
            prepare_checkpoint(source_path),
            "student",
        )
        _step_portable_checkpoint(trainer, -0.125)
        trainer.save_checkpoint(output_path, "student")
    finally:
        if lora_module.ps.model_parallel_is_initialized():
            lora_module.ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _advanced_topology_worker(
    rank: int,
    world_size: int,
    init_method: str,
    source_path: str,
    output_path: str,
    topology: Literal["pp", "ep", "etp", "cp"],
    counts_path: str | None,
) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        lora_module.ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2 if topology == "pp" else 1,
            context_parallel_size=2 if topology == "cp" else 1,
            expert_model_parallel_size=2 if topology == "ep" else 1,
            expert_tensor_parallel_size=2 if topology == "etp" else None,
        )
        if topology == "pp":
            modules = _pipeline_loras(rank)
            model_identifier = str(_PORTABLE_CONFIG["base_model_name_or_path"])
        elif topology == "ep":
            modules = [_moe_lora(num_local_experts=1)]
            model_identifier = str(_MOE_CONFIG["base_model_name_or_path"])
        elif topology == "etp":
            modules = [_moe_lora(num_local_experts=2, expert_tp=True)]
            model_identifier = str(_MOE_CONFIG["base_model_name_or_path"])
        else:
            modules = [
                LoRA(
                    _PORTABLE_PREFIX,
                    3,
                    4,
                    2,
                    2,
                    torch.float32,
                    torch.device("cpu"),
                )
            ]
            model_identifier = str(_PORTABLE_CONFIG["base_model_name_or_path"])
        trainer = _portable_trainer(
            torch.nn.Sequential(*modules),
            rank=rank,
            world_size=world_size,
            model_identifier=model_identifier,
        )
        setattr(lora_publish, "_rank_and_device", lambda: (rank, torch.device("cpu")))
        load_trainer_checkpoint(trainer, prepare_checkpoint(source_path), "student")
        _step_portable_checkpoint(trainer, -0.125)

        sends = 0
        original_send = torch.distributed.send

        def counted_send(*args, **kwargs):
            nonlocal sends
            sends += 1
            return original_send(*args, **kwargs)

        torch.distributed.send = counted_send
        trainer.save_checkpoint(output_path, "student")
        if counts_path is not None:
            counts: list[int | None] = [None] * world_size
            torch.distributed.all_gather_object(counts, sends)
            if rank == 0:
                Path(counts_path).write_text(json.dumps(counts))
    finally:
        if lora_module.ps.model_parallel_is_initialized():
            lora_module.ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _reversed_prefetch_order_worker(
    rank: int,
    world_size: int,
    init_method: str,
    first_path: str,
    second_path: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        lora_module.ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        lora = LoRA(
            _PORTABLE_PREFIX,
            3,
            4,
            2,
            2,
            torch.float32,
            torch.device("cpu"),
        )
        trainer = _portable_trainer(lora, rank=rank, world_size=world_size)
        import art.trainer_rank._checkpoint as checkpoint_module

        original_prepare = checkpoint_module.prepare_checkpoint
        preparation_barrier = threading.Barrier(2)
        completed: list[str] = []

        def reversed_prepare(path: str):
            preparation_barrier.wait(timeout=10)
            slow_path = first_path if rank == 0 else second_path
            if path == slow_path:
                time.sleep(0.25)
            source = original_prepare(path)
            completed.append(path)
            return source

        setattr(checkpoint_module, "prepare_checkpoint", reversed_prepare)

        async def load_both() -> None:
            first = trainer.load_checkpoint(first_path)
            second = trainer.load_checkpoint(second_path)
            await asyncio.wait_for(asyncio.gather(first, second), timeout=30)

        asyncio.run(load_both())
        assert trainer._default_slot_ref == trainer._slot_ref(second_path)
        assert set(trainer._checkpoint_slot_params_by_name) == {
            first_path,
            second_path,
        }
        orders: list[list[str] | None] = [None] * world_size
        torch.distributed.all_gather_object(orders, completed)
        assert orders[0] is not None and orders[0][0] == second_path
        assert orders[1] is not None and orders[1][0] == first_path
    finally:
        if lora_module.ps.model_parallel_is_initialized():
            lora_module.ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _transactional_load_failure_worker(
    rank: int,
    world_size: int,
    init_method: str,
    source_path: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        lora_module.ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        lora = LoRA(
            _PORTABLE_PREFIX,
            3,
            4,
            2,
            2,
            torch.float32,
            torch.device("cpu"),
        )
        trainer = _portable_trainer(lora, rank=rank, world_size=world_size)
        _install_portable_checkpoint(trainer)
        trainer._dynamic_optimizers["student"] = trainer._new_dynamic_optimizer(
            "student", AdamParams(learning_rate=1e-3)
        )
        _step_portable_checkpoint(trainer, -0.5)
        previous_params = trainer._checkpoint_slot_params_by_name["student"]
        previous_values = tuple(
            parameter.detach().clone() for parameter in previous_params
        )
        previous_dynamic = trainer._dynamic_optimizers["student"]
        original_replace = lora_module.replace_lora_slot_in_model
        if rank == 1:

            def fail_after_replacement(
                model: Sequence[torch.nn.Module],
                source: LoRASlotRef,
                destination: LoRASlotRef,
            ) -> None:
                original_replace(model, source, destination)
                raise RuntimeError("injected checkpoint commit failure")

            setattr(lora_module, "replace_lora_slot_in_model", fail_after_replacement)

        with pytest.raises(
            RuntimeError, match="checkpoint commit failure|failed to commit"
        ):
            load_trainer_checkpoint(trainer, prepare_checkpoint(source_path), "student")

        restored_params = trainer._checkpoint_slot_params_by_name["student"]
        assert tuple(map(id, restored_params)) == tuple(map(id, previous_params))
        for expected, actual in zip(previous_values, restored_params, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert trainer._dynamic_optimizers["student"] is previous_dynamic
        assert trainer._checkpoint_slot_adapter_configs["student"] == _PORTABLE_CONFIG
        assert trainer._checkpoint_revisions["student"] == 0
        assert all(
            not (ref.name or "").startswith("__art_loading_") for ref in lora._slot_keys
        )
    finally:
        if lora_module.ps.model_parallel_is_initialized():
            lora_module.ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _portable_replica_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_path: str,
    counts_path: str,
    diverge: Literal["lora", "optimizer"] | None = None,
) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        lora_module.ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        import art.trainer_rank._checkpoint as checkpoint_module

        setattr(checkpoint_module, "_device", lambda: torch.device("cpu"))
        setattr(
            lora_publish,
            "_rank_and_device",
            lambda: (rank, torch.device("cpu")),
        )
        lora = LoRA(
            _PORTABLE_PREFIX,
            3,
            4,
            2,
            2,
            torch.float32,
            torch.device("cpu"),
        )
        setattr(lora, "_should_export_parameter", lambda _param: True)
        trainer = _portable_trainer(lora, rank=rank, world_size=world_size)
        _install_portable_checkpoint(trainer)
        trainer._dynamic_optimizers["student"] = trainer._new_dynamic_optimizer(
            "student",
            AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95, weight_decay=0.1),
        )
        _step_portable_checkpoint(trainer, 0.25)
        if diverge == "lora" and rank == 1:
            with torch.no_grad():
                trainer._checkpoint_slot_params_by_name["student"][0].add_(1)
        elif diverge == "optimizer" and rank == 1:
            dynamic = trainer._dynamic_optimizers["student"]
            master = dynamic.master_params[0]
            dynamic.optimizer.state[master]["exp_avg"].add_(1)

        sends = 0
        original_send = torch.distributed.send

        def counted_send(*args, **kwargs):
            nonlocal sends
            sends += 1
            return original_send(*args, **kwargs)

        torch.distributed.send = counted_send
        trainer.save_checkpoint(output_path, "student")
        counts: list[int | None] = [None] * world_size
        torch.distributed.all_gather_object(counts, sends)
        if rank == 0:
            Path(counts_path).write_text(json.dumps(counts))
    finally:
        if lora_module.ps.model_parallel_is_initialized():
            lora_module.ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _exchange_preparation_failure_worker(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        if rank == 1:

            def fail_preparation(*_args: object, **_kwargs: object) -> None:
                raise OSError("injected tensor exchange preparation failure")

            setattr(
                lora_publish,
                "_prepare_exchange_buffers",
                fail_preparation,
            )
        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        metadata = (
            LoraShardMeta(
                key=key,
                owner_rank=1,
                shape=(2,),
                dtype_name="float32",
                manifest={"sharded": False, "shard_world_size": 1, "shard_rank": 0},
                block="base_model.model.model.layers.0",
            ),
        )
        with pytest.raises(
            (OSError, RuntimeError),
            match="injected tensor exchange preparation failure",
        ):
            lora_publish._exchange_tensors(
                metadata,
                local_tensors={key: torch.ones(2)} if rank == 1 else {},
                rank=rank,
                device=torch.device("cpu"),
            )
    finally:
        torch.distributed.destroy_process_group()


def test_lora_exchange_coordinates_asymmetric_preparation_failure(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _exchange_preparation_failure_worker,
        args=(2, f"file://{tmp_path / 'exchange-failure-init'}"),
        nprocs=2,
        join=True,
    )


def test_checkpoint_loads_follow_call_order_across_ranks(tmp_path: Path) -> None:
    first = tmp_path / "prefetch-first"
    second = tmp_path / "prefetch-second"
    save_vllm_lora_tensors(first, _PORTABLE_ADAPTER, _PORTABLE_CONFIG)
    save_vllm_lora_tensors(
        second,
        {key: value + 1 for key, value in _PORTABLE_ADAPTER.items()},
        _PORTABLE_CONFIG,
    )

    mp.spawn(
        _reversed_prefetch_order_worker,
        args=(
            2,
            f"file://{tmp_path / 'prefetch-order-init'}",
            str(first),
            str(second),
        ),
        nprocs=2,
        join=True,
    )


def test_checkpoint_commit_failure_rolls_back_every_rank(tmp_path: Path) -> None:
    source_trainer = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(source_trainer)
    source_trainer._dynamic_optimizers["student"] = (
        source_trainer._new_dynamic_optimizer("student", AdamParams(learning_rate=3e-4))
    )
    _step_portable_checkpoint(source_trainer, 0.25)
    source = tmp_path / "transaction-source"
    source_trainer.save_checkpoint(str(source), "student")

    mp.spawn(
        _transactional_load_failure_worker,
        args=(
            2,
            f"file://{tmp_path / 'transaction-init'}",
            str(source),
        ),
        nprocs=2,
        join=True,
    )


def test_trainer_rank_checkpoint_deduplicates_data_parallel_replicas(
    tmp_path: Path,
) -> None:
    expected = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(expected)
    expected._dynamic_optimizers["student"] = expected._new_dynamic_optimizer(
        "student",
        AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95, weight_decay=0.1),
    )
    _step_portable_checkpoint(expected, 0.25)

    output = tmp_path / "replicated"
    counts = tmp_path / "replicated-counts.json"
    mp.spawn(
        _portable_replica_worker,
        args=(
            2,
            f"file://{tmp_path / 'replica-init'}",
            str(output),
            str(counts),
            None,
        ),
        nprocs=2,
        join=True,
    )

    assert json.loads(counts.read_text()) == [0, 0]
    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    load_trainer_checkpoint(
        restored,
        prepare_checkpoint(str(output)),
        "student",
    )
    for expected_param, actual_param in zip(
        expected._checkpoint_slot_params_by_name["student"],
        restored._checkpoint_slot_params_by_name["student"],
        strict=True,
    ):
        torch.testing.assert_close(actual_param, expected_param, atol=0, rtol=0)
    expected_optimizer = expected._dynamic_optimizers["student"]
    actual_optimizer = restored._dynamic_optimizers["student"]
    for expected_master, actual_master in zip(
        expected_optimizer.master_params,
        actual_optimizer.master_params,
        strict=True,
    ):
        torch.testing.assert_close(actual_master, expected_master, atol=0, rtol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual_optimizer.optimizer.state[actual_master][key],
                expected_optimizer.optimizer.state[expected_master][key],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("diverge", ["lora", "optimizer"])
def test_trainer_rank_checkpoint_rejects_divergent_data_parallel_replicas(
    tmp_path: Path,
    diverge: Literal["lora", "optimizer"],
) -> None:
    output = tmp_path / "replica-mismatch"
    with pytest.raises(
        ProcessRaisedException, match="Inconsistent replicated tensor contents"
    ):
        mp.spawn(
            _portable_replica_worker,
            args=(
                2,
                f"file://{tmp_path / 'replica-mismatch-init'}",
                str(output),
                str(tmp_path / "unused-counts.json"),
                diverge,
            ),
            nprocs=2,
            join=True,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".replica-mismatch.snapshot-*"))


def test_trainer_rank_checkpoint_restores_1_to_2_to_1(
    tmp_path: Path,
) -> None:
    original = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(original)
    original._dynamic_optimizers["student"] = original._new_dynamic_optimizer(
        "student",
        AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95, weight_decay=0.1),
    )
    _step_portable_checkpoint(original, 0.25)
    one_rank = tmp_path / "one-rank"
    original.save_checkpoint(str(one_rank), "student")
    _step_portable_checkpoint(original, -0.125)

    two_rank = tmp_path / "two-rank"
    mp.spawn(
        _portable_topology_worker,
        args=(
            2,
            f"file://{tmp_path / 'topology-init'}",
            str(one_rank),
            str(two_rank),
        ),
        nprocs=2,
        join=True,
    )

    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    load_trainer_checkpoint(
        restored,
        prepare_checkpoint(str(two_rank)),
        "student",
    )
    for expected, actual in zip(
        original._checkpoint_slot_params_by_name["student"],
        restored._checkpoint_slot_params_by_name["student"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_optimizer = original._dynamic_optimizers["student"]
    actual_optimizer = restored._dynamic_optimizers["student"]
    for expected, actual in zip(
        expected_optimizer.master_params,
        actual_optimizer.master_params,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected_state = expected_optimizer.optimizer.state[expected]
        actual_state = actual_optimizer.optimizer.state[actual]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual_state[key], expected_state[key], atol=0, rtol=0
            )


def test_checkpoint_restores_pipeline_parallel_next_step(tmp_path: Path) -> None:
    original = _portable_trainer(torch.nn.Sequential(*_pipeline_loras(0, 1)))
    _install_checkpoint(original, _PIPELINE_ADAPTER, _PORTABLE_CONFIG)
    source = tmp_path / "pipeline-source"
    _save_before_next_step(original, source)
    output = tmp_path / "pipeline-output"

    mp.spawn(
        _advanced_topology_worker,
        args=(
            2,
            f"file://{tmp_path / 'pipeline-init'}",
            str(source),
            str(output),
            "pp",
            None,
        ),
        nprocs=2,
        join=True,
    )

    restored = _portable_trainer(torch.nn.Sequential(*_pipeline_loras(0, 1)))
    load_trainer_checkpoint(restored, prepare_checkpoint(str(output)), "student")
    _assert_checkpoint_state_equal(original, restored)


@pytest.mark.parametrize("topology", ["ep", "etp"])
def test_checkpoint_restores_expert_parallel_next_step(
    tmp_path: Path, topology: Literal["ep", "etp"]
) -> None:
    original = _portable_trainer(
        _moe_lora(num_local_experts=2),
        model_identifier=str(_MOE_CONFIG["base_model_name_or_path"]),
    )
    _install_checkpoint(original, _MOE_ADAPTER, _MOE_CONFIG)
    source = tmp_path / f"{topology}-source"
    _save_before_next_step(original, source)
    output = tmp_path / f"{topology}-output"

    mp.spawn(
        _advanced_topology_worker,
        args=(
            2,
            f"file://{tmp_path / f'{topology}-init'}",
            str(source),
            str(output),
            topology,
            None,
        ),
        nprocs=2,
        join=True,
    )

    restored = _portable_trainer(
        _moe_lora(num_local_experts=2),
        model_identifier=str(_MOE_CONFIG["base_model_name_or_path"]),
    )
    load_trainer_checkpoint(restored, prepare_checkpoint(str(output)), "student")
    _assert_checkpoint_state_equal(original, restored)


def test_checkpoint_deduplicates_context_parallel_replicas(
    tmp_path: Path,
) -> None:
    original = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    _install_portable_checkpoint(original)
    source = tmp_path / "context-source"
    _save_before_next_step(original, source)
    output = tmp_path / "context-output"
    counts = tmp_path / "context-counts.json"

    mp.spawn(
        _advanced_topology_worker,
        args=(
            2,
            f"file://{tmp_path / 'context-init'}",
            str(source),
            str(output),
            "cp",
            str(counts),
        ),
        nprocs=2,
        join=True,
    )

    restored = _portable_trainer(
        LoRA(_PORTABLE_PREFIX, 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    )
    load_trainer_checkpoint(restored, prepare_checkpoint(str(output)), "student")
    _assert_checkpoint_state_equal(original, restored)
    assert json.loads(counts.read_text()) == [0, 0]


@pytest.mark.parametrize(
    ("handler", "base_model"),
    (
        (QWEN3_5_MOE_HANDLER, "Qwen/Qwen3.5-35B-A3B"),
        (DSV4_HANDLER, "deepseek-ai/DeepSeek-V4-Flash"),
    ),
)
@pytest.mark.parametrize("dynamic_slot", [False, True])
def test_direct_3d_packed_expert_publish_matches_handler_vllm_exactly(
    tmp_path: Path,
    monkeypatch,
    handler,
    base_model: str,
    dynamic_slot: bool,
):
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)

    rank = 2
    hidden = 3
    intermediate = 4
    group_prefix = "base_model.model.model.layers.0.mlp.experts"
    full: dict[str, torch.Tensor] = {}
    gate_up_lora = LoRA(
        adapter_model_prefix=f"{group_prefix}.{{expert}}.gate_up_proj",
        in_features=hidden,
        out_features=2 * intermediate,
        rank=rank,
        alpha=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_local_experts=2,
    )
    down_lora = LoRA(
        adapter_model_prefix=f"{group_prefix}.{{expert}}.down_proj",
        in_features=intermediate,
        out_features=hidden,
        rank=rank,
        alpha=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_local_experts=2,
    )
    offset = 0
    for expert in range(2):
        expert_prefix = f"{group_prefix}.{expert}"
        tensors = {
            "gate_up_proj.lora_A.weight": torch.arange(
                rank * hidden,
                dtype=torch.float32,
            ).reshape(rank, hidden)
            + offset,
            "gate_up_proj.lora_B.weight": torch.arange(
                2 * intermediate * rank,
                dtype=torch.float32,
            ).reshape(2 * intermediate, rank)
            + offset
            + 100,
            "down_proj.lora_A.weight": torch.arange(
                rank * intermediate,
                dtype=torch.float32,
            ).reshape(rank, intermediate)
            + offset
            + 200,
            "down_proj.lora_B.weight": torch.arange(
                hidden * rank,
                dtype=torch.float32,
            ).reshape(hidden, rank)
            + offset
            + 300,
        }
        for suffix, tensor in tensors.items():
            full[f"{expert_prefix}.{suffix}"] = tensor
        gate_up_lora.A_T.data[expert].copy_(tensors["gate_up_proj.lora_A.weight"].T)
        gate_up_lora.B_T.data[expert].copy_(tensors["gate_up_proj.lora_B.weight"].T)
        down_lora.A_T.data[expert].copy_(tensors["down_proj.lora_A.weight"].T)
        down_lora.B_T.data[expert].copy_(tensors["down_proj.lora_B.weight"].T)
        offset += 1000

    slot_ref = LoRASlotRef("student") if dynamic_slot else None
    if slot_ref is not None:
        assert gate_up_lora.load_lora_slot(slot_ref, full, alpha=rank)
        assert down_lora.load_lora_slot(slot_ref, full, alpha=rank)

    adapter_config = _config(base_model, rank=rank, alpha=rank)
    old_dir = tmp_path / "old"
    current_dir = tmp_path / "current"
    old_tensors, old_config = handler.to_vllm_lora_tensors(
        full,
        adapter_config=dict(adapter_config),
    )
    save_vllm_lora_tensors(old_dir, old_tensors, old_config)
    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(gate_up_lora, down_lora)]),
        adapter_dtypes={key: tensor.dtype for key, tensor in full.items()},
        handler=handler,
        adapter_config=dict(adapter_config),
        output_dir=str(current_dir),
        rank=0,
        world_size=1,
        slot_ref=slot_ref,
    )

    _assert_tensors_equal(
        load_file(current_dir / "adapter_model.safetensors"),
        load_file(old_dir / "adapter_model.safetensors"),
    )
    assert (current_dir / "adapter_model.safetensors").read_bytes() == (
        old_dir / "adapter_model.safetensors"
    ).read_bytes()
    assert json.loads((current_dir / "adapter_config.json").read_text()) == json.loads(
        (old_dir / "adapter_config.json").read_text()
    )


def test_direct_gpt_oss_packed_expert_publish_matches_handler_vllm_exactly(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)

    rank = 2
    hidden = 128
    intermediate = 128
    group_prefix = "base_model.model.model.layers.0.mlp.experts"
    full = {
        key: tensor
        for key, tensor in _gpt_oss_moe_art_tensors(
            "base_model.model.model.layers.0",
            rank=rank,
        ).items()
        if ".mlp.experts." in key
    }
    gate_up_lora = LoRA(
        adapter_model_prefix=f"{group_prefix}.{{expert}}.gate_up_proj",
        in_features=hidden,
        out_features=2 * intermediate,
        rank=rank,
        alpha=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_local_experts=2,
    )
    down_lora = LoRA(
        adapter_model_prefix=f"{group_prefix}.{{expert}}.down_proj",
        in_features=intermediate,
        out_features=hidden,
        rank=rank,
        alpha=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_local_experts=2,
    )
    for expert in range(2):
        expert_prefix = f"{group_prefix}.{expert}"
        gate_up_lora.A_T.data[expert].copy_(
            full[f"{expert_prefix}.gate_up_proj.lora_A.weight"].T
        )
        gate_up_lora.B_T.data[expert].copy_(
            full[f"{expert_prefix}.gate_up_proj.lora_B.weight"].T
        )
        down_lora.A_T.data[expert].copy_(
            full[f"{expert_prefix}.down_proj.lora_A.weight"].T
        )
        down_lora.B_T.data[expert].copy_(
            full[f"{expert_prefix}.down_proj.lora_B.weight"].T
        )

    adapter_config = _gpt_oss_config(
        _gpt_oss_model_dir(tmp_path),
        rank=rank,
        alpha=rank,
    )
    old_dir = tmp_path / "old"
    current_dir = tmp_path / "current"
    old_tensors, old_config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        full,
        adapter_config=dict(adapter_config),
    )
    save_vllm_lora_tensors(old_dir, old_tensors, old_config)
    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(gate_up_lora, down_lora)]),
        adapter_dtypes={key: tensor.dtype for key, tensor in full.items()},
        handler=GPT_OSS_MOE_HANDLER,
        adapter_config=dict(adapter_config),
        output_dir=str(current_dir),
        rank=0,
        world_size=1,
    )

    _assert_tensors_equal(
        load_file(current_dir / "adapter_model.safetensors"),
        load_file(old_dir / "adapter_model.safetensors"),
    )
    assert (current_dir / "adapter_model.safetensors").read_bytes() == (
        old_dir / "adapter_model.safetensors"
    ).read_bytes()
    assert json.loads((current_dir / "adapter_config.json").read_text()) == json.loads(
        (old_dir / "adapter_config.json").read_text()
    )


def test_qwen35_megatron_shards_can_merge_to_separate_vllm_checkpoint(
    tmp_path: Path,
):
    prefix = "base_model.model.model.layers.0.mlp.experts.0"
    full = {
        f"{prefix}.gate_up_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        f"{prefix}.gate_up_proj.lora_B.weight": torch.arange(
            8,
            dtype=torch.float32,
        ).reshape(8, 1),
        f"{prefix}.down_proj.lora_A.weight": torch.arange(
            4,
            dtype=torch.float32,
        ).reshape(1, 4),
        f"{prefix}.down_proj.lora_B.weight": torch.arange(
            2,
            dtype=torch.float32,
        ).reshape(2, 1),
    }
    publish_dir = tmp_path / "published"
    adapter_config = _config("Qwen/Qwen3.5-35B-A3B", rank=1, alpha=1)
    entries_by_key = {
        key: [(_manifest(sharded=False, shard_world_size=1, shard_rank=0), tensor)]
        for key, tensor in full.items()
    }
    merged = merge_sharded_adapter_entries(entries_by_key)
    vllm_tensors, adapter_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        merged,
        adapter_config=adapter_config,
    )
    save_vllm_lora_tensors(publish_dir, vllm_tensors, adapter_config)

    assert (publish_dir / "adapter_model.safetensors").exists()
    roundtrip = load_lora_tensors_for_megatron(
        str(publish_dir),
        handler=QWEN3_5_MOE_HANDLER,
    )
    _assert_tensors_equal(roundtrip, full)
