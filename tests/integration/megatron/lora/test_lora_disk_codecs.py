import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

import pytest
from safetensors.torch import load_file, save_file
import torch

pytest.importorskip("megatron.bridge.models.gpt_provider")

from art.megatron import lora as lora_module
from art.megatron.lora import LoRA, LoRAParallelSpec, LoRAPublishPlanner
from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    GPT_OSS_MOE_HANDLER,
    QWEN3_5_MOE_HANDLER,
    QWEN3_MOE_HANDLER,
)
from art.megatron.model_support.handlers.gemma4 import GEMMA4_MOE_HANDLER
from art.megatron.model_support.lora_disk import (
    load_lora_tensors_for_megatron,
    normalize_lora_checkpoint_to_vllm,
    save_vllm_lora_tensors,
)
from art.megatron.weights import lora_publish
from art.megatron.weights.lora_publish import (
    LoraShardMeta,
    merge_sharded_adapter_entries,
    save_vllm_lora_from_model,
)
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
    handler,
    adapter_config: dict,
) -> None:
    entries_by_key: dict[str, list[tuple[dict, torch.Tensor]]] = {}
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
            entries_by_key.setdefault(key, []).append((manifest[key], tensor))

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


def _gpt_oss_moe_art_tensors(prefix: str, *, rank: int = 2) -> dict[str, torch.Tensor]:
    hidden = 3
    intermediate = 4
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
        adapter_config=_gpt_oss_config("openai/gpt-oss-20b"),
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

    roundtrip = GPT_OSS_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=vllm_config,
    )
    _assert_tensors_equal(roundtrip, original)

    adapter_dir = tmp_path / "gpt_oss"
    _save_adapter(adapter_dir, vllm_tensors, vllm_config)
    loaded_modules = _assert_stock_vllm_loads(
        adapter_dir,
        expected_modules={"q_proj", "experts"},
    )
    assert "model.layers.0.attn.q_proj" in loaded_modules
    assert "model.layers.0.mlp.experts" in loaded_modules
    assert "model.layers.0.mlp.experts.base_layer" in loaded_modules


def test_gpt_oss_expert_lora_is_not_emitted_as_merged_delta() -> None:
    module_path = VLLM_RUNTIME_SRC / "art_vllm_runtime/lora_delta.py"
    spec = importlib.util.spec_from_file_location("art_vllm_lora_delta", module_path)
    assert spec is not None and spec.loader is not None
    lora_delta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lora_delta)
    original = _gpt_oss_moe_art_tensors("base_model.model.model.layers.0")
    vllm_tensors, adapter_config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        original,
        adapter_config=_gpt_oss_config("openai/gpt-oss-20b"),
    )

    names = [
        name
        for name, _tensor in lora_delta._iter_lora_checkpoint_deltas(
            vllm_tensors,
            adapter_config=adapter_config,
            previous_lora_tensors=None,
        )
    ]

    assert adapter_config["art_merged_lora_delta_unsupported_target_modules"] == [
        "experts"
    ]
    assert "model.layers.0.attn.q_proj.weight" in names
    assert not any(".mlp.experts" in name for name in names)


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

    def unsharded() -> dict:
        return {"sharded": False, "shard_world_size": 1, "shard_rank": 0}

    def sharded(rank_id: int, dim: int) -> dict:
        return {
            "sharded": True,
            "shard_world_size": 2,
            "shard_rank": rank_id,
            "export_shard_dim": dim,
            "export_shard_strategy": "uniform",
        }

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
            manifest={**manifest, "shard_rank": 0},
            block="base_model.model.model.layers.0",
        ),
        LoraShardMeta(
            key=key,
            owner_rank=1,
            shape=tuple(shard1.shape),
            dtype_name="float32",
            manifest={**manifest, "shard_rank": 1},
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


def test_lora_publish_planner_derives_metadata_from_lora_modules():
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    b_parallel_spec = LoRAParallelSpec(sharded=True, shard_dim=-1)
    lora = LoRA(
        adapter_model_prefix=prefix,
        in_features=4,
        out_features=6,
        rank=2,
        alpha=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        b_parallel_spec=b_parallel_spec,
    )
    adapter_model = {
        f"{prefix}.lora_A.weight": torch.empty(2, 4, dtype=torch.float32),
        f"{prefix}.lora_B.weight": torch.empty(6, 2, dtype=torch.float32),
    }

    metadata = LoRAPublishPlanner([torch.nn.Sequential(lora)]).global_metadata(
        adapter_model
    )
    by_key = {meta.key: meta for meta in metadata}

    a_meta = by_key[f"{prefix}.lora_A.weight"]
    assert a_meta.shape == (2, 4)
    assert a_meta.dtype_name == "float32"
    assert a_meta.owner_rank == 0
    assert a_meta.manifest == {
        "sharded": False,
        "shard_world_size": 1,
        "shard_rank": 0,
    }
    assert a_meta.block == "base_model.model.model.layers.0"

    b_meta = by_key[f"{prefix}.lora_B.weight"]
    assert b_meta.shape == (6, 2)
    assert b_meta.dtype_name == "float32"
    assert b_meta.owner_rank == 0
    assert b_meta.manifest == {
        "sharded": True,
        "shard_world_size": 1,
        "shard_rank": 0,
        "export_shard_dim": 0,
        "export_shard_strategy": "uniform",
    }


def test_lora_publish_planner_maps_expert_owner_ranks(monkeypatch):
    monkeypatch.setattr(lora_module, "_distributed_initialized", lambda: True)
    monkeypatch.setattr(
        lora_module,
        "_get_shard_world_size",
        lambda domain: 2 if domain == "expert_tp" else 1,
    )
    monkeypatch.setattr(
        lora_module.ps,
        "get_expert_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        lora_module.ps,
        "get_expert_tensor_and_model_parallel_group",
        lambda check_initialized=False: "joint",
    )
    monkeypatch.setattr(
        lora_module.ps,
        "get_expert_model_parallel_group",
        lambda: "ep",
    )
    monkeypatch.setattr(
        lora_module.ps,
        "get_expert_tensor_parallel_group",
        lambda check_initialized=False: "etp",
    )

    row_major = {"joint": (0, 1, 2, 3, 4, 5, 6, 7), "ep": (0, 2, 4, 6), "etp": (0, 1)}
    monkeypatch.setattr(
        lora_module,
        "_process_group_ranks",
        lambda group: row_major[group],
    )
    assert LoRAPublishPlanner._expert_owner_rank(ep_rank=3, shard_rank=1) == 7

    column_major = {
        "joint": (0, 1, 2, 3, 4, 5, 6, 7),
        "ep": (0, 1, 2, 3),
        "etp": (0, 4),
    }
    monkeypatch.setattr(
        lora_module,
        "_process_group_ranks",
        lambda group: column_major[group],
    )
    assert LoRAPublishPlanner._expert_owner_rank(ep_rank=3, shard_rank=1) == 7


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
    unsharded_manifest = {"sharded": False, "shard_world_size": 1, "shard_rank": 0}
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
        uniform_key: {**uniform_manifest, "shard_rank": 0},
        componentwise_key: {**componentwise_manifest, "shard_rank": 0},
    }
    manifest1 = {
        uniform_key: {**uniform_manifest, "shard_rank": 1},
        componentwise_key: {**componentwise_manifest, "shard_rank": 1},
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
        handler=handler,
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
    lora_publish._save_rank0_vllm_lora(
        metadata=metadata,
        tensors_by_owner_key={
            **{(0, key): tensor for key, tensor in shard0.items()},
            **{(1, key): tensor for key, tensor in shard1.items()},
        },
        handler=handler,
        adapter_config=adapter_config,
        output_dir=str(current_dir),
    )

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
        adapter_model=full,
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


def test_direct_qwen35_packed_expert_publish_matches_old_vllm_exactly(
    tmp_path: Path,
    monkeypatch,
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

    adapter_config = _config("Qwen/Qwen3.5-35B-A3B", rank=rank, alpha=rank)
    old_dir = tmp_path / "old"
    current_dir = tmp_path / "current"
    old_tensors, old_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        full,
        adapter_config=dict(adapter_config),
    )
    save_vllm_lora_tensors(old_dir, old_tensors, old_config)
    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(gate_up_lora, down_lora)]),
        adapter_model=full,
        handler=QWEN3_5_MOE_HANDLER,
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


def test_direct_gpt_oss_packed_expert_publish_matches_handler_vllm_exactly(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_module.ps, "get_expert_data_parallel_rank", lambda: 0)

    rank = 2
    hidden = 3
    intermediate = 4
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

    adapter_config = _gpt_oss_config("openai/gpt-oss-20b", rank=rank, alpha=rank)
    old_dir = tmp_path / "old"
    current_dir = tmp_path / "current"
    old_tensors, old_config = GPT_OSS_MOE_HANDLER.to_vllm_lora_tensors(
        full,
        adapter_config=dict(adapter_config),
    )
    save_vllm_lora_tensors(old_dir, old_tensors, old_config)
    save_vllm_lora_from_model(
        model=cast(Any, [torch.nn.Sequential(gate_up_lora, down_lora)]),
        adapter_model=full,
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
        key: [({"sharded": False, "shard_world_size": 1, "shard_rank": 0}, tensor)]
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
