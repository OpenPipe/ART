"""CPU-only rehearsal of real Bridge/provider imports and adapter loading.

Keep native TE constructors and conversions. Redirect only unavailable CUDA
allocation/pinning and DDP boundaries; no forward is run or qualified here.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import patch

from megatron.bridge.models import model_provider
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.core import parallel_state
import torch
import torch.distributed as dist

from art.megatron.glm5next.bridge import ArtGlm5NextBridge
from art.megatron.lora import LoraLayout
from art.megatron.runtime import bridge_runtime
from art.megatron.train import build_training_runtime
from tests.integration.megatron.glm5next_qualification import (
    configure_qualification_provider,
    identity,
    sha,
    write,
)


def _family(name: str) -> str:
    if ".hc_" in name:
        return "mhc"
    if ".experts." in name:
        return "routed_experts"
    if ".router." in name:
        return "router"
    if ".shared_experts." in name:
        return "shared_experts"
    if ".self_attention." in name:
        return "attention"
    return "other"


def _layer(name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


class _LoadedMeta(Exception):
    pass


def check_pretrained_load(
    model: Path,
    output: Path,
    *,
    device: str,
    legacy: bool = False,
    task_only: bool = False,
) -> dict[str, Any]:
    assert not torch.cuda.is_initialized() and not dist.is_initialized()
    result: dict[str, Any] = {
        "device": device,
        "legacy": legacy,
        "config_sha256": sha(model / "config.json"),
        "index_sha256": sha(model / "model.safetensors.index.json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    validate = bridge_runtime._validate_local_pretrained_tasks
    loaded: dict[str, Any] = {}

    def coverage(bridge: Any, chunks: Any, tasks: Any) -> None:
        loaded.update(bridge=bridge, tasks=tasks)
        covered = {
            id(task.param_weight)
            for task in tasks
            if task is not None and task.megatron_module is not None
        }
        missing = [
            name
            for chunk in chunks
            for name, parameter in chunk.named_parameters()
            if id(parameter) not in covered
        ]
        buffers = [
            name
            for chunk in chunks
            for name, _ in chunk.named_buffers()
            if name.endswith("router.expert_bias")
            and not any(
                task is not None
                and task.megatron_module is not None
                and task.param_name == name
                for task in tasks
            )
        ]
        counts = Counter((_layer(name), _family(name)) for name in missing)
        result.update(
            missing_parameters=missing,
            missing_parameter_count=len(missing),
            missing_router_buffers=buffers,
            missing_by_layer_family=[
                {"layer": int(layer), "family": family, "count": count}
                for (layer, family), count in sorted(counts.items())
            ],
            local_tasks=sum(
                task is not None and task.megatron_module is not None for task in tasks
            ),
        )
        write(output / "coverage.json", result)
        (output / "missing-parameters.tsv").write_text(
            "layer\tfamily\tparameter\n"
            + "".join(f"{_layer(name)}\t{_family(name)}\t{name}\n" for name in missing)
        )
        validate(bridge, chunks, tasks)
        if task_only:
            raise _LoadedMeta

    def before_ddp(chunks: Any, *args: Any, **kwargs: Any) -> Any:
        assert all(p.device.type == device for c in chunks for p in c.parameters())
        result["parameter_elements"] = sum(
            p.numel() for c in chunks for p in c.parameters()
        )
        result["router_bias_exact"] = True
        if device == "cpu":
            bridge = loaded["bridge"]
            state = bridge.hf_pretrained.state
            verified = 0
            for task in loaded["tasks"]:
                if task is None or task.megatron_module is None:
                    continue
                expected = bridge.maybe_modify_loaded_hf_weight(
                    task.mapping.hf_param, state
                )
                converted = task.mapping.hf_to_megatron(expected, task.megatron_module)
                assert torch.equal(task.param_weight, converted), task.param_name
                verified += 1
                del expected, converted
            result["exactly_verified_local_tensors"] = verified
            from safetensors import safe_open

            index = json.loads((model / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            for chunk in chunks:
                for layer in chunk.module.decoder.layers:
                    if not hasattr(layer.mlp, "router"):
                        continue
                    key = f"model.language_model.layers.{layer.layer_number - 1}.mlp.gate.e_score_correction_bias"
                    with safe_open(
                        model / index[key], framework="pt", device="cpu"
                    ) as handle:
                        assert layer.mlp.router.expert_bias.dtype == torch.float32
                        assert torch.equal(
                            layer.mlp.router.expert_bias, handle.get_tensor(key)
                        )
        if device == "meta":
            raise _LoadedMeta
        return chunks

    with tempfile.TemporaryDirectory(prefix="glm-pretrained-load-") as temporary:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dist.init_process_group(
            "gloo",
            init_method="env://" if world_size > 1 else f"file://{temporary}/rdzv",
            rank=rank,
            world_size=world_size,
        )
        try:
            parallel_state.initialize_model_parallel()
            with ExitStack() as stack:
                for target, name, replacement in (
                    (torch.cuda, "is_available", lambda: True),
                    (torch.cuda, "current_device", lambda: torch.device(device)),
                    (torch.Tensor, "pin_memory", lambda tensor, *a, **kw: tensor),
                    (
                        bridge_runtime,
                        "_materialization_device",
                        lambda: torch.device(device),
                    ),
                    (bridge_runtime, "_validate_local_pretrained_tasks", coverage),
                    (model_provider, "_ddp_wrap", before_ddp),
                ):
                    stack.enter_context(patch.object(target, name, replacement))
                if legacy:
                    stack.enter_context(
                        patch.object(
                            ArtGlm5NextBridge,
                            "build_conversion_tasks",
                            MegatronModelBridge.build_conversion_tasks,
                        )
                    )
                try:
                    runtime = build_training_runtime(
                        model_identifier=str(model),
                        model_support_key="glm5next",
                        allow_unvalidated_arch=True,
                        build_optimizer=False,
                        lora_layout=LoraLayout(expert="shared_outer"),
                        provider_configure=configure_qualification_provider,
                    )
                    from art.trainer_rank import TrainerRank

                    adapter = output / "initial"
                    identity(model, adapter)
                    trainer = TrainerRank(runtime)
                    trainer.load_checkpoint(str(adapter))
                    result["load_checkpoint"] = "pass"
                    result["status"] = "pass"
                    del trainer, runtime
                except _LoadedMeta:
                    result["status"] = "pass"
                    result["load_checkpoint"] = (
                        "not_run; meta has no values; exercised on CPU"
                    )
                except RuntimeError as exc:
                    if not legacy or "required local parameter(s)" not in str(exc):
                        raise
                    result["status"] = "expected_legacy_failure"
                    result["error"] = str(exc)
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
    gc.collect()
    result["cuda_initialized"] = torch.cuda.is_initialized()
    assert not result["cuda_initialized"]
    result["cpu_topology"] = f"real Gloo TP1/PP1/CP1/EP1/DP{world_size}"
    result["rank"] = rank
    result["path"] = (
        "build_training_runtime/AutoBridge/provider.provide_distributed_model/native TE/pretrained load/Float16Module/rank32 LoRA/TrainerRank.load_checkpoint"
    )
    result["hardware_substitutions"] = [
        "CUDA constructor guards/allocation -> CPU or meta",
        "pin_memory -> identity",
        "CUDA DDP -> model chunks",
    ]
    write(output / "result.json", result)
    return result


def rehearse_pretrained_load(model: Path, output: Path) -> None:
    results = []
    for legacy in (True, False):
        for device in ("meta", "cpu"):
            label = f"{'legacy' if legacy else 'fixed'}-{device}"
            results.append(
                check_pretrained_load(
                    model, output / label, device=device, legacy=legacy
                )
            )
    assert all(r["missing_parameter_count"] == 2970 for r in results[:2])
    assert all(
        r["status"] == "pass" and r["missing_parameter_count"] == 0 for r in results[2:]
    )
    # Bound the full CPU import to one replica. Check task discovery on the
    # production EP1/DP2 groups too, on both ranks, without checkpoint I/O.
    topology = []
    for legacy in (True, False):
        destination = output / ("dp2-legacy" if legacy else "dp2-fixed")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(Path(__file__).resolve()),
                "--model",
                str(model),
                "--output",
                str(destination),
                *(["--legacy"] if legacy else []),
            ],
            check=True,
            timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        for rank in range(2):
            row = json.loads((destination / f"rank{rank}/result.json").read_text())
            assert row["missing_parameter_count"] == (2970 if legacy else 0)
            topology.append(row)
    write(
        output / "pretrained-load.json",
        {
            "status": "pass",
            "results": results,
            "DP2_task_coverage": topology,
            "cuda_initialized": False,
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    check_pretrained_load(
        args.model,
        args.output / f"rank{os.environ.get('RANK', '0')}",
        device="meta",
        legacy=args.legacy,
        task_only=True,
    )
