from __future__ import annotations

import hashlib
from itertools import groupby
import json
import os
from typing import Literal, cast

from art import dev
from art._source_revision import art_source_revision
from art.dev.get_model_config import default_target_modules
from art.distributed.art_runtime import ArtRuntime
from art.distributed.specs import NixlTransportSpec, TrainerMeshSpec

from ..lora_config import LORA_ALPHA, default_lora_rank_for_handler
from ..model_support import (
    get_model_support_handler_for_spec,
    get_model_support_spec,
    model_uses_expert_parallel,
)
from ..runtime_config import get_megatron_runtime_config
from .run_residency import RunResidencyConfig
from .specs import HybridEpRuntimeSpec, TrainerRuntimeSpec


def build_trainer_runtime_spec(
    runtime: ArtRuntime,
    *,
    base_model: str,
    config: dev.BackendModelConfig,
    enable_expert_replay: bool,
    offload_between_jobs: bool,
    run_residency: RunResidencyConfig | None = None,
) -> TrainerRuntimeSpec:
    mesh = runtime.topology.trainer
    if mesh is None:
        raise RuntimeError("ART runtime has no trainer mesh")
    runtime_config = get_megatron_runtime_config()
    if runtime_config.topology != mesh.topology:
        raise ValueError(
            "Megatron runtime topology does not match the ART trainer mesh"
        )
    lora = cast(dev.LoRAConfig, config.get("lora_config") or {})
    allow_unvalidated_arch = bool(config.get("allow_unvalidated_arch", False))
    support_spec = get_model_support_spec(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    handler = get_model_support_handler_for_spec(support_spec)
    targets = lora.get("target_modules") or default_target_modules(base_model)
    init_args = config.get("init_args", {})
    model_identifier = init_args.get("model_name", base_model)
    if not isinstance(model_identifier, str) or not model_identifier:
        raise ValueError("init_args.model_name must be a non-empty string")
    revision = str(init_args.get("revision") or "default")
    compile_enabled = os.environ.get(
        "ART_DISABLE_MEGATRON_COMPILE", "0"
    ).lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }
    model_semantics = {
        "model": model_identifier,
        "support_model": base_model,
        "revision": revision,
        "handler": handler.key,
        "model_initialization": config.get(
            "megatron_model_initialization", "pretrained"
        ),
    }
    identity = {
        "art": art_source_revision(),
        **model_semantics,
        "mesh": mesh.model_dump(mode="json"),
    }
    dtype = trainer_dtype(config)
    return TrainerRuntimeSpec(
        art_revision=identity["art"],
        model_identifier=model_identifier,
        model_revision=revision,
        model_initialization=identity["model_initialization"],
        cache_root=runtime.topology.cluster.cache_root,
        model_support_key=support_spec.key,
        handler_name=handler.key,
        lora_rank=int(lora.get("rank") or default_lora_rank_for_handler(handler)),
        lora_alpha=float(lora.get("alpha", LORA_ALPHA)),
        lora_target_modules=tuple(targets),
        lora_moe_parameterization=lora.get("moe_parameterization", "per_expert"),
        dtype=dtype,
        trainer_mesh=mesh,
        packed_sequence_length=runtime_config.packed_sequence_length,
        snapshot_pool_capacity=runtime_config.snapshot_pool_capacity,
        run_residency=run_residency,
        compile_enabled=compile_enabled,
        compile_cache=runtime_config.compile_cache and compile_enabled,
        compile_fingerprint=_digest({**identity, "compile": compile_enabled}),
        optimizer_semantic_fingerprint=_digest(
            {
                **model_semantics,
                "dtype": dtype,
                "lora_rank": int(
                    lora.get("rank") or default_lora_rank_for_handler(handler)
                ),
                "lora_alpha": float(lora.get("alpha", LORA_ALPHA)),
                # Target order does not change the logical LoRA parameter set.
                "lora_target_modules": tuple(sorted(targets)),
                "lora_moe_parameterization": lora.get(
                    "moe_parameterization", "per_expert"
                ),
            }
        ),
        optimizer_layout_fingerprint=_digest({"mesh": mesh.model_dump(mode="json")}),
        allow_unvalidated_arch=allow_unvalidated_arch,
        enable_moe_routing_replay=enable_expert_replay
        and model_uses_expert_parallel(
            base_model, allow_unvalidated_arch=allow_unvalidated_arch
        ),
        streaming_weight_offload=runtime_config.streaming_weight_offload,
        offload_between_jobs=offload_between_jobs,
        random_state=_random_state(config),
        hybrid_ep=hybrid_ep_runtime_spec(
            mesh,
            run_id=runtime.runtime_id,
            transport=runtime.topology.cluster.nixl_transport,
        ),
    )


def hybrid_ep_runtime_spec(
    mesh: TrainerMeshSpec,
    *,
    run_id: str,
    transport: NixlTransportSpec | None,
) -> HybridEpRuntimeSpec | None:
    if mesh.topology.ep <= 1:
        return None
    group_size = mesh.topology.etp * mesh.topology.ep
    domain_sizes: set[int] = set()
    multinode = False
    for offset in range(0, len(mesh.ranks), group_size):
        group = mesh.ranks[offset : offset + group_size]
        domains = [
            (host_id, tuple(ranks))
            for host_id, ranks in groupby(group, key=lambda rank: rank.host_id)
        ]
        if len({host_id for host_id, _ in domains}) != len(domains):
            raise ValueError("HybridEP ranks for each host must be contiguous")
        domain_sizes.update(len(ranks) for _, ranks in domains)
        multinode |= len(domains) > 1
    if len(domain_sizes) != 1:
        raise ValueError(
            "HybridEP TP x EP groups require equal ranks per NVLink domain"
        )
    if multinode and transport is None:
        raise ValueError("cross-host expert parallelism requires NIXL transport")
    return HybridEpRuntimeSpec(
        ranks_per_nvlink_domain=domain_sizes.pop(),
        run_id=run_id,
        nixl_transport=transport if multinode else None,
    )


def trainer_dtype(
    config: dev.BackendModelConfig,
) -> Literal["bfloat16", "float16", "float32"]:
    value = str(config.get("init_args", {}).get("dtype") or "bfloat16").lower()
    value = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "torch.bfloat16": "bfloat16",
        "torch.float16": "float16",
        "torch.float32": "float32",
    }.get(value, value)
    if value not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported Megatron trainer dtype {value!r}")
    return cast(Literal["bfloat16", "float16", "float32"], value)


def _random_state(config: dev.BackendModelConfig) -> int | None:
    for key in ("lora_config", "init_args"):
        value = config.get(key, {}).get("random_state")
        if value is not None:
            return int(value)
    return None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
