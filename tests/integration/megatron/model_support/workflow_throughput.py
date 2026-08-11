from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
import hashlib
import json
import math
from multiprocessing import resource_tracker, shared_memory
import os
from pathlib import Path
import shutil
from statistics import fmean, median, quantiles
import struct
import subprocess
import sys
from typing import Any, Literal, NamedTuple, cast
import uuid

from art.megatron.model_support.registry import get_model_support_spec
from art.megatron.model_support.spec import ArchitectureReport

from .validation_spec import ValidationStageResult
from .workflow_fixtures import (
    FIXTURE_PATH_ENV,
    _flatten_token_ids,
    _validate_tokenizer_compatible_fixture,
)
from .workflow_resources import (
    ThroughputThresholds,
    ThroughputWorkflowConfig,
    handler_workflow_resources_for_base_model,
    resolve_stage_resources_for_visible_gpus,
)

_STAGE_DIR_ENV = "ART_MODEL_SUPPORT_WORKFLOW_STAGE_DIR"
_LAYER_LIST_FIELDS = (
    "layer_types",
    "mlp_layer_types",
    "indexer_types",
    "compress_ratios",
)
_WIDTH_TERMS = ("hidden", "intermediate", "head", "expert", "lora_rank", "topk")
_POLICY_AGE_MEAN = "offpolicy/token_weighted_policy_age_steps"
_POLICY_AGE_P95 = "offpolicy/token_weighted_policy_age_p95_steps"
_FRESHNESS_DISCOUNT = "sample_efficiency/freshness_discount"
_STALE_GROUPS = "discarded/step/stale_groups"
_ZERO_VARIANCE_GROUPS = "discarded/step/zero_variance_groups"
_MEASUREMENT_CONTRACT_VERSION = 8
_ISOLATED_WARMUP_STEPS = 1
_MATCHED_MEASURED_STEPS = 2
_CAPTURE_GUARD_STEPS = 1
_REPO_ROOT = Path(__file__).parents[4]
_MAIN_RUNTIME_PACKAGES = (
    "openpipe-art",
    "torchmonarch",
    "torch",
    "triton",
    "transformer-engine",
    "megatron-core",
    "megatron-bridge",
    "transformers",
    "flashinfer-python",
    "nvidia-nccl-cu13",
    "nvidia-nvshmem-cu13",
)
_VLLM_RUNTIME_PACKAGES = (
    "art-vllm-runtime",
    "vllm",
    "torch",
    "triton",
    "transformers",
    "flashinfer-python",
    "nvidia-nccl-cu13",
)
_LOCAL_SOURCE_PACKAGES = ("openpipe-art", "art-vllm-runtime")
_H200_THROUGHPUT_NUM_LAYERS = {"dsv4": 4, "glm52": 6}


class ThroughputFixture(NamedTuple):
    model_key: str
    path: str
    num_layers: int
    width_fingerprint: dict[str, int]
    manifest: dict[str, Any]


class TrainerPhaseEvidence(NamedTuple):
    phase: Literal["isolated", "e2e"]
    runtime_fingerprint: str
    trajectory_input_fingerprint: str
    packed_input_fingerprint: str
    workload_fingerprint: str
    sample_count: int
    policy_steps: tuple[int, ...]
    train_s: float
    metrics: tuple[dict[str, float], ...]

    @property
    def train_tok_s(self) -> float:
        return (
            sum(
                metrics["data/step_nonpadding_logical_tokens"]
                for metrics in self.metrics
            )
            / self.train_s
        )


def _same_executed_settings(decisions: list[Any]) -> bool:
    first, second = decisions[-2:]
    return first.previous == first.updated == second.previous


@contextmanager
def _hold_pipeline_settings_after_step(trainer: Any, step: int) -> Iterator[None]:
    apply = trainer.apply_pipeline_settings

    def apply_before_step(settings: Any) -> None:
        # The terminal decision is future evidence; matched capture must retain the
        # setting that executed the measured windows.
        if trainer.state.next_training_step < step:
            apply(settings)

    setattr(trainer, "apply_pipeline_settings", apply_before_step)
    try:
        yield
    finally:
        setattr(trainer, "apply_pipeline_settings", apply)


def _current_pipeline_settings(trainer: Any) -> dict[str, int]:
    return {
        name: int(getattr(trainer, name))
        for name in (
            "num_rollout_workers",
            "min_batch_size",
            "max_batch_size",
            "queue_maxsize",
            "target_groups_per_step",
        )
    }


def _text(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config", config)


def _width_fingerprint(config: dict[str, Any]) -> dict[str, int]:
    text = _text(config)
    return {
        key: value
        for key, value in text.items()
        if key != "num_hidden_layers"
        and type(value) is int
        and any(term in key for term in _WIDTH_TERMS)
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _files_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise RuntimeError(f"calibration provenance file is missing: {path}")
        relative = path.relative_to(_REPO_ROOT).as_posix().encode()
        payload = path.read_bytes()
        digest.update(struct.pack("<Q", len(relative)))
        digest.update(relative)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _environment_provenance(python: Path, packages: tuple[str, ...]) -> dict[str, Any]:
    script = """
import hashlib
from importlib import metadata
import json
import platform
import sys
import torch

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None

distributions = {}
local_source_packages = set(json.loads(sys.argv[2]))
for name in json.loads(sys.argv[1]):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    provenance = {
        "version": dist.version,
        "metadata_sha256": sha(dist.read_text("METADATA")),
    }
    if name not in local_source_packages:
        provenance.update({
            "direct_url_sha256": sha(dist.read_text("direct_url.json")),
            "record_sha256": sha(dist.read_text("RECORD")),
        })
    distributions[name] = provenance
print(json.dumps({
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "cache_tag": sys.implementation.cache_tag,
        "abi_flags": sys.abiflags,
    },
    "torch": {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    },
    "distributions": distributions,
}, sort_keys=True))
"""
    if not python.is_file():
        raise RuntimeError(f"calibration runtime Python is missing: {python}")
    result = subprocess.run(
        [
            str(python),
            "-c",
            script,
            json.dumps(packages),
            json.dumps(_LOCAL_SOURCE_PACKAGES),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, Any], json.loads(result.stdout))


def _source_provenance() -> dict[str, Any]:
    return {
        "art_source_sha256": _files_digest(
            list((_REPO_ROOT / "src/art").rglob("*.py"))
        ),
        "vllm_runtime_source_sha256": _files_digest(
            list((_REPO_ROOT / "vllm_runtime/src/art_vllm_runtime").rglob("*.py"))
        ),
        "workflow_runtime_sha256": _files_digest(
            [
                Path(__file__),
                Path(__file__).with_name("workflow.py"),
                Path(__file__).with_name("workflow_fixtures.py"),
                Path(__file__).with_name("workflow_stage_worker.py"),
                Path(__file__).with_name("validation_spec.py"),
            ]
        ),
        "build_contract_sha256": _files_digest(
            [
                _REPO_ROOT / "pyproject.toml",
                _REPO_ROOT / "vllm_runtime/pyproject.toml",
                _REPO_ROOT / "vllm_runtime/setup.sh",
            ]
        ),
        "root_lock_sha256": _files_digest([_REPO_ROOT / "uv.lock"]),
        "vllm_runtime_lock_sha256": _files_digest(
            [_REPO_ROOT / "vllm_runtime/uv.lock"]
        ),
        "main_environment": _environment_provenance(
            Path(sys.executable), _MAIN_RUNTIME_PACKAGES
        ),
        "vllm_environment": _environment_provenance(
            _REPO_ROOT / "vllm_runtime/.venv/bin/python", _VLLM_RUNTIME_PACKAGES
        ),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    _require(
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value).is_integer()
        and value >= 0,
        f"{name} must be a nonnegative integer, got {value!r}",
    )
    return int(value)


_PHASE_WORKLOAD_KEYS = (
    "data/step_num_groups_trainable",
    "data/step_packed_sequences",
    "data/step_nonpadding_logical_tokens",
    "data/step_loss_bearing_tokens",
    "data/step_executed_token_equivalents",
    "data/step_dummy_executed_token_equivalents",
    "data/step_nominal_schedule_capacity_tokens",
    "data/step_dummy_schedule_capacity_tokens",
    "data/step_unused_packed_capacity_tokens",
    "data/step_num_gradient_steps",
    "pipeline/global_real_microbatches",
    "pipeline/global_dummy_microbatches",
)


def _phase_evidence(
    *,
    phase: Literal["isolated", "e2e"],
    runtime_fingerprint: str,
    trajectory_input_fingerprint: str,
    packed_input_fingerprint: str,
    samples: list[tuple[Mapping[str, Any], int]],
) -> TrainerPhaseEvidence:
    if not samples:
        raise RuntimeError(f"{phase} trainer phase produced no samples")
    numeric_samples = tuple(
        {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
        for metrics, _ in samples
    )
    workloads = [
        {
            key: _nonnegative_integer(metrics.get(key), name=f"{phase} {key}")
            for key in _PHASE_WORKLOAD_KEYS
        }
        for metrics in numeric_samples
    ]
    if any(workload != workloads[0] for workload in workloads[1:]):
        raise RuntimeError(f"{phase} trainer samples executed different workloads")
    train_s = sum(metrics.get("time/step_train_s", 0.0) for metrics in numeric_samples)
    if not math.isfinite(train_s) or train_s <= 0.0:
        raise RuntimeError(f"{phase} trainer timing is invalid: train={train_s}")
    workload = workloads[0]
    workload_fingerprint = _digest(workload)
    return TrainerPhaseEvidence(
        phase=phase,
        runtime_fingerprint=runtime_fingerprint,
        trajectory_input_fingerprint=trajectory_input_fingerprint,
        packed_input_fingerprint=packed_input_fingerprint,
        workload_fingerprint=workload_fingerprint,
        sample_count=len(samples),
        policy_steps=tuple(step for _, step in samples),
        train_s=train_s,
        metrics=numeric_samples,
    )


def _bundle_bytes(bundles: tuple[Any, ...]) -> bytes:
    from msgspec import msgpack

    return msgpack.encode(tuple(bundle.model_dump(mode="python") for bundle in bundles))


def _matched_input_fingerprints(
    trajectory_fingerprints: list[str], packed_fingerprints: list[str]
) -> tuple[str, str]:
    _require(
        len(trajectory_fingerprints)
        == len(packed_fingerprints)
        == _MATCHED_MEASURED_STEPS,
        "matched trainer phases require complete paired inputs",
    )
    _require(
        len(set(trajectory_fingerprints)) == _MATCHED_MEASURED_STEPS,
        "matched E2E samples must use distinct trajectory inputs",
    )
    return _digest(trajectory_fingerprints), _digest(
        list(zip(trajectory_fingerprints, packed_fingerprints, strict=True))
    )


async def _discard_prepared_pipeline_batch(backend: Any, groups: list[Any]) -> None:
    for group in groups:
        group._distributed_lease = None
    await backend.discard_pipeline_batch(groups)


def _matched_capture_steps(max_steps: int) -> tuple[int, ...]:
    first = max_steps + _CAPTURE_GUARD_STEPS + 1
    return tuple(range(first, first + _MATCHED_MEASURED_STEPS))


def _packed_input_fingerprint(groups: list[Any]) -> str:
    prepared = groups[0]._prepared_training_batch
    if prepared is None or any(
        group._prepared_training_batch is not prepared for group in groups
    ):
        raise RuntimeError("trainer groups do not share one prepared data-plane batch")
    batch = prepared.batch
    packed = batch.payload.packed
    ref = packed.leases.ref
    stable_ref = ref.model_dump(
        mode="json",
        exclude={
            "batch_id",
            "owner_actor_id",
            "lease_id",
            "shared_memory_name",
            "owner_process_id",
            "group_ids",
            "record_ids",
            "min_source_version",
            "max_source_version",
        },
    )
    manifest = {
        "packing_config": prepared.packing_config.model_dump(mode="json"),
        "batch": batch.model_dump(mode="json", exclude={"payload"}),
        "packed_ref": stable_ref,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    digest.update(b"packed_group_shapes:v1")
    digest.update(struct.pack("<Q", len(packed.packed_group_shapes)))
    for shape in packed.packed_group_shapes:
        digest.update(b"\0" if shape is None else b"\1")
        if shape is None:
            continue
        digest.update(struct.pack("<Q", len(shape.leaves)))
        for leaf in shape.leaves:
            typecode = leaf.token_ids.typecode.encode("ascii")
            digest.update(struct.pack("<Q", len(typecode)))
            digest.update(typecode)
            digest.update(
                struct.pack("<QQ", len(leaf.token_ids), leaf.shareable_length)
            )
            digest.update(leaf.token_ids.tobytes())
    shm = shared_memory.SharedMemory(name=ref.shared_memory_name)
    if ref.owner_process_id != os.getpid():
        resource_tracker.unregister(getattr(shm, "_name"), "shared_memory")
    try:
        buffer = shm.buf
        assert buffer is not None
        for tensor in ref.tensors:
            digest.update(buffer[tensor.offset : tensor.offset + tensor.byte_count])
        del buffer
    finally:
        shm.close()
    return digest.hexdigest()


def _load_bundles(path: Path) -> tuple[Any, ...]:
    from msgspec import msgpack

    from art.distributed.trajectory_store import TrajectoryGroupBundle

    values = msgpack.decode(path.read_bytes())
    return tuple(TrajectoryGroupBundle.model_validate(value) for value in values)


async def _capture_training_bundles(groups: Any) -> tuple[Any, ...]:
    from art.distributed.trajectory_store import TrajectoryGroupBundle

    prepared = groups[0]._prepared_training_batch
    selections = (
        tuple(getattr(prepared.batch.payload, "selections", ()))
        if prepared is not None
        else ()
    )
    if len(selections) != len(groups):
        raise RuntimeError("prepared throughput batch lacks exact queue selections")
    materialized = await asyncio.gather(
        *(selection.queue.materialize_selection(selection) for selection in selections)
    )
    return await asyncio.to_thread(
        lambda: tuple(TrajectoryGroupBundle.from_group(group) for group in materialized)
    )


def _collect_matched_packing_shapes(groups: Any) -> None:
    for group in groups:
        group._collect_packing_shape = True


def _reduced_config(
    source: dict[str, Any], *, model_key: str, num_layers: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    reduced = json.loads(json.dumps(source))
    text = _text(reduced)
    source_text = _text(source)
    source_layers = int(source_text["num_hidden_layers"])
    if num_layers > source_layers:
        raise ValueError(
            f"requested {num_layers} layers from {source_layers}-layer model"
        )
    text["num_hidden_layers"] = num_layers
    for field in _LAYER_LIST_FIELDS:
        if field not in source_text:
            continue
        values = source_text[field]
        if len(values) < num_layers:
            raise ValueError(f"{model_key} {field} has only {len(values)} entries")
        text[field] = values[:num_layers]
    source_width = _width_fingerprint(source)
    if not source_width or source_width != _width_fingerprint(reduced):
        raise ValueError("throughput fixture changed or lost production-width fields")
    prefix = "text_config." if "text_config" in source else ""
    return reduced, {
        "source_num_layers": source_layers,
        "changed_paths": [
            f"{prefix}{field}"
            for field in ("num_hidden_layers", *_LAYER_LIST_FIELDS)
            if field == "num_hidden_layers" or field in source_text
        ],
        "width_fingerprint": source_width,
    }


def _copy_metadata(source: Path, target: Path) -> None:
    excluded = {"config.json", "fixture_manifest.json", "model.safetensors.index.json"}
    for path in source.iterdir():
        if (
            path.is_file()
            and path.name not in excluded
            and not path.name.endswith((".safetensors", ".bin", ".pt", ".pth"))
        ):
            shutil.copy2(path, target / path.name)


def _config_only_tensors(config: dict[str, Any], *, model_key: str) -> dict[str, Any]:
    import torch

    tensors = {"_art_config_only": torch.zeros(1)}
    if model_key != "gemma4_moe":
        return tensors
    text = _text(config)
    for layer in range(int(text["num_hidden_layers"])):
        for suffix in (
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
        ):
            tensors[f"model.language_model.layers.{layer}.{suffix}.weight"] = (
                torch.ones(int(text["hidden_size"]), dtype=torch.bfloat16)
            )
    return tensors


def ensure_throughput_fixture(
    *,
    canonical_model: str,
    model_key: str,
    correctness_fixture: Path,
    num_layers: int,
    initialization_version: str,
    random_seed: int,
    output: Path,
) -> ThroughputFixture:
    source_config_path = correctness_fixture / "production_config" / "config.json"
    if not source_config_path.is_file():
        raise RuntimeError(
            f"correctness fixture lacks pinned production config: {source_config_path}"
        )
    source = json.loads(source_config_path.read_text())
    reduced, reduction = _reduced_config(
        source, model_key=model_key, num_layers=num_layers
    )
    vocabulary_contract: dict[str, object] = {
        "config_vocab_size": int(_text(reduced)["vocab_size"])
    }
    _validate_tokenizer_compatible_fixture(correctness_fixture, vocabulary_contract)
    manifest = {
        "version": 1,
        "canonical_model": canonical_model,
        "model_key": model_key,
        "num_layers": num_layers,
        "source_config_sha256": _digest(source),
        "reduced_config_sha256": _digest(reduced),
        "initialization": initialization_version,
        "random_seed": random_seed,
        "vocabulary_contract": vocabulary_contract,
        **reduction,
    }
    output.mkdir()
    _copy_metadata(correctness_fixture, output)
    (output / "config.json").write_text(json.dumps(reduced, indent=2) + "\n")
    from safetensors.torch import save_file

    tensors = _config_only_tensors(reduced, model_key=model_key)
    checkpoint = output / "model.safetensors"
    save_file(tensors, checkpoint)
    if model_key == "gemma4_moe":
        (output / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {}, "weight_map": dict.fromkeys(tensors, checkpoint.name)},
                indent=2,
            )
            + "\n"
        )
    (output / "throughput_fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return ThroughputFixture(
        model_key=model_key,
        path=str(output),
        num_layers=num_layers,
        width_fingerprint=reduction["width_fingerprint"],
        manifest=manifest,
    )


class PolicyActivationEvent(NamedTuple):
    step: int
    trainer_completed_monotonic_s: float
    serving_active_monotonic_s: float

    @property
    def lag_s(self) -> float:
        return self.serving_active_monotonic_s - self.trainer_completed_monotonic_s


async def _activation_event(service: Any, step: int) -> PolicyActivationEvent:
    await service.wait_for_serving(step)
    completed, active = service.policy_activation_timing(step)
    return PolicyActivationEvent(step, completed, active)


async def _cancel_activation_tasks(
    tasks: Mapping[int, asyncio.Task[PolicyActivationEvent]],
) -> None:
    for task in tasks.values():
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)


def _gpu_identities(
    *, trainer_gpu_ids: list[int], inference_gpu_ids: list[int]
) -> list[dict[str, Any]]:
    import torch

    from art.distributed.host_admission import _query_gpu_inventory

    roles = [
        *(("trainer", gpu_id) for gpu_id in trainer_gpu_ids),
        *(("inference", gpu_id) for gpu_id in inference_gpu_ids),
    ]
    if (len(trainer_gpu_ids), len(inference_gpu_ids)) != (2, 2):
        raise RuntimeError(
            "throughput stage requires two trainer and two inference CUDA roles, "
            f"got {roles}"
        )

    def uuid_key(value: str) -> str:
        return value.casefold().removeprefix("gpu-").removeprefix("mig-")

    inventory = _query_gpu_inventory(include_mig=True)
    by_uuid: dict[str, list[tuple[Any, str]]] = {}
    for gpu, driver in inventory:
        by_uuid.setdefault(uuid_key(gpu.uuid), []).append((gpu, driver))
    identities = []
    for role, logical_index in roles:
        properties = torch.cuda.get_device_properties(logical_index)
        cuda_uuid = str(getattr(properties, "uuid", ""))
        matches = by_uuid.get(uuid_key(cuda_uuid), [])
        if len(matches) != 1:
            raise RuntimeError(
                "could not map CUDA-visible GPU to one physical identity: "
                f"logical={logical_index}, uuid={cuda_uuid!r}, matches={len(matches)}"
            )
        gpu, driver = matches[0]
        identities.append(
            {
                "role": role,
                "logical_index": logical_index,
                "uuid": gpu.uuid,
                "parent_uuid": gpu.parent_uuid,
                "pci_bus_id": gpu.pci_bus_id,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "driver_version": driver,
            }
        )
    physical_uuids = {
        str(identity["uuid"]).casefold()
        for identity in identities
        if identity["uuid"] == identity["parent_uuid"]
        and not str(identity["uuid"]).startswith("MIG-")
    }
    if len(physical_uuids) != 4:
        raise RuntimeError(
            "throughput stage requires four unique non-MIG physical GPU UUIDs, "
            f"got {[identity['uuid'] for identity in identities]}"
        )
    return identities


def _hardware(gpu_identities: list[dict[str, Any]]) -> Literal["h200", "b300"]:
    names = {str(identity["name"]).upper() for identity in gpu_identities}
    if len(names) != 1:
        raise RuntimeError(
            f"throughput stage requires homogeneous GPUs, got {sorted(names)}"
        )
    name = next(iter(names))
    if "B300" in name or "GB300" in name:
        return "b300"
    if "H200" in name:
        return "h200"
    raise RuntimeError(f"throughput thresholds are unavailable for {name}")


def _throughput_config_for_hardware(
    model_key: str,
    config: ThroughputWorkflowConfig,
    hardware: Literal["h200", "b300"],
) -> ThroughputWorkflowConfig:
    num_layers = _H200_THROUGHPUT_NUM_LAYERS.get(model_key)
    if hardware != "h200" or num_layers is None:
        return config
    return config.model_copy(update={"num_layers": num_layers})


def _stable_gpu_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": identity["name"],
        "total_memory_bytes": identity["total_memory_bytes"],
        "compute_capability": identity["compute_capability"],
        "driver_version": identity["driver_version"],
    }


def _calibration_contract(
    *,
    base_model: str,
    fixture: ThroughputFixture,
    stage: Any,
    config: ThroughputWorkflowConfig,
    autotune: Any,
    actual_prompt_tokens: int,
    gpu_identities: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = fixture.manifest
    _require(
        all(
            manifest.get(key)
            for key in ("source_config_sha256", "reduced_config_sha256")
        )
        and manifest.get("width_fingerprint") == fixture.width_fingerprint,
        "throughput fixture lacks source/reduced hashes or production width",
    )
    workload = config.model_dump(
        mode="json",
        exclude={"thresholds", "random_initialization_version", "random_seed"},
    )
    role_counts = {
        role: sum(identity["role"] == role for identity in gpu_identities)
        for role in ("trainer", "inference")
    }
    accelerator_specs = {
        json.dumps(_stable_gpu_identity(identity), sort_keys=True)
        for identity in gpu_identities
    }
    _require(
        len(accelerator_specs) == 1,
        "throughput stage requires one homogeneous accelerator specification",
    )
    return {
        "measurement_contract_version": _MEASUREMENT_CONTRACT_VERSION,
        "source_provenance": _source_provenance(),
        "fixture_manifest": manifest,
        "model_identity": {"base_model": base_model, "model_key": fixture.model_key},
        "topology": stage.megatron.topology.model_dump(mode="json"),
        "hardware": {
            "role_counts": role_counts,
            "class": _hardware(gpu_identities),
            "accelerator": json.loads(accelerator_specs.pop()),
        },
        "engine_args": {
            **stage.vllm.engine_args(),
            "seed": config.random_seed,
            "model": f"fixture-sha256:{manifest['reduced_config_sha256']}",
        },
        "autotuner_config": autotune.model_dump(mode="json"),
        "workload_config": {**workload, "actual_prompt_tokens": actual_prompt_tokens},
        "random_initialization": {
            "version": config.random_initialization_version,
            "seed": config.random_seed,
        },
        "trainer_config": {
            "learning_rate": 1e-6,
            "loss_fn": "cispo",
            "eval_fn": None,
            "eval_every_n_steps": 0,
            "eval_at_start": False,
            "save_checkpoint": False,
            "resume": False,
            "score_reference_groups_per_step": config.groups_per_step,
            "score_reference_rollouts_per_group": config.rollouts_per_group,
            "max_steps_off_policy": config.max_steps_off_policy,
            "isolated_warmup_steps": _ISOLATED_WARMUP_STEPS,
            "matched_measured_steps": _MATCHED_MEASURED_STEPS,
        },
        "packed_sequence_length": config.packed_sequence_length,
        "prompt_tokens": actual_prompt_tokens,
        "completion_tokens": config.completion_tokens,
        "rollouts_per_group": config.rollouts_per_group,
        "groups_per_step": config.groups_per_step,
    }


def _chat_token_count(tokenizer: Any, prompt: str) -> int:
    return len(
        _flatten_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    )


def _sized_prompt(tokenizer: Any, *, target_tokens: int) -> str:
    prefix = "Throughput scenario 00000000. Process the following neutral record.\n"
    unit = " measured context item"
    lower, upper = 0, target_tokens
    while lower < upper:
        middle = (lower + upper + 1) // 2
        candidate = prefix + unit * middle
        if _chat_token_count(tokenizer, candidate) <= target_tokens:
            lower = middle
        else:
            upper = middle - 1
    prompt = prefix + unit * lower
    actual = _chat_token_count(tokenizer, prompt)
    if actual < target_tokens - 64:
        raise RuntimeError(
            f"could not size throughput prompt near {target_tokens} tokens: {actual}"
        )
    return prompt


async def _scenarios(prompt: str) -> AsyncIterator[dict[str, str]]:
    index = 0
    while True:
        scenario_id = f"throughput-{index:08d}"
        yield {
            "scenario_id": scenario_id,
            "prompt": prompt.replace("00000000", f"{index:08d}", 1),
        }
        index += 1


def _training_rows(model_output_dir: Path) -> list[dict[str, Any]]:
    history_path = model_output_dir / "history.jsonl"
    if not history_path.is_file():
        raise RuntimeError(f"throughput history is missing: {history_path}")
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    return [row for row in rows if "data/step_nonpadding_logical_tokens" in row]


_COUNT_METRICS = {
    "original_trajectory_tokens": "train/prefix_tree/logical_tokens",
    "nonpadding_logical_tokens": "data/step_nonpadding_logical_tokens",
    "loss_bearing_tokens": "data/step_loss_bearing_tokens",
    "accepted_train_tokens": "data/step_trainable_assistant_tokens",
    "executed_token_equivalents": "data/step_executed_token_equivalents",
    "dummy_token_equivalents": "data/step_dummy_executed_token_equivalents",
    "nominal_capacity_tokens": "data/step_nominal_schedule_capacity_tokens",
    "dummy_schedule_capacity_tokens": "data/step_dummy_schedule_capacity_tokens",
    "unused_packed_capacity_tokens": "data/step_unused_packed_capacity_tokens",
    "packed_sequences": "data/step_packed_sequences",
    "real_microbatches": "pipeline/global_real_microbatches",
    "dummy_microbatches": "pipeline/global_dummy_microbatches",
}


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [row.get(key) for row in rows]
    _require(
        all(
            isinstance(value, int | float) and math.isfinite(float(value))
            for value in values
        ),
        f"throughput rows lack finite numeric {key}: {values}",
    )
    return [float(value) for value in values if isinstance(value, int | float)]


def _total(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_numeric_values(rows, key))


def _runtime_workload_counts(
    rows: list[dict[str, Any]], *, packed_sequence_length: int
) -> dict[str, int]:
    count_rows = [
        {
            name: _nonnegative_integer(
                row.get(key), name=f"step {row.get('step')} {key}"
            )
            for name, key in _COUNT_METRICS.items()
        }
        for row in rows
    ]
    for counts in count_rows:
        real_capacity = (
            counts["nominal_capacity_tokens"] - counts["dummy_schedule_capacity_tokens"]
        )
        real_executed = (
            counts["executed_token_equivalents"] - counts["dummy_token_equivalents"]
        )
        _require(
            counts["packed_sequences"] == counts["real_microbatches"]
            and counts["nominal_capacity_tokens"]
            == (counts["real_microbatches"] + counts["dummy_microbatches"])
            * packed_sequence_length
            and counts["dummy_schedule_capacity_tokens"]
            == counts["dummy_microbatches"] * packed_sequence_length
            and counts["unused_packed_capacity_tokens"]
            == real_capacity - counts["nonpadding_logical_tokens"]
            and 0
            < counts["accepted_train_tokens"]
            == counts["loss_bearing_tokens"]
            <= counts["nonpadding_logical_tokens"]
            <= real_executed
            <= real_capacity
            and 0
            <= counts["dummy_token_equivalents"]
            <= counts["dummy_schedule_capacity_tokens"],
            f"runtime token accounting does not reconcile: {counts}",
        )
    totals = {
        name: sum(counts[name] for counts in count_rows) for name in _COUNT_METRICS
    }
    _require(
        totals["packed_sequences"] > 0 and totals["real_microbatches"] > 0,
        f"runtime workload contains no real packed sequences: {totals}",
    )
    return totals


def _accepted_token_weighted(rows: list[dict[str, Any]], key: str) -> float:
    values = _numeric_values(rows, key)
    weights = _numeric_values(rows, "data/step_trainable_assistant_tokens")
    total_weight = sum(weights)
    _require(total_weight > 0.0, "throughput rows contain no accepted assistant tokens")
    return sum(
        value * weight for value, weight in zip(values, weights, strict=True)
    ) / (total_weight)


def _discard_rates(rows: list[dict[str, Any]]) -> tuple[float, float]:
    stale = _total(rows, _STALE_GROUPS)
    zero_variance = _total(rows, _ZERO_VARIANCE_GROUPS)
    _require(
        stale >= 0.0 and zero_variance >= 0.0, "discard counts must be nonnegative"
    )
    denominator = max(
        _total(rows, "data/step_num_groups_trainable") + stale + zero_variance,
        1.0,
    )
    return stale / denominator, zero_variance / denominator


def _window_measurements(stats: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration_s = float(stats.window_end_s) - float(stats.window_start_s)
    stale_rate, zero_variance_rate = _discard_rates(rows)
    _require(
        math.isfinite(duration_s) and duration_s > 0.0,
        f"autotuner window {stats.start_step}..{stats.end_step} has invalid duration",
    )
    _require(
        math.isclose(
            stale_rate, float(stats.actual_stale_frac), rel_tol=0.0, abs_tol=1e-12
        ),
        f"history and autotuner stale rates disagree at step {stats.end_step}",
    )
    return {
        "start_step": stats.start_step,
        "end_step": stats.end_step,
        "duration_s": duration_s,
        "vllm_pressure": float(stats.vllm_pressure),
        "vllm_waiting_capacity_request_s": float(stats.vllm_waiting_capacity_request_s),
        "vllm_running_request_s": float(stats.vllm_running_request_s),
        "trainer_underfeed": float(stats.trainer_underfeed_score),
        _POLICY_AGE_MEAN: _accepted_token_weighted(rows, _POLICY_AGE_MEAN),
        _POLICY_AGE_P95: max(_numeric_values(rows, _POLICY_AGE_P95)),
        _FRESHNESS_DISCOUNT: _accepted_token_weighted(rows, _FRESHNESS_DISCOUNT),
        "discarded/rate/stale_groups": stale_rate,
        "discarded/rate/zero_variance_groups": zero_variance_rate,
    }


async def _run_isolated_backend_phase(
    *,
    backend: Any,
    model: Any,
    service: Any,
    train: Callable[..., Awaitable[Any]],
    bundles_paths: tuple[Path, ...],
) -> TrainerPhaseEvidence:
    from art.distributed.trajectory_store import TrajectoryGroupBundle

    _require(
        len(bundles_paths) == _MATCHED_MEASURED_STEPS,
        "isolated phase requires every matched E2E input",
    )
    trajectory_input_fingerprints = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in bundles_paths
    ]
    benchmark_paths = (bundles_paths[0],) * _ISOLATED_WARMUP_STEPS + bundles_paths
    packed_input_fingerprints: list[str] = []
    samples: list[tuple[Mapping[str, Any], int]] = []
    for sample_index, bundles_path in enumerate(benchmark_paths):
        expected_trajectory_fingerprint = hashlib.sha256(
            bundles_path.read_bytes()
        ).hexdigest()
        source_bundles = _load_bundles(bundles_path)
        groups = [bundle.build() for bundle in source_bundles]
        rebuilt = tuple(TrajectoryGroupBundle.from_group(group) for group in groups)
        if (
            hashlib.sha256(_bundle_bytes(rebuilt)).hexdigest()
            != expected_trajectory_fingerprint
        ):
            raise RuntimeError("isolated trajectory input changed during round trip")
        _collect_matched_packing_shapes(groups)
        packing = await backend.prepare_pipeline_batch(model, groups)
        if packing is None:
            raise RuntimeError("isolated backend benchmark produced no packed batch")
        try:
            current_packed_fingerprint = _packed_input_fingerprint(groups)
        except BaseException:
            await _discard_prepared_pipeline_batch(backend, groups)
            raise
        result = await train(
            model,
            groups,
            learning_rate=1e-6,
            loss_fn="cispo",
            loss_fn_config=None,
            normalize_advantages=True,
            save_checkpoint=False,
            adam_params=None,
            optimizer_save_interval=5,
        )
        if sample_index >= _ISOLATED_WARMUP_STEPS:
            packed_input_fingerprints.append(current_packed_fingerprint)
            samples.append((result.metrics, int(result.step)))
        await service.wait_for_serving(int(result.step))
    trajectory_input_fingerprint, packed_input_fingerprint = (
        _matched_input_fingerprints(
            trajectory_input_fingerprints, packed_input_fingerprints
        )
    )
    return _phase_evidence(
        phase="isolated",
        runtime_fingerprint=service._runtime_spec().fingerprint,
        trajectory_input_fingerprint=trajectory_input_fingerprint,
        packed_input_fingerprint=packed_input_fingerprint,
        samples=samples,
    )


def _collect_measurements(
    *,
    fixture: ThroughputFixture,
    config: ThroughputWorkflowConfig,
    hardware: Literal["h200", "b300"],
    model_output_dir: Path,
    profile: Any,
    events: list[PolicyActivationEvent],
    isolated: TrainerPhaseEvidence,
    e2e: TrainerPhaseEvidence,
    capture_settings: Mapping[str, int],
    calibration_fingerprint: str,
) -> dict[str, Any]:
    from art.pipeline_tuner.autotune import _trainer_underfeed_score

    _require(
        profile.config.mode == "online",
        f"throughput stage requires online autotuning, got {profile.config.mode}",
    )
    policy_age_limit = profile.policy_age_limit_steps
    _require(
        isinstance(policy_age_limit, int | float)
        and math.isfinite(float(policy_age_limit))
        and float(policy_age_limit) >= 0.0,
        f"online autotuner lacks a policy-age limit: {policy_age_limit}",
    )
    policy_age_limit = float(policy_age_limit)
    _require(
        policy_age_limit == config.max_steps_off_policy,
        "autotuner policy-age limit does not match the throughput contract: "
        f"{policy_age_limit} != {config.max_steps_off_policy}",
    )
    decisions = [
        decision for decision in profile.decisions if decision.stats is not None
    ]
    selected = decisions[-2:]
    _require(
        len(selected) == 2 and _same_executed_settings(selected),
        "throughput evidence requires two trailing windows executed with one setting",
    )
    executed_settings = selected[-1].previous.model_dump(mode="json")
    _require(
        dict(capture_settings) == executed_settings,
        "matched capture did not use the measured pipeline settings: "
        f"{dict(capture_settings)} != {executed_settings}",
    )
    stats = [decision.stats for decision in selected]
    assert all(window is not None for window in stats)
    first_stats, last_stats = stats
    expected_window = (
        config.max_steps - profile.config.window_steps + 1,
        config.max_steps,
    )
    _require(
        (last_stats.start_step, last_stats.end_step) == expected_window,
        f"final autotuner window is not {expected_window[0]}..{expected_window[1]}",
    )
    for left, right in zip(stats, stats[1:]):
        _require(
            left.end_step + 1 == right.start_step
            and math.isclose(
                float(left.window_end_s),
                float(right.window_start_s),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "trailing autotuner windows are not contiguous",
        )
    history_rows = _training_rows(model_output_dir)
    by_step = {int(row["step"]): row for row in history_rows}
    _require(
        len(by_step) == len(history_rows),
        "throughput history contains duplicate training steps",
    )
    steps = list(range(first_stats.start_step, last_stats.end_step + 1))
    missing = [step for step in steps if step not in by_step]
    _require(not missing, f"autotuner decision window lacks train rows: {missing}")
    rows = [by_step[step] for step in steps]
    window_rows = [
        [by_step[step] for step in range(window.start_step, window.end_step + 1)]
        for window in stats
    ]
    windows = [
        _window_measurements(window, selected_rows)
        for window, selected_rows in zip(stats, window_rows, strict=True)
    ]
    e2e_elapsed_s = float(last_stats.window_end_s) - float(first_stats.window_start_s)
    _require(
        math.isclose(
            sum(window["duration_s"] for window in windows),
            e2e_elapsed_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "autotuner window durations do not reconcile",
    )

    events_by_step = {event.step: event for event in events}
    _require(
        len(events_by_step) == len(events),
        "throughput stage observed duplicate policy activations",
    )
    activation_steps = [first_stats.start_step - 1, *steps]
    missing_events = [step for step in activation_steps if step not in events_by_step]
    _require(
        not missing_events,
        f"throughput decision intervals lack policy activations: {missing_events}",
    )
    interval_events = [events_by_step[step] for step in activation_steps]
    window_events = interval_events[1:]
    activation_times = [event.serving_active_monotonic_s for event in interval_events]
    intervals = [
        right - left for left, right in zip(activation_times, activation_times[1:])
    ]
    _require(
        all(interval > 0.0 for interval in intervals),
        f"policy activations were not ordered in time: {intervals}",
    )
    lags = [event.lag_s for event in window_events]
    _require(
        all(lag >= 0.0 for lag in lags),
        f"policy activation preceded trainer completion: {lags}",
    )

    counts = _runtime_workload_counts(
        rows, packed_sequence_length=config.packed_sequence_length
    )
    logical = counts["nonpadding_logical_tokens"]
    train_s = _total(rows, "time/step_train_s")
    wall_s = _total(rows, "time/step_wall_s")
    _require(
        0.0 < train_s <= wall_s <= e2e_elapsed_s + 1e-6,
        f"invalid throughput durations: {train_s}, {wall_s}, {e2e_elapsed_s}",
    )

    stale_rate, zero_variance_rate = _discard_rates(rows)
    waiting_request_s = sum(
        window["vllm_waiting_capacity_request_s"] for window in windows
    )
    running_request_s = sum(window["vllm_running_request_s"] for window in windows)
    stable_vllm_pressure = (
        waiting_request_s / running_request_s
        if running_request_s > 0.0
        else math.inf
        if waiting_request_s > 0.0
        else 0.0
    )
    capacities = _numeric_values(rows, "data/step_nominal_schedule_capacity_tokens")
    nonpadding = _numeric_values(rows, "data/step_nonpadding_logical_tokens")
    stable_trainer_underfeed = _trainer_underfeed_score(
        idle_frac=_total(rows, "time/step_collect_batch_s") / wall_s,
        unused_and_dummy_ratio=fmean(
            max(0.0, (capacity - used) / capacity)
            for capacity, used in zip(capacities, nonpadding, strict=True)
        ),
    )
    thresholds = config.thresholds.get(hardware)
    measurements = {
        "hardware": hardware,
        "calibration_basis": (
            thresholds.calibration_basis if thresholds is not None else None
        ),
        "calibration_fingerprint": calibration_fingerprint,
        "model_key": fixture.model_key,
        "model_path": fixture.path,
        "num_layers": fixture.num_layers,
        "packed_sequence_length": config.packed_sequence_length,
        "width_fingerprint": fixture.width_fingerprint,
        **counts,
        "isolated_train_tok_s": isolated.train_tok_s,
        "matched_e2e_core_train_tok_s": e2e.train_tok_s,
        "e2e_core_train_tok_s": logical / train_s,
        "e2e_train_tok_s": logical / e2e_elapsed_s,
        "accepted_train_tok_s": counts["accepted_train_tokens"] / e2e_elapsed_s,
        _POLICY_AGE_MEAN: _accepted_token_weighted(rows, _POLICY_AGE_MEAN),
        _POLICY_AGE_P95: max(_numeric_values(rows, _POLICY_AGE_P95)),
        _FRESHNESS_DISCOUNT: _accepted_token_weighted(rows, _FRESHNESS_DISCOUNT),
        "discarded/rate/stale_groups": stale_rate,
        "discarded/rate/zero_variance_groups": zero_variance_rate,
        "policy_age_limit_steps": policy_age_limit,
        "mean_ready_batch_idle_s": fmean(
            [float(row["queue/packed_get_wait_s"]) for row in rows]
        ),
        "mean_train_gap_s": (e2e_elapsed_s - train_s) / len(rows),
        "e2e_elapsed_s": e2e_elapsed_s,
        "autotuner_windows": windows,
        "stable_vllm_pressure": stable_vllm_pressure,
        "stable_trainer_underfeed": stable_trainer_underfeed,
        "matched_capture_pipeline_settings": dict(capture_settings),
        "mean_policy_activation_lag_s": fmean(lags),
        "p50_policy_activation_lag_s": median(lags),
        "p95_policy_activation_lag_s": quantiles(lags, n=20, method="inclusive")[18],
        "max_policy_activation_lag_s": max(lags),
        "post_warmup_policy_activation_count": len(window_events),
        "mean_policy_activation_interval_s": fmean(intervals),
        "p50_policy_activation_interval_s": median(intervals),
        "p95_policy_activation_interval_s": quantiles(
            intervals, n=20, method="inclusive"
        )[18],
        "second_max_policy_activation_interval_s": sorted(intervals)[-2],
        "max_policy_activation_interval_s": max(intervals),
    }
    matched_fields = (
        "runtime_fingerprint",
        "trajectory_input_fingerprint",
        "packed_input_fingerprint",
        "workload_fingerprint",
    )
    mismatches = {
        name: (getattr(e2e, name), getattr(isolated, name))
        for name in matched_fields
        if getattr(e2e, name) != getattr(isolated, name)
    }
    _require(
        not mismatches,
        f"isolated and E2E phases did not execute the same packed input: {mismatches}",
    )
    capture_steps = _matched_capture_steps(config.max_steps)
    _require(
        e2e.policy_steps == capture_steps,
        f"matched E2E inputs were not captured at reserved steps {capture_steps}",
    )
    _require(
        e2e.sample_count == isolated.sample_count == _MATCHED_MEASURED_STEPS,
        "matched trainer phases have asymmetric sample counts",
    )
    expected_isolated_steps = tuple(
        range(
            capture_steps[-1] + 1 + _ISOLATED_WARMUP_STEPS,
            capture_steps[-1] + 1 + _ISOLATED_WARMUP_STEPS + isolated.sample_count,
        )
    )
    _require(
        isolated.policy_steps == expected_isolated_steps,
        "isolated measured steps do not follow the configured warmup: "
        f"{isolated.policy_steps}",
    )
    return measurements


def acceptance_failures(
    measurements: Mapping[str, Any],
    config: ThroughputWorkflowConfig,
    thresholds: ThroughputThresholds | None,
) -> list[str]:
    checks = {
        "stable_min_vllm_pressure": measurements["stable_vllm_pressure"]
        >= config.min_vllm_pressure,
        "stable_trainer_underfeed": measurements["stable_trainer_underfeed"]
        <= config.max_trainer_underfeed,
    }
    for window in measurements["autotuner_windows"]:
        prefix = f"window_{window['start_step']}_{window['end_step']}"
        checks.update(
            {
                f"{prefix}_policy_age_p95": window[_POLICY_AGE_P95]
                <= measurements["policy_age_limit_steps"],
                f"{prefix}_zero_variance_rate": window[
                    "discarded/rate/zero_variance_groups"
                ]
                == 0.0,
            }
        )
    failures = [name for name, passed in checks.items() if not passed]
    if thresholds is None:
        return [f"missing_{measurements['hardware']}_calibration", *failures]
    floor_checks = {
        "isolated_train_tok_s": measurements["isolated_train_tok_s"]
        >= thresholds.min_isolated_train_tok_s,
        "e2e_train_tok_s": measurements["e2e_train_tok_s"]
        >= thresholds.min_e2e_train_tok_s,
        "accepted_train_tok_s": measurements["accepted_train_tok_s"]
        >= thresholds.min_accepted_train_tok_s,
        "e2e_to_isolated_ratio": measurements["e2e_train_tok_s"]
        / measurements["isolated_train_tok_s"]
        >= thresholds.min_e2e_to_isolated_ratio,
        "matched_core_to_isolated_ratio": measurements["matched_e2e_core_train_tok_s"]
        / measurements["isolated_train_tok_s"]
        >= thresholds.min_matched_core_to_isolated_ratio,
        "matched_core_to_isolated_ratio_max": measurements[
            "matched_e2e_core_train_tok_s"
        ]
        / measurements["isolated_train_tok_s"]
        <= thresholds.max_matched_core_to_isolated_ratio,
        "mean_policy_activation_lag_s": measurements["mean_policy_activation_lag_s"]
        <= thresholds.max_mean_policy_activation_lag_s,
        "max_policy_activation_lag_s": measurements["max_policy_activation_lag_s"]
        <= thresholds.max_policy_activation_lag_s,
        "repeated_policy_activation_cadence_s": measurements[
            "second_max_policy_activation_interval_s"
        ]
        <= thresholds.max_repeated_policy_activation_interval_s,
    }
    if thresholds.calibration_fingerprint is not None:
        floor_checks["calibration_fingerprint"] = (
            measurements["calibration_fingerprint"]
            == thresholds.calibration_fingerprint
        )
    if measurements["hardware"] == "b300":
        floor_checks["calibration_basis"] = thresholds.calibration_basis == "measured"
    return [
        *failures,
        *(name for name, passed in floor_checks.items() if not passed),
    ]


async def _run_e2e_throughput_async(
    *,
    base_model: str,
    allow_unvalidated_arch: bool,
    stage: Any,
    config: ThroughputWorkflowConfig,
    fixture: ThroughputFixture,
    gpu_identities: list[dict[str, Any]],
    hardware: Literal["h200", "b300"],
) -> ValidationStageResult:
    from transformers import AutoTokenizer

    import art
    from art.megatron.backend import MegatronBackend
    from art.pipeline_trainer import PipelineTrainer
    from art.pipeline_tuner import PipelineAutotuneConfig, PipelineAutotunerProfile
    from art.preprocessing.policy_spans import validate_complete_policy_token_spans
    from art.preprocessing.vllm_tokens import choice_completion_tokens

    if stage.megatron is None or stage.vllm is None:
        raise RuntimeError(
            "E2E throughput requires separate Megatron and vLLM resources"
        )
    stage_dir = Path(os.environ[_STAGE_DIR_ENV])
    stage_dir.mkdir(parents=True, exist_ok=True)
    topology = stage.megatron.topology
    art.init_megatron_runtime_config(
        topology=topology.to_megatron_config(),
        packed_sequence_length=config.packed_sequence_length,
    )
    engine_args = stage.vllm.engine_args()
    engine_args["seed"] = config.random_seed
    engine_args["model"] = fixture.path
    max_model_len = int(engine_args["max_model_len"])
    if config.prompt_tokens + config.completion_tokens > max_model_len:
        raise RuntimeError(
            "throughput prompt and completion exceed vLLM context: "
            f"{config.prompt_tokens}+{config.completion_tokens}>{max_model_len}"
        )
    internal_config = {
        "trainer_gpu_ids": stage.megatron.gpu_ids,
        "inference_gpu_ids": stage.vllm.gpu_ids,
        "rollout_weights_mode": "lora",
        "rollout_weight_update_mode": "in_flight_lora",
        "engine_args": engine_args,
        "init_args": {
            "model_name": fixture.path,
            "max_seq_length": config.packed_sequence_length,
            "random_state": config.random_seed,
        },
        "allow_unvalidated_arch": allow_unvalidated_arch,
        "megatron_model_initialization": "random",
    }
    from art.megatron.model_support.tokenizer import (
        configure_tokenizer_for_model_support,
    )

    tokenizer = configure_tokenizer_for_model_support(
        cast(Any, AutoTokenizer.from_pretrained(fixture.path, local_files_only=True)),
        base_model=base_model,
        internal_config=internal_config,
    )
    prompt = _sized_prompt(tokenizer, target_tokens=config.prompt_tokens)
    actual_prompt_tokens = _chat_token_count(tokenizer, prompt)
    run_name = f"throughput-{fixture.model_key}-{uuid.uuid4().hex[:8]}"
    model_output_dir: Path | None = None
    events: list[PolicyActivationEvent] = []
    e2e_phase: TrainerPhaseEvidence | None = None
    isolated_phase: TrainerPhaseEvidence | None = None
    captured_training_inputs: list[
        tuple[tuple[Any, ...], str, dict[str, int], Mapping[str, Any], int]
    ] = []
    bundles_paths = tuple(
        stage_dir / f"matched_packed_input_{index}.msgpack"
        for index in range(_MATCHED_MEASURED_STEPS)
    )
    autotune = PipelineAutotuneConfig(
        mode="online",
        output_name="throughput",
        initial_model_calls_per_inference_gpu=(
            config.initial_model_calls_per_inference_gpu
        ),
        initial_min_groups_per_packed_sequence=config.groups_per_step,
        initial_max_groups_per_packed_sequence=config.groups_per_step,
        vllm_metric_interval_s=0.25,
    )
    measured_steps = config.max_steps - autotune.warmup_ignore_steps
    if (
        measured_steps < 2 * autotune.window_steps
        or measured_steps % autotune.window_steps
    ):
        raise RuntimeError(
            "throughput stage must end on a whole autotuner window after at least "
            f"two measured windows: max_steps={config.max_steps}, "
            f"warmup={autotune.warmup_ignore_steps}, window={autotune.window_steps}"
        )
    # Keep test-only trajectory materialization off the final measured publication.
    capture_train_calls = _matched_capture_steps(config.max_steps)
    runtime_contract = _calibration_contract(
        base_model=base_model,
        fixture=fixture,
        stage=stage,
        config=config,
        autotune=autotune,
        actual_prompt_tokens=actual_prompt_tokens,
        gpu_identities=gpu_identities,
    )
    calibration_fingerprint = _digest(runtime_contract)

    async with MegatronBackend(
        path=str(stage_dir / "art"),
        enable_expert_replay=topology.ep > 1,
        in_process=False,
    ) as backend:
        model = cast(
            Any,
            art.TrainableModel(
                name=run_name,
                run_name=run_name,
                project="model-support-throughput",
                base_model=base_model,
                _internal_config=cast(art.dev.InternalModelConfig, internal_config),
                report_metrics=[],
            ),
        )
        await model.register(backend)
        model_output_dir = Path(model._get_output_dir())
        client = model.openai_client()
        try:
            await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model.get_inference_name(),
                max_tokens=config.completion_tokens,
                temperature=0.0,
                timeout=1200.0,
                extra_body={
                    "ignore_eos": True,
                    "min_tokens": config.completion_tokens,
                },
            )

            async def rollout_fn(
                rollout_model: Any,
                scenario: dict[str, str],
                _rollout_config: None,
            ) -> Any:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": scenario["prompt"]}],
                    model=rollout_model.get_inference_name(),
                    max_tokens=config.completion_tokens,
                    n=config.rollouts_per_group,
                    temperature=1.0,
                    seed=int(scenario["scenario_id"].rsplit("-", 1)[-1]),
                    logprobs=True,
                    top_logprobs=0,
                    timeout=1200.0,
                    extra_body={
                        "ignore_eos": True,
                        "min_tokens": config.completion_tokens,
                    },
                )
                if len(response.choices) != config.rollouts_per_group:
                    raise RuntimeError(
                        "vLLM returned an incomplete rollout group: "
                        f"{len(response.choices)} != {config.rollouts_per_group}"
                    )
                trajectories = []
                for index, choice in enumerate(response.choices):
                    completion_tokens = choice_completion_tokens(choice)
                    if not isinstance(completion_tokens, int) or (
                        completion_tokens != config.completion_tokens
                    ):
                        raise RuntimeError(
                            "throughput completion length changed: "
                            f"{completion_tokens} != {config.completion_tokens}"
                        )
                    validate_complete_policy_token_spans(
                        choice, completion_tokens=completion_tokens
                    )
                    trajectories.append(
                        art.Trajectory(
                            messages_and_choices=[
                                {"role": "user", "content": scenario["prompt"]},
                                choice,
                            ],
                            reward=index / (config.rollouts_per_group - 1),
                            metrics={"completion_tokens": completion_tokens},
                            metadata={"scenario_id": scenario["scenario_id"]},
                        )
                    )
                return art.TrajectoryGroup(
                    trajectories,
                    metadata={"scenario_id": scenario["scenario_id"]},
                )

            trainer = PipelineTrainer(
                model=model,
                backend=backend,
                rollout_fn=rollout_fn,
                scenarios=_scenarios(prompt),
                config=None,
                autotune=autotune,
                learning_rate=1e-6,
                loss_fn="cispo",
                max_steps=capture_train_calls[-1],
                eval_fn=None,
                eval_every_n_steps=0,
                eval_at_start=False,
                save_checkpoint=False,
                resume=False,
                log_interval_seconds=30.0,
                score_reference_groups_per_step=float(config.groups_per_step),
                score_reference_rollouts_per_group=float(config.rollouts_per_group),
                max_steps_off_policy=config.max_steps_off_policy,
            )
            from art.megatron.distributed_service import DistributedMegatronService

            service = cast(
                DistributedMegatronService, await backend._get_service(model)
            )
            activation_tasks: dict[int, asyncio.Task[PolicyActivationEvent]] = {}
            original_train = backend.train
            train_call_count = 0

            async def tracked_train(*args: Any, **kwargs: Any) -> Any:
                nonlocal train_call_count
                train_call_count += 1
                if len(args) < 2:
                    raise RuntimeError(
                        "PipelineTrainer did not pass trajectory groups positionally"
                    )
                groups = args[1]
                captured: tuple[tuple[Any, ...], str, dict[str, int]] | None = None
                if train_call_count in capture_train_calls:
                    try:
                        _collect_matched_packing_shapes(groups)
                        captured = (
                            await _capture_training_bundles(groups),
                            _packed_input_fingerprint(groups),
                            _current_pipeline_settings(trainer),
                        )
                    except BaseException:
                        await _discard_prepared_pipeline_batch(backend, groups)
                        raise
                result = await original_train(*args, **kwargs)
                step = int(result.step)
                if step in activation_tasks:
                    raise RuntimeError(
                        f"duplicate trainer completion for policy {step}"
                    )
                activation_tasks[step] = asyncio.create_task(
                    _activation_event(service, step)
                )
                if captured is not None:
                    captured_training_inputs.append((*captured, result.metrics, step))
                return result

            setattr(backend, "train", tracked_train)
            try:
                with _hold_pipeline_settings_after_step(trainer, config.max_steps):
                    await trainer.train(handle_signals=False)
                if train_call_count != capture_train_calls[-1]:
                    raise RuntimeError(
                        "online pipeline did not execute measurement plus capture steps: "
                        f"{train_call_count} != {capture_train_calls[-1]}"
                    )
                events = sorted(
                    await asyncio.gather(*activation_tasks.values()),
                    key=lambda event: event.step,
                )
            finally:
                setattr(backend, "train", original_train)
                await _cancel_activation_tasks(activation_tasks)
            if len(captured_training_inputs) != _MATCHED_MEASURED_STEPS:
                raise RuntimeError(
                    "online pipeline did not capture every matched train batch"
                )
            bundle_payloads = await asyncio.gather(
                *(
                    asyncio.to_thread(_bundle_bytes, captured[0])
                    for captured in captured_training_inputs
                )
            )
            await asyncio.gather(
                *(
                    asyncio.to_thread(path.write_bytes, payload)
                    for path, payload in zip(
                        bundles_paths, bundle_payloads, strict=True
                    )
                )
            )
            capture_settings = captured_training_inputs[0][2]
            _require(
                all(
                    captured[2] == capture_settings
                    for captured in captured_training_inputs[1:]
                ),
                "matched E2E samples used different pipeline settings",
            )
            trajectory_input_fingerprint, packed_input_fingerprint = (
                _matched_input_fingerprints(
                    [
                        hashlib.sha256(payload).hexdigest()
                        for payload in bundle_payloads
                    ],
                    [captured[1] for captured in captured_training_inputs],
                )
            )
            e2e_phase = _phase_evidence(
                phase="e2e",
                runtime_fingerprint=service._runtime_spec().fingerprint,
                trajectory_input_fingerprint=trajectory_input_fingerprint,
                packed_input_fingerprint=packed_input_fingerprint,
                samples=[
                    (captured[3], captured[4]) for captured in captured_training_inputs
                ],
            )
            isolated_phase = await _run_isolated_backend_phase(
                backend=backend,
                model=model,
                service=service,
                train=original_train,
                bundles_paths=bundles_paths,
            )
        finally:
            await client.close()

    assert model_output_dir is not None
    assert e2e_phase is not None and isolated_phase is not None
    profile_path = model_output_dir / "pipeline_tuner" / "throughput.json"
    profile = PipelineAutotunerProfile.model_validate_json(profile_path.read_text())
    measurements = _collect_measurements(
        fixture=fixture,
        config=config,
        hardware=hardware,
        model_output_dir=model_output_dir,
        profile=profile,
        events=events,
        isolated=isolated_phase,
        e2e=e2e_phase,
        capture_settings=capture_settings,
        calibration_fingerprint=calibration_fingerprint,
    )
    activation_path = stage_dir / "policy_activation_timeline.json"
    activation_path.write_text(
        json.dumps([event._asdict() for event in events], indent=2) + "\n"
    )
    thresholds = config.thresholds.get(hardware)
    failures = acceptance_failures(measurements, config, thresholds)
    metrics = {
        **measurements,
        "gpu_identities": [
            {"role": identity["role"], **_stable_gpu_identity(identity)}
            for identity in gpu_identities
        ],
        "isolated": isolated_phase._asdict(),
        "e2e": e2e_phase._asdict(),
        "runtime_contract": runtime_contract,
        "matched_trajectory_inputs": [str(path) for path in bundles_paths],
        "autotuner_profile": str(profile_path),
        "policy_activation_timeline": str(activation_path),
        "thresholds": thresholds.model_dump(mode="json") if thresholds else None,
        "acceptance_failures": failures,
    }
    (stage_dir / "throughput_measurements.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    return ValidationStageResult(
        name="e2e_throughput",
        passed=not failures,
        metrics=metrics,
        artifact_dir=str(stage_dir),
    )


def run_e2e_throughput(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    del architecture
    resources = handler_workflow_resources_for_base_model(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    if resources is None or resources.e2e_throughput is None:
        raise RuntimeError(f"missing E2E throughput resources for {base_model}")
    spec = get_model_support_spec(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    import torch

    stage = resolve_stage_resources_for_visible_gpus(
        "e2e_throughput",
        resources.e2e_throughput,
        visible_gpu_count=int(torch.cuda.device_count()),
    )
    config = stage.throughput
    if config is None:
        raise RuntimeError("E2E throughput resources lack throughput configuration")
    if stage.megatron is None or stage.vllm is None:
        raise RuntimeError(
            "E2E throughput requires separate Megatron and vLLM resources"
        )
    gpu_identities = _gpu_identities(
        trainer_gpu_ids=stage.megatron.gpu_ids,
        inference_gpu_ids=stage.vllm.gpu_ids,
    )
    hardware = _hardware(gpu_identities)
    config = _throughput_config_for_hardware(spec.key, config, hardware)
    correctness_path = os.environ.get(FIXTURE_PATH_ENV)
    if correctness_path is None:
        raise RuntimeError(f"missing {FIXTURE_PATH_ENV}")
    fixture = ensure_throughput_fixture(
        canonical_model=base_model,
        model_key=spec.key,
        correctness_fixture=Path(correctness_path),
        num_layers=config.num_layers,
        initialization_version=config.random_initialization_version,
        random_seed=config.random_seed,
        output=Path(os.environ[_STAGE_DIR_ENV]) / "production_width_model",
    )
    os.environ["WANDB_MODE"] = "disabled"
    return asyncio.run(
        _run_e2e_throughput_async(
            base_model=base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
            stage=stage,
            config=config,
            fixture=fixture,
            gpu_identities=gpu_identities,
            hardware=hardware,
        )
    )
