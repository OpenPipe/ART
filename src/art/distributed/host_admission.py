from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .specs import HostServiceHealth, HostSpec

_SCHEMA = "art-host-runtime-v1"
_SHA256 = r"^[0-9a-f]{64}$"
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_BASE_PACKAGES = ("openpipe-art", "pydantic", "torchmonarch")
_TRAINER_PACKAGES = (
    "flash-attn-4",
    "megatron-bridge",
    "megatron-core",
    "numpy",
    "torch",
    "transformer_engine",
    "transformer_engine_torch",
    "transformers",
    "triton",
)
_RUNTIME_ENV = {
    "ART_DISABLE_MEGATRON_COMPILE",
    "ART_MEGATRON_ALLOW_UNVALIDATED_ARCH",
    "ART_MEGATRON_ENABLE_MOE_ROUTING_REPLAY",
    "ART_MEGATRON_OFFLOAD_BETWEEN_JOBS",
    "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD",
    "ART_VLLM_RUNTIME_BIN",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "NCCL_ALGO",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_NET",
    "NCCL_NET_PLUGIN",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_PROTO",
    "NCCL_SOCKET_IFNAME",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "VLLM_USE_V1",
    "VLLM_WORKER_MULTIPROC_METHOD",
}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GpuIdentity(_Contract):
    index: int = Field(ge=0)
    uuid: str = Field(pattern=r"^GPU-[0-9A-Fa-f-]+$")
    pci_bus_id: str = Field(
        pattern=r"^(?:[0-9A-F]{4}|[0-9A-F]{8}):[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]$"
    )


class RuntimeFingerprint(_Contract):
    schema_version: Literal["art-host-runtime-v1"] = _SCHEMA
    art_build_sha256: str = Field(pattern=_SHA256)
    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    packages: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _validate_digest(self) -> RuntimeFingerprint:
        manifest = self.model_dump(mode="json", exclude={"sha256"})
        if self.sha256 != _json_sha256(manifest):
            raise ValueError("runtime fingerprint digest does not match its manifest")
        return self


class HostAdmissionRequest(_Contract):
    host_id: str = Field(min_length=1)
    node_rank: int = Field(ge=0)
    expected_gpu_ids: tuple[Annotated[int, Field(ge=0)], ...]
    runtime_packages: tuple[Annotated[str, Field(min_length=1)], ...]


class HostAdmissionReport(HostServiceHealth):
    node_rank: int = Field(ge=0)
    boot_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    assigned_gpus: tuple[GpuIdentity, ...]
    nvidia_driver_version: str | None = Field(
        default=None, pattern=r"^[0-9]+(?:\.[0-9]+)*$"
    )
    runtime: RuntimeFingerprint


def runtime_package_names(*, trainer: bool) -> tuple[str, ...]:
    return tuple(sorted((*_BASE_PACKAGES, *(_TRAINER_PACKAGES if trainer else ()))))


def build_runtime_fingerprint(
    package_names: Sequence[str] = _BASE_PACKAGES,
) -> RuntimeFingerprint:
    libc = platform.libc_ver()
    values = {
        "schema_version": _SCHEMA,
        "art_build_sha256": _art_build_sha256(),
        "python": f"{platform.python_implementation()}-{platform.python_version()}-"
        f"{sys.implementation.cache_tag}",
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}-"
        f"{libc[0]}-{libc[1]}",
        "packages": tuple((name, metadata.version(name)) for name in package_names),
        "environment": _runtime_environment(os.environ),
    }
    return RuntimeFingerprint(**values, sha256=_json_sha256(values))


def inspect_host(request: HostAdmissionRequest) -> HostAdmissionReport:
    runtime = build_runtime_fingerprint(request.runtime_packages)
    inventory = {
        gpu.index: (gpu, driver)
        for gpu, driver in (_query_gpu_inventory() if request.expected_gpu_ids else ())
    }
    missing = sorted(set(request.expected_gpu_ids) - inventory.keys())
    if missing:
        raise RuntimeError(
            f"host {request.host_id!r} is missing configured physical GPUs {missing}; "
            f"nvidia-smi reported {sorted(inventory)}"
        )
    assigned = tuple(inventory[index][0] for index in request.expected_gpu_ids)
    drivers = {inventory[index][1] for index in request.expected_gpu_ids}
    if len(drivers) > 1:
        raise RuntimeError(f"host {request.host_id!r} has multiple NVIDIA drivers")
    hostname = socket.gethostname().strip()
    if not hostname:
        raise RuntimeError("host returned an empty hostname")
    return HostAdmissionReport(
        host_id=request.host_id,
        node_rank=request.node_rank,
        hostname=hostname,
        boot_id=_read_boot_id(),
        process_id=os.getpid(),
        assigned_gpus=assigned,
        nvidia_driver_version=next(iter(drivers), None),
        runtime=runtime,
    )


def validate_host_admission(
    hosts: Sequence[HostSpec],
    reports: Sequence[HostAdmissionReport],
    *,
    expected_runtime: RuntimeFingerprint,
) -> dict[str, HostAdmissionReport]:
    expected = {host.host_id: host for host in hosts}
    actual = {report.host_id: report for report in reports}
    if len(actual) != len(reports) or actual.keys() != expected.keys():
        raise RuntimeError(
            f"host-service membership mismatch: expected={sorted(expected)} "
            f"actual={sorted(actual)}"
        )
    controller_contract = expected_runtime.model_dump(exclude={"environment", "sha256"})
    for host_id, host in expected.items():
        report = actual[host_id]
        if report.node_rank != host.node_rank:
            raise RuntimeError(f"host {host_id!r} reported an unexpected node rank")
        if tuple(gpu.index for gpu in report.assigned_gpus) != host.gpu_ids:
            raise RuntimeError(f"host {host_id!r} reported unexpected GPU indices")
        host_contract = report.runtime.model_dump(exclude={"environment", "sha256"})
        if host_contract != controller_contract:
            fields = sorted(
                name
                for name, value in controller_contract.items()
                if host_contract[name] != value
            )
            raise RuntimeError(
                f"host {host_id!r} runtime contract differs from controller: {fields}"
            )
    runtime_digests = {report.runtime.sha256 for report in actual.values()}
    if len(runtime_digests) > 1:
        detail = " ".join(
            f"{host_id}={report.runtime.sha256}" for host_id, report in actual.items()
        )
        raise RuntimeError(f"runtime fingerprints differ across hosts: {detail}")
    drivers = {
        report.nvidia_driver_version
        for report in actual.values()
        if report.nvidia_driver_version is not None
    }
    if len(drivers) > 1:
        raise RuntimeError(f"NVIDIA driver versions differ across hosts: {drivers}")
    _require_unique(
        "physical host boot IDs",
        [(report.boot_id, host_id) for host_id, report in actual.items()],
    )
    _require_unique(
        "GPU UUIDs",
        [
            (gpu.uuid.casefold(), f"{host_id}:{gpu.index}")
            for host_id, report in actual.items()
            for gpu in report.assigned_gpus
        ],
    )
    _require_unique(
        "physical GPU PCI identities",
        [
            (f"{report.boot_id}/{gpu.pci_bus_id}", f"{host_id}:{gpu.index}")
            for host_id, report in actual.items()
            for gpu in report.assigned_gpus
        ],
    )
    return actual


def _query_gpu_inventory() -> tuple[tuple[GpuIdentity, str], ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for GPU host admission")
    try:
        result = subprocess.run(
            (
                executable,
                "--query-gpu=index,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"nvidia-smi GPU identity query failed: {error}") from None
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"nvidia-smi exited {result.returncode}: {detail}")
    rows: list[tuple[GpuIdentity, str]] = []
    for line_number, row in enumerate(csv.reader(result.stdout.splitlines()), start=1):
        values = tuple(value.strip() for value in row)
        try:
            if len(values) != 4:
                raise ValueError(f"expected 4 fields, received {len(values)}")
            gpu = GpuIdentity(
                index=int(values[0]), uuid=values[1], pci_bus_id=values[2].upper()
            )
        except ValueError as error:
            raise RuntimeError(
                f"invalid nvidia-smi row {line_number}: {error}"
            ) from None
        rows.append((gpu, values[3]))
    _require_unique(
        "nvidia-smi GPU indices", [(gpu.index, gpu.uuid) for gpu, _ in rows]
    )
    _require_unique(
        "nvidia-smi GPU UUIDs", [(gpu.uuid.casefold(), gpu.index) for gpu, _ in rows]
    )
    _require_unique(
        "nvidia-smi PCI identities", [(gpu.pci_bus_id, gpu.index) for gpu, _ in rows]
    )
    return tuple(rows)


def _art_build_sha256(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[1]
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise RuntimeError(f"ART package root {root} contains no build files")
    digest = hashlib.sha256()
    for path in files:
        _update_digest(digest, path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            _update_digest(digest, handle.read())
    return digest.hexdigest()


def _runtime_environment(
    environment: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, environment[name])
            for name in _RUNTIME_ENV & environment.keys()
            if environment[name]
        )
    )


def _read_boot_id() -> str:
    try:
        return str(UUID(_BOOT_ID_PATH.read_text(encoding="ascii").strip()))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot read Linux physical host boot ID: {error}"
        ) from None


def _require_unique(name: str, values: Sequence[tuple[object, object]]) -> None:
    owners: dict[object, object] = {}
    for value, owner in values:
        if value in owners:
            raise RuntimeError(
                f"duplicate {name}: {value!r} belongs to {owners[value]!r} and {owner!r}"
            )
        owners[value] = owner


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _update_digest(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
