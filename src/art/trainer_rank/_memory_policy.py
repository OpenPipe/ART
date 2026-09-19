"""Host memory limits and bounded graph/output placement admission.

The calibrated GPU estimator remains authoritative for a physical forward. These
policies only change how much state coexists between physical forwards; they do
not assume offload or replay reduces the workspace needed to execute one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

Retention = Literal["gpu", "cpu", "replay"]
OutputDevice = Literal["model", "cpu"]
_HOST_RESERVE_FRACTION = 0.1
_HOST_RESERVE_MAX_BYTES = 4 * 1024**3


def choose_output_placements(
    outputs: Sequence[tuple[int, Literal["auto", "model", "cpu"]]],
    *,
    gpu_available_bytes: int,
) -> tuple[OutputDevice, ...]:
    """Place already gathered CPU outputs before making any model-device copy.

    The caller admits the CPU receive/serialization buffers before gathering.
    Fresh GPU headroom excludes live storage and pending restore/staging reserves.
    Explicit model placement is reserved before optional model placement.
    """
    if any(
        size < 0 or device not in ("auto", "model", "cpu") for size, device in outputs
    ):
        raise ValueError("Expected nonnegative output bytes and auto/model/cpu policy")
    required = sum(size for size, device in outputs if device == "model")
    if required > max(0, gpu_available_bytes):
        raise MemoryError(
            f"Gathered model-device outputs require {required} GPU bytes; "
            f"only {max(0, gpu_available_bytes)} bytes are available"
        )
    remaining = max(0, gpu_available_bytes) - required
    placements: list[OutputDevice] = []
    for size, device in outputs:
        if device == "auto":
            device = "model" if size <= remaining else "cpu"
            if device == "model":
                remaining -= size
        placements.append(device)
    return tuple(placements)


@dataclass(frozen=True)
class MemoryScope:
    """A shared capacity limit, with the ranks whose allocations it counts."""

    name: str
    limit_bytes: int
    available_bytes: int
    rank_count: int

    @property
    def per_rank_available_bytes(self) -> int:
        if self.rank_count < 1:
            raise ValueError("Memory scope rank_count must be positive")
        reserve = min(
            _HOST_RESERVE_MAX_BYTES,
            int(max(0, self.limit_bytes) * _HOST_RESERVE_FRACTION),
        )
        return max(0, min(self.limit_bytes, self.available_bytes) - reserve) // (
            self.rank_count
        )


@dataclass(frozen=True)
class HostMemoryBudget:
    scopes: tuple[MemoryScope, ...]

    @property
    def available_bytes(self) -> int:
        """Additional CPU allocation allowed per rank, excluding existing use."""
        return min((scope.per_rank_available_bytes for scope in self.scopes), default=0)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _mount_path(value: str) -> str:
    for code, char in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(code, char)
    return value


def _cgroup_paths(proc_root: Path) -> tuple[tuple[Path, Path, bool], ...]:
    """Resolve membership through mount roots, including cgroup namespaces."""
    try:
        return _resolve_cgroup_paths(
            (proc_root / "self/cgroup").read_text(),
            (proc_root / "self/mountinfo").read_text(),
        )
    except OSError:
        return ()


@lru_cache(maxsize=8)
def _resolve_cgroup_paths(
    membership_text: str, mounts: str
) -> tuple[tuple[Path, Path, bool], ...]:
    # Cache only pure discovery. Membership/mount changes invalidate immediately;
    # every available-memory, limit and usage counter is read on every query.
    memberships = [line.split(":", 2) for line in membership_text.splitlines()]
    result = []
    for line in mounts.splitlines():
        if " - cgroup" not in line:
            continue
        before, separator, after = line.partition(" - ")
        fields, fs = before.split(), after.split()
        if not separator or len(fields) < 5 or len(fs) < 3:
            continue
        v2 = fs[0] == "cgroup2"
        if not v2 and not (fs[0] == "cgroup" and "memory" in fs[2].split(",")):
            continue
        mount_root, mount = Path(_mount_path(fields[3])), Path(_mount_path(fields[4]))
        for membership in memberships:
            if len(membership) != 3:
                continue
            _, controllers, name = membership
            if not (controllers == "" if v2 else "memory" in controllers.split(",")):
                continue
            try:
                relative = Path(name).relative_to(mount_root)
            except ValueError:
                # A namespaced membership can already be relative to its mount.
                if name != "/":
                    continue
                relative = Path()
            current = mount / relative
            if ".." not in current.parts:
                result.append((current, mount, v2))
    return tuple(result)


def host_memory_budget(
    *, local_world_size: int, proc_root: Path = Path("/proc")
) -> HostMemoryBudget:
    """Read fresh host/cgroup headroom, conservatively shared by local ranks.

    Each scope subtracts existing allocations before division. A per-process
    cgroup is also divided by the host rank count: conservative when ranks have
    separate limits, safe when a pod or ancestor limit covers all local ranks.
    Missing required counters grant no memory credit.
    """
    if local_world_size < 1:
        raise ValueError("local_world_size must be positive")
    try:
        memory = {
            key: int(value.split()[0]) * 1024
            for key, value in (
                line.split(":", 1)
                for line in (proc_root / "meminfo").read_text().splitlines()
                if ":" in line
            )
        }
        total, available = memory["MemTotal"], memory["MemAvailable"]
    except (OSError, ValueError, KeyError):
        total, available = 0, 0
    scopes = [MemoryScope("host", total, available, local_world_size)]
    visited = set()
    for current, mount, v2 in _cgroup_paths(proc_root):
        while True:
            if current not in visited:
                visited.add(current)
                limit = _read_int(
                    current / ("memory.max" if v2 else "memory.limit_in_bytes")
                )
                if limit is not None and limit < (1 << 60):
                    used = _read_int(
                        current / ("memory.current" if v2 else "memory.usage_in_bytes")
                    )
                    scopes.append(
                        MemoryScope(
                            str(current),
                            limit,
                            0 if used is None else max(0, limit - used),
                            local_world_size,
                        )
                    )
            if current == mount:
                break
            current = current.parent
    return HostMemoryBudget(tuple(scopes))


def local_rank_count(*, world_size: int = 1) -> int:
    """Launcher rank count; callers with explicit topology should pass it instead."""
    for name in ("LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"):
        value = os.environ.get(name)
        if value is not None:
            count = int(value)
            if count < 1:
                raise ValueError(f"{name} must be positive")
            return count
    # WORLD_SIZE is conservative across hosts and safe without local topology.
    return max(1, world_size, int(os.environ.get("WORLD_SIZE", "1")))


@dataclass(frozen=True)
class ForwardMemoryCost:
    peak_bytes: int
    retained_bytes: int
    output_bytes: int
    replay_bytes: int = 0
    backward_required: bool = True
    persistent_bytes: int = 0
    correction_workspace_bytes: int = 0
    gradient_staging_bytes: int = 0
    replay_seconds: float | None = None
    cpu_resident_bytes: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.output_bytes <= self.retained_bytes <= self.peak_bytes:
            raise ValueError("Expected 0 <= output <= retained <= peak bytes")
        if (
            min(
                self.replay_bytes,
                self.persistent_bytes,
                self.correction_workspace_bytes,
                self.gradient_staging_bytes,
                self.cpu_resident_bytes,
            )
            < 0
        ):
            raise ValueError(
                "Replay, persistent, correction and staging bytes must be nonnegative"
            )
        if self.cpu_resident_bytes > self.retained_bytes:
            raise ValueError("CPU-mode GPU residency cannot exceed retained bytes")


@dataclass(frozen=True)
class MemoryPlacement:
    backward_state: Retention
    output_device: OutputDevice
    gpu_required_bytes: int
    cpu_required_bytes: int
    gpu_retained_bytes: int
    gpu_backward_bytes: int = 0
    execution_peak_bytes: int = 0


def placement_cost(
    costs: Sequence[ForwardMemoryCost],
    *,
    backward_state: Retention,
    output_device: OutputDevice,
) -> MemoryPlacement:
    """Bound one complete root's live state plus one child's restore workspace."""
    gpu_retained, cpu_retained, workspace, staging = 0, 0, 0, 0
    for cost in costs:
        output_cpu = output_device == "cpu"
        if not cost.backward_required:
            gpu = 0 if output_cpu else cost.output_bytes
            cpu = cost.output_bytes if output_cpu else 0
            transient = cost.peak_bytes - gpu
        else:
            # Graph outputs keep their original CUDA storage while a graph lives,
            # even if caller-facing copies are on CPU. The detached caller copy
            # has distinct storage to isolate caller mutation and oversized views.
            physical_gpu = (
                cost.retained_bytes
                if backward_state == "gpu"
                else max(cost.output_bytes, cost.cpu_resident_bytes)
                if backward_state == "cpu"
                else 0
            )
            gpu = physical_gpu + (0 if output_cpu else cost.output_bytes)
            cpu = cost.replay_bytes + (cost.output_bytes if output_cpu else 0)
            if backward_state == "cpu":
                cpu += cost.retained_bytes - cost.output_bytes
            transient = cost.peak_bytes - physical_gpu
        gpu_retained += gpu + cost.persistent_bytes
        cpu_retained += cpu
        workspace = max(workspace, transient, cost.correction_workspace_bytes)
        staging += cost.gradient_staging_bytes
    return MemoryPlacement(
        backward_state,
        output_device,
        gpu_retained + workspace + staging,
        cpu_retained,
        gpu_retained,
        staging,
        max((cost.peak_bytes for cost in costs), default=0),
    )


def choose_memory_placement(
    costs: Sequence[ForwardMemoryCost],
    *,
    gpu_available_bytes: int,
    cpu_available_bytes: int,
    backward_state: Literal["auto", "gpu", "cpu", "replay"] = "auto",
    output_device: Literal["auto", "model", "cpu"] = "model",
    allow_cpu_offload: bool = True,
    allow_replay: bool = True,
) -> MemoryPlacement | None:
    """Prefer retained GPU, then CPU saved state, then replay; O(children)."""
    if backward_state not in ("auto", "gpu", "cpu", "replay"):
        raise ValueError(f"Unknown backward_state: {backward_state}")
    if output_device not in ("auto", "model", "cpu"):
        raise ValueError(f"Unknown output_device: {output_device}")
    if backward_state == "cpu" and not allow_cpu_offload:
        raise ValueError("CPU backward state conflicts with allow_cpu_offload=False")
    if backward_state == "replay" and not allow_replay:
        raise ValueError("Replay backward state conflicts with allow_replay=False")
    states: tuple[Retention, ...] = (
        tuple(
            state
            for state, enabled in (
                ("gpu", True),
                ("cpu", allow_cpu_offload),
                ("replay", allow_replay),
            )
            if enabled
        )
        if backward_state == "auto"
        else (backward_state,)
    )
    devices: tuple[OutputDevice, ...] = (
        ("model", "cpu") if output_device == "auto" else (output_device,)
    )
    for state in states:
        for device in devices:
            placement = placement_cost(
                costs, backward_state=state, output_device=device
            )
            if (
                placement.gpu_required_bytes <= gpu_available_bytes
                and placement.cpu_required_bytes <= cpu_available_bytes
            ):
                return placement
    return None
