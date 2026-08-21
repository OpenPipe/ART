from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import Condition, Lock, RLock
import time
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from ..tensor_snapshot import SnapshotReadBarrier
from .nvme_residency import (
    NvmeResidencyManifest,
    NvmeResidencyStore,
    NvmeResidencyStoreConfig,
)
from .residency import (
    ResidencyCapacityUnavailable,
    ResidencyDemand,
    ResidencyKey,
    ResidencyLedger,
    ResidencyLimits,
    ResidencyReservation,
    ResidencyTier,
)
from .tensor_residency import (
    HostTensorImage,
    TensorResidencyMover,
    TensorResidencySnapshot,
    TensorResidencyTransition,
)


class RunResidencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limits: ResidencyLimits
    nvme: NvmeResidencyStoreConfig
    device: str = "cuda"
    shutdown_timeout_s: float = Field(default=20.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_lossless_capacity(self) -> "RunResidencyConfig":
        if self.limits.l2_cpu.max_bytes < self.limits.l1_gpu.max_bytes:
            raise ValueError("L2 must hold at least the complete L1 residency budget")
        if self.limits.l3_nvme.max_bytes < self.limits.l1_gpu.max_bytes:
            raise ValueError("L3 must hold at least the complete L1 residency budget")
        return self


class _ManagedState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    key: ResidencyKey
    tensors: tuple[torch.Tensor, ...]
    l2: TensorResidencySnapshot | HostTensorImage | None = None
    l3: HostTensorImage | None = None
    l3_manifest: NvmeResidencyManifest | None = None
    l2_future: Future[HostTensorImage] | None = None
    l3_future: Future[NvmeResidencyManifest] | None = None
    l2_demotion: Future[None] | None = None
    l1_transition: TensorResidencyTransition | None = None


class RunResidencyManager:
    """Bounded rank-local ownership of movable run tensors in L1-L3."""

    def __init__(
        self,
        config: RunResidencyConfig,
        *,
        snapshot_barrier: SnapshotReadBarrier,
    ) -> None:
        self.config = config
        self.ledger = ResidencyLedger(config.limits)
        self._store = NvmeResidencyStore(config.nvme)
        self._mover = TensorResidencyMover(
            staging_capacity=config.limits.max_concurrent_transitions
        )
        self._snapshot_barrier = snapshot_barrier
        self._mutation_barriers: dict[ResidencyKey, SnapshotReadBarrier] = {}
        self._states: dict[ResidencyKey, _ManagedState] = {}
        self._installed_components: dict[tuple[str, str], ResidencyKey] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=config.limits.max_concurrent_transitions,
            thread_name_prefix="art-residency",
        )
        self._futures: set[Future[Any]] = set()
        self._retirements: dict[ResidencyKey, Future[None]] = {}
        self._l2_retains: dict[ResidencyKey, int] = {}
        self._l1_demands: set[ResidencyKey] = set()
        self._failures: list[BaseException] = []
        self._lock = RLock()
        self._admission_locks = {
            tier: Lock() for tier in ("l1_gpu", "l2_cpu", "l3_nvme")
        }
        self._transition_slots = Condition(self._lock)
        self._active_transitions = 0
        self._closing = False
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def register_l1(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> Future[HostTensorImage]:
        self._register_l1(key, tensors)
        return self.ensure_l2(key)

    def register_mutable_l1(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        """Admit mutable L1 state without eagerly copying it to a lower tier."""
        self._register_l1(key, tensors)

    def _register_l1(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        self._require_open()
        if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("new L1 residency must be a non-empty CUDA tensor tuple")
        component = self._exclusive_component(key)
        if component is not None and component in self._installed_components:
            raise RuntimeError(
                f"a run may have only one installed {key.representation} generation"
            )
        byte_count = self._mover.byte_count(tensors, "cuda")
        managed = _ManagedState(key=key, tensors=tensors)
        with self._admission_locks["l1_gpu"]:
            self._reclaim_l1(byte_count, demanded_bytes=byte_count, protected={key})
            reservation = self._reserve(
                key, source=None, target="l1_gpu", byte_count=byte_count
            )
            with self._lock:
                self._commit(reservation)
                self._states[key] = managed
                self._mutation_barriers[key] = SnapshotReadBarrier()
                if component is not None:
                    self._installed_components[component] = key

    def begin_l1_mutation(self, key: ResidencyKey) -> None:
        """Invalidate exact lower-tier images before mutating an L1 component."""
        self._require_open()
        with self._lock:
            state = self._state(key)
            if not self.ledger.has_copy(key, "l1_gpu"):
                raise RuntimeError("mutable residency is not L1 resident")
            if state.l1_transition is not None or self.ledger.has_reservation(key):
                raise RuntimeError("cannot mutate residency during a transition")
            if any(
                future is not None and not future.done()
                for future in (state.l2_future, state.l3_future, state.l2_demotion)
            ):
                raise RuntimeError("cannot mutate residency during a lower-tier copy")
            if self.ledger.has_copy(key, "l3_nvme"):
                self._store.delete(key)
                self.ledger.drop(key, "l3_nvme")
                state.l3 = None
                state.l3_manifest = None
                state.l3_future = None
            if self.ledger.has_copy(key, "l2_cpu"):
                self.ledger.drop(key, "l2_cpu")
                state.l2 = None
                state.l2_future = None
            self.ledger.touch(key)

    def register_l2(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> HostTensorImage:
        """Admit an off-GPU generation without making it compute resident."""
        self._require_open()
        if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
            raise RuntimeError("new L2 residency must be a non-empty CPU tensor tuple")
        byte_count = self._mover.byte_count(tensors, "cpu")
        with self._admission_locks["l2_cpu"]:
            self._reclaim("l2_cpu", byte_count, protected={key})
            reservation = self._reserve(
                key, source=None, target="l2_cpu", byte_count=byte_count
            )
        try:
            image = self._mover.host_image(tensors)
            managed = _ManagedState(key=key, tensors=tensors, l2=image)
            with self._lock:
                if key in self._states:
                    raise RuntimeError("residency key is already registered")
                self._commit(reservation)
                self._states[key] = managed
                self._mutation_barriers[key] = SnapshotReadBarrier()
        except BaseException as error:
            self._abort(reservation, error)
            raise
        return image

    def register_l2_working_set(
        self,
        working_set: Iterable[tuple[ResidencyKey, tuple[torch.Tensor, ...]]],
    ) -> tuple[HostTensorImage, ...]:
        """Atomically adopt a complete immutable off-GPU working set.

        Success transfers exclusive tensor-storage ownership to this manager. Failure
        leaves every input tensor unchanged and owned by the caller.
        """
        self._require_open()
        working_set = tuple(working_set)
        if not working_set:
            raise ValueError("L2 working set must be non-empty")
        keys = tuple(key for key, _tensors in working_set)
        if len(set(keys)) != len(keys):
            raise ValueError("L2 working-set keys must be unique")
        if any(
            not tensors or any(tensor.device.type != "cpu" for tensor in tensors)
            for _key, tensors in working_set
        ):
            raise RuntimeError("new L2 residency must be a non-empty CPU tensor tuple")
        byte_counts = tuple(
            self._mover.byte_count(tensors, "cpu") for _key, tensors in working_set
        )
        demands = tuple(
            ResidencyDemand(
                key=key,
                source=None,
                target="l2_cpu",
                byte_count=byte_count,
            )
            for (key, _tensors), byte_count in zip(
                working_set, byte_counts, strict=True
            )
        )
        with self._admission_locks["l2_cpu"]:
            with self._lock:
                if any(key in self._states for key in keys):
                    raise RuntimeError("residency key is already registered")
            self._reclaim("l2_cpu", sum(byte_counts), protected=set(keys))
            reservations = self._reserve_many(demands)
        try:
            images = tuple(
                self._mover.adopt_host_image(tensors) for _key, tensors in working_set
            )
            if any(
                image.stats.byte_count != byte_count
                for image, byte_count in zip(images, byte_counts, strict=True)
            ):
                raise RuntimeError("L2 working-set image byte count changed")
            states = tuple(
                _ManagedState(key=key, tensors=tensors, l2=image)
                for (key, tensors), image in zip(working_set, images, strict=True)
            )
            barriers = tuple(SnapshotReadBarrier() for _state in states)
            with self._lock:
                if any(key in self._states for key in keys):
                    raise RuntimeError("residency key is already registered")
                try:
                    for state, barrier in zip(states, barriers, strict=True):
                        self._states[state.key] = state
                        self._mutation_barriers[state.key] = barrier
                    self._commit_many(reservations)
                except BaseException:
                    for key in keys:
                        self._states.pop(key, None)
                        self._mutation_barriers.pop(key, None)
                    raise
        except BaseException as error:
            self._abort_many(reservations, error)
            raise
        return images

    def advance_l1(
        self,
        source: ResidencyKey,
        target: ResidencyKey,
        tensors: tuple[torch.Tensor, ...],
        *,
        retire_source: bool = False,
    ) -> Future[HostTensorImage]:
        self._require_open()
        if (
            source.representation not in ("weights", "optimizer", "accumulator")
            or target.representation != source.representation
        ):
            raise ValueError(
                "only committed trainer component generations may advance in place"
            )
        if retire_source:
            with self._lock:
                source_state = self._state(source)
                lower_pending = source_state.l2_future
                has_lower = self.ledger.has_copy(
                    source, "l2_cpu"
                ) or self.ledger.has_copy(source, "l3_nvme")
            if lower_pending is not None and lower_pending.done():
                lower_pending.result()
            if not has_lower and lower_pending is None:
                raise RuntimeError(
                    "committed L1 source has no immutable lower-tier image"
                )
        byte_count = self._mover.byte_count(tensors, "cuda")
        old_bytes = self.ledger.copy(source, "l1_gpu").byte_count
        with self._admission_locks["l1_gpu"]:
            self._reclaim_l1(
                max(byte_count - old_bytes, 0),
                demanded_bytes=byte_count,
                protected={source, target},
            )
            with self._lock:
                component = self._exclusive_component(source)
                if (
                    component is not None
                    and self._installed_components.get(component) != source
                ):
                    raise RuntimeError(
                        f"residency source is not the installed {source.representation}"
                    )
                if source.run_id != target.run_id:
                    raise ValueError("residency advance cannot change runs")
                self.ledger.advance(source, target, "l1_gpu", byte_count=byte_count)
                self._states[target] = _ManagedState(key=target, tensors=tensors)
                self._mutation_barriers[target] = SnapshotReadBarrier()
                if component is not None:
                    self._installed_components[component] = target
        if retire_source:
            self.retire_async(source)
        return self.ensure_l2(target)

    def ensure_l2(self, key: ResidencyKey) -> Future[HostTensorImage]:
        return self._ensure_l2(key, protected={key})

    def _ensure_l2(
        self, key: ResidencyKey, *, protected: set[ResidencyKey]
    ) -> Future[HostTensorImage]:
        self._require_open()
        with self._lock:
            self._require_not_retiring(key)
            state = self._state(key)
            if state.l2 is not None:
                return _resolved_future(state.l2)
            if state.l2_future is not None:
                return state.l2_future
        with self._admission_locks["l2_cpu"]:
            with self._lock:
                state = self._state(key)
                if state.l2 is not None:
                    return _resolved_future(state.l2)
                if state.l2_future is not None:
                    return state.l2_future
                if self.ledger.has_copy(key, "l1_gpu"):
                    byte_count = self._mover.byte_count(state.tensors, "cuda")
                    source: ResidencyTier = "l1_gpu"
                elif self.ledger.has_copy(key, "l3_nvme"):
                    if state.l3_manifest is None:
                        raise RuntimeError("L3 residency has no committed manifest")
                    byte_count = state.l3_manifest.payload_bytes
                    source = "l3_nvme"
                else:
                    raise RuntimeError(
                        "state has no local source for L2 materialization"
                    )
            self._reclaim("l2_cpu", byte_count, protected=protected)
            with self._lock:
                state = self._state(key)
                if state.l2 is not None:
                    return _resolved_future(state.l2)
                if state.l2_future is not None:
                    return state.l2_future
                if not self.ledger.has_copy(key, source):
                    raise RuntimeError("L2 residency source changed during reclamation")
                if source == "l1_gpu":
                    reservation = self._reserve(
                        key,
                        source="l1_gpu",
                        target="l2_cpu",
                        byte_count=byte_count,
                    )
                    try:
                        snapshot = self._mover.snapshot(state.tensors)
                        self._snapshot_barrier.register(snapshot.pending)
                        try:
                            barrier = self._mutation_barriers[key]
                        except KeyError as error:
                            raise RuntimeError(
                                "residency snapshot has no mutation fence"
                            ) from error
                        barrier.register(snapshot.pending)
                    except BaseException:
                        self._cancel(reservation)
                        raise
                    future = self._submit(
                        self._finish_l2_snapshot, state, reservation, snapshot
                    )
                else:
                    reservation = self._reserve(
                        key,
                        source="l3_nvme",
                        target="l2_cpu",
                        byte_count=byte_count,
                    )
                    future = self._submit(self._restore_l2, state, reservation)
                state.l2_future = future
                return future

    def ensure_l3(self, key: ResidencyKey) -> Future[NvmeResidencyManifest]:
        self._require_open()
        with self._lock:
            self._require_not_retiring(key)
            state = self._state(key)
            if state.l3_manifest is not None:
                return _resolved_future(state.l3_manifest)
            if state.l3_future is not None:
                return state.l3_future
            l2 = self.ensure_l2(key)
            future = self._submit(self._write_l3, state, l2)
            state.l3_future = future
            return future

    def prepare_l1(self, key: ResidencyKey) -> None:
        self.prepare_l1_working_set((key,))

    def prepare_l1_working_set(self, keys: Iterable[ResidencyKey]) -> None:
        """Atomically admit and launch one demanded trainer working set."""
        self._require_open()
        keys = self._working_set_keys(keys)
        if not keys:
            return
        protected = set(keys)
        with self._admission_locks["l1_gpu"], self._protect_l1_demand(keys):
            with self._lock:
                states = tuple(self._state(key) for key in keys)
                for key in keys:
                    self._require_not_retiring(key)
                demanded_bytes = sum(self._l1_demand_bytes(state) for state in states)
                missing = tuple(
                    state
                    for state in states
                    if not self.ledger.has_copy(state.key, "l1_gpu")
                    and state.l1_transition is None
                )
                conflicts = {
                    installed
                    for state in missing
                    if (component := self._exclusive_component(state.key)) is not None
                    and (installed := self._installed_components.get(component))
                    is not None
                    and installed not in protected
                }
            capacity = self.config.limits.l1_gpu.max_bytes
            if demanded_bytes > capacity:
                raise ResidencyCapacityUnavailable(
                    "l1_gpu demanded working set exceeds capacity: "
                    f"demanded={demanded_bytes}, max={capacity}"
                )
            for conflict in conflicts:
                try:
                    self.evict_l1(conflict)
                except RuntimeError as error:
                    if "pinned" not in str(error):
                        raise
                    raise ResidencyCapacityUnavailable(
                        "a pinned L1 component prevents demanded working-set admission"
                    ) from error
            incoming_bytes = sum(self._l1_demand_bytes(state) for state in missing)
            self._reclaim_l1(
                incoming_bytes,
                demanded_bytes=demanded_bytes,
                protected=protected,
            )
            with self._lock:
                missing = tuple(
                    self._state(key)
                    for key in keys
                    if not self.ledger.has_copy(key, "l1_gpu")
                    and self._state(key).l1_transition is None
                )
                if not missing:
                    return
                sources = tuple(self._l1_source(state) for state in missing)
                demands = tuple(
                    ResidencyDemand(
                        key=state.key,
                        source=source,
                        target="l1_gpu",
                        byte_count=self._l1_demand_bytes(state),
                    )
                    for state, source in zip(missing, sources, strict=True)
                )
                reservations = self._reserve_many(demands)
            authoritative: list[HostTensorImage | TensorResidencySnapshot] = []
            transition: TensorResidencyTransition | None = None
            staging = None
            try:
                if "l3_nvme" in sources:
                    staging = self._mover.staging()
                for state, source in zip(missing, sources, strict=True):
                    image = state.l2 if source == "l2_cpu" else state.l3
                    if image is None:
                        raise RuntimeError(
                            f"L1 admission lost its committed {source} source"
                        )
                    authoritative.append(image)
                    if source == "l3_nvme":
                        manifest = state.l3_manifest
                        if manifest is None or staging is None:
                            raise RuntimeError(
                                "L3 residency lost its committed transfer metadata"
                            )
                        targets = tuple(
                            staging.stager.target_bytes(record.byte_count)
                            for record in manifest.storages
                        )
                        image = self._store.read_committed(
                            state.key, manifest, state.tensors, targets
                        )
                    elif staging is not None:
                        image = self._mover.stage_host_image(image, staging)
                    image.activate()
                transfer_staging = staging
                staging = None
                transition = self._mover.move(
                    tuple(tensor for state in missing for tensor in state.tensors),
                    self.config.device,
                    staging=transfer_staging,
                )
                if transition.stats.byte_count != incoming_bytes:
                    raise RuntimeError("L1 working-set transfer byte count changed")
                with self._lock:
                    # Storage is allocated; the retained event gates authoritative use.
                    self.ledger.commit_many(reservations)
                    for state in missing:
                        state.l1_transition = transition
            except BaseException:
                if transition is not None:
                    transition.wait_on_current_stream()
                elif staging is not None:
                    staging.release()
                with self._lock:
                    for image in authoritative:
                        image.activate()
                    self._cancel_many(reservations)
                raise

    def acquire_l1(self, key: ResidencyKey) -> None:
        self.acquire_l1_working_set((key,))

    def acquire_l1_working_set(self, keys: Iterable[ResidencyKey]) -> None:
        """Fence, commit, and pin a complete demanded trainer working set."""
        self._require_open()
        keys = self._working_set_keys(keys)
        if not keys:
            return
        with self._lock:
            needs_prepare = any(
                not self.ledger.has_copy(key, "l1_gpu")
                and self._state(key).l1_transition is None
                for key in keys
            )
        if needs_prepare:
            self.prepare_l1_working_set(keys)
        with self._lock:
            for key in keys:
                self._require_not_retiring(key)
            transitions = {
                id(transition): transition
                for key in keys
                if (transition := self._state(key).l1_transition) is not None
            }
            for transition in transitions.values():
                self._finish_l1_transfer_locked(transition)
            if any(not self.ledger.has_copy(key, "l1_gpu") for key in keys):
                raise RuntimeError("demanded working set did not become L1 resident")
            self.ledger.pin_many((key, "l1_gpu") for key in keys)

    def prefetch_l1(self, key: ResidencyKey) -> None:
        self.prefetch_l1_working_set((key,))

    def prefetch_l1_working_set(self, keys: Iterable[ResidencyKey]) -> None:
        """Launch a demanded working set without synchronizing the CPU."""
        self.prepare_l1_working_set(keys)

    def release_l1(self, key: ResidencyKey) -> None:
        self.release_l1_working_set((key,))

    def release_l1_working_set(self, keys: Iterable[ResidencyKey]) -> None:
        keys = self._working_set_keys(keys)
        with self._lock:
            self.ledger.unpin_many((key, "l1_gpu") for key in keys)
        for key in keys:
            self._try_retire(key)

    def wait_before_mutation_working_set(self, keys: Iterable[ResidencyKey]) -> None:
        """Order a trainer working set's next mutation after all D2H reads."""
        self._require_open()
        keys = self._working_set_keys(keys)
        with self._lock:
            for key in keys:
                self._state(key)
            missing = tuple(key for key in keys if key not in self._mutation_barriers)
            if missing:
                components = ", ".join(key.representation for key in missing)
                raise RuntimeError(
                    f"mutation working set is missing mutation fence: {components}"
                )
            barriers = tuple(self._mutation_barriers[key] for key in keys)
        for barrier in barriers:
            barrier.wait_before_mutation()

    def touch(self, key: ResidencyKey) -> None:
        self._require_open()
        self.ledger.touch(key)

    def evict_l1(self, key: ResidencyKey) -> None:
        if not (
            self.ledger.has_copy(key, "l2_cpu") or self.ledger.has_copy(key, "l3_nvme")
        ):
            self.ensure_l2(key).result()
        with self._lock:
            state = self._state(key)
            if state.l1_transition is not None:
                self._finish_l1_transfer_locked(state.l1_transition)
            self._evict_l1_locked(key)

    def evict_l2(self, key: ResidencyKey) -> None:
        self.ensure_l3(key).result()
        if not self._evict_l2(key) and self.ledger.has_copy(key, "l2_cpu"):
            raise RuntimeError("cannot evict pinned L2 residency")

    def l2_image(self, key: ResidencyKey) -> HostTensorImage:
        return self.ensure_l2(key).result()

    @contextmanager
    def borrow_l2(self, key: ResidencyKey) -> Iterator[HostTensorImage]:
        """Pin one host copy while a consumer reads its L2 image."""
        self._require_open()
        while True:
            image = self.ensure_l2(key).result()
            with self._lock:
                state = self._state(key)
                self._require_not_retiring(key)
                if state.l2 is not image or not self.ledger.has_copy(key, "l2_cpu"):
                    continue
                self.ledger.pin(key, "l2_cpu")
                break
        try:
            yield image
        finally:
            with self._lock:
                self.ledger.unpin(key, "l2_cpu")
            self._try_retire(key)

    def retain_l2(self, key: ResidencyKey) -> Future[HostTensorImage]:
        """Keep one exact L2 image available for a deferred consumer."""
        self._require_open()
        with self._lock:
            self._require_not_retiring(key)
            self._state(key)
            self._l2_retains[key] = self._l2_retains.get(key, 0) + 1
        try:
            return self.ensure_l2(key)
        except BaseException:
            self.release_l2(key)
            raise

    def release_l2(self, key: ResidencyKey) -> None:
        with self._lock:
            count = self._l2_retains.get(key, 0)
            if count < 1:
                raise RuntimeError("residency L2 retain is not active")
            if count == 1:
                self._l2_retains.pop(key)
            else:
                self._l2_retains[key] = count - 1
        self._try_retire(key)

    def retire(self, key: ResidencyKey) -> None:
        self._require_open()
        self._retire(key)

    def _retire(self, key: ResidencyKey) -> None:
        with self._lock:
            state = self._state(key)
            if state.l1_transition is not None:
                raise RuntimeError("cannot retire a state with an active L1 transfer")
            self._retire_now_locked(key)
            retirement = self._retirements.pop(key, None)
        if retirement is not None:
            retirement.set_result(None)

    def retire_async(self, key: ResidencyKey) -> Future[None]:
        self._require_open()
        with self._lock:
            if existing := self._retirements.get(key):
                return existing
            state = self._state(key)
            future: Future[None] = Future()
            self._retirements[key] = future
            dependencies = tuple(
                dependency
                for dependency in (
                    state.l2_future,
                    state.l3_future,
                    state.l2_demotion,
                )
                if dependency is not None and not dependency.done()
            )
            if state.l1_transition is not None:
                self._finish_l1_transfer_locked(state.l1_transition)
        for dependency in dependencies:
            dependency.add_done_callback(
                lambda _completed, retiring=key: self._try_retire(retiring)
            )
        self._try_retire(key)
        return future

    def demote_l2_async(self, key: ResidencyKey) -> Future[None]:
        self._require_open()
        with self._lock:
            state = self._state(key)
            if state.l2_demotion is not None:
                return state.l2_demotion
            l3 = self.ensure_l3(key)
            state.l2_demotion = self._submit(self._demote_after, key, l3)
            state.l2_demotion.add_done_callback(
                lambda completed: self._demotion_completed(key, completed)
            )
            return state.l2_demotion

    def keys(self, run_id: str) -> tuple[ResidencyKey, ...]:
        with self._lock:
            return tuple(key for key in self._states if key.run_id == run_id)

    def close(self, *, deadline: float | None = None) -> None:
        deadline = (
            time.monotonic() + self.config.shutdown_timeout_s
            if deadline is None
            else deadline
        )
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._transition_slots.notify_all()
            futures = tuple(self._futures)
        for future in futures:
            future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        _done, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        failures: list[BaseException] = []
        if not pending:
            with self._lock:
                transitions = {
                    id(transition): transition
                    for state in self._states.values()
                    if (transition := state.l1_transition) is not None
                }
                for transition in transitions.values():
                    if time.monotonic() >= deadline:
                        break
                    try:
                        self._finish_l1_transfer_locked(transition, synchronize=True)
                    except BaseException as error:
                        failures.append(error)
                states = tuple(self._states.values())
            for state in states:
                if time.monotonic() >= deadline:
                    break
                try:
                    self._retire(state.key)
                except BaseException as error:
                    failures.append(error)
        try:
            self._raise_failures()
        except BaseException as error:
            failures.append(error)
        with self._lock:
            live_futures = tuple(
                future for future in self._futures if not future.done()
            )
            live_transitions = self._active_transitions + sum(
                state.l1_transition is not None for state in self._states.values()
            )
            remaining_states = len(self._states)
        unsafe = bool(
            pending
            or live_futures
            or live_transitions
            or remaining_states
            or time.monotonic() > deadline
        )
        if unsafe:
            failures.append(
                TimeoutError(
                    "run residency shutdown deadline left "
                    f"{len(set(pending).union(live_futures))} worker futures and "
                    f"{live_transitions} active transitions and "
                    f"{remaining_states} resident states"
                )
            )
        else:
            self._pool.shutdown(wait=True, cancel_futures=True)
            with self._lock:
                self._closed = True
                self._closing = False
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("run residency shutdown failed", failures)

    def _finish_l2_snapshot(
        self,
        state: _ManagedState,
        reservation: ResidencyReservation,
        snapshot: TensorResidencySnapshot,
    ) -> HostTensorImage:
        try:
            image = snapshot.materialize_l2()
            with self._lock:
                self._commit(reservation)
                state.l2 = image
                return image
        except BaseException as error:
            self._abort(reservation, error)
            raise

    def _finish_l1_transfer_locked(
        self,
        transition: TensorResidencyTransition,
        *,
        synchronize: bool = False,
    ) -> None:
        states = tuple(
            state
            for state in self._states.values()
            if state.l1_transition is transition
        )
        if not states:
            return
        if synchronize:
            transition.synchronize()
        else:
            transition.wait_on_current_stream()
        with self._transition_slots:
            self._release_transition_slot()
        for state in states:
            state.l1_transition = None
            if component := self._exclusive_component(state.key):
                self._installed_components[component] = state.key

    def _try_retire(self, key: ResidencyKey) -> None:
        retirement: Future[None] | None = None
        error: BaseException | None = None
        with self._lock:
            retirement = self._retirements.get(key)
            state = self._states.get(key)
            if retirement is None or state is None:
                return
            if key in self._l1_demands:
                return
            if self._l2_retains.get(key, 0):
                return
            dependencies = (
                state.l2_future,
                state.l3_future,
                state.l2_demotion,
            )
            if any(
                dependency is not None and not dependency.done()
                for dependency in dependencies
            ):
                return
            if state.l1_transition is not None:
                self._finish_l1_transfer_locked(state.l1_transition)
            try:
                entry = self.ledger.entry(key)
            except KeyError:
                failed_dependency = next(
                    (
                        dependency
                        for dependency in dependencies
                        if dependency is not None
                        and dependency.done()
                        and not dependency.cancelled()
                        and dependency.exception() is not None
                    ),
                    None,
                )
                if failed_dependency is None:
                    raise
                error = failed_dependency.exception()
                assert error is not None
                self._states.pop(key)
                self._mutation_barriers.pop(key, None)
                self._retirements.pop(key, None)
            else:
                if any(entry.pin_counts.values()) or self.ledger.has_reservation(key):
                    return
                try:
                    self._retire_now_locked(key)
                except BaseException as caught:
                    error = caught
                    self._failures.append(caught)
                self._retirements.pop(key, None)
        if error is None:
            retirement.set_result(None)
        else:
            retirement.set_exception(error)

    def _retire_now_locked(self, key: ResidencyKey) -> None:
        state = self._state(key)
        if state.l1_transition is not None or any(
            future is not None and not future.done()
            for future in (state.l2_future, state.l3_future, state.l2_demotion)
        ):
            raise RuntimeError("cannot retire a state with an active transition")
        entry = self.ledger.entry(key)
        if any(entry.pin_counts.values()):
            raise RuntimeError("cannot retire a pinned residency state")
        if self._l2_retains.get(key, 0):
            raise RuntimeError("cannot retire a retained L2 residency state")
        if self.ledger.has_reservation(key):
            raise RuntimeError("cannot retire a state with an active transition")
        if self.ledger.has_copy(key, "l1_gpu"):
            host = state.l2 if state.l2 is not None else state.l3
            if host is not None:
                host.activate()
        if state.l3_manifest is not None:
            self._store.delete(key)
        self.ledger.drop_entry(key)
        self._states.pop(key)
        self._mutation_barriers.pop(key, None)
        component = self._exclusive_component(key)
        if component is not None and self._installed_components.get(component) == key:
            self._installed_components.pop(component)

    def _demote_after(
        self, key: ResidencyKey, dependency: Future[NvmeResidencyManifest]
    ) -> None:
        dependency.result()
        self._evict_l2(key)

    def _restore_l2(
        self, state: _ManagedState, reservation: ResidencyReservation
    ) -> HostTensorImage:
        try:
            with self._lock:
                manifest = state.l3_manifest
            if manifest is None:
                raise RuntimeError("L3 residency has no verified mapped image")
            targets = tuple(
                torch.empty(record.byte_count, dtype=torch.uint8)
                for record in manifest.storages
            )
            image = self._store.read_committed(
                state.key, manifest, state.tensors, targets
            )
            with self._lock:
                self._commit(
                    reservation,
                    immutable_ref=str(self._store.path(state.key)),
                )
                state.l3_manifest = manifest
                state.l2 = image
                return image
        except BaseException as error:
            self._abort(reservation, error)
            raise

    def _write_l3(
        self,
        state: _ManagedState,
        l2_future: Future[HostTensorImage],
    ) -> NvmeResidencyManifest:
        reservation: ResidencyReservation | None = None
        try:
            image = l2_future.result()
            plan = self._store.prepare_write(state.key, image)
            byte_count = plan.physical_bytes
            with self._admission_locks["l3_nvme"]:
                self._reclaim("l3_nvme", byte_count, protected={state.key})
                reservation = self._reserve(
                    state.key,
                    source="l2_cpu",
                    target="l3_nvme",
                    byte_count=byte_count,
                )
            manifest = self._store.write_prepared(plan, image)
            if manifest != plan.manifest:
                raise RuntimeError("L3 residency manifest changed during commit")
            mapped = self._store.map_newly_committed(plan, state.tensors)
            with self._lock:
                self._commit(
                    reservation,
                    immutable_ref=str(self._store.path(state.key)),
                )
                state.l3 = mapped
                state.l3_manifest = manifest
                return manifest
        except BaseException as error:
            if reservation is not None:
                self._abort(reservation, error)
            else:
                self._record_failure(error)
            raise

    def _evict_installed_component(self, component: tuple[str, str]) -> None:
        current = self._installed_components.get(component)
        if current is not None:
            self.evict_l1(current)

    def _reclaim_l1(
        self,
        incoming_bytes: int,
        *,
        demanded_bytes: int,
        protected: set[ResidencyKey],
    ) -> None:
        with self._lock:
            protected = (
                protected
                | self._retirements.keys()
                | self._l1_demands
                | {
                    state.key
                    for state in self._states.values()
                    if state.l1_transition is not None
                }
            )
        candidates = self.ledger.admission_evictions(
            "l1_gpu",
            incoming_bytes,
            demanded_bytes,
            protected=protected,
            require_other_copy=False,
        )
        for candidate in candidates:
            if not (
                self.ledger.has_copy(candidate, "l2_cpu")
                or self.ledger.has_copy(candidate, "l3_nvme")
            ):
                self._ensure_l2(candidate, protected=protected | {candidate}).result()
            with self._lock:
                self._evict_l1_locked(candidate)

    def _reclaim(
        self,
        tier: ResidencyTier,
        incoming_bytes: int,
        *,
        protected: set[ResidencyKey],
    ) -> None:
        with self._lock:
            protected = (
                protected
                | self._retirements.keys()
                | self._l2_retains.keys()
                | self._l1_demands
                | {
                    state.key
                    for state in self._states.values()
                    if state.l1_transition is not None
                }
            )
        demanded_bytes = incoming_bytes + self.ledger.accounted_bytes(tier, protected)
        while candidates := self.ledger.admission_evictions(
            tier,
            incoming_bytes,
            demanded_bytes,
            protected=protected,
            require_other_copy=tier != "l2_cpu",
        ):
            for candidate in candidates:
                if tier == "l2_cpu":
                    if not (
                        self.ledger.has_copy(candidate, "l1_gpu")
                        or self.ledger.has_copy(candidate, "l3_nvme")
                    ):
                        self.ensure_l3(candidate).result()
                    self._evict_l2(candidate)
                elif tier == "l3_nvme":
                    self._evict_l3(candidate)
                else:
                    raise ValueError(f"unsupported background residency tier: {tier}")

    def _evict_l2(self, key: ResidencyKey) -> bool:
        with self._lock:
            state = self._state(key)
            if not self.ledger.has_copy(key, "l2_cpu"):
                return False
            if state.l1_transition is not None or self.ledger.has_reservation(key):
                return False
            if self._l2_retains.get(key, 0):
                return False
            if self.ledger.entry(key).pin_counts["l2_cpu"]:
                return False
            if not self.ledger.has_copy(key, "l1_gpu"):
                if state.l3 is None:
                    raise RuntimeError("cannot evict L2 before L3 is ready")
                state.l3.activate()
            self.ledger.drop(key, "l2_cpu")
            state.l2 = None
            state.l2_future = None
            return True

    def _evict_l3(self, key: ResidencyKey) -> bool:
        with self._lock:
            state = self._state(key)
            if not self.ledger.has_copy(key, "l3_nvme"):
                return False
            if state.l1_transition is not None or self.ledger.has_reservation(key):
                return False
            if self.ledger.entry(key).pin_counts["l3_nvme"]:
                return False
            if not (
                self.ledger.has_copy(key, "l1_gpu")
                or self.ledger.has_copy(key, "l2_cpu")
            ):
                raise RuntimeError("cannot evict the only exact residency copy")
            # L4 is a service checkpoint bundle, not a rank-local tensor image.
            # It becomes executable only after service-level rematerialization.
            self._store.delete(key)
            self.ledger.drop(key, "l3_nvme")
            state.l3 = None
            state.l3_manifest = None
            state.l3_future = None
            return True

    def _evict_l1_locked(self, key: ResidencyKey) -> None:
        state = self._state(key)
        if state.l1_transition is not None:
            raise RuntimeError("cannot evict an active L1 transfer")
        host = state.l2 if state.l2 is not None else state.l3
        if host is None:
            raise RuntimeError("cannot evict L1 without a lossless host copy")
        if self.ledger.entry(key).pin_counts["l1_gpu"]:
            raise RuntimeError("cannot evict a pinned residency state")
        host.activate()
        self.ledger.drop(key, "l1_gpu")
        component = self._exclusive_component(key)
        if component is not None and self._installed_components.get(component) == key:
            self._installed_components.pop(component)

    @staticmethod
    def _exclusive_component(key: ResidencyKey) -> tuple[str, str] | None:
        if key.representation not in ("weights", "optimizer"):
            return None
        return key.run_id, key.representation

    @staticmethod
    def _working_set_keys(
        keys: Iterable[ResidencyKey],
    ) -> tuple[ResidencyKey, ...]:
        keys = tuple(keys)
        if len(set(keys)) != len(keys):
            raise ValueError("L1 working-set keys must be unique")
        if len({key.run_id for key in keys}) > 1:
            raise ValueError("one L1 working set cannot span runs")
        if any(
            key.representation not in ("weights", "optimizer", "accumulator")
            for key in keys
        ):
            raise ValueError("L1 working sets contain trainer components only")
        if len({key.representation for key in keys}) != len(keys):
            raise ValueError("an L1 working set may demand each component once")
        return keys

    def _l1_demand_bytes(self, state: _ManagedState) -> int:
        if self.ledger.has_copy(state.key, "l1_gpu"):
            return self.ledger.copy(state.key, "l1_gpu").byte_count
        if state.l2 is not None:
            return state.l2.stats.byte_count
        if state.l3_manifest is not None:
            return state.l3_manifest.payload_bytes
        byte_count = sum(
            self._mover.byte_count(state.tensors, device_type)
            for device_type in ("cpu", "cuda")
        )
        if byte_count < 1:
            raise RuntimeError("residency state has no materializable tensor storage")
        return byte_count

    def _l1_source(self, state: _ManagedState) -> ResidencyTier:
        if self.ledger.has_copy(state.key, "l2_cpu"):
            if state.l2 is None:
                raise RuntimeError("L2 ledger copy has no host image")
            return "l2_cpu"
        if self.ledger.has_copy(state.key, "l3_nvme"):
            if state.l3 is None or state.l3_manifest is None:
                raise RuntimeError("L3 ledger copy has no committed image")
            return "l3_nvme"
        raise RuntimeError("L1 admission lost every lower-tier source")

    def _require_not_retiring(self, key: ResidencyKey) -> None:
        if key in self._retirements:
            raise RuntimeError("residency state is retiring")

    @contextmanager
    def _protect_l1_demand(self, keys: tuple[ResidencyKey, ...]) -> Iterator[None]:
        with self._lock:
            self._l1_demands.update(keys)
        try:
            yield
        finally:
            with self._lock:
                self._l1_demands.difference_update(keys)
            for key in keys:
                self._try_retire(key)

    def _submit(self, fn: Any, *args: Any) -> Future[Any]:
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("run residency manager is closed")
            future = self._pool.submit(fn, *args)
            self._futures.add(future)
        future.add_done_callback(self._completed)
        return future

    def _completed(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def _demotion_completed(self, key: ResidencyKey, future: Future[None]) -> None:
        with self._lock:
            state = self._states.get(key)
            if state is not None and state.l2_demotion is future:
                state.l2_demotion = None

    def _reserve(
        self,
        key: ResidencyKey,
        *,
        source: ResidencyTier | None,
        target: ResidencyTier,
        byte_count: int,
    ) -> ResidencyReservation:
        return self._reserve_many(
            (
                ResidencyDemand(
                    key=key,
                    source=source,
                    target=target,
                    byte_count=byte_count,
                ),
            )
        )[0]

    def _reserve_many(
        self, demands: tuple[ResidencyDemand, ...]
    ) -> tuple[ResidencyReservation, ...]:
        with self._transition_slots:
            limit = self.config.limits.max_concurrent_transitions
            while self._active_transitions >= limit:
                if self._closing or self._closed:
                    raise RuntimeError("run residency manager is closed")
                self._transition_slots.wait()
                self._raise_failures()
            reservations = self.ledger.reserve_many(demands)
            self._active_transitions += 1
            return reservations

    def _commit(
        self,
        reservation: ResidencyReservation,
        *,
        immutable_ref: str | None = None,
        digest: str | None = None,
    ) -> None:
        with self._transition_slots:
            self.ledger.commit(reservation, immutable_ref=immutable_ref, digest=digest)
            self._release_transition_slot()

    def _commit_many(self, reservations: tuple[ResidencyReservation, ...]) -> None:
        with self._transition_slots:
            self.ledger.commit_many(reservations)
            self._release_transition_slot()

    def _cancel(self, reservation: ResidencyReservation) -> None:
        with self._transition_slots:
            self.ledger.abort(reservation)
            self._release_transition_slot()

    def _cancel_many(self, reservations: tuple[ResidencyReservation, ...]) -> None:
        with self._transition_slots:
            self.ledger.abort_many(reservations)
            self._release_transition_slot()

    def _release_transition_slot(self) -> None:
        if self._active_transitions < 1:
            raise RuntimeError("residency transition accounting underflow")
        self._active_transitions -= 1
        self._transition_slots.notify()

    def _abort(self, reservation: ResidencyReservation, error: BaseException) -> None:
        with self._lock:
            try:
                self._cancel(reservation)
            finally:
                self._failures.append(error)

    def _abort_many(
        self,
        reservations: tuple[ResidencyReservation, ...],
        error: BaseException,
    ) -> None:
        with self._lock:
            try:
                self._cancel_many(reservations)
            finally:
                self._failures.append(error)

    def _record_failure(self, error: BaseException) -> None:
        with self._lock:
            self._failures.append(error)

    def _state(self, key: ResidencyKey) -> _ManagedState:
        try:
            return self._states[key]
        except KeyError as exc:
            raise KeyError(f"unknown residency state: {key}") from exc

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("run residency manager is closed")
        self._raise_failures()

    def _raise_failures(self) -> None:
        with self._lock:
            failures = tuple(self._failures)
            self._failures.clear()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("run residency transitions failed", failures)


def _resolved_future(value: Any) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(value)
    return future
