from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import Condition, Lock, RLock
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
    l1_reservation: ResidencyReservation | None = None
    l1_transition: TensorResidencyTransition | None = None
    evict_l2_after_l1: bool = False


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

    def register_l1(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> Future[HostTensorImage]:
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
                if component is not None:
                    self._installed_components[component] = key
        return self.ensure_l2(key)

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
        except BaseException as error:
            self._abort(reservation, error)
            raise
        return image

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
            if not (
                self.ledger.has_copy(source, "l2_cpu")
                or self.ledger.has_copy(source, "l3_nvme")
            ):
                self.ensure_l2(source).result()
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
                        self._mutation_barriers.setdefault(
                            key, SnapshotReadBarrier()
                        ).register(snapshot.pending)
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
                transient_l2 = {
                    state.key
                    for state in missing
                    if not self.ledger.has_copy(state.key, "l2_cpu")
                    and self.ledger.has_copy(state.key, "l3_nvme")
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
            for state in missing:
                if not self.ledger.has_copy(state.key, "l2_cpu"):
                    self._ensure_l2(state.key, protected=protected).result()
            with self._lock:
                missing = tuple(
                    self._state(key)
                    for key in keys
                    if not self.ledger.has_copy(key, "l1_gpu")
                    and self._state(key).l1_transition is None
                )
                if not missing:
                    return
            demands = tuple(
                ResidencyDemand(
                    key=state.key,
                    source="l2_cpu",
                    target="l1_gpu",
                    byte_count=self._l1_demand_bytes(state),
                )
                for state in missing
            )
            with self._lock:
                reservations = self._reserve_many(demands)
                images: list[HostTensorImage | TensorResidencySnapshot] = []
                transition: TensorResidencyTransition | None = None
                try:
                    for state in missing:
                        image = self._state(state.key).l2
                        if image is None:
                            raise RuntimeError(
                                "L1 admission lost its committed L2 source"
                            )
                        image.activate()
                        images.append(image)
                    transition = self._mover.move(
                        tuple(tensor for state in missing for tensor in state.tensors),
                        self.config.device,
                    )
                    if transition.stats.byte_count != incoming_bytes:
                        raise RuntimeError("L1 working-set transfer byte count changed")
                    # Storage is allocated; the retained event gates authoritative use.
                    self.ledger.commit_many(reservations)
                except BaseException:
                    if transition is not None:
                        transition.wait_on_current_stream()
                    for image in images:
                        image.activate()
                    self._cancel_many(reservations)
                    raise
                for state in missing:
                    state.l1_reservation = None
                    state.l1_transition = transition
                    state.evict_l2_after_l1 = state.key in transient_l2

    def acquire_l1(self, key: ResidencyKey) -> None:
        self.acquire_l1_working_set((key,))

    def acquire_l1_working_set(self, keys: Iterable[ResidencyKey]) -> None:
        """Fence, commit, and pin a complete demanded trainer working set."""
        keys = self._working_set_keys(keys)
        if not keys:
            return
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

    def wait_before_mutation(self, key: ResidencyKey) -> None:
        """Order this key's next CUDA mutation after its recovery snapshot."""
        self._require_open()
        with self._lock:
            self._state(key)
            barrier = self._mutation_barriers.pop(key, None)
        if barrier is not None:
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
                self._drop_transient_l2_locked(key)
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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("run residency manager close is already in progress")
            self._closing = True
            self._transition_slots.notify_all()
            futures = tuple(self._futures)
        _done, pending = wait(futures, timeout=self.config.shutdown_timeout_s)
        self._pool.shutdown(wait=False, cancel_futures=True)
        failures: list[BaseException] = []
        if not pending:
            with self._lock:
                transitions = {
                    id(transition): transition
                    for state in self._states.values()
                    if (transition := state.l1_transition) is not None
                }
                for transition in transitions.values():
                    self._finish_l1_transfer_locked(transition, synchronize=True)
                states = tuple(self._states.values())
            for state in states:
                try:
                    self._retire(state.key)
                except BaseException as error:
                    failures.append(error)
        try:
            self._raise_failures()
        except BaseException as error:
            failures.append(error)
        with self._lock:
            self._closed = True
            self._closing = False
        if pending:
            failures.append(
                TimeoutError(
                    f"{len(pending)} residency transitions exceeded shutdown timeout"
                )
            )
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
        reservations = tuple(
            state.l1_reservation for state in states if state.l1_reservation is not None
        )
        if reservations and len(reservations) != len(states):
            raise RuntimeError("L1 transfer has an incomplete reservation set")
        if synchronize:
            transition.synchronize()
        else:
            transition.wait_on_current_stream()
        if reservations:
            self._commit_many(reservations)
        else:
            with self._transition_slots:
                self._release_transition_slot()
        for state in states:
            state.l1_transition = None
            state.l1_reservation = None
            if component := self._exclusive_component(state.key):
                self._installed_components[component] = state.key
        for state in states:
            self._drop_transient_l2_locked(state.key)

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
            entry = self.ledger.entry(key)
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

    def _drop_transient_l2_locked(self, key: ResidencyKey) -> None:
        state = self._states.get(key)
        if (
            state is not None
            and state.evict_l2_after_l1
            and state.l1_transition is None
            and self.ledger.has_copy(key, "l1_gpu")
            and self._evict_l2(key)
        ):
            state.evict_l2_after_l1 = False

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
                    digest=manifest.data_sha256,
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
            byte_count = self._store.physical_bytes(state.key, image)
            with self._admission_locks["l3_nvme"]:
                self._reclaim("l3_nvme", byte_count, protected={state.key})
                reservation = self._reserve(
                    state.key,
                    source="l2_cpu",
                    target="l3_nvme",
                    byte_count=byte_count,
                )
            manifest = self._store.write(state.key, image)
            if manifest.physical_bytes != byte_count:
                raise RuntimeError("L3 residency physical size changed during commit")
            mapped = self._store.map_committed(state.key, manifest, state.tensors)
            with self._lock:
                self._commit(
                    reservation,
                    immutable_ref=str(self._store.path(state.key)),
                    digest=manifest.data_sha256,
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
            if self.ledger.entry(key).pin_counts["l2_cpu"]:
                return False
            if not self.ledger.has_copy(key, "l1_gpu"):
                if state.l3 is None:
                    raise RuntimeError("cannot evict L2 before L3 is ready")
                state.l3.activate()
            self.ledger.drop(key, "l2_cpu")
            state.l2 = None
            state.l2_future = None
            state.evict_l2_after_l1 = False
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
        if state.l1_reservation is not None:
            return state.l1_reservation.byte_count
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
        future = self._pool.submit(fn, *args)
        with self._lock:
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
