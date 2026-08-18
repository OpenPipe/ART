from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from threading import Condition, RLock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from .nvme_residency import (
    NvmeResidencyManifest,
    NvmeResidencyStore,
    NvmeResidencyStoreConfig,
)
from .residency import (
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


class RunResidencyManager:
    """Bounded rank-local ownership of movable run tensors in L1-L3."""

    def __init__(
        self,
        config: RunResidencyConfig,
        *,
        snapshot_barrier: Any,
    ) -> None:
        self.config = config
        self.ledger = ResidencyLedger(config.limits)
        self._store = NvmeResidencyStore(config.nvme)
        self._mover = TensorResidencyMover()
        self._snapshot_barrier = snapshot_barrier
        self._states: dict[ResidencyKey, _ManagedState] = {}
        self._installed_base: dict[str, ResidencyKey] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=config.limits.max_concurrent_transitions,
            thread_name_prefix="art-residency",
        )
        self._futures: set[Future[Any]] = set()
        self._retirements: dict[ResidencyKey, Future[None]] = {}
        self._failures: list[BaseException] = []
        self._lock = RLock()
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
        if key.accumulator_revision == 0 and key.run_id in self._installed_base:
            raise RuntimeError("a run may have only one installed base generation")
        byte_count = self._mover.byte_count(tensors, "cuda")
        self._reclaim_l1(byte_count, protected={key})
        reservation = self._reserve(
            key, source=None, target="l1_gpu", byte_count=byte_count
        )
        self._commit(reservation)
        with self._lock:
            self._states[key] = _ManagedState(key=key, tensors=tensors)
            if key.accumulator_revision == 0:
                self._installed_base[key.run_id] = key
        return self.ensure_l2(key)

    def register_l2(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> HostTensorImage:
        """Admit an off-GPU generation without making it compute resident."""
        self._require_open()
        if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
            raise RuntimeError("new L2 residency must be a non-empty CPU tensor tuple")
        byte_count = self._mover.byte_count(tensors, "cpu")
        self._reclaim("l2_cpu", byte_count, protected={key})
        reservation = self._reserve(
            key, source=None, target="l2_cpu", byte_count=byte_count
        )
        try:
            image = self._mover.host_image(tensors)
            self._commit(reservation)
        except BaseException as error:
            self._abort(reservation, error)
            raise
        with self._lock:
            if key in self._states:
                raise RuntimeError("residency key is already registered")
            self._states[key] = _ManagedState(key=key, tensors=tensors, l2=image)
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
        if source.accumulator_revision or target.accumulator_revision:
            raise ValueError("only committed base generations may advance in place")
        if retire_source:
            self.ensure_l2(source).result()
        byte_count = self._mover.byte_count(tensors, "cuda")
        old_bytes = self.ledger.copy(source, "l1_gpu").byte_count
        self._reclaim_l1(max(byte_count - old_bytes, 0), protected={source, target})
        with self._lock:
            if self._installed_base.get(source.run_id) != source:
                raise RuntimeError("residency source is not the installed generation")
            if source.run_id != target.run_id:
                raise ValueError("residency advance cannot change runs")
            self.ledger.advance(source, target, "l1_gpu", byte_count=byte_count)
            self._states[target] = _ManagedState(key=target, tensors=tensors)
            self._installed_base[target.run_id] = target
        if retire_source:
            self._retire(source)
        return self.ensure_l2(target)

    def ensure_l2(self, key: ResidencyKey) -> Future[HostTensorImage]:
        self._require_open()
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
                raise RuntimeError("state has no local source for L2 materialization")
        self._reclaim("l2_cpu", byte_count, protected={key})
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
        """Launch L2/L3 -> L1 transfer; compute later waits on its CUDA event."""
        self._require_open()
        if self.ledger.has_copy(key, "l1_gpu"):
            return
        if key.accumulator_revision == 0:
            self._evict_installed_base(key.run_id)
        with self._lock:
            state = self._state(key)
            if self.ledger.has_copy(key, "l2_cpu") and state.l2 is not None:
                byte_count = state.l2.stats.byte_count
            elif self.ledger.has_copy(key, "l3_nvme") and state.l3_manifest is not None:
                byte_count = state.l3_manifest.payload_bytes
            else:
                raise RuntimeError("state has no local source for L1 materialization")
        self._reclaim_l1(byte_count, protected={key})
        l2 = self.ensure_l2(key).result()
        with self._lock:
            state = self._state(key)
            if state.l1_transition is not None:
                return
            l2.activate()
            reservation = self._reserve(
                key,
                source="l2_cpu",
                target="l1_gpu",
                byte_count=byte_count,
            )
            try:
                transition = self._mover.move(state.tensors, self.config.device)
            except BaseException:
                self._cancel(reservation)
                raise
            state.l1_reservation = reservation
            state.l1_transition = transition

    def acquire_l1(self, key: ResidencyKey) -> None:
        """Fence the current compute stream and pin the selected L1 generation."""
        self.prepare_l1(key)
        with self._lock:
            state = self._state(key)
            transition = state.l1_transition
            reservation = state.l1_reservation
            if transition is not None:
                assert reservation is not None
                transition.wait_on_current_stream()
                self._commit(reservation)
                state.l1_transition = None
                state.l1_reservation = None
                if key.accumulator_revision == 0:
                    self._installed_base[key.run_id] = key
            if not self.ledger.has_copy(key, "l1_gpu"):
                raise RuntimeError("requested generation did not become L1 resident")
            self.ledger.pin(key)

    def release_l1(self, key: ResidencyKey) -> None:
        self.ledger.unpin(key)

    def evict_l1(self, key: ResidencyKey) -> None:
        self.ensure_l2(key).result()
        with self._lock:
            self._evict_l1_locked(key)

    def evict_l2(self, key: ResidencyKey) -> None:
        self.ensure_l3(key).result()
        self._evict_l2(key)

    def record_l4(
        self,
        key: ResidencyKey,
        *,
        immutable_ref: str,
        digest: str,
        byte_count: int,
    ) -> None:
        self._require_open()
        if self.ledger.has_copy(key, "l4_archive"):
            current = self.ledger.copy(key, "l4_archive")
            if (
                current.immutable_ref,
                current.digest,
                current.byte_count,
            ) != (immutable_ref, digest, byte_count):
                raise RuntimeError("L4 residency identity changed after commit")
            return
        reservation = self._reserve(
            key,
            source=("l3_nvme" if self.ledger.has_copy(key, "l3_nvme") else "l2_cpu"),
            target="l4_archive",
            byte_count=byte_count,
        )
        self._commit(reservation, immutable_ref=immutable_ref, digest=digest)

    def l2_image(self, key: ResidencyKey) -> HostTensorImage:
        return self.ensure_l2(key).result()

    def retire(self, key: ResidencyKey) -> None:
        self._require_open()
        self._retire(key)

    def _retire(self, key: ResidencyKey) -> None:
        with self._lock:
            state = self._state(key)
            if state.l1_transition is not None:
                raise RuntimeError("cannot retire a state with an active L1 transfer")
            entry = self.ledger.entry(key)
            if entry.pin_count:
                raise RuntimeError("cannot retire a pinned residency state")
            if state.l3_manifest is not None:
                self._store.delete(key)
            self.ledger.drop_entry(key)
            self._states.pop(key)
            if self._installed_base.get(key.run_id) == key:
                self._installed_base.pop(key.run_id)

    def retire_async(self, key: ResidencyKey) -> Future[None]:
        self._require_open()
        with self._lock:
            if existing := self._retirements.get(key):
                return existing
            state = self._state(key)
            dependencies = tuple(
                future
                for future in (
                    state.l2_future,
                    state.l3_future,
                    state.l2_demotion,
                )
                if future is not None
            )
            future = self._submit(self._retire_after, key, dependencies)
            self._retirements[key] = future
        future.add_done_callback(
            lambda completed: self._retirement_completed(key, completed)
        )
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
                states = tuple(self._states.values())
            for state in states:
                try:
                    if state.l1_transition is not None:
                        assert state.l1_reservation is not None
                        state.l1_transition.synchronize()
                        self._commit(state.l1_reservation)
                        state.l1_transition = None
                        state.l1_reservation = None
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
            snapshot.resolve()
            with self._lock:
                self._commit(reservation)
                state.l2 = snapshot
                return snapshot
        except BaseException as error:
            self._abort(reservation, error)
            raise

    def _retire_after(
        self, key: ResidencyKey, dependencies: tuple[Future[Any], ...]
    ) -> None:
        for dependency in dependencies:
            dependency.result()
        self._retire(key)

    def _demote_after(
        self, key: ResidencyKey, dependency: Future[NvmeResidencyManifest]
    ) -> None:
        dependency.result()
        self._evict_l2(key)

    def _restore_l2(
        self, state: _ManagedState, reservation: ResidencyReservation
    ) -> HostTensorImage:
        try:
            mapped, manifest = self._store.load(state.key, state.tensors)
            image = mapped.pinned_copy()
            with self._lock:
                self._commit(
                    reservation,
                    immutable_ref=str(self._store.path(state.key)),
                    digest=manifest.data_sha256,
                )
                state.l3 = mapped
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
            with self._lock:
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
            mapped, loaded = self._store.load(state.key, state.tensors)
            if loaded != manifest:
                raise RuntimeError("L3 residency changed immediately after commit")
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

    def _evict_installed_base(self, run_id: str) -> None:
        current = self._installed_base.get(run_id)
        if current is not None:
            self.evict_l1(current)

    def _reclaim_l1(self, incoming_bytes: int, *, protected: set[ResidencyKey]) -> None:
        reclaim = self.ledger.required_reclaim("l1_gpu", incoming_bytes)
        if reclaim <= 0:
            return
        candidates = self.ledger.eviction_candidates(
            "l1_gpu", reclaim, protected=protected, require_other_copy=True
        )
        for candidate in candidates:
            self.ensure_l2(candidate).result()
            with self._lock:
                self._evict_l1_locked(candidate)

    def _reclaim(
        self,
        tier: ResidencyTier,
        incoming_bytes: int,
        *,
        protected: set[ResidencyKey],
    ) -> None:
        reclaim = self.ledger.required_reclaim(tier, incoming_bytes)
        if reclaim <= 0:
            return
        candidates = self.ledger.eviction_candidates(
            tier,
            reclaim,
            protected=protected,
            require_other_copy=tier != "l2_cpu",
        )
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

    def _evict_l2(self, key: ResidencyKey) -> None:
        with self._lock:
            state = self._state(key)
            if not self.ledger.has_copy(key, "l2_cpu"):
                return
            if not self.ledger.has_copy(key, "l1_gpu"):
                if state.l3 is None:
                    raise RuntimeError("cannot evict L2 before L3 is ready")
                state.l3.activate()
            self.ledger.drop(key, "l2_cpu")
            state.l2 = None
            state.l2_future = None

    def _evict_l3(self, key: ResidencyKey) -> None:
        with self._lock:
            state = self._state(key)
            if not self.ledger.has_copy(key, "l3_nvme"):
                return
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

    def _evict_l1_locked(self, key: ResidencyKey) -> None:
        state = self._state(key)
        if state.l1_transition is not None:
            raise RuntimeError("cannot evict an active L1 transfer")
        if state.l2 is None:
            raise RuntimeError("cannot evict L1 before L2 is ready")
        if self.ledger.entry(key).pin_count:
            raise RuntimeError("cannot evict a pinned residency state")
        state.l2.activate()
        self.ledger.drop(key, "l1_gpu")
        if self._installed_base.get(key.run_id) == key:
            self._installed_base.pop(key.run_id)

    def _submit(self, fn: Any, *args: Any) -> Future[Any]:
        future = self._pool.submit(fn, *args)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._completed)
        return future

    def _completed(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def _retirement_completed(self, key: ResidencyKey, future: Future[None]) -> None:
        with self._lock:
            if self._retirements.get(key) is future:
                self._retirements.pop(key)

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
        with self._transition_slots:
            limit = self.config.limits.max_concurrent_transitions
            while self._active_transitions >= limit:
                if self._closing or self._closed:
                    raise RuntimeError("run residency manager is closed")
                self._transition_slots.wait()
                self._raise_failures()
            reservation = self.ledger.reserve(
                key, source=source, target=target, byte_count=byte_count
            )
            self._active_transitions += 1
            return reservation

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

    def _cancel(self, reservation: ResidencyReservation) -> None:
        with self._transition_slots:
            self.ledger.abort(reservation)
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
