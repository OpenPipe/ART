from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from threading import RLock
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
        self._installed: dict[str, ResidencyKey] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=config.limits.max_concurrent_transitions,
            thread_name_prefix="art-residency",
        )
        self._futures: set[Future[Any]] = set()
        self._failures: list[BaseException] = []
        self._lock = RLock()
        self._closed = False

    def register_l1(
        self, key: ResidencyKey, tensors: tuple[torch.Tensor, ...]
    ) -> Future[HostTensorImage]:
        self._require_open()
        if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("new L1 residency must be a non-empty CUDA tensor tuple")
        if key.run_id in self._installed:
            raise RuntimeError("a run may have only one installed L1 generation")
        byte_count = self._mover.byte_count(tensors, "cuda")
        reservation = self.ledger.reserve(
            key, source=None, target="l1_gpu", byte_count=byte_count
        )
        self.ledger.commit(reservation)
        with self._lock:
            self._states[key] = _ManagedState(key=key, tensors=tensors)
            self._installed[key.run_id] = key
        return self.ensure_l2(key)

    def advance_l1(
        self,
        source: ResidencyKey,
        target: ResidencyKey,
        tensors: tuple[torch.Tensor, ...],
    ) -> Future[HostTensorImage]:
        self._require_open()
        with self._lock:
            if self._installed.get(source.run_id) != source:
                raise RuntimeError("residency source is not the installed generation")
            if source.run_id != target.run_id:
                raise ValueError("residency advance cannot change runs")
            byte_count = self._mover.byte_count(tensors, "cuda")
            self.ledger.advance(source, target, "l1_gpu", byte_count=byte_count)
            self._states[target] = _ManagedState(key=target, tensors=tensors)
            self._installed[target.run_id] = target
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
                reservation = self.ledger.reserve(
                    key,
                    source="l1_gpu",
                    target="l2_cpu",
                    byte_count=byte_count,
                )
                try:
                    snapshot = self._mover.snapshot(state.tensors)
                    self._snapshot_barrier.register(snapshot.pending)
                except BaseException:
                    self.ledger.abort(reservation)
                    raise
                future = self._submit(
                    self._finish_l2_snapshot, state, reservation, snapshot
                )
            elif self.ledger.has_copy(key, "l3_nvme"):
                byte_count = self.ledger.copy(key, "l3_nvme").byte_count
                reservation = self.ledger.reserve(
                    key,
                    source="l3_nvme",
                    target="l2_cpu",
                    byte_count=byte_count,
                )
                future = self._submit(self._restore_l2, state, reservation)
            else:
                raise RuntimeError("state has no local source for L2 materialization")
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
        if self._installed.get(key.run_id) == key:
            return
        l2 = self.ensure_l2(key).result()
        self._evict_installed_run(key.run_id)
        byte_count = l2.stats.byte_count
        self._reclaim_l1(byte_count, protected={key})
        with self._lock:
            state = self._state(key)
            if state.l1_transition is not None:
                return
            l2.activate()
            reservation = self.ledger.reserve(
                key,
                source="l2_cpu",
                target="l1_gpu",
                byte_count=byte_count,
            )
            try:
                transition = self._mover.move(state.tensors, self.config.device)
            except BaseException:
                self.ledger.abort(reservation)
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
                self.ledger.commit(reservation)
                state.l1_transition = None
                state.l1_reservation = None
                self._installed[key.run_id] = key
            if self._installed.get(key.run_id) != key:
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
        with self._lock:
            state = self._state(key)
            if not self.ledger.has_copy(key, "l1_gpu"):
                assert state.l3 is not None
                state.l3.activate()
            self.ledger.drop(key, "l2_cpu")
            state.l2 = None
            state.l2_future = None

    def record_l4(
        self,
        key: ResidencyKey,
        *,
        immutable_ref: str,
        digest: str,
        byte_count: int,
    ) -> None:
        self._require_open()
        reservation = self.ledger.reserve(
            key,
            source=("l3_nvme" if self.ledger.has_copy(key, "l3_nvme") else "l2_cpu"),
            target="l4_archive",
            byte_count=byte_count,
        )
        self.ledger.commit(reservation, immutable_ref=immutable_ref, digest=digest)

    def l2_image(self, key: ResidencyKey) -> HostTensorImage:
        return self.ensure_l2(key).result()

    def retire(self, key: ResidencyKey) -> None:
        self._require_open()
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
            if self._installed.get(key.run_id) == key:
                self._installed.pop(key.run_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(self._futures)
        _done, pending = wait(futures, timeout=self.config.shutdown_timeout_s)
        self._pool.shutdown(wait=False, cancel_futures=True)
        if pending:
            raise TimeoutError(
                f"{len(pending)} residency transitions exceeded shutdown timeout"
            )
        self._raise_failures()

    def _finish_l2_snapshot(
        self,
        state: _ManagedState,
        reservation: ResidencyReservation,
        snapshot: TensorResidencySnapshot,
    ) -> HostTensorImage:
        try:
            snapshot.resolve()
            with self._lock:
                self.ledger.commit(reservation)
                state.l2 = snapshot
                return snapshot
        except BaseException as error:
            self._abort(reservation, error)
            raise

    def _restore_l2(
        self, state: _ManagedState, reservation: ResidencyReservation
    ) -> HostTensorImage:
        try:
            mapped, manifest = self._store.load(state.key, state.tensors)
            image = mapped.pinned_copy()
            with self._lock:
                self.ledger.commit(
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
            with self._lock:
                reservation = self.ledger.reserve(
                    state.key,
                    source="l2_cpu",
                    target="l3_nvme",
                    byte_count=image.stats.byte_count,
                )
            manifest = self._store.write(state.key, image)
            mapped, loaded = self._store.load(state.key, state.tensors)
            if loaded != manifest:
                raise RuntimeError("L3 residency changed immediately after commit")
            with self._lock:
                self.ledger.commit(
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

    def _evict_installed_run(self, run_id: str) -> None:
        current = self._installed.get(run_id)
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
            self.evict_l1(candidate)

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
        if self._installed.get(key.run_id) == key:
            self._installed.pop(key.run_id)

    def _submit(self, fn: Any, *args: Any) -> Future[Any]:
        future = self._pool.submit(fn, *args)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._completed)
        return future

    def _completed(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def _abort(self, reservation: ResidencyReservation, error: BaseException) -> None:
        with self._lock:
            try:
                self.ledger.abort(reservation)
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
        if self._closed:
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
