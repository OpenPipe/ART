"""Immutable forward identities and transactional routing to current parameters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING, Any, cast
import weakref

import torch

if TYPE_CHECKING:
    from ._impl import TrainerRank


@dataclass(frozen=True)
class CheckpointVersion:
    checkpoint: str
    generation: int
    revision: int


VersionedGradient = tuple[CheckpointVersion, int, torch.nn.Parameter, torch.Tensor]


@dataclass
class _GradientBatch:
    gradients: dict[int, tuple[torch.nn.Parameter, torch.Tensor]] = field(
        default_factory=dict
    )
    origins: set[tuple[CheckpointVersion, int, int]] = field(default_factory=set)
    failed: bool = False

    def add(
        self,
        version: CheckpointVersion,
        maximum: int,
        parameter: torch.nn.Parameter,
        gradient: torch.Tensor,
    ) -> None:
        key = id(parameter)
        if key in self.gradients:
            current = self.gradients[key][1]
            if current.layout != torch.strided and gradient.layout == torch.strided:
                self.gradients[key] = parameter, gradient.detach() + current
            else:
                current.add_(gradient.detach())
        else:
            self.gradients[key] = (parameter, gradient.detach().clone())
        self.origins.add((version, maximum, key))

    def validations(self) -> Iterator[VersionedGradient]:
        for version, maximum, key in self.origins:
            yield version, maximum, *self.gradients[key]

    def clear(self) -> None:
        self.gradients.clear()
        self.origins.clear()


@dataclass
class _PreparedGradients:
    parameters: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor | None]]
    origins: dict[str, set[tuple[CheckpointVersion, int]]]

    def clear(self) -> None:
        self.parameters.clear()
        self.origins.clear()


class CheckpointVersions:
    def __init__(self, trainer: TrainerRank) -> None:
        self._trainer = weakref.ref(trainer)
        self._transaction: _GradientBatch | None = None
        self._lock = threading.RLock()
        self._origins: dict[str, set[tuple[CheckpointVersion, int]]] = {}
        self.generation = 0
        self.lora: weakref.WeakValueDictionary[
            tuple[CheckpointVersion, CheckpointVersion, int], Any
        ] = weakref.WeakValueDictionary()

    def capture(self, name: str) -> CheckpointVersion:
        trainer = self._trainer()
        if trainer is None:
            raise RuntimeError("TrainerRank no longer exists")
        slot = trainer._checkpoint_slots[name]
        return CheckpointVersion(name, slot.generation, slot.revision)

    def validate(self, version: CheckpointVersion, maximum: int = 2) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError("max_gradient_staleness must be a nonnegative integer")
        trainer = self._trainer()
        if trainer is None:
            raise RuntimeError("TrainerRank no longer exists")
        slot = trainer._checkpoint_slots.get(version.checkpoint)
        if slot is None or slot.generation != version.generation:
            raise trainer._slot_state_error(
                f"Checkpoint {version.checkpoint!r} was replaced after forward"
            )
        age = slot.revision - version.revision
        if not 0 <= age <= maximum:
            raise trainer._slot_state_error(
                f"Checkpoint {version.checkpoint!r} gradient staleness {age} exceeds "
                f"max_gradient_staleness={maximum} (forward revision "
                f"{version.revision}, current revision {slot.revision})"
            )

    def snapshot(
        self,
        parameter: torch.nn.Parameter,
        version: CheckpointVersion,
        maximum: int = 2,
    ) -> torch.nn.Parameter:
        self.validate(version, maximum)
        with torch._C.DisableTorchFunctionSubclass():
            result = torch.nn.Parameter(
                parameter.detach().clone(), requires_grad=parameter.requires_grad
            )
        self.track(result, parameter, version, maximum)
        return result

    def track(
        self,
        snapshot: torch.nn.Parameter,
        parameter: torch.nn.Parameter,
        version: CheckpointVersion,
        maximum: int = 2,
    ) -> None:
        if snapshot.requires_grad:
            snapshot.register_hook(
                lambda grad: self._stage(version, maximum, parameter, grad)
            )
            snapshot.register_post_accumulate_grad_hook(
                lambda parameter: setattr(parameter, "grad", None)
            )

    def _stage(
        self,
        version: CheckpointVersion,
        maximum: int,
        parameter: torch.nn.Parameter,
        gradient: torch.Tensor,
    ) -> None:
        with self._lock:
            batch = self._transaction
            if batch is None:
                raise RuntimeError(
                    "Backward through captured parameters requires TrainerRank.backward() "
                    "or an explicit _gradient_transaction()"
                )
            try:
                self.validate(version, maximum)
                batch.add(version, maximum, parameter, gradient)
            except BaseException:
                batch.failed = True
                batch.clear()
                raise

    @contextmanager
    def transaction(
        self, *, before_commit: Callable[[Callable[[], None]], None] | None = None
    ) -> Iterator[None]:
        """Coordinate one exit phase on success or failure, then publish once."""
        owns_batch = self._transaction is None
        batch = self._transaction = self._transaction or _GradientBatch()
        error: BaseException | None = None
        prepared: _PreparedGradients | None = None

        def validate() -> None:
            nonlocal prepared
            if error is not None:
                raise error
            if batch.failed:
                raise RuntimeError("A nested gradient transaction failed")
            if owns_batch:
                prepared = self._prepare_batch(batch)
            else:
                self.validate_gradients(batch.validations())

        try:
            try:
                yield
            except BaseException as exc:
                error = exc
            try:
                if before_commit is None:
                    validate()
                else:
                    before_commit(validate)
                if error is not None:
                    raise error
                if owns_batch:
                    if prepared is None:
                        raise RuntimeError(
                            "before_commit must invoke its validation callback"
                        )
                    self._publish(prepared)
            except BaseException:
                batch.failed = True
                if error is not None:
                    raise error
                raise
        finally:
            if prepared is not None:
                prepared.clear()
            if owns_batch:
                batch.clear()
                self._transaction = None

    def validate_gradients(self, gradients: Iterable[VersionedGradient]) -> None:
        with torch._C.DisableTorchFunctionSubclass():
            trainer = self._trainer()
            if trainer is None:
                raise RuntimeError("TrainerRank no longer exists")
            targets: dict[str, set[int]] = {}
            parameter = gradient = None
            try:
                for version, maximum, parameter, gradient in gradients:
                    self.validate(version, maximum)
                    if version.checkpoint not in targets:
                        targets[version.checkpoint] = {
                            id(current)
                            for current in trainer._checkpoint_slots[
                                version.checkpoint
                            ].params
                        }
                    if id(parameter) not in targets[version.checkpoint]:
                        raise trainer._slot_state_error(
                            f"Checkpoint {version.checkpoint!r} gradient target was replaced"
                        )
                    if (
                        gradient.shape != parameter.shape
                        or gradient.device != parameter.device
                        or gradient.dtype != parameter.dtype
                    ):
                        raise ValueError(
                            "Versioned gradient shape/device/dtype differs from parameter"
                        )
                    if any(
                        tensor.layout != torch.strided
                        for tensor in (parameter, gradient, parameter.grad)
                        if tensor is not None
                    ):
                        raise ValueError(
                            "Versioned gradients require strided tensor layouts"
                        )
            finally:
                parameter = gradient = None

    def accumulate(self, gradients: Sequence[VersionedGradient]) -> None:
        with self._lock:
            self.validate_gradients(gradients)
            if self._transaction is None:
                self.commit(gradients)
            else:
                for entry in gradients:
                    self._transaction.add(*entry)

    def commit(self, gradients: Sequence[VersionedGradient]) -> None:
        batch = _GradientBatch()
        try:
            self.validate_gradients(gradients)
            for entry in gradients:
                batch.add(*entry)
            self._commit_batch(batch)
        finally:
            batch.clear()

    def _prepare_batch(self, batch: _GradientBatch) -> _PreparedGradients:
        from ._parameter_hooks import apply_parameter_hooks

        prepared = _PreparedGradients([], {})
        parameter = gradient = previous = combined = None
        try:
            with self._lock, torch.no_grad():
                self.validate_gradients(batch.validations())
                # Prepare every allocation before coordinated exit/publication.
                for parameter, gradient in batch.gradients.values():
                    gradient = apply_parameter_hooks(parameter, gradient)
                    with torch._C.DisableTorchFunctionSubclass():
                        previous = parameter.grad
                        combined = gradient if previous is None else previous + gradient
                    prepared.parameters.append((parameter, combined, previous))
                prepared.origins = {
                    name: origins.copy() for name, origins in self._origins.items()
                }
                for version, maximum, _ in batch.origins:
                    prepared.origins.setdefault(version.checkpoint, set()).add(
                        (version, maximum)
                    )
            return prepared
        except BaseException:
            prepared.clear()
            raise
        finally:
            # A retained exception traceback must not own unpublished tensors.
            parameter = gradient = previous = combined = None

    def _publish(self, prepared: _PreparedGradients) -> None:
        with self._lock, torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
            parameter = gradient = previous = None
            try:
                for parameter, gradient, _ in prepared.parameters:
                    parameter.grad = gradient
            except BaseException:
                for parameter, gradient, previous in prepared.parameters:
                    cast(Any, torch.Tensor.grad).__set__(parameter, previous)
                raise
            else:
                self._origins, prepared.origins = prepared.origins, {}
            finally:
                parameter = gradient = previous = None

    def _commit_batch(self, batch: _GradientBatch) -> None:
        prepared = None
        try:
            prepared = self._prepare_batch(batch)
            self._publish(prepared)
        finally:
            batch.clear()
            if prepared is not None:
                prepared.clear()

    def validate_accumulated(self, names: Sequence[str]) -> None:
        for name in names:
            for version, maximum in self._origins.get(name, ()):
                self.validate(version, maximum)

    def clear(self, names: Sequence[str] | None = None) -> None:
        if names is None:
            self._origins.clear()
        else:
            for name in names:
                self._origins.pop(name, None)
