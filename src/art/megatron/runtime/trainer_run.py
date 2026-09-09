from typing import Protocol

from .publication import TrainerPublicationEvent


class TrainingCancelledError(RuntimeError):
    pass


class EventSink(Protocol):
    def progress(
        self, *, step_index: int, num_steps: int, metrics: dict[str, float]
    ) -> None: ...

    def adapter_ready(self, *, learner_version: int, adapter_path: str) -> None: ...

    def publication(self, event: TrainerPublicationEvent) -> None: ...
