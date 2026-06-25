from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os

import torch

from .model_chunks import ModelChunks
from .offload import (
    OffloadState,
    offload_to_cpu,
    offload_trainable_buffers_to_cpu,
    reload_to_gpu,
    reload_trainable_buffers_to_gpu,
)
from .streaming_weight_offload import (
    StreamingWeightOffloadConfig,
    StreamingWeightOffloader,
    install_streaming_weight_offload,
    streaming_weight_offload_config_from_env,
)

OFFLOAD_BETWEEN_JOBS_ENV = "ART_MEGATRON_OFFLOAD_BETWEEN_JOBS"


class WeightOffloadManager:
    def __init__(
        self,
        *,
        model: ModelChunks,
        rank: int,
        compile_enabled: bool,
        offload_between_jobs: bool,
        streaming_config: StreamingWeightOffloadConfig,
    ) -> None:
        self.model = model
        self.rank = rank
        self.compile_enabled = compile_enabled
        self.offload_between_jobs = offload_between_jobs
        self.streaming_config = streaming_config
        self.offload_state = OffloadState()
        self.streaming: StreamingWeightOffloader | None = None

    @classmethod
    def from_env(
        cls,
        *,
        model: ModelChunks,
        rank: int,
        compile_enabled: bool,
    ) -> WeightOffloadManager:
        return cls(
            model=model,
            rank=rank,
            compile_enabled=compile_enabled,
            offload_between_jobs=_env_flag(OFFLOAD_BETWEEN_JOBS_ENV, default=True),
            streaming_config=streaming_weight_offload_config_from_env(),
        )

    @classmethod
    def from_config(
        cls,
        *,
        model: ModelChunks,
        rank: int,
        compile_enabled: bool,
        offload_between_jobs: bool = True,
        streaming_config: StreamingWeightOffloadConfig | None = None,
    ) -> WeightOffloadManager:
        return cls(
            model=model,
            rank=rank,
            compile_enabled=compile_enabled,
            offload_between_jobs=offload_between_jobs,
            streaming_config=streaming_config or StreamingWeightOffloadConfig(),
        )

    def install(self) -> None:
        self.streaming = install_streaming_weight_offload(
            model=self.model,
            rank=self.rank,
            compile_enabled=self.compile_enabled,
            config=self.streaming_config,
        )

    def before_job(self) -> None:
        if self.offload_between_jobs:
            if self.streaming is None:
                reload_to_gpu(self.model, self.rank, self.offload_state)
            else:
                reload_trainable_buffers_to_gpu(self.model, self.rank)
        if self.streaming is not None:
            self.streaming.begin_job()

    def after_job(self) -> None:
        if self.streaming is not None:
            self.streaming.finish_job()
        if self.offload_between_jobs:
            if self.streaming is None:
                offload_to_cpu(self.model, self.rank, self.offload_state)
            else:
                offload_trainable_buffers_to_cpu(self.model, self.rank)
            torch.cuda.empty_cache()

    @contextmanager
    def job(self) -> Iterator[None]:
        self.before_job()
        try:
            yield
        finally:
            self.after_job()


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
