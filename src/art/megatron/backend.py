import asyncio
from typing import Any, Iterable, cast

from mp_actors import move_to_child_process

from ..backend import AnyTrainableModel
from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import Model, TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult
from ..utils.lifecycle import process_shutdown_timeout
from ..utils.output_dirs import get_model_dir
from .migrations import apply_megatron_migrations, optimizer_state_path
from .optimizer_state import (
    format_megatron_resume_message,
    prepare_megatron_resume_state,
    read_optimizer_commit,
)
from .runtime_config import get_megatron_runtime_config


class MegatronBackend(LocalBackend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        enable_expert_replay: bool = True,
    ) -> None:
        super().__init__(
            in_process=in_process,
            path=path,
            enable_expert_replay=enable_expert_replay,
        )
        self._requires_explicit_packed_sequence_length = True
        self._packed_sequence_length_requires_chunk_alignment = False
        self._supports_result_packing = True
        self._resume_prepared_models: set[str] = set()

    async def register(self, model: Model) -> None:
        await super().register(model)
        if model.trainable:
            # Keep durable Megatron state migrations centralized behind this call.
            apply_megatron_migrations(get_model_dir(model=model, art_path=self._path))

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> LocalTrainResult:
        for removed_kwarg in ("packed_sequence_length", "megatron_topology"):
            if removed_kwarg in kwargs:
                raise TypeError(
                    f"MegatronBackend.train gets {removed_kwarg} from "
                    "art.init_megatron_runtime_config(...)."
                )
        return await super().train(
            model,
            trajectory_groups,
            packed_sequence_length=get_megatron_runtime_config().packed_sequence_length,
            **kwargs,
        )

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from .service import MegatronService

        if model.name not in self._services:
            output_dir = get_model_dir(model=model, art_path=self._path)
            config = get_model_config(
                base_model=model.base_model,
                output_dir=output_dir,
                config=model._internal_config,
                lora_config=model.lora_config,
            )
            self._services[model.name] = MegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=output_dir,
                enable_expert_replay=self._enable_expert_replay,
            )
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="megatron-service",
                )
        return self._services[model.name]

    async def _get_step(self, model: AnyTrainableModel) -> int:
        if not model.trainable:
            return 0
        if model.name in self._resume_prepared_models:
            return await super()._get_step(model)
        output_dir = get_model_dir(model=model, art_path=self._path)
        info = prepare_megatron_resume_state(
            output_dir=output_dir,
            optimizer_state_path=optimizer_state_path(output_dir),
        )
        print(format_megatron_resume_message(info))
        self._resume_prepared_models.add(model.name)
        return await super()._get_step(model)

    async def finalize_training_session(self, model: AnyTrainableModel) -> None:
        service = self._services.get(model.name)
        if service is not None:
            await cast(Any, service).finalize_training_session()

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        output_dir = get_model_dir(model=model, art_path=self._path)
        commit = read_optimizer_commit(optimizer_state_path(output_dir))
        if commit is not None:
            steps_to_keep = sorted(set(steps_to_keep) | {commit.step})
        await super()._delete_checkpoint_files(model, steps_to_keep)

    async def close(self) -> None:
        failures: list[BaseException] = []
        for service in self._services.values():
            try:
                await asyncio.wait_for(
                    cast(Any, service).finalize_training_session(),
                    timeout=process_shutdown_timeout(1),
                )
            except BaseException as exc:
                failures.append(exc)
        await super().close()
        if failures:
            raise BaseExceptionGroup(
                "Failed to persist Megatron optimizer state during shutdown",
                failures,
            )

    def _default_sft_batch_size(self) -> int:
        import torch

        num_gpus = max(int(torch.cuda.device_count()), 1)
        tensor_parallel_size = min(2, num_gpus)
        return max(num_gpus // tensor_parallel_size, 1)
