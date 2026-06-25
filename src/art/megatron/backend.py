from typing import Any, Iterable

from mp_actors import move_to_child_process

from ..backend import AnyTrainableModel
from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult
from ..utils.output_dirs import get_model_dir
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
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
                lora_config=model.lora_config,
            )
            self._services[model.name] = MegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
                enable_expert_replay=self._enable_expert_replay,
            )
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="megatron-service",
                )
        return self._services[model.name]

    def _default_sft_batch_size(self) -> int:
        import torch

        num_gpus = max(int(torch.cuda.device_count()), 1)
        tensor_parallel_size = min(2, num_gpus)
        return max(num_gpus // tensor_parallel_size, 1)
