import os

from mp_actors import move_to_child_process

from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import TrainableModel
from ..utils.output_dirs import get_model_dir


class MegatronBackend(LocalBackend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
    ) -> None:
        super().__init__(in_process=in_process, path=path)
        self._requires_explicit_packed_sequence_length = True
        self._packed_sequence_length_requires_chunk_alignment = False

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from ..dev.validate import is_dedicated_mode, validate_dedicated_config
        from .service import MegatronService

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            validate_dedicated_config(config)
            dedicated = is_dedicated_mode(config)
            if dedicated:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(gpu_id) for gpu_id in config["trainer_gpu_ids"]
                )
            self._services[model.name] = MegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not dedicated and not self._in_process:
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
