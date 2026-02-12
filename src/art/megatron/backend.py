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

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from .service import MegatronService

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            self._services[model.name] = MegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="megatron-service",
                )
        return self._services[model.name]
