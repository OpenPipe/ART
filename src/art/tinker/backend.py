import os
from typing import cast

from mp_actors import move_to_child_process

from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import TrainableModel
from ..utils.output_dirs import get_model_dir
from .renderers import get_renderer_name


class TinkerBackend(LocalBackend):
    def __init__(
        self,
        *,
        tinker_api_key: str | None = None,
        in_process: bool = False,
        path: str | None = None,
    ) -> None:
        if not "TINKER_API_KEY" in os.environ or tinker_api_key is not None:
            assert tinker_api_key is not None, (
                "TINKER_API_KEY is not set and no tinker_api_key was provided"
            )
            print("Setting TINKER_API_KEY to", tinker_api_key, "in environment")
            os.environ["TINKER_API_KEY"] = tinker_api_key
        super().__init__(in_process=in_process, path=path)

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from ..dev.model import TinkerArgs, TinkerTrainingClientArgs
        from .service import TinkerService

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            config["tinker_args"] = config.get("tinker_args") or TinkerArgs(
                renderer_name=get_renderer_name(model.base_model)
            )
            config["tinker_args"]["training_client_args"] = cast(
                TinkerTrainingClientArgs,
                config["tinker_args"].get("training_client_args") or {},
            )
            self._services[model.name] = TinkerService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="tinker-service",
                )
        return self._services[model.name]
