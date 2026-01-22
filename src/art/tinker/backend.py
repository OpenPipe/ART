import os
from typing import Any, cast

from mp_actors import move_to_child_process

from .. import dev
from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import TrainableModel
from ..utils.output_dirs import get_model_dir


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

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        """Start the local OpenAI server and return its base URL + API key."""
        service = await self._get_service(model)
        raw_config: dict[str, Any] = cast(dict[str, Any], config) if config else {}

        server_args = cast(dict[str, Any], raw_config.get("server_args", {}))
        host = server_args.get("host", raw_config.get("host", "0.0.0.0"))
        port = server_args.get("port", raw_config.get("port"))
        if port is None:
            from .service import get_free_port

            port = get_free_port()
        api_key = server_args.get("api_key", raw_config.get("api_key")) or "default"

        # Ensure the Tinker server binds to the same host/port we return.
        tinker_config = cast(
            dev.OpenAIServerConfig,
            {
                **raw_config,
                "host": host,
                "port": port,
            },
        )
        await service.start_openai_server(config=tinker_config)

        base_url = f"http://{host}:{port}/v1"
        return base_url, api_key

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


renderer_name_message = """
To manually specify a renderer (and silence this message), you can set the "renderer_name" field like so:

model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen3-8B",
    _internal_config=art.dev.InternalModelConfig(
        tinker_args=art.dev.TinkerArgs(renderer_name="qwen3_disable_thinking"),
    ),
)

Valid renderer names are:

- llama3
- qwen3
- qwen3_disable_thinking
- qwen3_instruct
- deepseekv3
- deepseekv3_disable_thinking
- gpt_oss_no_sysprompt
- gpt_oss_low_reasoning
- gpt_oss_medium_reasoning
- gpt_oss_high_reasoning
""".strip()


def get_renderer_name(base_model: str) -> str:
    if base_model.startswith("meta-llama/"):
        return "llama3"
    elif base_model.startswith("Qwen/Qwen3-"):
        if "Instruct" in base_model:
            return "qwen3_instruct"
        else:
            print("Defaulting to Qwen3 renderer without thinking for", base_model)
            print(renderer_name_message)
            return "qwen3_disable_thinking"
    elif base_model.startswith("deepseek-ai/DeepSeek-V3"):
        print("Defaulting to DeepSeekV3 renderer without thinking for", base_model)
        print(renderer_name_message)
        return "deepseekv3_disable_thinking"
    elif base_model.startswith("openai/gpt-oss"):
        print("Defaulting to GPT-OSS renderer without system prompt for", base_model)
        print(renderer_name_message)
        return "gpt_oss_no_sysprompt"
    else:
        raise ValueError(f"Unknown base model: {base_model}")
