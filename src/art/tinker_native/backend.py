from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import re
import time
from typing import Any, Awaitable, Iterable, Literal, TypeVar, cast
import uuid

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.completion_usage import CompletionUsage
import tinker
import uvicorn

from art.tinker.cookbook_v import renderers, tokenizer_utils

from .. import dev
from ..backend import Backend
from ..model import Model, TrainableModel
from ..tinker.backend import get_renderer_name
from ..tinker.server import get_free_port
from ..trajectories import TrajectoryGroup
from ..types import TrainResult
from ..utils.output_dirs import get_model_dir
from ..utils.trajectory_migration import auto_migrate_on_register
from .data import (
    convert_openai_messages_to_renderer_format,
    parse_completion_to_openai_message,
    trajectory_groups_to_datums,
)

STATE_KEY_RUN_IDS = "tinker_run_ids"
STATE_KEY_LATEST_STEP = "latest_step"
T = TypeVar("T")


@dataclass
class ModelState:
    service_client: tinker.ServiceClient
    rest_client: Any
    training_client: tinker.TrainingClient
    sampler_clients: dict[int, tinker.SamplingClient]
    sampler_checkpoint_paths: dict[int, str]
    training_checkpoint_paths: dict[int, str]
    current_step: int
    renderer: Any
    tokenizer: Any
    output_dir: str
    tinker_run_ids: list[str]
    model_name: str
    server_task: asyncio.Task[None] | None = None
    server_host: str | None = None
    server_port: int | None = None
    server_api_key: str | None = None


@dataclass
class TinkerNativeModelConfig:
    renderer_name: str
    training_client_args: dict[str, Any]


class TinkerNativeBackend(Backend):
    _tinker_train_log_env = "ART_TINKER_TRAIN_LOG"
    _tinker_sample_log_env = "ART_TINKER_SAMPLE_LOG"

    def __init__(
        self,
        *,
        tinker_api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        if not "TINKER_API_KEY" in os.environ or tinker_api_key is not None:
            assert tinker_api_key is not None, (
                "TINKER_API_KEY is not set and no tinker_api_key was provided"
            )
            print("Setting TINKER_API_KEY to", tinker_api_key, "in environment")
            os.environ["TINKER_API_KEY"] = tinker_api_key

        self._path = path or ".art"
        os.makedirs(self._path, exist_ok=True)
        self._model_state: dict[str, ModelState] = {}

    def _env_enabled(self, env_name: str) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return False
        return value.lower() not in ("", "0", "false", "no")

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    async def _tinker_call(
        self,
        label: str,
        awaitable: Awaitable[T],
        *,
        env_name: str,
        prefix: str,
    ) -> T:
        if not self._env_enabled(env_name):
            return await awaitable
        start = time.perf_counter()
        print(f"[tinker:{prefix}] {label} start {self._timestamp()}")
        try:
            return await awaitable
        finally:
            elapsed = time.perf_counter() - start
            print(
                f"[tinker:{prefix}] {label} done in {elapsed:.2f}s "
                f"at {self._timestamp()}"
            )

    async def _tinker_train_call(self, label: str, awaitable: Awaitable[T]) -> T:
        return await self._tinker_call(
            label,
            awaitable,
            env_name=self._tinker_train_log_env,
            prefix="train",
        )

    async def _tinker_sample_call(self, label: str, awaitable: Awaitable[T]) -> T:
        return await self._tinker_call(
            label,
            awaitable,
            env_name=self._tinker_sample_log_env,
            prefix="sample",
        )

    async def close(self) -> None:
        for state in self._model_state.values():
            if state.server_task is not None:
                state.server_task.cancel()

    async def register(self, model: Model) -> None:
        model.base_path = self._path
        output_dir = get_model_dir(model=model, art_path=self._path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.json", "w") as f:
            import json

            json.dump(model.model_dump(), f)

        auto_migrate_on_register(output_dir)

        if not model.trainable:
            return
        trainable_model = cast(TrainableModel, model)
        state = await self._build_model_state(trainable_model)
        self._model_state[model.name] = state

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        state = self._model_state[model.name]

        raw_config: dict[str, Any] = cast(dict[str, Any], config) if config else {}
        server_args = cast(dict[str, Any], raw_config.get("server_args", {}))
        host = server_args.get("host", raw_config.get("host", "0.0.0.0"))
        port = server_args.get("port", raw_config.get("port"))
        if port is None:
            port = get_free_port()
        api_key = server_args.get("api_key", raw_config.get("api_key")) or "default"

        if state.server_task is None:
            state.server_host = host
            state.server_port = port
            state.server_api_key = api_key
            state.server_task = asyncio.create_task(
                self._run_openai_server(state, host=host, port=port)
            )
            state.server_task.add_done_callback(self._crash_on_server_exit)

        base_url = f"http://{host}:{port}/v1"
        await self._wait_for_server_ready(base_url, api_key, model)
        return base_url, api_key

    async def train(  # type: ignore[override]
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        learning_rate: float = 1e-5,
        loss_fn: Literal["cispo", "ppo", "importance_sampling", "dro"] = "cispo",
        normalize_advantages: bool = True,
        save_checkpoint: bool = False,
        loss_fn_config: dict | None = None,
        adam_params: tinker.AdamParams | None = None,
    ) -> TrainResult:
        state = self._model_state[model.name]
        groups_list = list(trajectory_groups)

        datums = trajectory_groups_to_datums(
            groups_list,
            state.renderer,
            state.tokenizer,
            normalize_advantages,
        )

        metrics: dict[str, float] = {
            "num_groups_submitted": float(len(groups_list)),
            "num_datums": float(len(datums)),
        }

        if not datums:
            return TrainResult(step=state.current_step, metrics=metrics)

        if adam_params is None:
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

        def remove_mask(datum: tinker.Datum) -> tinker.Datum:
            if "mask" not in datum.loss_fn_inputs:
                return datum
            loss_fn_inputs = {
                key: value
                for key, value in datum.loss_fn_inputs.items()
                if key != "mask"
            }
            return tinker.Datum(
                model_input=datum.model_input, loss_fn_inputs=loss_fn_inputs
            )

        forward_output = await self._tinker_train_call(
            "forward_backward",
            state.training_client.forward_backward(
                [remove_mask(datum) for datum in datums],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
        )
        optim_output = await self._tinker_train_call(
            "optim_step", state.training_client.optim_step(adam_params)
        )

        if forward_output.metrics:
            for key, value in forward_output.metrics.items():
                if value is None:
                    continue
                metrics[key] = float(value)
        if optim_output.metrics:
            for key, value in optim_output.metrics.items():
                if value is None:
                    continue
                metrics[key] = float(value)

        next_step = state.current_step + 1
        checkpoint_name = f"step_{next_step:06d}"

        if save_checkpoint:
            state_response, sampler_response = await self._save_checkpoint(
                state.training_client, checkpoint_name
            )
            state.training_checkpoint_paths[next_step] = state_response.path
        else:
            sampler_response = await self._save_sampler_weights(
                state.training_client, checkpoint_name
            )
        sampler_client = await self._tinker_train_call(
            "create_sampling_client_async",
            state.training_client.create_sampling_client_async(
                model_path=sampler_response.path
            ),
        )
        state.sampler_clients[next_step] = sampler_client
        state.sampler_checkpoint_paths[next_step] = sampler_response.path

        state.current_step = next_step
        self._persist_model_state(model, state)

        return TrainResult(step=state.current_step, metrics=metrics)

    async def _get_step(self, model: TrainableModel) -> int:
        if model.name in self._model_state:
            return self._model_state[model.name].current_step
        state = model.read_state() or {}
        return int(state.get(STATE_KEY_LATEST_STEP, 0))

    async def _delete_checkpoint_files(
        self,
        model: TrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        print("Checkpoint deletion is not yet implemented for TinkerNativeBackend.")

    def _model_inference_name(self, model: Model, step: int | None = None) -> str:
        base_name = model.inference_model_name or model.name
        if "@" in base_name:
            base_name = base_name.split("@", 1)[0]
        if step is None:
            state = self._model_state.get(model.name)
            step = state.current_step if state is not None else 0
        return f"{base_name}@{step}"

    async def _run_openai_server(
        self,
        state: ModelState,
        host: str,
        port: int,
    ) -> None:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(body: CompletionCreateParams) -> ChatCompletion:
            model_name = body.get("model")
            _, step = self._parse_model_name(model_name)
            sampler_client = await self._get_sampler_client(state, step)

            messages = self._normalize_messages(body["messages"])
            tools = self._normalize_tools(body.get("tools"))
            renderer_messages = convert_openai_messages_to_renderer_format(
                messages=messages,
                tools=tools,
                renderer=state.renderer,
            )
            prompt = state.renderer.build_generation_prompt(renderer_messages)
            prompt_tokens = list(prompt.to_ints())

            max_tokens = body.get("max_completion_tokens")
            if max_tokens is None:
                max_tokens = body.get("max_tokens")
            temperature = body.get("temperature")
            top_k = body.get("top_k")
            top_p = body.get("top_p")
            sampling_params = tinker.SamplingParams(
                max_tokens=max_tokens,
                seed=body.get("seed"),
                temperature=temperature if temperature is not None else 1.0,
                top_k=top_k if top_k is not None else -1,
                top_p=top_p if top_p is not None else 1.0,
                stop=state.renderer.get_stop_sequences(),
            )
            sample_response = await self._tinker_sample_call(
                "sample_async",
                sampler_client.sample_async(
                    prompt=prompt,
                    num_samples=body.get("n") or 1,
                    sampling_params=sampling_params,
                ),
            )

            choices: list[Choice] = []
            for i, sequence in enumerate(sample_response.sequences):
                if sequence.logprobs is None:
                    raise HTTPException(status_code=400, detail="Logprobs are required")
                if len(sequence.tokens) != len(sequence.logprobs):
                    raise HTTPException(
                        status_code=400,
                        detail="Tokens and logprobs must have the same length",
                    )
                parsed_message = parse_completion_to_openai_message(
                    list(sequence.tokens), state.renderer
                )
                content = parsed_message.get("content")
                tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None
                if parsed_message.get("tool_calls"):
                    tool_calls = [
                        ChatCompletionMessageFunctionToolCall(
                            type="function",
                            id=tool_call.get("id") or f"call_{idx}",
                            function=Function(
                                name=tool_call["function"]["name"],
                                arguments=tool_call["function"]["arguments"],
                            ),
                        )
                        for idx, tool_call in enumerate(parsed_message["tool_calls"])
                    ]
                choices.append(
                    Choice(
                        finish_reason=sequence.stop_reason,
                        index=i,
                        message=ChatCompletionMessage(
                            content=content or None,
                            role="assistant",
                            tool_calls=tool_calls,
                        ),
                        logprobs=ChoiceLogprobs(
                            content=[
                                ChatCompletionTokenLogprob(
                                    token=f"token_id:{token}",
                                    logprob=logprob,
                                    top_logprobs=[],
                                )
                                for token, logprob in zip(
                                    sequence.tokens, sequence.logprobs
                                )
                            ]
                        ),
                    )
                )

            completion_tokens = sum(
                len(sequence.tokens) for sequence in sample_response.sequences
            )
            return ChatCompletion(
                id=str(uuid.uuid4()),
                choices=choices,
                created=int(time.time()),
                model=self._format_response_model(model_name, step, state),
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=len(prompt_tokens),
                    total_tokens=completion_tokens + len(prompt_tokens),
                ),
            )

        server_config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(server_config)
        await server.serve()

    def _crash_on_server_exit(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"OpenAI server crashed: {exc}")
        else:
            print("OpenAI server exited unexpectedly.")
        os._exit(1)

    async def _wait_for_server_ready(
        self, base_url: str, api_key: str, model: TrainableModel
    ) -> None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        with_timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 300.0))
        start = time.time()
        while True:
            if time.time() - start > with_timeout:
                raise TimeoutError(
                    f"Unable to reach OpenAI-compatible server within {with_timeout} seconds."
                )
            try:
                await client.chat.completions.create(
                    model=self._model_inference_name(model),
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    max_completion_tokens=1,
                )
                return
            except Exception:
                await asyncio.sleep(0.1)

    async def _build_model_state(self, model: TrainableModel) -> ModelState:
        config = self._resolve_model_config(model)
        service_client = tinker.ServiceClient()
        rest_client = service_client.create_rest_client()

        tokenizer = tokenizer_utils.get_tokenizer(model.base_model)
        renderer = renderers.get_renderer(
            name=config.renderer_name,
            tokenizer=tokenizer,
        )

        saved_state = model.read_state() or {}
        tinker_run_ids = list(saved_state.get(STATE_KEY_RUN_IDS, []))
        training_paths, sampler_paths = await self._list_checkpoints(
            rest_client, tinker_run_ids
        )

        training_client: tinker.TrainingClient
        current_step = 0

        if training_paths:
            current_step = max(training_paths.keys())
            checkpoint_path = training_paths[current_step]
            training_client = await self._create_training_client_from_checkpoint(
                service_client=service_client,
                checkpoint_state_path=checkpoint_path,
                base_model=model.base_model,
                training_client_args=config.training_client_args,
                reset_optimizer=False,
            )
        else:
            training_client = await self._tinker_train_call(
                "create_lora_training_client_async",
                service_client.create_lora_training_client_async(
                    model.base_model, **config.training_client_args
                ),
            )
            checkpoint_name = f"step_{current_step:06d}"
            sampler_response = await self._save_sampler_weights(
                training_client, checkpoint_name
            )
            sampler_paths[current_step] = sampler_response.path

        run_id = training_client.model_id
        if run_id not in tinker_run_ids:
            tinker_run_ids.append(run_id)

        sampler_clients: dict[int, tinker.SamplingClient] = {}
        if current_step in sampler_paths:
            sampler_clients[current_step] = await self._tinker_train_call(
                "create_sampling_client_async",
                training_client.create_sampling_client_async(
                    model_path=sampler_paths[current_step]
                ),
            )
        else:
            checkpoint_name = f"step_{current_step:06d}"
            sampler_response = await self._save_sampler_weights(
                training_client, checkpoint_name
            )
            sampler_paths[current_step] = sampler_response.path
            sampler_clients[current_step] = await self._tinker_train_call(
                "create_sampling_client_async",
                training_client.create_sampling_client_async(
                    model_path=sampler_response.path
                ),
            )

        state = ModelState(
            service_client=service_client,
            rest_client=rest_client,
            training_client=training_client,
            sampler_clients=sampler_clients,
            sampler_checkpoint_paths=sampler_paths,
            training_checkpoint_paths=training_paths,
            current_step=current_step,
            renderer=renderer,
            tokenizer=tokenizer,
            output_dir=get_model_dir(model=model, art_path=self._path),
            tinker_run_ids=tinker_run_ids,
            model_name=((model.inference_model_name or model.name).split("@", 1)[0]),
        )
        self._persist_model_state(model, state)
        return state

    def _resolve_model_config(self, model: TrainableModel) -> TinkerNativeModelConfig:
        internal_config = model._internal_config or {}
        tinker_native_args = cast(
            dev.TinkerNativeArgs | None,
            internal_config.get("tinker_native_args"),
        )
        renderer_name = (
            tinker_native_args.get("renderer_name")
            if tinker_native_args is not None
            else None
        )
        if renderer_name is None:
            renderer_name = get_renderer_name(model.base_model)

        training_client_args = dict(
            tinker_native_args.get("training_client_args", {})
            if tinker_native_args is not None
            else {}
        )
        if "rank" not in training_client_args:
            training_client_args["rank"] = 8
        if "train_unembed" not in training_client_args:
            training_client_args["train_unembed"] = False

        return TinkerNativeModelConfig(
            renderer_name=renderer_name,
            training_client_args=training_client_args,
        )

    async def _list_checkpoints(
        self, rest_client: Any, tinker_run_ids: list[str]
    ) -> tuple[dict[int, str], dict[int, str]]:
        training_paths: dict[int, str] = {}
        sampler_paths: dict[int, str] = {}
        step_pattern = re.compile(r"(?:weights/)?step_(\d+)$")

        for run_id in tinker_run_ids:
            try:
                response = await self._tinker_train_call(
                    f"list_checkpoints_async {run_id}",
                    rest_client.list_checkpoints_async(run_id),
                )
            except Exception as exc:
                print(f"Warning: Could not list checkpoints for {run_id}: {exc}")
                continue
            for checkpoint in response.checkpoints:
                match = step_pattern.match(checkpoint.checkpoint_id)
                if not match:
                    continue
                step = int(match.group(1))
                path = f"tinker://{run_id}/{checkpoint.checkpoint_id}"
                if checkpoint.checkpoint_type == "training":
                    training_paths[step] = path
                elif checkpoint.checkpoint_type == "sampler":
                    sampler_paths[step] = path
        return training_paths, sampler_paths

    async def _get_sampler_client(
        self, state: ModelState, step: int | None
    ) -> tinker.SamplingClient:
        actual_step = step if step is not None else state.current_step
        if actual_step in state.sampler_clients:
            return state.sampler_clients[actual_step]

        if actual_step not in state.sampler_checkpoint_paths:
            training_paths, sampler_paths = await self._list_checkpoints(
                state.rest_client, state.tinker_run_ids
            )
            state.training_checkpoint_paths.update(training_paths)
            state.sampler_checkpoint_paths.update(sampler_paths)

        if actual_step not in state.sampler_checkpoint_paths:
            available = sorted(state.sampler_checkpoint_paths.keys())
            raise HTTPException(
                status_code=404,
                detail=f"No sampler checkpoint for step {actual_step}. Available: {available}",
            )

        sampler_client = await self._tinker_train_call(
            "create_sampling_client_async",
            state.training_client.create_sampling_client_async(
                model_path=state.sampler_checkpoint_paths[actual_step]
            ),
        )
        state.sampler_clients[actual_step] = sampler_client
        return sampler_client

    def _normalize_messages(self, messages: Iterable[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if hasattr(message, "model_dump"):
                normalized.append(message.model_dump())
            else:
                normalized.append(dict(message))
        return normalized

    def _normalize_tools(
        self, tools: Iterable[Any] | None
    ) -> list[dict[str, Any]] | None:
        if tools is None:
            return None
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if hasattr(tool, "model_dump"):
                normalized.append(tool.model_dump())
            else:
                normalized.append(dict(tool))
        return normalized

    def _parse_model_name(
        self, model_name: str | None
    ) -> tuple[str | None, int | None]:
        if model_name and "@" in model_name:
            base_name, step_str = model_name.rsplit("@", 1)
            try:
                return base_name, int(step_str)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid model step: {model_name}"
                ) from exc
        return model_name, None

    def _format_response_model(
        self, model_name: str | None, step: int | None, state: ModelState
    ) -> str:
        if model_name is None:
            return f"{state.model_name}@{state.current_step}"
        if step is None and "@" not in model_name:
            return f"{model_name}@{state.current_step}"
        return model_name

    async def _create_training_client_from_checkpoint(
        self,
        service_client: tinker.ServiceClient,
        checkpoint_state_path: str,
        base_model: str,
        training_client_args: dict[str, Any],
        reset_optimizer: bool = False,
    ) -> tinker.TrainingClient:
        training_client = await self._tinker_train_call(
            "create_lora_training_client_async",
            service_client.create_lora_training_client_async(
                base_model, **training_client_args
            ),
        )

        if reset_optimizer:
            load_future = await self._tinker_train_call(
                "load_state_async",
                training_client.load_state_async(checkpoint_state_path),
            )
            await self._tinker_train_call(
                "load_state_result_async", load_future.result_async()
            )
        else:
            load_future = await self._tinker_train_call(
                "load_state_with_optimizer_async",
                training_client.load_state_with_optimizer_async(checkpoint_state_path),
            )
            await self._tinker_train_call(
                "load_state_with_optimizer_result_async", load_future.result_async()
            )

        return training_client

    async def _save_checkpoint(
        self,
        training_client: tinker.TrainingClient,
        checkpoint_name: str,
    ) -> tuple[Any, Any]:
        state_future, sampler_future = await asyncio.gather(
            self._tinker_train_call(
                "save_state_async",
                training_client.save_state_async(checkpoint_name),
            ),
            self._tinker_train_call(
                "save_weights_for_sampler_async",
                training_client.save_weights_for_sampler_async(checkpoint_name),
            ),
        )
        state_result = await self._tinker_train_call(
            "save_state_result_async", state_future.result_async()
        )
        sampler_result = await self._tinker_train_call(
            "save_weights_for_sampler_result_async", sampler_future.result_async()
        )
        return state_result, sampler_result

    async def _save_sampler_weights(
        self,
        training_client: tinker.TrainingClient,
        checkpoint_name: str,
    ) -> Any:
        sampler_future = await self._tinker_train_call(
            "save_weights_for_sampler_async",
            training_client.save_weights_for_sampler_async(checkpoint_name),
        )
        return await self._tinker_train_call(
            "save_weights_for_sampler_result_async", sampler_future.result_async()
        )

    async def _save_training_state(
        self,
        training_client: tinker.TrainingClient,
        checkpoint_name: str,
    ) -> Any:
        state_future = await self._tinker_train_call(
            "save_state_async",
            training_client.save_state_async(checkpoint_name),
        )
        return await self._tinker_train_call(
            "save_state_result_async", state_future.result_async()
        )

    def _persist_model_state(self, model: TrainableModel, state: ModelState) -> None:
        model.merge_state(
            {
                STATE_KEY_RUN_IDS: state.tinker_run_ids,
                STATE_KEY_LATEST_STEP: state.current_step,
            }
        )
