import asyncio
from dataclasses import dataclass, field
from itertools import cycle
import json
import os
import socket
import time
from typing import Annotated, Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI
from openai.types import Model, ModelDeleted
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, SkipValidation
import tinker
from transformers.tokenization_utils_base import BatchEncoding
import uvicorn

from art.tinker.cookbook_v import renderers
from art.tinker.cookbook_v.tokenizer_utils import get_tokenizer
from art.tinker.prefix_cache import LRUTrieCache
from art.tinker.renderers import get_renderer_name
from mp_actors import close_proxy, move_to_child_process


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


class ModelUpsert(BaseModel):
    target: str


@dataclass
class OpenAICompatibleTinkerServer:
    host: str | None = None
    port: int | None = None
    num_workers: int | None = None
    _prefix_cache: LRUTrieCache = field(default_factory=LRUTrieCache)
    _task: asyncio.Task[None] | None = None
    _tenants: dict[str, "OpenAICompatibleTinkerServerTenant"] = field(
        default_factory=dict
    )
    _workers: list["OpenAICompatibleTinkerServerWorker"] = field(default_factory=list)

    @property
    def models(self) -> dict[str, str]:
        if "TINKER_API_KEY" not in os.environ:
            raise ValueError("TINKER_API_KEY is not set")
        return self._get_tenant(os.environ["TINKER_API_KEY"])._models

    @models.setter
    def models(self, models: dict[str, str]) -> None:
        if "TINKER_API_KEY" not in os.environ:
            raise ValueError("TINKER_API_KEY is not set")
        self._get_tenant(os.environ["TINKER_API_KEY"])._models = models

    async def start(self) -> tuple[str, int]:
        host = self.host or "0.0.0.0"
        port = self.port or get_free_port(host)
        self._workers = [
            move_to_child_process(
                OpenAICompatibleTinkerServerWorker(),
                process_name=f"openai-compatible-tinker-server-worker-{i}",
            )
            for i in range(self.num_workers or self._default_num_workers())
        ]
        self._task = asyncio.create_task(self._run(host, port))
        client = AsyncOpenAI(api_key="default", base_url=f"http://{host}:{port}/v1")
        start = time.time()
        while True:
            timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 300.0))
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Unable to reach OpenAI-compatible server within {timeout} seconds. You can increase this timeout by setting the ART_SERVER_TIMEOUT environment variable."
                )
            try:
                await client.completions.create(model="", prompt="")
                break  # Server is ready
            except Exception:
                await asyncio.sleep(0.1)
        return host, port

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None
        for worker in self._workers:
            close_proxy(worker)

    def _get_request_tenant(
        self, request: Request
    ) -> "OpenAICompatibleTinkerServerTenant":
        auth = request.headers.get("authorization", "")
        scheme, _, api_key = auth.partition(" ")
        api_key = api_key.strip()
        if scheme.lower() != "bearer" or not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return self._get_tenant(api_key)

    async def _run(self, host: str, port: int) -> None:
        workers = cycle(self._workers)
        app = FastAPI()

        @app.get("/metrics")
        async def metrics() -> str:
            # Minimal Prometheus-style metrics to satisfy the health monitor
            return "# Tinker service metrics\n"

        @app.post("/v1/completions")
        async def completions() -> dict:
            # Minimal completions endpoint for health checks
            return {"choices": [{"text": ""}]}

        @app.get("/v1/models")
        async def list_models(request: Request) -> ModelList:
            tenant = self._get_request_tenant(request)
            return ModelList(
                object="list",
                data=[
                    Model(
                        id=model,
                        created=tenant._model_timestamps.get(model, 0),
                        object="model",
                        owned_by="tinker",
                    )
                    for model in tenant._models
                ],
            )

        @app.get("/v1/models/{model}")
        async def get_model(request: Request, model: str) -> Model:
            tenant = self._get_request_tenant(request)
            if model not in tenant._models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {model}",
                )
            return Model(
                id=model,
                created=tenant._model_timestamps.get(model, 0),
                object="model",
                owned_by="tinker",
            )

        @app.put("/v1/models/{model}")
        async def put_model(
            request: Request,
            model: str,
            body: ModelUpsert,
        ) -> Model:
            tenant = self._get_request_tenant(request)
            tenant._models[model] = body.target
            tenant._model_timestamps.setdefault(model, int(time.time()))
            return Model(
                id=model,
                created=tenant._model_timestamps[model],
                object="model",
                owned_by="tinker",
            )

        @app.delete("/v1/models/{model}")
        async def delete_model(request: Request, model: str) -> ModelDeleted:
            tenant = self._get_request_tenant(request)
            if model not in tenant._models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {model}",
                )
            tenant._models.pop(model)
            tenant._model_timestamps.pop(model, None)
            return ModelDeleted(
                id=model,
                deleted=True,
                object="model",
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request, body: Annotated[CompletionCreateParams, SkipValidation]
        ) -> ChatCompletion:
            worker = next(workers)
            tenant = self._get_request_tenant(request)
            samplable_model = await tenant._get_samplable_model(body["model"])
            rendered_prompt_tokens = await worker.prompt_tokens(
                base_model=samplable_model.base_model,
                messages=list(body["messages"]),
                tools=list(body.get("tools", [])) if "tools" in body else None,
            )
            prompt_tokens = rendered_prompt_tokens
            prefix_entry = self._prefix_cache.lookup(rendered_prompt_tokens)
            if prefix_entry is not None and prefix_entry.rendered_len <= len(
                rendered_prompt_tokens
            ):
                prompt_tokens = (
                    list(prefix_entry.raw_prefix)
                    + rendered_prompt_tokens[prefix_entry.rendered_len :]
                )
            try:
                sample_response = await samplable_model.sampling_client.sample_async(
                    prompt=tinker.ModelInput.from_ints(tokens=prompt_tokens),
                    num_samples=body.get("n") or 1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=body.get("max_completion_tokens")
                        or body.get("max_tokens"),
                        seed=body.get("seed"),
                        temperature=(
                            t if (t := body.get("temperature")) is not None else 1.0
                        ),
                        top_k=body.get("top_k") or -1,
                        top_p=body.get("top_p") or 1.0,
                    ),
                )
            except tinker.APIStatusError as e:
                error_body = e.body
                if isinstance(error_body, dict) and "detail" in error_body:
                    detail = error_body["detail"]  # ty:ignore[invalid-argument-type]
                elif error_body is not None:
                    detail = error_body
                else:
                    detail = str(e)
                raise HTTPException(status_code=e.status_code, detail=detail) from e
            (
                chat_completion,
                token_discrepancies,
            ) = await worker.chat_completion_and_token_discrepancies(
                base_model=samplable_model.base_model,
                sample_response=sample_response,
                model_name=body["model"],
                prompt_tokens=len(prompt_tokens),
            )
            for rendered_response_tokens, raw_response_tokens in token_discrepancies:
                self._prefix_cache.insert(
                    rendered_prompt_tokens + rendered_response_tokens,
                    prompt_tokens + raw_response_tokens,
                )
            return chat_completion

        server_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error",
        )
        server = uvicorn.Server(server_config)
        await server.serve()

    def _default_num_workers(self) -> int:
        try:
            return max(1, len(os.sched_getaffinity(0)))  # ty:ignore[unresolved-attribute]
        except (AttributeError, OSError):
            return os.cpu_count() or 1

    def _get_tenant(self, api_key: str) -> "OpenAICompatibleTinkerServerTenant":
        if api_key not in self._tenants:
            self._tenants[api_key] = OpenAICompatibleTinkerServerTenant(api_key)
        return self._tenants[api_key]


@dataclass
class OpenAICompatibleTinkerServerSamplableModel:
    sampling_client: tinker.SamplingClient
    base_model: str


class OpenAICompatibleTinkerServerTenant:
    def __init__(self, api_key: str) -> None:
        self._models: dict[str, str] = {}
        self._model_timestamps: dict[str, int] = {}
        self._service_client = tinker.ServiceClient(api_key=api_key)
        self._rest_client = self._service_client.create_rest_client()
        self._samplable_models: dict[
            str, asyncio.Task[OpenAICompatibleTinkerServerSamplableModel]
        ] = dict()

    async def _get_samplable_model(
        self, model: str
    ) -> OpenAICompatibleTinkerServerSamplableModel:
        model_path_or_base_model = self._models.get(model, model)
        if not model_path_or_base_model.startswith("tinker://"):
            try:
                get_renderer_name(model_path_or_base_model)
            except ValueError:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Model not found: {model_path_or_base_model}. "
                        "A model must be either a valid `tinker://...` path, supported base model, or registered model alias."
                    ),
                )
        if (task := self._samplable_models.get(model_path_or_base_model)) and (
            not task.done() or task.exception() is None
        ):
            return await task
        self._samplable_models[model_path_or_base_model] = asyncio.create_task(
            self._load_samplable_model(model_path_or_base_model)
        )
        return await self._samplable_models[model_path_or_base_model]

    async def _load_samplable_model(
        self, model_path_or_base_model: str
    ) -> OpenAICompatibleTinkerServerSamplableModel:
        is_model_path = model_path_or_base_model.startswith("tinker://")
        sampling_client = await self._service_client.create_sampling_client_async(
            model_path=model_path_or_base_model if is_model_path else None,
            base_model=model_path_or_base_model if not is_model_path else None,
        )
        if is_model_path:
            sampler_response = await self._rest_client.get_sampler_async(
                sampling_client._sampling_session_id
            )
            base_model = sampler_response.base_model
        else:
            base_model = model_path_or_base_model
        return OpenAICompatibleTinkerServerSamplableModel(
            sampling_client=sampling_client,
            base_model=base_model,
        )


@dataclass
class OpenAICompatibleTinkerServerWorker:
    _renderers: dict[str, renderers.Renderer] = field(default_factory=dict)

    async def prompt_tokens(
        self,
        base_model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolUnionParam] | None,
    ) -> list[int]:
        encoding = self._get_renderer(base_model).tokenizer.apply_chat_template(
            messages,  # type: ignore
            tools=tools,  # type: ignore
            add_generation_prompt=True,
        )
        if isinstance(encoding, BatchEncoding):
            return encoding.input_ids
        else:
            return encoding  # type: ignore

    async def chat_completion_and_token_discrepancies(
        self,
        base_model: str,
        sample_response: tinker.SampleResponse,
        model_name: str,
        prompt_tokens: int,
    ) -> tuple[ChatCompletion, list[tuple[list[int], list[int]]]]:
        renderer = self._get_renderer(base_model)
        choices: list[Choice] = []
        token_discrepancies: list[tuple[list[int], list[int]]] = []
        for i, sequence in enumerate(sample_response.sequences):
            assert sequence.logprobs is not None, "Logprobs are required"
            assert len(sequence.tokens) == len(sequence.logprobs), (
                "Tokens and logprobs must have the same length"
            )
            rendered_response_tokens = renderer.tokenizer.encode(
                renderer.tokenizer.decode(sequence.tokens)
            )
            if rendered_response_tokens != sequence.tokens:
                token_discrepancies.append((rendered_response_tokens, sequence.tokens))
            message, _ = renderer.parse_response(sequence.tokens)
            openai_message = renderer.to_openai_message(message)
            tool_calls = (
                [
                    ChatCompletionMessageFunctionToolCall(
                        type="function",
                        id=tool_call.get("id") or "",
                        function=Function(
                            name=tool_call["function"]["name"],
                            arguments=(
                                tool_call["function"]["arguments"]
                                if isinstance(tool_call["function"]["arguments"], str)
                                else json.dumps(tool_call["function"]["arguments"])
                            ),
                        ),
                    )
                    for tool_call in openai_message.get("tool_calls", [])
                ]
                if openai_message.get("tool_calls")
                else None
            )
            choices.append(
                Choice(
                    finish_reason=sequence.stop_reason,
                    index=i,
                    message=ChatCompletionMessage(
                        content=openai_message.get("content") or None,
                        role="assistant",
                        tool_calls=tool_calls,  # type: ignore
                    ),
                    logprobs=ChoiceLogprobs(
                        content=[
                            ChatCompletionTokenLogprob(
                                token=f"token_id:{token}",
                                bytes=list(renderer.tokenizer.decode(token).encode()),
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
        return (
            ChatCompletion(
                id=str(uuid.uuid4()),
                choices=choices,
                created=int(time.time()),
                model=model_name,
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=completion_tokens + prompt_tokens,
                ),
            ),
            token_discrepancies,
        )

    def _get_renderer(self, base_model: str) -> renderers.Renderer:
        if base_model not in self._renderers:
            self._renderers[base_model] = renderers.get_renderer(
                name=get_renderer_name(base_model),
                tokenizer=get_tokenizer(base_model),
                model_name=base_model,
            )
        return self._renderers[base_model]


def get_free_port(host: str | None = None) -> int:
    """
    Returns the first free port >= 8000.
    """
    port = 8000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host or "", port))
                return port
            except OSError:
                port += 1
