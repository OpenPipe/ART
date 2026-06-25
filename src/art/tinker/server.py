import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import cycle
import json
import os
import socket
import time
from typing import Annotated, Any, AsyncGenerator, Literal, cast
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
from pydantic import BaseModel, Field, SkipValidation, TypeAdapter
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers.tokenization_utils_base import BatchEncoding
import uvicorn

from art.tinker.prefix_cache import LRUTrieCache
from art.tinker.renderers import get_renderer_name, is_qwen3_dot_family_model
from art.types import Message, Tools
from art.utils.chat_template import default_chat_template_kwargs_for_tokenizer
from mp_actors import close_proxy, move_to_child_process


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


class ModelUpsert(BaseModel):
    target: str


WireMessagesAndChoices = list[Choice | Message]
_MESSAGE_ADAPTER = TypeAdapter(ChatCompletionMessageParam)


class MessagesAndChoicesWithLogprobsArgs(BaseModel):
    messages_and_choices: WireMessagesAndChoices
    models: list[str]
    model_aliases: dict[str, str] = Field(default_factory=dict)
    tools: Tools | None


class MessagesAndChoicesWithLogprobs(BaseModel):
    messages_and_choices: WireMessagesAndChoices
    usages: list[CompletionUsage]


def _normalize_message_or_choice(
    message_or_choice: Choice | Message,
) -> Choice | Message:
    if isinstance(message_or_choice, Choice):
        return message_or_choice
    return cast(Message, _MESSAGE_ADAPTER.validate_python(message_or_choice))


def _normalize_qwen3_dot_messages(
    base_model: str, messages: list[ChatCompletionMessageParam]
) -> list[dict[str, Any]]:
    normalized_messages = [cast(dict[str, Any], message) for message in messages]
    if not is_qwen3_dot_family_model(base_model):
        return normalized_messages
    for i, message in enumerate(normalized_messages):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        normalized_tool_calls: list[Any] = []
        changed = False
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                normalized_tool_calls.append(tool_call)
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                normalized_tool_calls.append(tool_call)
                continue
            arguments_json = function.get("arguments")
            if not isinstance(arguments_json, str):
                normalized_tool_calls.append(tool_call)
                continue
            try:
                arguments = json.loads(arguments_json)
            except json.JSONDecodeError:
                normalized_tool_calls.append(tool_call)
                continue
            if not isinstance(arguments, dict):
                normalized_tool_calls.append(tool_call)
                continue
            changed = True
            normalized_tool_calls.append(
                {**tool_call, "function": {**function, "arguments": arguments}}
            )
        if changed:
            normalized_messages[i] = {**message, "tool_calls": normalized_tool_calls}
    return normalized_messages


@dataclass
class OpenAICompatibleTinkerServer:
    host: str | None = None
    port: int | None = None
    num_workers: int | None = None
    max_concurrent_sampling_clients: int | None = None
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
        return self._get_tenant(os.environ["TINKER_API_KEY"]).models

    @models.setter
    def models(self, models: dict[str, str]) -> None:
        if "TINKER_API_KEY" not in os.environ:
            raise ValueError("TINKER_API_KEY is not set")
        self._get_tenant(os.environ["TINKER_API_KEY"]).models = models

    async def start(self) -> tuple[str, int]:
        host = self.host or "0.0.0.0"
        port = self.port or get_free_port(host)
        try:
            self._workers = []
            for i in range(self.num_workers or self._default_num_workers()):
                self._workers.append(
                    move_to_child_process(
                        OpenAICompatibleTinkerServerWorker(),
                        process_name=f"openai-compatible-tinker-server-worker-{i}",
                    )
                )
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
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        try:
            if self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
                self._task = None
        finally:
            for worker in self._workers:
                close_proxy(worker)
            self._workers.clear()

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

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/v1/completions")
        async def completions() -> dict:
            # Minimal completions endpoint for health checks
            return {"choices": [{"text": ""}]}

        @app.post("/v1/messages_and_choices/with_logprobs")
        async def messages_and_choices_with_logprobs(
            request: Request, args: MessagesAndChoicesWithLogprobsArgs
        ) -> MessagesAndChoicesWithLogprobs:
            tenant = self._get_request_tenant(request)

            async def add_logprobs(model: str, alias: str | None) -> CompletionUsage:
                worker = next(workers)
                samplable_model = await tenant.get_samplable_model(model)
                prompt_tokens_and_choice_offsets = (
                    await worker.messages_and_choices_prompt_tokens_and_choice_offsets(
                        base_model=samplable_model.base_model,
                        messages_and_choices=args.messages_and_choices,
                        tools=args.tools,
                    )
                )
                if prompt_tokens_and_choice_offsets is None:
                    return CompletionUsage(
                        completion_tokens=0, prompt_tokens=0, total_tokens=0
                    )
                prompt_tokens, choice_offsets = prompt_tokens_and_choice_offsets
                try:
                    async with samplable_model.sampling_client() as sampling_client:
                        sample_response = await sampling_client.sample_async(
                            prompt=tinker.ModelInput.from_ints(tokens=prompt_tokens),
                            num_samples=1,
                            sampling_params=tinker.SamplingParams(max_tokens=1),
                            include_prompt_logprobs=True,
                        )
                        assert sample_response.prompt_logprobs is not None
                        for choice in args.messages_and_choices:
                            if not isinstance(choice, Choice):
                                continue
                            if choice.logprobs is None:
                                continue
                            token_logprobs = (
                                choice.logprobs.content or choice.logprobs.refusal or []
                            )
                            offset = choice_offsets.pop(0)
                            for i, token_logprob in enumerate(token_logprobs):
                                assert token_logprob.model_extra is not None
                                if token_logprob.token.startswith("token_id:"):
                                    assert (
                                        int(token_logprob.token.split(":")[1])
                                        == prompt_tokens[offset + i]
                                    )
                                token_logprob.model_extra.setdefault(
                                    "extra_logprobs", {}
                                )[alias or model] = sample_response.prompt_logprobs[
                                    offset + i
                                ]
                        return CompletionUsage(
                            completion_tokens=1,
                            prompt_tokens=len(prompt_tokens),
                            total_tokens=1 + len(prompt_tokens),
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

            usages = await asyncio.gather(
                *[
                    add_logprobs(model, args.model_aliases.get(model))
                    for model in args.models
                ]
            )
            return MessagesAndChoicesWithLogprobs(
                messages_and_choices=[
                    _normalize_message_or_choice(item)
                    for item in args.messages_and_choices
                ],
                usages=usages,
            )

        @app.get("/v1/models")
        async def list_models(request: Request) -> ModelList:
            tenant = self._get_request_tenant(request)
            return ModelList(
                object="list",
                data=[
                    Model(
                        id=model,
                        created=tenant.model_timestamps.get(model, 0),
                        object="model",
                        owned_by="tinker",
                    )
                    for model in tenant.models
                ],
            )

        @app.get("/v1/models/{model}")
        async def get_model(request: Request, model: str) -> Model:
            tenant = self._get_request_tenant(request)
            if model not in tenant.models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {model}",
                )
            return Model(
                id=model,
                created=tenant.model_timestamps.get(model, 0),
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
            tenant.models[model] = body.target
            tenant.model_timestamps.setdefault(model, int(time.time()))
            return Model(
                id=model,
                created=tenant.model_timestamps[model],
                object="model",
                owned_by="tinker",
            )

        @app.delete("/v1/models/{model}")
        async def delete_model(request: Request, model: str) -> ModelDeleted:
            tenant = self._get_request_tenant(request)
            if model not in tenant.models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {model}",
                )
            tenant.models.pop(model)
            tenant.model_timestamps.pop(model, None)
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
            samplable_model = await tenant.get_samplable_model(body["model"])
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
                async with samplable_model.sampling_client() as sampling_client:
                    sample_response = await sampling_client.sample_async(
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
            self._tenants[api_key] = OpenAICompatibleTinkerServerTenant(
                api_key, self.max_concurrent_sampling_clients or 32
            )
        return self._tenants[api_key]


@dataclass
class OpenAICompatibleTinkerServerSamplableModel:
    base_model: str
    _sampling_client: tinker.SamplingClient
    _concurrent_sampling_client_semaphore: asyncio.Semaphore
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _yields: int = 0

    @asynccontextmanager
    async def sampling_client(self) -> AsyncGenerator[tinker.SamplingClient, None]:
        async with self._lock:
            if self._yields == 0:
                await self._concurrent_sampling_client_semaphore.acquire()
            self._yields += 1
        try:
            yield self._sampling_client
        finally:
            async with self._lock:
                self._yields -= 1
                if self._yields == 0:
                    self._concurrent_sampling_client_semaphore.release()


class OpenAICompatibleTinkerServerTenant:
    def __init__(self, api_key: str, max_concurrent_sampling_clients: int) -> None:
        self.models: dict[str, str] = {}
        self.model_timestamps: dict[str, int] = {}
        self._service_client = tinker.ServiceClient(api_key=api_key)
        self._rest_client = self._service_client.create_rest_client()
        self._samplable_models: dict[
            str, asyncio.Task[OpenAICompatibleTinkerServerSamplableModel]
        ] = dict()
        self._concurrent_sampling_client_semaphores: defaultdict[
            str, asyncio.Semaphore
        ] = defaultdict(lambda: asyncio.Semaphore(max_concurrent_sampling_clients))

    async def get_samplable_model(
        self, model: str
    ) -> OpenAICompatibleTinkerServerSamplableModel:
        model_path_or_base_model = self.models.get(model, model)
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
        # on_queue_state_change = sampling_client.on_queue_state_change

        # def patched_on_queue_state_change(
        #     queue_state: TinkerQueueState, queue_state_reason: str | None
        # ) -> None:
        #     on_queue_state_change(queue_state, queue_state_reason)
        #     if queue_state == TinkerQueueState.PAUSED_RATE_LIMIT:
        #         # implicit upper-bound on the number of concurrent sampling clients found
        #         # do not allow this number of concurrent sampling clients again
        #         semaphore = self._concurrent_sampling_client_semaphores[base_model]
        #         semaphore._value = max(semaphore._value - 1, -4)

        # sampling_client.on_queue_state_change = patched_on_queue_state_change
        return OpenAICompatibleTinkerServerSamplableModel(
            base_model=base_model,
            _sampling_client=sampling_client,
            _concurrent_sampling_client_semaphore=self._concurrent_sampling_client_semaphores[
                base_model
            ],
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
        normalized_messages = _normalize_qwen3_dot_messages(base_model, messages)
        tokenizer = self._get_renderer(base_model).tokenizer
        chat_template_kwargs = default_chat_template_kwargs_for_tokenizer(tokenizer)
        encoding = tokenizer.apply_chat_template(
            cast(Any, normalized_messages),
            tools=cast(Any, tools),
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if isinstance(encoding, BatchEncoding):
            return list(cast(list[int], encoding.input_ids))
        else:
            return encoding  # type: ignore

    async def messages_and_choices_prompt_tokens_and_choice_offsets(
        self,
        base_model: str,
        messages_and_choices: WireMessagesAndChoices,
        tools: Tools | None,
    ) -> tuple[list[int], list[int]] | None:
        from art.preprocessing.tokenize import tokenize_trajectory
        from art.trajectories import History, Trajectory

        result = tokenize_trajectory(
            tokenizer=self._get_renderer(base_model).tokenizer,
            image_processor=None,
            history=History(
                messages_and_choices=messages_and_choices,
                tools=tools,
            ),
            advantage=0.0,
            allow_training_without_logprobs=False,
            trajectory=Trajectory(
                messages_and_choices=messages_and_choices,
                tools=tools,
            ),
        )
        return (result.token_ids, result.choice_offsets) if result is not None else None

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
                                bytes=list(
                                    cast(str, renderer.tokenizer.decode(token)).encode()
                                ),
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
