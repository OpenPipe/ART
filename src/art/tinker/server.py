import asyncio
from dataclasses import dataclass, field
import os
import socket
import time
from typing import cast
import uuid

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.completion_usage import CompletionUsage
import tinker
import uvicorn

from art.tinker.cookbook_v import renderers
from art.tinker.prefix_cache import LRUTrieCache


@dataclass
class OpenAICompatibleTinkerServer:
    host: str | None = None
    port: int | None = None
    sampling_clients_and_renderers: dict[
        str, tuple[tinker.SamplingClient, renderers.Renderer]
    ] = field(default_factory=dict)
    prefix_cache: LRUTrieCache = field(
        default_factory=lambda: LRUTrieCache(max_entries=1000)
    )
    _task: asyncio.Task[None] | None = None

    async def start(self) -> tuple[str, int]:
        host = self.host or "0.0.0.0"
        port = self.port or get_free_port(host)
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
                await client.completions.create(
                    model="",
                    prompt="",
                )
                break  # Server is ready
            except:  # noqa: E722
                await asyncio.sleep(0.1)
        return host, port

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None

    async def _run(self, host: str, port: int) -> None:
        app = FastAPI()

        @app.get("/metrics")
        async def metrics() -> str:
            # Minimal Prometheus-style metrics to satisfy the health monitor
            return "# Tinker service metrics\n"

        @app.post("/v1/completions")
        async def completions() -> dict:
            # Minimal completions endpoint for health checks
            return {"choices": [{"text": ""}]}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request, body: CompletionCreateParams
        ) -> ChatCompletion:
            try:
                sampler_client, renderer = self.sampling_clients_and_renderers[
                    body["model"]
                ]
            except KeyError:
                raise HTTPException(
                    status_code=404, detail=f"Model {body['model']} not found"
                )
            rendered_prompt_tokens = cast(
                list[int],
                renderer.tokenizer.apply_chat_template(
                    list(body["messages"]),  # type: ignore
                    tools=body.get("tools"),  # type: ignore
                    add_generation_prompt=True,
                ),
            )
            prompt_tokens = rendered_prompt_tokens
            prefix_entry = self.prefix_cache.lookup(rendered_prompt_tokens)
            if prefix_entry is not None and prefix_entry.rendered_len <= len(
                rendered_prompt_tokens
            ):
                prompt_tokens = (
                    list(prefix_entry.raw_prefix)
                    + rendered_prompt_tokens[prefix_entry.rendered_len :]
                )
            try:
                sample_response = await sampler_client.sample_async(
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
            choices: list[Choice] = []
            for i, sequence in enumerate(sample_response.sequences):
                assert sequence.logprobs is not None, "Logprobs are required"
                assert len(sequence.tokens) == len(sequence.logprobs), (
                    "Tokens and logprobs must have the same length"
                )
                rendered_response_tokens = renderer.tokenizer.encode(
                    renderer.tokenizer.decode(sequence.tokens)
                )
                if rendered_response_tokens != sequence.tokens:
                    self.prefix_cache.insert(
                        rendered_prompt_tokens + rendered_response_tokens,
                        prompt_tokens + sequence.tokens,
                    )
                message, _ = renderer.parse_response(sequence.tokens)
                openai_message = renderer.to_openai_message(message)
                tool_calls = (
                    [
                        ChatCompletionMessageFunctionToolCall(
                            type="function",
                            id=tool_call.get("id") or "",
                            function=Function(
                                name=tool_call["function"]["name"],
                                arguments=tool_call["function"]["arguments"],
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
                                        renderer.tokenizer.decode(token).encode()
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
            return ChatCompletion(
                id=str(uuid.uuid4()),
                choices=choices,
                created=int(time.time()),
                model=body["model"],
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=len(prompt_tokens),
                    total_tokens=completion_tokens + len(prompt_tokens),
                ),
            )

        server_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error",
        )
        server = uvicorn.Server(server_config)
        await server.serve()


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
