import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
import os
from pathlib import Path
import shutil
import socket
import time
from typing import AsyncIterator, Generator
import uuid

from fastapi import FastAPI, Request
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
from tinker.lib.public_interfaces.rest_client import RestClient as TinkerRestClient
from tinker_cookbook import renderers, tokenizer_utils
import torch
import uvicorn
import yaml

from .. import dev, types
from ..loss import loss_fn, shift_tensor
from ..preprocessing.inputs import TrainInputs, create_train_inputs
from ..preprocessing.pack import (
    DiskPackedTensors,
    packed_tensors_from_dir,
)

# Patch Tinker's Qwen3InstructRenderer which mistakenly expects "args" instead of "arguments" in tool calls.
_parse_tool_call = renderers.Qwen3InstructRenderer._parse_tool_call


def _patched_parse_tool_call(
    self, tool_call_str: str
) -> list[renderers.ToolCall] | None:
    return _parse_tool_call(self, tool_call_str.replace('"arguments": ', '"args": '))


renderers.Qwen3InstructRenderer._parse_tool_call = _patched_parse_tool_call


@contextmanager
def log_timing(msg: str) -> Generator[None, None, None]:
    """Context manager that logs a message with timestamp and duration."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}...", end="", flush=True)
    t0 = time.time()
    yield
    print(f" ✓ ({time.time() - t0:.1f}s)", flush=True)


@dataclass
class TinkerService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _openai_server_task: asyncio.Task[None] | None = None

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        self._openai_server_task = asyncio.create_task(
            self._run_openai_server(config, await self._state_task)
        )
        client = AsyncOpenAI(
            base_url=f"http://{(config or {}).get('host', '0.0.0.0')}:{(config or {}).get('port', 8000)}/v1"
        )
        with log_timing("Waiting for server"):
            start = time.time()
            while True:
                timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 300.0))
                if time.time() - start > timeout:
                    raise TimeoutError(
                        f"Unable to reach OpenAI-compatible server within {timeout} seconds. You can increase this timeout by setting the ART_SERVER_TIMEOUT environment variable."
                    )
                try:
                    await client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": "Hello, world!"}],
                        max_completion_tokens=1,
                    )
                    break  # Server is ready
                except:  # noqa: E722
                    await asyncio.sleep(0.1)

    async def vllm_engine_is_sleeping(self) -> bool:
        return False

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        state = await self._state_task

        def custom_loss_fn(
            _: list[tinker.Datum],
            logprobs_list: list[torch.Tensor],
            *,
            masks: list[torch.Tensor],
            inputs: "TrainInputs",
        ) -> tuple[torch.Tensor, dict[str, float]]:
            logprobs = torch.zeros(
                inputs["tokens"].shape[1],
                dtype=logprobs_list[0].dtype,
                device=logprobs_list[0].device,
            )
            for mask, lp in zip(masks, logprobs_list):
                logprobs[mask] = lp
            loss = loss_fn(inputs, logprobs.unsqueeze(0), None, None, _config)
            return loss.mean_policy_loss, {"policy_loss": loss.mean_policy_loss.item()}

        shifted_tokens = shift_tensor(packed_tensors["tokens"], 0)

        for i in range(packed_tensors["tokens"].shape[0]):
            masks = [
                (packed_tensors["group_ids"][i] == group_id)
                | (packed_tensors["parent_ids"][i] == parent_id)
                for group_id in packed_tensors["group_ids"][i].unique()
                for parent_id in [
                    packed_tensors["parent_ids"][i][
                        packed_tensors["group_ids"][i] == group_id
                    ][0]
                ]
            ]
            forward_backward_output_future = (
                await state.training_client.forward_backward_custom_async(
                    data=[
                        tinker.Datum(
                            loss_fn_inputs={
                                "target_tokens": tinker.TensorData.from_torch(
                                    shifted_tokens[i][mask]
                                ),
                                "weights": tinker.TensorData.from_torch(
                                    torch.ones_like(
                                        shifted_tokens[i][mask], dtype=torch.float32
                                    )
                                ),
                            },
                            model_input=tinker.ModelInput.from_ints(
                                packed_tensors["tokens"][i][mask].tolist()
                            ),
                        )
                        for mask in masks
                    ],
                    loss_fn=partial(
                        custom_loss_fn,
                        masks=masks,
                        inputs=create_train_inputs(
                            packed_tensors, i, config, _config, False
                        ),
                    ),
                )
            )
            optim_step_future = await state.training_client.optim_step_async(
                adam_params=tinker.AdamParams(learning_rate=config.learning_rate),
            )
            forward_backward_output, optim_step_response = await asyncio.gather(
                forward_backward_output_future, optim_step_future
            )
            yield {
                **forward_backward_output.metrics,
                **(optim_step_response.metrics or {}),
            }
        last_checkpoint_dir = self._get_last_checkpoint_dir()
        assert last_checkpoint_dir is not None, "No checkpoint found"
        state.sampler_client = await self._save_checkpoint(
            last_checkpoint_dir.with_name(f"{int(last_checkpoint_dir.name) + 1:04d}"),
            state.training_client,
        )

    async def delete_checkpoints(self, steps_to_keep: list[int]) -> None:
        state = await self._state_task
        await asyncio.gather(
            *[
                delete_checkpoint(checkpoint_dir, state.rest_client)
                for checkpoint_dir in self._checkpoints_path.iterdir()
                if int(checkpoint_dir.name) not in steps_to_keep
            ]
        )

    @cached_property
    def _state_task(self) -> asyncio.Task["TinkerState"]:
        return asyncio.create_task(self._get_state())

    async def _get_state(self) -> "TinkerState":
        config = self.config.get("tinker_args") or {"renderer_name": "qwen3_instruct"}
        service_client = tinker.ServiceClient()
        rest_client = service_client.create_rest_client()
        checkpoint_dir = self._get_last_checkpoint_dir()
        if checkpoint_dir:
            info = yaml.safe_load(open(checkpoint_dir / "info.yaml", "r"))
            with log_timing("Creating Tinker training client from checkpoint"):
                training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                    path=info["state_with_optimizer_path"],
                    user_metadata=(self.config.get("tinker_args") or {}).get(
                        "user_metadata", None
                    ),
                )
            with log_timing("Creating Tinker sampling client from checkpoint"):
                sampler_client = await training_client.create_sampling_client_async(
                    model_path=info["sampler_weights_path"],
                )
        else:
            with log_timing("Creating Tinker training client"):
                training_client = (
                    await service_client.create_lora_training_client_async(
                        base_model=self.base_model,
                        **config.get("training_client_args", {}),
                    )
                )
            sampler_client = await self._save_checkpoint(
                self._checkpoints_path / "0000", training_client
            )
        return TinkerState(
            service_client=service_client,
            rest_client=rest_client,
            training_client=training_client,
            sampler_client=sampler_client,
            renderer=renderers.get_renderer(
                name=config.get("renderer_name"),
                tokenizer=tokenizer_utils.get_tokenizer(self.base_model),
            ),
        )

    @property
    def _checkpoints_path(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    def _get_last_checkpoint_dir(self) -> Path | None:
        checkpoint_dirs = (
            sorted(self._checkpoints_path.iterdir())
            if self._checkpoints_path.is_dir()
            else []
        )
        checkpoint_dir: Path | None = checkpoint_dirs[-1] if checkpoint_dirs else None
        return checkpoint_dir

    async def _save_checkpoint(
        self, checkpoint_dir: Path, training_client: tinker.TrainingClient
    ) -> tinker.SamplingClient:
        with log_timing("Saving Tinker checkpoint"):
            state_response, sampler_response = await asyncio.gather(
                *await asyncio.gather(
                    training_client.save_state_async(checkpoint_dir.name),
                    training_client.save_weights_for_sampler_async(checkpoint_dir.name),
                )
            )
        os.makedirs(checkpoint_dir, exist_ok=True)
        yaml.safe_dump(
            {
                "model_id": training_client.model_id,
                "state_with_optimizer_path": state_response.path,
                "sampler_weights_path": sampler_response.path,
            },
            open(checkpoint_dir / "info.yaml", "w"),
        )
        with log_timing("Creating Tinker sampling client"):
            sampling_client = await training_client.create_sampling_client_async(
                model_path=sampler_response.path
            )
        return sampling_client

    async def _run_openai_server(
        self, config: dev.OpenAIServerConfig | None, state: "TinkerState"
    ) -> None:
        config = config or {}
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
            prompt = tinker.ModelInput.from_ints(
                tokens=state.renderer.tokenizer.apply_chat_template(
                    list(body["messages"]),  # type: ignore
                    tools=body.get("tools"),  # type: ignore
                    add_generation_prompt=True,
                )
            )
            sample_response = await state.sampler_client.sample_async(
                prompt=prompt,
                num_samples=body.get("n") or 1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=body.get("max_completion_tokens")
                    or body.get("max_tokens"),
                    seed=body.get("seed"),
                    temperature=body.get("temperature") or 1.0,
                    top_k=body.get("top_k") or -1,
                    top_p=body.get("top_p") or 1.0,
                ),
            )
            choices: list[Choice] = []
            for i, sequence in enumerate(sample_response.sequences):
                assert sequence.logprobs is not None, "Logprobs are required"
                assert len(sequence.tokens) == len(sequence.logprobs), (
                    "Tokens and logprobs must have the same length"
                )
                message, _ = state.renderer.parse_response(sequence.tokens)
                choices.append(
                    Choice(
                        finish_reason=sequence.stop_reason,
                        index=i,
                        message=ChatCompletionMessage(
                            content=message["content"],
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageFunctionToolCall(
                                    type="function",
                                    id=tool_call.id or "",
                                    function=Function(
                                        name=tool_call.function.name,
                                        arguments=tool_call.function.arguments,
                                    ),
                                )
                                for tool_call in message.get("tool_calls", [])
                            ]
                            or None,
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
                model=self.model_name,
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt.length,
                    total_tokens=completion_tokens + prompt.length,
                ),
            )

        server_config = uvicorn.Config(
            app,
            host=config.get("host", "0.0.0.0"),
            port=config.get("port", get_free_port()),
            log_level="error",
        )
        server = uvicorn.Server(server_config)
        await server.serve()


async def delete_checkpoint(
    checkpoint_dir: Path, rest_client: TinkerRestClient
) -> None:
    info = yaml.safe_load(open(checkpoint_dir / "info.yaml", "r"))
    await asyncio.gather(
        rest_client.delete_checkpoint_from_tinker_path_async(
            tinker_path=info["state_with_optimizer_path"],
        ),
        rest_client.delete_checkpoint_from_tinker_path_async(
            tinker_path=info["sampler_weights_path"],
        ),
    )
    shutil.rmtree(checkpoint_dir)
    print(f"Deleted checkpoint {checkpoint_dir.name}")


def get_free_port() -> int:
    """
    Returns the first free port >= 8000.
    """
    port = 8000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1


@dataclass
class TinkerState:
    service_client: tinker.ServiceClient
    rest_client: TinkerRestClient
    training_client: tinker.TrainingClient
    sampler_client: tinker.SamplingClient
    renderer: renderers.Renderer
