import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
import os
from pathlib import Path
import shutil
import time
from typing import AsyncIterator, Generator

import tinker
from tinker.lib.public_interfaces.rest_client import RestClient as TinkerRestClient
import torch
import yaml

from art.tinker.cookbook_v import renderers, tokenizer_utils

from .. import dev, types
from ..loss import loss_fn, shift_tensor
from ..preprocessing.inputs import TrainInputs, create_train_inputs
from ..preprocessing.pack import (
    DiskPackedTensors,
    packed_tensors_from_dir,
)
from .server import OpenAICompatibleTinkerServer


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
    _server: OpenAICompatibleTinkerServer | None = None

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        state = await self._state_task
        self._server = OpenAICompatibleTinkerServer(
            host=config.get("host") if config else None,
            port=config.get("port") if config else None,
            sampling_clients_and_renderers=state.sampling_clients_and_renderers,
        )
        with log_timing("Starting OpenAI-compatible Tinker server"):
            return await self._server.start()

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
        next_step = int(last_checkpoint_dir.name) + 1
        new_sampling_client = await self._save_checkpoint(
            last_checkpoint_dir.with_name(f"{next_step:04d}"),
            state.training_client,
        )
        state.sampling_clients_and_renderers[self.model_name] = (
            new_sampling_client,
            state.renderer,
        )
        state.sampling_clients_and_renderers[f"{self.model_name}@{next_step}"] = (
            new_sampling_client,
            state.renderer,
        )

    async def delete_checkpoints(self, steps_to_keep: list[int]) -> None:
        state = await self._state_task
        # Find steps to delete
        steps_to_delete = [
            int(checkpoint_dir.name)
            for checkpoint_dir in self._checkpoints_path.iterdir()
            if int(checkpoint_dir.name) not in steps_to_keep
        ]
        # Delete checkpoints from disk and Tinker
        await asyncio.gather(
            *[
                delete_checkpoint(
                    self._checkpoints_path / f"{step:04d}", state.rest_client
                )
                for step in steps_to_delete
            ]
        )
        # Also remove corresponding sampler clients from state
        for step in steps_to_delete:
            if f"{self.model_name}@{step}" in state.sampling_clients_and_renderers:
                del state.sampling_clients_and_renderers[f"{self.model_name}@{step}"]
                print(f"Removed sampler client for step {step}")

    @cached_property
    def _state_task(self) -> asyncio.Task["TinkerState"]:
        return asyncio.create_task(self._get_state())

    async def _get_state(self) -> "TinkerState":
        config = self.config.get("tinker_args")
        assert config is not None, "Tinker args are required"
        service_client = tinker.ServiceClient()
        rest_client = service_client.create_rest_client()
        checkpoint_dir = self._get_last_checkpoint_dir()
        if checkpoint_dir:
            current_step = int(checkpoint_dir.name)
            info = yaml.safe_load(open(checkpoint_dir / "info.yaml", "r"))
            with log_timing("Creating Tinker training client from checkpoint"):
                training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                    path=info["state_with_optimizer_path"],
                    user_metadata=config.get("user_metadata", None),
                )
            with log_timing("Creating Tinker sampling client from checkpoint"):
                sampling_client = await training_client.create_sampling_client_async(
                    model_path=info["sampler_weights_path"],
                )
        else:
            current_step = 0
            with log_timing("Creating Tinker training client"):
                training_client_args = config.get("training_client_args", {})
                if "rank" not in training_client_args:
                    training_client_args["rank"] = 8
                if "train_unembed" not in training_client_args:
                    training_client_args["train_unembed"] = False
                training_client = (
                    await service_client.create_lora_training_client_async(
                        base_model=self.base_model,
                        **training_client_args,
                    )
                )
            sampling_client = await self._save_checkpoint(
                self._checkpoints_path / "0000", training_client
            )
        renderer = renderers.get_renderer(
            name=config["renderer_name"],
            tokenizer=tokenizer_utils.get_tokenizer(self.base_model),
        )
        return TinkerState(
            service_client=service_client,
            rest_client=rest_client,
            training_client=training_client,
            sampling_clients_and_renderers={
                self.model_name: (sampling_client, renderer),
                f"{self.model_name}@{current_step}": (sampling_client, renderer),
            },
            renderer=renderer,
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


@dataclass
class TinkerState:
    service_client: tinker.ServiceClient
    rest_client: TinkerRestClient
    training_client: tinker.TrainingClient
    sampling_clients_and_renderers: dict[
        str, tuple[tinker.SamplingClient, renderers.Renderer]
    ]
    renderer: renderers.Renderer
