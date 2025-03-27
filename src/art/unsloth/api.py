import asyncio
import httpx
from openai import (
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
import os
from transformers import PreTrainedTokenizerBase
from typing import cast
import wandb
from wandb.sdk.wandb_run import Run
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from ..api import API
from ..model import Model
from ..types import BaseModel, Message, Trajectory, TuneConfig, Verbosity
from ..utils import format_message
from .pack import (
    packed_tensors_from_tokenized_results,
    packed_tensors_to_dir,
    PackedTensors,
    plot_packed_tensors,
)
from ..model_service import ModelService, StartOpenaiServer
from .tokenization import tokenize_trajectory_groups
from .checkpoints import (
    clear_iteration_dirs,
    get_iteration,
)
from .vllm import max_concurrent_tokens


class UnslothAPI(API):

    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str = "./.art",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ) -> None:
        """
        Initializes a local, directory-based API interface at the given path.

        Note:
            The local API uses Weights & Biases for training monitoring.
            If you don't have a W&B account, you can create one at https://wandb.ai.

        Args:
            in_process: Whether to run the Unsloth service in-process.
            path: The path to the local directory. Defaults to "./.art".
            wandb_entity: The preferred Weights & Biases entity.
            wandb_project: The preferred Weights & Biases project.
        """
        self._in_process = in_process
        self._path = path
        os.makedirs(self._path, exist_ok=True)
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project

        # Other initialization
        self._services: dict[str, ModelService] = {}
        self._tokenizers: dict[str, "PreTrainedTokenizerBase"] = {}
        self._wandb_runs: dict[str, Run] = {}

    async def get_or_create_model(self, name: str, base_model: BaseModel) -> Model:
        """
        Retrieves an existing model or creates a new one.

        Args:
            name: The model's name.
            base_model: The model's base model.
            tool_use: Whether to enable tool use.

        Returns:
            Model: A model instance.
        """
        os.makedirs(self._get_output_dir(name), exist_ok=True)
        return Model(api=self, name=name, base_model=base_model)

    async def _get_service(self, model: Model) -> ModelService:
        if model.name not in self._services:
            self._services[model.name] = ModelService(
                host="localhost",
                port=8089 + len(self._services),
                model_name=model.name,
                base_model=model.base_model,
                output_dir=self._get_output_dir(model.name),
            )
            if not self._in_process:
                await self._services[model.name].serve()
        return self._services[model.name]

    def _get_packed_tensors(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory | BaseException]],
        sequence_length: int,
        verbosity: Verbosity,
        plot_tensors: bool,
    ) -> PackedTensors:
        from transformers import AutoTokenizer

        if not model.base_model in self._tokenizers:
            self._tokenizers[model.base_model] = AutoTokenizer.from_pretrained(
                model.base_model
            )
        tokenizer = self._tokenizers[model.base_model]
        tokenized_results = list(
            tokenize_trajectory_groups(
                tokenizer,
                [
                    [t for t in g if isinstance(t, Trajectory)]
                    for g in trajectory_groups
                ],
            )
        )
        packed_tensors = packed_tensors_from_tokenized_results(
            tokenized_results,
            sequence_length,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
            verbosity=verbosity,
        )
        if plot_tensors:
            plot_packed_tensors(packed_tensors)
        elif verbosity > 0:
            print(
                f"Prepared tuning data with {packed_tensors['tokens'].shape[0]} sequences of length {packed_tensors['tokens'].shape[1]}"
            )
        return packed_tensors

    def _get_output_dir(self, model_name: str) -> str:
        return f"{self._path}/models/{model_name}"

    async def _get_iteration(self, model: Model) -> int:
        return self.__get_iteration(model)

    def __get_iteration(self, model: Model) -> int:
        return get_iteration(self._get_output_dir(model.name))

    async def _clear_iterations(
        self,
        model: Model,
        benchmark: str,
        benchmark_smoothing: float = 1.0,
        verbosity: Verbosity = 1,
    ) -> None:
        run = self._get_wandb_run(model)
        output_dir = self._get_output_dir(model.name)
        # Keep the latest iteration
        iterations_to_keep = [get_iteration(output_dir)]
        try:
            history_df = (
                wandb.Api()
                .run(f"{run.entity}/{run.project}/{run.id}")
                .history()
                .dropna(subset=[benchmark])
                .groupby("iteration")
                .mean()
                .sort_index()
            )
            # Keep the best iteration so far, potentially smoothing to account for variance
            best_iteration = (
                history_df[benchmark].ewm(alpha=benchmark_smoothing).mean().idxmax()
            )
            iterations_to_keep.append(best_iteration)
        except KeyError:
            if verbosity > 1:
                print(f'No "{benchmark}" metric found in history')
        clear_iteration_dirs(output_dir, iterations_to_keep)

    async def _get_openai_client(
        self,
        model: Model,
        estimated_completion_tokens: int,
        tool_use: bool,
        verbosity: Verbosity,
    ) -> tuple[AsyncOpenAI, asyncio.Semaphore]:
        service = await self._get_service(model)
        await service.start_openai_server(StartOpenaiServer(tool_use=tool_use))
        return (
            AsyncOpenAI(
                base_url=f"http://localhost:{service.port - 89}/v1",
                api_key="default",
                http_client=DefaultAsyncHttpxClient(
                    timeout=httpx.Timeout(timeout=1200, connect=5.0),
                    limits=httpx.Limits(
                        max_connections=2048, max_keepalive_connections=2048
                    ),
                ),
            ),
            asyncio.Semaphore(
                int(
                    max_concurrent_tokens(
                        "./logs/vllm.log"
                        if self._in_process
                        else f"{self._get_output_dir(model.name)}/logs/vllm.log"
                    )
                    / estimated_completion_tokens
                )
            ),
        )

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        await client.close()

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory | BaseException]],
        name: str = "val",
    ) -> None:
        # Save logs for each trajectory
        for i, group in enumerate(trajectory_groups):
            for j, trajectory in enumerate(group):
                if isinstance(trajectory, BaseException):
                    continue
                directory = f"{self._get_output_dir(model.name)}/trajectories/{name}/{self.__get_iteration(model):04d}"
                os.makedirs(directory, exist_ok=True)
                i_digits = len(str(len(trajectory_groups) - 1))
                j_digits = len(str(len(group) - 1))
                with open(
                    f"{directory}/{i:0{i_digits}d}-{j:0{j_digits}d}.log", "w"
                ) as f:
                    f.write(self._trajectory_log(trajectory))

        # Collect all metrics (including reward) across all trajectories
        all_metrics: dict[str, list[float]] = {"reward": [], "exception_rate": []}

        for group in trajectory_groups:
            for trajectory in group:
                if isinstance(trajectory, BaseException):
                    all_metrics["exception_rate"].append(1)
                    continue
                else:
                    all_metrics["exception_rate"].append(0)
                # Add reward metric
                all_metrics["reward"].append(trajectory.reward)

                # Collect other custom metrics
                for metric, value in trajectory.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        # Calculate averages for all metrics
        averages = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        self._log_wandb_data(model, averages, name)

    def _trajectory_log(self, trajectory: Trajectory) -> str:
        """Format a trajectory into a readable log string."""
        header = f"reward: {trajectory.reward} {' '.join(f'{k}: {v}' for k, v in trajectory.metrics.items())}\n\n"
        formatted_messages = []
        for message_or_choice in trajectory.messages_and_choices:
            if isinstance(message_or_choice, dict):
                message = message_or_choice
            else:
                message = cast(Message, message_or_choice.message.model_dump())
            formatted_messages.append(format_message(message))
        return header + "\n".join(formatted_messages)

    async def _tune_model(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory | BaseException]],
        config: TuneConfig,
    ) -> None:
        service = await self._get_service(model)
        await self._log(model, trajectory_groups, "train")
        packed_tensors = self._get_packed_tensors(
            model,
            trajectory_groups,
            config.sequence_length,
            config.verbosity,
            config.plot_tensors,
        )
        disk_packed_tensors = packed_tensors_to_dir(
            packed_tensors, f"{self._get_output_dir(model.name)}/tensors"
        )
        await service.tune(disk_packed_tensors)

    def _log_wandb_data(
        self,
        model: Model,
        data: dict[str, float],
        namespace: str,
        iteration_offset: int = 0,
    ) -> None:
        # Add namespacing if needed
        data = (
            {f"{namespace}/{metric}": value for metric, value in data.items()}
            if namespace
            else data
        )

        # Log the data
        self._get_wandb_run(model).log(
            {
                "iteration": self.__get_iteration(model) + iteration_offset,
                **data,
            }
        )

    def _get_wandb_run(self, model: Model) -> Run:
        if model.name not in self._wandb_runs:
            run = wandb.init(
                entity=self._wandb_entity,
                project=self._wandb_project,
                name=model.name,
                id=model.name,
                resume="allow",
            )
            self._wandb_runs[model.name] = run
        return self._wandb_runs[model.name]
