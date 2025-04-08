import asyncio
from openai import AsyncOpenAI
import os
import torch
from typing import cast
import wandb
from wandb.sdk.wandb_run import Run


from ..api import API
from ..config.openai_server import OpenAIServerConfig
from ..model import Model
from ..types import BaseModel, Message, Trajectory, TuneConfig, Verbosity
from ..utils import format_message
from .grpo import GRPO
from .pack import packed_tensors_from_tokenized_results, plot_packed_tensors
from .model_configs import model_configs
from .recipe import ComponentConfig, TuneRecipeConfig
from .tokenize import tokenize_trajectory_groups
from .tune import (
    clear_iteration_dirs,
    get_iteration,
    get_last_iteration_dir,
    get_last_tune_metrics,
    tune,
)
from .vllm import kill_vllm_workers, start_vllm, vLLM


class LocalAPI(API):

    def __init__(
        self,
        *,
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
            path: The path to the local directory. Defaults to "./.art".
            wandb_entity: The preferred Weights & Biases entity.
            wandb_project: The preferred Weights & Biases project.
        """
        self._path = path
        os.makedirs(self._path, exist_ok=True)
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._vllm: vLLM | None = None
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
        config: OpenAIServerConfig | None,
    ) -> tuple[AsyncOpenAI, asyncio.Semaphore]:
        model_config = model_configs[model.base_model]()
        self._vllm = await start_vllm(
            get_last_iteration_dir(self._get_output_dir(model.name))
            or model.base_model,
            model.name,
            max_concurrent_requests=2048,
            named_arguments=dict(
                block_size=32,
                disable_log_requests=True,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                enforce_eager=True,
                gpu_memory_utilization=0.95,
                max_num_seqs=2048,
                max_num_batched_tokens=16384,
                num_scheduler_steps=1,
                preemption_mode="swap",
                return_tokens_as_token_ids=True,
                swap_space=80,
                tensor_parallel_size=torch.cuda.device_count(),
                enable_auto_tool_choice=tool_use or None,
                tool_call_parser=model_config.vllm_tool_call_parser,
            ),
            timeout=360 + 15 * torch.cuda.device_count(),
            verbosity=verbosity,
        )
        try:
            run = self._get_wandb_run(model)
            history_df = (
                wandb.Api()
                .run(f"{run.entity}/{run.project}/{run.id}")
                .history()
                .dropna(subset=["train/completion_tokens"])
                .sort_values("iteration")
            )
            estimated_completion_tokens = history_df["train/completion_tokens"].iloc[-1]
            if verbosity > 1:
                print(
                    f"Using previous iteration {estimated_completion_tokens} completion tokens per request as estimate"
                )
        except KeyError:
            if verbosity > 1:
                print(f'"train/completion_tokens" not found in run history')
        return self._vllm.client, asyncio.Semaphore(
            int(self._vllm.max_concurrent_tokens / estimated_completion_tokens)
        )

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        if self._vllm:
            self._vllm.process.kill()
            kill_vllm_workers()

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory | BaseException]],
        split: str = "val",
    ) -> None:
        # Save logs for each trajectory
        for i, group in enumerate(trajectory_groups):
            for j, trajectory in enumerate(group):
                if isinstance(trajectory, BaseException):
                    continue
                directory = f"{self._get_output_dir(model.name)}/trajectories/{split}/{self.__get_iteration(model):04d}"
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

        self._log_wandb_data(model, averages, split)

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
        from transformers import AutoTokenizer

        await self._log(model, trajectory_groups, "train")
        tokenizer = AutoTokenizer.from_pretrained(model.base_model)
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
            config.sequence_length,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
            verbosity=config.verbosity,
        )
        if config.plot_tensors:
            plot_packed_tensors(packed_tensors)
        elif config.verbosity > 0:
            print(
                f"Prepared tuning data with {packed_tensors['tokens'].shape[0]} sequences of length {packed_tensors['tokens'].shape[1]}"
            )
        model_config = model_configs[model.base_model]()
        optimizer = ComponentConfig(
            (
                "torch.optim.AdamW"
                if torch.cuda.device_count() > 1
                else "bitsandbytes.optim.PagedAdam8bit"
            ),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        if torch.cuda.device_count() > 1:
            optimizer.fused = True
        await tune(
            base_model=model.base_model,
            output_dir=self._get_output_dir(model.name),
            packed_tensors=packed_tensors,
            model=model_config.tune_model,
            model_type=model_config.tune_model_type,
            config=TuneRecipeConfig(
                optimizer=optimizer,
                loss=ComponentConfig(
                    GRPO,
                    clip_epsilon=config.clip_epsilon,
                    entropy_coef=0.0,
                    kl_coef=0.0,
                ),
                shuffle=True,
                batch_size=32768 // config.sequence_length,
                fsdp_cpu_offload=True,
                enable_activation_checkpointing=True,
                enable_activation_offloading=True,
                custom_sharded_layers=["tok_embeddings", "output"],
                num_output_chunks=model_config.tune_num_output_chunks,
                compile=True,
            ),
            verbosity=config.verbosity,
        )
        self._log_wandb_data(
            model,
            {
                **get_last_tune_metrics(self._get_output_dir(model.name)),
            },
            "train",
            iteration_offset=-1,
        )

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
