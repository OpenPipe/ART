# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modifications:
# Bradley Hilton, OpenPipe Inc., and other ART contributors.

import os
import sys
import time

from functools import partial
from typing import Any, Optional, Union
from warnings import warn

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator, ConfigDict
import torch
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.embedding_utils import resize_token_embeddings
from torchtune.modules.loss import SFTLoss
from torchtune.modules.moe import utils as moe_utils
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    DummyProfiler,
    VALID_BACKENDS_FOR_MEMORY_STATS,
)
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
)
from tqdm import tqdm
from typing import List, Literal


# Pydantic Configuration Models
class ComponentConfig(BaseModel):
    """Base config for components that can be instantiated via config.instantiate"""

    component: str = Field(
        ..., alias="_component_", description="Path to the component class"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for component-specific params
        populate_by_name=True,  # Allow using either field name or alias
    )


class ModelConfig(ComponentConfig):
    """Configuration for model instantiation"""

    pass  # Additional fields handled by extra="allow"


class TokenizerConfig(ComponentConfig):
    """Configuration for tokenizer instantiation"""

    pass  # Additional fields handled by extra="allow"


class OptimizerConfig(ComponentConfig):
    """Configuration for optimizer instantiation"""

    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class LossConfig(ComponentConfig):
    """Configuration for loss function instantiation"""

    ignore_index: int = Field(default=-100)


class DatasetConfig(ComponentConfig):
    """Configuration for dataset instantiation"""

    packed: bool = Field(default=False)


class MetricLoggerConfig(ComponentConfig):
    """Configuration for metric logger instantiation"""

    pass  # Additional fields handled by extra="allow"


class LRSchedulerConfig(ComponentConfig):
    """Configuration for learning rate scheduler instantiation"""

    pass  # Additional fields handled by extra="allow"


class CheckpointerConfig(BaseModel):
    """Configuration for checkpointer"""

    model_type: str = Field(..., description="Model type for checkpointing")

    model_config = ConfigDict(extra="allow")


class CompileConfig(BaseModel):
    """Configuration for torch.compile settings"""

    model: bool = Field(default=True)
    loss: bool = Field(default=True)
    optimizer_step: bool = Field(default=False)
    scale_grads: bool = Field(default=True)


class ProfilerConfig(ComponentConfig):
    """Configuration for profiler"""

    enabled: bool = Field(default=False)
    profile_memory: bool = Field(default=False)
    wait_steps: int = Field(default=0)
    warmup_steps: int = Field(default=1)
    active_steps: int = Field(default=1)

    @field_validator("component", mode="before")
    @classmethod
    def set_default_component(cls, v):
        # v can be None when profiler is not configured
        return v if v is not None else "torchtune.training.setup_torch_profiler"


class TensorParallelPlanConfig(ComponentConfig):
    """Configuration for tensor parallel plan"""

    pass  # Additional fields handled by extra="allow"


class RecipeConfig(BaseModel):
    """Main configuration for FullFinetuneRecipeDistributed"""

    # Core components
    model: ModelConfig
    tokenizer: TokenizerConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    dataset: Union[DatasetConfig, List[DatasetConfig]]
    metric_logger: MetricLoggerConfig
    checkpointer: CheckpointerConfig

    # Device and precision settings
    device: str = Field(default="cuda")
    dtype: Literal["fp32", "bf16"] = Field(default="bf16")

    # Training parameters
    seed: Optional[int] = Field(default=None)
    epochs: int = Field(default=1, gt=0)
    max_steps_per_epoch: Optional[int] = Field(default=None, gt=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    shuffle: bool = Field(default=True)

    # Output and logging
    output_dir: str = Field(default="./outputs")
    log_level: str = Field(default="INFO")
    log_every_n_steps: int = Field(default=1, gt=0)
    log_peak_memory_stats: bool = Field(default=False)

    # Checkpointing and resumption
    resume_from_checkpoint: bool = Field(default=False)
    enable_async_checkpointing: bool = Field(default=False)

    # Distributed training settings
    fsdp_cpu_offload: bool = Field(default=False)
    fsdp_reshard_after_forward: bool = Field(default=True)
    tensor_parallel_dim: int = Field(default=1, ge=1)
    tensor_parallel_plan: Optional[TensorParallelPlanConfig] = Field(default=None)
    context_parallel_dim: int = Field(default=1, ge=1)
    data_parallel_shard_dim: int = Field(default=-1)
    data_parallel_replicate_dim: int = Field(default=1, ge=1)

    # Optimization settings
    optimizer_in_bwd: bool = Field(default=False)
    clip_grad_norm: Optional[float] = Field(default=None, gt=0)
    lr_scheduler: Optional[LRSchedulerConfig] = Field(default=None)

    # Activation checkpointing
    enable_activation_checkpointing: bool = Field(default=False)
    enable_activation_offloading: bool = Field(default=False)
    activation_offloading_use_streams: bool = Field(default=True)
    ac_mode: Optional[str] = Field(default=None)
    ac_option: Optional[int] = Field(default=None)

    # Float8 training
    enable_fp8_training: bool = Field(default=False)
    fp8_recipe_name: Optional[str] = Field(default=None)

    # Compilation settings
    compile: Union[bool, CompileConfig] = Field(default=False)

    # Optional components
    dataset_val: Optional[Union[DatasetConfig, List[DatasetConfig]]] = Field(
        default=None
    )
    batch_size_val: Optional[int] = Field(default=None, gt=0)
    run_val_every_n_steps: Optional[int] = Field(default=None, gt=0)
    profiler: Optional[ProfilerConfig] = Field(default=None)

    # Additional settings
    resize_token_embeddings: bool = Field(default=False)
    custom_sharded_layers: Optional[List[str]] = Field(default=None)
    collate_fn: str = Field(default="torchtune.data.padded_collate_sft")
    cudnn_deterministic_mode: Optional[bool] = Field(default=None)


class FullFinetuneRecipeDistributed(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: RecipeConfig) -> None:
        device_type = cfg.device
        self._device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.enable_async_checkpointing
        self.fsdp_cpu_offload = cfg.fsdp_cpu_offload
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self.tp_plan = cfg.tensor_parallel_plan
        self.tp_degree = cfg.tensor_parallel_dim
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        if self.tp_degree > 1:
            # DTensor does not support grouped_mm yet
            moe_utils.use_grouped_mm = False
        self.cp_degree = cfg.context_parallel_dim
        data_shard = cfg.data_parallel_shard_dim  # -1 means to infer
        data_replicate = cfg.data_parallel_replicate_dim

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            cp=self.cp_degree,
            world_size=self.world_size,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        # Logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.log_every_n_steps
        self._log_peak_memory_stats = cfg.log_peak_memory_stats
        self._logger = utils.get_logger(cfg.log_level)
        if (
            self._log_peak_memory_stats
            and self._device.type not in VALID_BACKENDS_FOR_MEMORY_STATS
        ):
            self._logger.info(
                f"log_peak_memory_stats was set to True; however, training device is not in {VALID_BACKENDS_FOR_MEMORY_STATS}."
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.optimizer_in_bwd
        self._clip_grad_norm = cfg.clip_grad_norm

        self._checkpoint_client = CheckpointClient(
            DictConfig(
                {
                    "checkpointer": cfg.checkpointer.model_dump(by_alias=True),
                    "resume_from_checkpoint": cfg.resume_from_checkpoint,
                    "output_dir": cfg.output_dir,
                }
            )
        )
        self._enable_fp8_training = cfg.enable_fp8_training
        self._fp8_recipe_name = cfg.fp8_recipe_name
        self._run_val_every_n_steps = cfg.run_val_every_n_steps
        if self._run_val_every_n_steps is not None:
            assert (
                cfg.dataset_val is not None
            ), "run_val_every_n_steps is set but dataset_val is not configured"

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.enable_activation_checkpointing
        self._enable_activation_offloading = cfg.enable_activation_offloading
        self._activation_offloading_use_streams = cfg.activation_offloading_use_streams
        if (
            self._enable_activation_offloading
            and self._activation_offloading_use_streams
            and self.parallel_dims.tp_enabled
        ):
            warn(
                message=(
                    "Using activation offloading with streams is not advised in tensor parallel, and may "
                    "cause unstable training. It is advised to set activation_offloading_use_streams: False"
                )
            )
        if self._enable_activation_offloading:
            if device_type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.cudnn_deterministic_mode
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: RecipeConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(
                cfg.metric_logger.model_dump(by_alias=True)
            )
            # log config with parameter override
            self._metric_logger.log_config(DictConfig(cfg.model_dump(by_alias=True)))

        # Load the base model
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()

        compile = cfg.compile
        compile_bool = bool(compile)
        self._compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

        self._compile_model = compile_bool
        self._compile_loss = compile_bool
        self._compile_optimizer_step = compile_bool
        self._compile_scale_grads = compile_bool
        if isinstance(compile, CompileConfig):
            self._compile_model = compile.model
            self._compile_loss = compile.loss
            self._compile_optimizer_step = compile.optimizer_step
            self._compile_scale_grads = compile.scale_grads
        if self._compile_model:
            # Capture scalar outputs is required to compile MoE
            torch._dynamo.config.capture_scalar_outputs = True

        # This indirection is needed to apply torch.compile to scale_grads step.
        self._grad_scaler = training.scale_grads_
        if self._compile_scale_grads:
            self._grad_scaler = torch.compile(
                self._grad_scaler, backend=self._compile_backend
            )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            activation_offloading_use_streams=self._activation_offloading_use_streams,
            custom_sharded_layers=cfg.custom_sharded_layers,
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=cfg.fsdp_reshard_after_forward,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.ac_mode,
            ac_option=cfg.ac_option,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer.model_dump(by_alias=True))

        if cfg.resize_token_embeddings:
            resize_token_embeddings(self._model, self._tokenizer.vocab_size)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )
        if self._compile_optimizer_step:
            if self._optimizer_in_bwd:
                raise ValueError(
                    "optimizer_in_bwd not supported with compiling the optimizer step"
                )
            self._optimizer.step = torch.compile(
                self._optimizer.step,
                backend=self._compile_backend,
            )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    checkpoint_dict = (
                        self._checkpoint_client.load_distributed_checkpoint(
                            self._model,
                            (
                                self._optim_ckpt_wrapper
                                if self._optimizer_in_bwd
                                else self._optimizer
                            ),
                        )
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss.model_dump(by_alias=True))
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile_loss:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        utils.log_rank_zero(self._logger, "Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.collate_fn
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # Setup validation dataloader if validation dataset is provided
        self._val_dataloader = None
        if cfg.dataset_val is not None:
            batch_size_val = (
                cfg.batch_size_val if cfg.batch_size_val is not None else cfg.batch_size
            )
            self._val_dataloader = self._setup_data(
                cfg_dataset=cfg.dataset_val,
                batch_size=batch_size_val,
                collate_fn=collate_name,
                shuffle=False,
            )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.profiler)

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[LRSchedulerConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                self._logger.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler.model_dump(by_alias=True),
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[ProfilerConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            return DummyProfiler()

        profiler, profiler_cfg = config.instantiate(
            cfg_profiler.model_dump(by_alias=True)
        )

        utils.log_rank_zero(
            self._logger, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = cfg_profiler.profile_memory
            if cfg_profiler.enabled:
                self.profiler_wait_steps = cfg_profiler.wait_steps
                self.profiler_warmup_steps = cfg_profiler.warmup_steps
                self.profiler_active_steps = cfg_profiler.active_steps

        return profiler

    def _setup_model(
        self,
        cfg_model: ModelConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        activation_offloading_use_streams: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        utils.log_rank_zero(
            self._logger,
            "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model.model_dump(by_alias=True))

        if self._compile_model:
            training.compile_model(model, verbose=self._is_rank_zero)

        if self._enable_fp8_training:
            # Requires https://github.com/pytorch/pytorch/pull/148922
            if torch.__version__ < "2.8.0.dev20250318":
                raise RuntimeError(
                    "Float8 fine-tuning requires PyTorch 2.8.0.dev20250318 or later."
                )
            if self.tp_plan is not None:
                raise ValueError(
                    "FP8 training does not support tensor parallelism yet. "
                    "This will be enabled in the near future."
                )
            if self.cp_degree > 1:
                raise ValueError(
                    "Context Parallel for fp8 training is not currently supported"
                )
            model = convert_to_float8_training(model, self._fp8_recipe_name)

        # Apply tensor parallelism to the model
        if self.parallel_dims.tp_enabled:
            if not self.parallel_dims.dp_enabled and self.fsdp_cpu_offload:
                raise ValueError(
                    "Tensor parallelism is not supported with FSDP CPU offloading when data parallelism is disabled."
                )
            # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor parallel
            model = training.prepare_mha_for_tp(model, self.world_mesh["tp"])
            if self.tp_plan is not None:
                self.tp_plan = config.instantiate(
                    self.tp_plan.model_dump(by_alias=True),
                    model=model,
                )
            parallelize_module(
                model,
                self.world_mesh["tp"],
                parallelize_plan=self.tp_plan,
            )

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply Fully Sharded Data Parallelism to the model
        if self.parallel_dims.dp_shard_enabled:
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]

            if self.parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard")
            else:
                dp_mesh_dim_names = ("dp_shard",)

            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=self.world_mesh[dp_mesh_dim_names],
            )

        # Define context manager for context parallelism
        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            world_mesh=self.world_mesh,
            model=model,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading, activation_offloading_use_streams
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier(device_ids=[self._device.index])

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: OptimizerConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: config.instantiate(
                    cfg_optimizer.model_dump(by_alias=True), [param]
                )
                for param in self._model.parameters()
            }

            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states for each param. If optimizer states are being restored in an optimizer in
            # backward run, these need to have been saved with the same setting. Cannot restore from runs that
            # did not use optimizer in backward.
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.optim_map[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            utils.log_rank_zero(self._logger, "In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(
                cfg_optimizer.model_dump(by_alias=True), self._model.parameters()
            )
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(self._logger, "Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self,
        cfg_dataset: Union[DatasetConfig, List[DatasetConfig]],
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """
        if isinstance(cfg_dataset, list):
            datasets = [
                config.instantiate(
                    single_cfg_dataset.model_dump(by_alias=True), self._tokenizer
                )
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            ds = config.instantiate(
                cfg_dataset.model_dump(by_alias=True), self._tokenizer
            )
            packed = cfg_dataset.packed

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=shuffle, seed=0
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                    pad_to_multiple_of=self.parallel_dims.min_seq_len_divisor,
                )
                if not packed
                else padded_collate_packed
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )

        return dataloader

    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        with self.activations_handling_ctx:
            outputs = self._model(**batch)

        # post process for third party loss functions
        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
            if isinstance(outputs, DTensor):
                outputs = outputs.full_tensor()

        # Compute loss
        loss = self._loss_fn(outputs, labels)

        # free logits otherwise it peaks backward memory
        del outputs

        return loss

    def validate(self) -> dict[str, float]:
        """
        Run validation loop and return average validation loss.
        """
        self._model.eval()
        total_val_loss = torch.tensor(0.0, device=self._device)
        total_val_tokens = torch.tensor(0.0, device=self._device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._val_dataloader):
                utils.batch_to_device(batch, self._device)

                # Count tokens excluding padding
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()

                # Compute loss
                val_loss = self._loss_step(batch) * current_num_tokens

                total_val_loss += val_loss
                total_val_tokens += current_num_tokens

        # Aggregate validation metrics across all ranks
        torch.distributed.all_reduce(total_val_loss)
        torch.distributed.all_reduce(total_val_tokens)

        avg_val_loss = (
            (total_val_loss / total_val_tokens).item()
            if total_val_tokens > 0
            else float("inf")
        )
        log_dict = {"val_loss": avg_val_loss}

        if self._is_rank_zero:
            self._logger.info(f"Validation loss: {avg_val_loss:.4f}")
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )

        self._model.train()
        return log_dict

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens

                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                with self.context_parallel_manager(list(batch.values())):
                    current_loss = self._loss_step(batch) * current_num_tokens
                    running_loss += current_loss
                    # For optimizer in backward, we need to normalize before calling backward
                    # This case and gradient accumulation are mutually exclusive
                    if self._optimizer_in_bwd:
                        torch.distributed.all_reduce(num_tokens)
                        torch.distributed.all_reduce(running_loss)
                        current_loss = current_loss * (self.dp_degree / num_tokens)
                    current_loss.backward()

                # Optimizer step (if not fused in backward call)
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        self._grad_scaler(
                            list(self._model.parameters()),
                            self.world_size / num_tokens,
                            False if self.parallel_dims.tp_enabled else None,
                        )

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                            # If sharded, collect the DTensor here
                            if isinstance(grad_norm, DTensor):
                                grad_norm = grad_norm.full_tensor()
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # If float8 training is enabled, perform a single all-reduce to compute the
                    # scale for all float8 parameters efficiently instead of doing many small
                    # all-reduces for each parameter
                    if (
                        self._enable_fp8_training
                        and is_fp8_tensorwise_scaling(self._fp8_recipe_name)
                        and self.dp_degree > 1
                    ):
                        precompute_float8_dynamic_scale_for_fsdp(self._model)

                    loss_to_log = running_loss.detach().item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": (
                                num_tokens / self.parallel_dims.non_data_parallel_size
                            )
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                    # Run validation after gradient update
                    if (
                        self._run_val_every_n_steps is not None
                        and self.global_step % self._run_val_every_n_steps == 0
                    ):
                        pbar.refresh()
                        self.validate()

                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1
            self._checkpoint_client.save_checkpoint(
                model=self._model,
                optimizer=(
                    self._optimizer
                    if not self._optimizer_in_bwd
                    else self._optim_ckpt_wrapper
                ),
                training_progress=TrainingProgress(
                    seed=self.seed,
                    epochs_run=self.epochs_run,
                    total_epochs=self.total_epochs,
                    max_steps_per_epoch=self.max_steps_per_epoch,
                    dataloader_state_dict=self._dataloader.state_dict(),
                ),
                epoch=curr_epoch,
            )

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeDistributed", cfg=cfg)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Config must be a dictionary")

    # Create the RecipeConfig - Pydantic handles nested conversions automatically
    recipe_cfg = RecipeConfig(**cfg_dict)

    recipe = FullFinetuneRecipeDistributed(cfg=recipe_cfg)
    recipe.setup(cfg=recipe_cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
