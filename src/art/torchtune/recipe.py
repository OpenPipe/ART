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
from typing import Any, Optional, Union, cast
from warnings import warn

import setproctitle
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchtune import config, modules, training, utils
from torchtune.modules import TransformerDecoder
from torchtune.modules.moe import utils as moe_utils
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    VALID_BACKENDS_FOR_MEMORY_STATS,
    DummyProfiler,
    FullModelHFCheckpointer,
)
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.memory import OptimizerInBackwardWrapper
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
)
from tqdm import tqdm

from .. import dev, types
from ..preprocessing.pack import PackedTensors, packed_tensors_from_dir
from ..unsloth.train import gc_and_empty_cuda_cache
from .batch import Batch
from .config import (
    CompileConfig,
    ModelConfig,
    OptimizerConfig,
    ProfilerConfig,
    RecipeConfig,
)


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
        self._device = self._current_device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.enable_async_checkpointing
        self.fsdp_cpu_offload = cfg.fsdp_cpu_offload
        self.is_distributed = os.getenv("WORLD_SIZE", "1") != "1"
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        if self.is_distributed:
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
        self.world_mesh = (
            self.parallel_dims.build_mesh(device_type=device_type)
            if self.is_distributed
            else DeviceMesh(device_type="xla", mesh=[0])
        )
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
        self.max_steps_per_epoch = cfg.max_steps_per_epoch or 0
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
                self.max_steps_per_epoch = int(ckpt_dict[training.MAX_STEPS_KEY])

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

    def setup(self, *, cfg: RecipeConfig, **kwargs: Any) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, optimizer, and metric logger.
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
            from torch._dynamo import config as dynamo_config

            # Capture scalar outputs is required to compile MoE
            dynamo_config.capture_scalar_outputs = True  # type: ignore

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
            assert self._optimizer is not None
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
                            self._optimizer_or_optim_ckpt_wrapper,
                        )
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # Skip output layer since we implement loss calculation directly
        self._model.skip_output_layer = True

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Since we no longer have a dataloader, we set _steps_per_epoch to 1.
        # The max_steps_per_epoch param set by the user controls the actual training length.
        self._steps_per_epoch = 1
        if (
            self.max_steps_per_epoch
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.profiler)

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
    ) -> TransformerDecoder:
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

        with (
            training.set_default_dtype(self._dtype),
            torch.device("meta") if self.is_distributed else self._device,
        ):
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

        assert isinstance(model, TransformerDecoder)

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
                    m.rope_init()  # type: ignore

        if isinstance(model, FSDPModule):
            # This method will convert the full model state dict into a sharded state
            # dict and load into the model
            training.load_from_full_model_state_dict(
                model,
                model_state_dict,
                self._device,
                strict=True,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            model.load_state_dict(model_state_dict)

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

        if self.is_distributed:
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
            if opt_state_dict is not None and isinstance(self._model, FSDPModule):
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
            if opt_state_dict and isinstance(self._model, FSDPModule):
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(self._logger, "Optimizer is initialized.")
            return optimizer

    @property
    def _optimizer_or_optim_ckpt_wrapper(
        self,
    ) -> Optimizer | OptimizerInBackwardWrapper:
        if self._optimizer_in_bwd:
            return self._optim_ckpt_wrapper
        else:
            assert self._optimizer is not None
            return self._optimizer

    def _loss_step(
        self,
        inputs: PackedTensors,
        config: types.TrainConfig,
        dev_config: dev.TrainConfig,
        return_new_logprobs: bool = False,
    ) -> torch.Tensor:
        import torch
        from torch.nn.attention.flex_attention import BlockMask, create_block_mask

        from ..unsloth.train import shift_tensor

        def make_block_mask(
            group_ids: torch.Tensor,  # [B, S]  int32/64
            parent_ids: torch.Tensor,  # [B, S]  int32/64
            block_size: int = 128,
        ) -> BlockMask:
            """
            FlexAttention equivalent of

                causal_mask & (group_ids[q]==group_ids[kv]  |  parent_ids[kv]==group_ids[q])

            * group_ids : id shared by all tokens of the same sampled trajectory
            * parent_ids: id identifying the prompt that produced each token
            """
            B, S = group_ids.shape  # batch, sequence length

            # the closure captures the two id tensors; that's fine for torch.compile
            def mask_mod(b, h, q_idx, kv_idx):
                # causal constraint
                causal = kv_idx <= q_idx

                same_group = group_ids[b, q_idx] == group_ids[b, kv_idx]
                prompt_link = parent_ids[b, q_idx] == group_ids[b, kv_idx]

                return causal & (same_group | prompt_link)

            return create_block_mask(
                mask_mod,
                B=B,
                H=None,
                Q_LEN=S,
                KV_LEN=S,
                BLOCK_SIZE=block_size,
            )

        # mask = calculate_mask(
        #     batch_size=batch["tokens"].shape[0],
        #     seq_len=batch["tokens"].shape[1],
        #     device=self._device,
        #     group_ids=batch["group_ids"],
        #     parent_ids=batch["parent_ids"],
        # )

        block_mask = make_block_mask(
            group_ids=inputs["group_ids"],
            parent_ids=inputs["parent_ids"],
        )

        import torch
        from torch.nn.functional import pad

        def dense_to_block(dense_mask: torch.Tensor, block_size: int) -> torch.Tensor:
            """
            Compress [B, S, S] token mask into [B, Q_blk, KV_blk] block mask
            where each tile is True iff *any* token pair inside it is visible.
            """
            B, S, _ = dense_mask.shape
            pad_len = (-S) % block_size  # make S divisible by block_size
            if pad_len:
                dense_mask = pad(dense_mask, (0, pad_len, 0, pad_len))

            S_pad = S + pad_len
            q_blk = S_pad // block_size
            kv_blk = q_blk

            dense_mask = dense_mask.view(
                B, q_blk, block_size, kv_blk, block_size
            )  # [B,Qb,Bs,Kb,Bs]
            return dense_mask.any(-1).any(-2)  # [B,Qb,Kb]

        def explain_block_diffs(
            dense_mask: torch.Tensor,  # [B,S,S] token-level
            block_mask,  # flex_attention.BlockMask
            block_size: int = 128,
            max_print: int = 20,
        ):
            dense_blk = dense_to_block(dense_mask, block_size)  # [B,Qb,Kb]
            bm_blk = block_mask.to_dense().squeeze(1).bool()  # [B,Qb,Kb]

            diff = dense_blk ^ bm_blk
            idx = diff.nonzero(as_tuple=False)
            if idx.numel() == 0:
                print("✅  token mask and BlockMask agree at block granularity.")
                return

            print(f"⚠️  {idx.size(0)} block mismatches (showing first {max_print})")
            print("b  q_blk  kv_blk | dense  block")
            for b, q, kv in idx[:max_print]:
                print(
                    f"{b:2d} {q:6d} {kv:7d} |   {int(dense_blk[b, q, kv])}      {int(bm_blk[b, q, kv])}"
                )
            if idx.size(0) > max_print:
                print(f"... (+{idx.size(0) - max_print} more)")

        # ---------------------------------------------------------------------
        # EXAMPLE USAGE inside your training loop -----------------------------
        # explain_block_diffs(mask, block_mask, block_size=128, max_print=20)

        # assert torch.equal(mask, block_mask.to_dense()), "masks differ"

        with self.activations_handling_ctx:
            hidden_states = self._model(
                tokens=inputs["tokens"],
                # mask=mask,
                mask=block_mask,
                input_pos=inputs["input_pos"],
            )

        # del mask
        del block_mask

        assistant_mask = shift_tensor(inputs["assistant_mask"], False)
        if assistant_mask.sum() == 0:
            # Like in LinearCrossEntropyLoss, mask 1 token to allow loss to sync with all data parallel workers
            assistant_mask[0] = True
        if return_new_logprobs:
            # If we're returning new logprobs, we need to mask all tokens
            assistant_mask = torch.ones_like(assistant_mask, dtype=torch.bool)

        if isinstance(hidden_states, DTensor):
            # DTensor doesn't support masks so we have to mask locally
            mesh = hidden_states.device_mesh
            placements = hidden_states.placements
            local_hidden_states = hidden_states.to_local()[assistant_mask]
            hidden_states = DTensor.from_local(local_hidden_states, mesh, placements)
        else:
            hidden_states = hidden_states[assistant_mask]

        next_token_ids = shift_tensor(inputs["tokens"], 0)[assistant_mask]
        old_logprobs = shift_tensor(inputs["logprobs"], 0.0)[assistant_mask]
        advantages = shift_tensor(inputs["advantages"], 0.0)[assistant_mask]
        weights = shift_tensor(inputs["weights"], 0.0)[assistant_mask]
        chunk_size = dev_config.get("logprob_calculation_chunk_size", 1024)

        selected_logits = torch.zeros_like(old_logprobs)
        new_logprobs = torch.zeros_like(old_logprobs)
        logits: torch.Tensor | None = None

        for start in range(0, hidden_states.size(0), chunk_size):
            end = min(start + chunk_size, hidden_states.size(0))
            logits = cast(torch.Tensor, self._model.output(hidden_states[start:end]))
            selected_logits[start:end] = torch.gather(
                logits, dim=-1, index=next_token_ids[start:end].unsqueeze(-1)
            ).squeeze(-1)
            new_logprobs[start:end] = selected_logits[start:end] - torch.logsumexp(
                logits, dim=-1
            )

        del hidden_states
        if logits is not None:
            del logits

        if return_new_logprobs:
            return torch.nn.functional.pad(new_logprobs[:-1], (1, 0), value=0.0)

        old_logprobs = torch.where(
            torch.isnan(old_logprobs),
            new_logprobs.detach(),
            old_logprobs,
        )
        prob_ratio = torch.exp(new_logprobs - old_logprobs)
        epsilon = dev_config.get("epsilon", 0.2)
        epsilon_high = dev_config.get("epsilon_high", epsilon)
        if epsilon_high is None:
            epsilon_high = epsilon
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
        loss = (policy_loss * weights).sum()

        return loss

    def train(self, *args: Any, **kwargs: Any) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            assert self._optimizer is not None
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
            micro_batches, batch = self._get_micro_batches(curr_epoch)
            pbar = tqdm(total=len(micro_batches), disable=not self._is_rank_zero)
            for idx, micro_batch in enumerate(micro_batches):
                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(micro_batch, self._device)  # type: ignore

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = micro_batch["assistant_mask"].sum()
                num_tokens += current_num_tokens

                # We'll normalize by the total number of tokens when accumulating gradients
                with self.context_parallel_manager(list(micro_batch.values())):  # type: ignore
                    current_loss = (
                        self._loss_step(micro_batch, batch.config, batch.dev_config)
                        # * current_num_tokens
                    )
                    running_loss += current_loss
                    # For optimizer in backward, we need to normalize before calling backward
                    # This case and gradient accumulation are mutually exclusive
                    if self._optimizer_in_bwd and self.is_distributed:
                        torch.distributed.all_reduce(num_tokens)
                        torch.distributed.all_reduce(running_loss)
                        current_loss = current_loss * (self.dp_degree / num_tokens)
                    current_loss.backward()

                # Optimizer step (if not fused in backward call)
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    grad_norm: float | torch.Tensor | None = None
                    if not self._optimizer_in_bwd:
                        if self.is_distributed:
                            # Get total number of tokens across all ranks to normalize gradients
                            torch.distributed.all_reduce(num_tokens)
                            # Ensure consistency across all ranks for logging
                            torch.distributed.all_reduce(running_loss)

                        # Manually scale the gradients by total # of tokens
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
                        assert self._optimizer is not None
                        for param_group in self._optimizer.param_groups:
                            param_group["lr"] = batch.config.learning_rate
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

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
                            "lr": (
                                self._optimizer.param_groups[0]["lr"]
                                if self._optimizer is not None
                                else list(self._optim_ckpt_wrapper.optim_map.values())[
                                    0
                                ].param_groups[0]["lr"]
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
                        if self._clip_grad_norm is not None and grad_norm is not None:
                            log_dict.update(
                                {
                                    "grad_norm": (
                                        float(grad_norm)
                                        if not isinstance(grad_norm, torch.Tensor)
                                        else float(grad_norm.detach().item())
                                    )
                                }
                            )
                        log_dict["num_gradient_steps"] = (
                            len(micro_batches) // self._gradient_accumulation_steps
                        )
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

                if (
                    self.max_steps_per_epoch
                    and (idx + 1) // self._gradient_accumulation_steps
                    == self.max_steps_per_epoch
                ):
                    break

            self.epochs_run += 1

        self._profiler.stop()

    def _get_micro_batches(self, curr_epoch: int) -> tuple[list[PackedTensors], Batch]:
        import math
        import time
        from pathlib import Path

        from safetensors.torch import save_file

        while True:
            with open(f"{self._output_dir}/batches.jsonl", "r") as f:
                try:
                    batch = Batch.model_validate_json(f.readlines()[curr_epoch].strip())
                except (IndexError, ValidationError):
                    if self._current_device == self._device:
                        if self.is_distributed:
                            gather_cpu_state_dict = training.gather_cpu_state_dict

                            def _gather_cpu_state_dict(
                                model: FSDPModule,
                                is_rank_zero: bool,
                                device: torch.device | None = None,
                                adapter_weights_only: bool = False,
                            ) -> dict[str, Any]:
                                state_dict = gather_cpu_state_dict(
                                    model, is_rank_zero, device, adapter_weights_only
                                )
                                # signal that the GPUs are free
                                Path(f"{self._output_dir}/pids.txt").unlink(
                                    missing_ok=True
                                )
                                training.gather_cpu_state_dict = gather_cpu_state_dict
                                return state_dict

                            training.gather_cpu_state_dict = _gather_cpu_state_dict
                        checkpointer: FullModelHFCheckpointer = (
                            self._checkpoint_client._get_checkpointer()
                        )
                        save_checkpoint = checkpointer.save_checkpoint

                        def _save_checkpoint(
                            state_dict: dict[str, Any],
                            epoch: int,
                            intermediate_checkpoint: bool = False,
                            adapter_only: bool = False,
                            *,
                            step: int | None = None,
                        ) -> None:
                            logger = self._logger

                            class DictWrapper(dict):
                                def __init__(self, original_dict: dict) -> None:
                                    super().__init__(original_dict)

                                def __setitem__(self, key: str, value: Any) -> None:
                                    if key == training.MODEL_KEY:
                                        start_time = time.perf_counter()
                                        save_file(
                                            value, "/dev/shm/weights.safetensors.tmp"
                                        )
                                        os.rename(
                                            "/dev/shm/weights.safetensors.tmp",
                                            "/dev/shm/weights.safetensors",
                                        )
                                        end_time = time.perf_counter()
                                        logger.info(
                                            f"Saving state dict took {end_time - start_time:.2f} seconds"
                                        )
                                    super().__setitem__(key, value)

                            save_checkpoint(
                                DictWrapper(state_dict),
                                epoch,
                                intermediate_checkpoint,
                                adapter_only,
                                step=step,
                            )
                            checkpointer.save_checkpoint = save_checkpoint

                        checkpointer.save_checkpoint = _save_checkpoint
                        self._move_to(torch.device("cpu"))
                        if not self.is_distributed:
                            # signal that the GPUs are free
                            Path(f"{self._output_dir}/pids.txt").unlink(missing_ok=True)
                        self._checkpoint_client.save_checkpoint(
                            model=self._model,
                            optimizer=self._optimizer_or_optim_ckpt_wrapper,
                            training_progress=TrainingProgress(
                                seed=self.seed,
                                epochs_run=self.epochs_run,
                                total_epochs=1,
                                max_steps_per_epoch=self.max_steps_per_epoch,
                            ),
                            epoch=0,
                            single_device=not self.is_distributed,
                        )
                        if self._is_rank_zero:
                            # Ensure checkpoints directory exists
                            checkpoint_dir = os.path.join(
                                self._output_dir, "checkpoints"
                            )
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Get the next step number from the checkpoints directory
                            next_step = (
                                max(
                                    (
                                        int(subdir)
                                        for subdir in os.listdir(checkpoint_dir)
                                        if os.path.isdir(
                                            os.path.join(checkpoint_dir, subdir)
                                        )
                                        and subdir.isdigit()
                                    ),
                                    default=0,
                                )
                                + 1
                            )

                            os.rename(
                                f"{self._output_dir}/epoch_0",
                                f"{checkpoint_dir}/{next_step:04d}",
                            )
                    time.sleep(0.5)
                    continue
            packed_tensors = packed_tensors_from_dir(**batch.disk_packed_tensors)
            if self._current_device != self._device:
                self._move_to(self._device)
            n = batch.disk_packed_tensors["num_sequences"]
            micro_batches = [
                cast(
                    PackedTensors,
                    {
                        k: cast(torch.Tensor, v)[i % n : i % n + 1]
                        for k, v in packed_tensors.items()
                    },
                )
                for i in range(
                    self.dp_rank,
                    math.ceil(n / self.dp_degree) * self.dp_degree,
                    self.dp_degree,
                )
            ]
            if batch.dev_config.get("precalculate_logprobs", False):
                for micro_batch in micro_batches:
                    utils.batch_to_device(micro_batch, self._device)  # type: ignore

                    # Disable gradient tracking for logprob calculation to save memory
                    with torch.no_grad():
                        # We'll normalize by the total number of tokens when accumulating gradients
                        with self.context_parallel_manager(list(micro_batch.values())):  # type: ignore
                            micro_batch["logprobs"] = self._loss_step(
                                micro_batch,
                                batch.config,
                                batch.dev_config,
                                return_new_logprobs=True,
                            ).unsqueeze(0)

                    # Move tensors back to CPU after calculation
                    utils.batch_to_device(micro_batch, torch.device("cpu"))  # type: ignore
                    # # Free GPU memory immediately
                    # if self._device.type == "cuda":
                    #     torch.cuda.empty_cache()
            return micro_batches, batch

    def _move_to(self, device: torch.device) -> None:
        """
        Move the model and optimizer to the given device.
        """
        move_start = time.perf_counter()

        # For FSDP models, we need to handle device movement carefully
        # FSDP models can't be moved with simple .to() calls

        # Method 1: Try to move parameters individually for FSDP compatibility
        try:
            # Move model parameters one by one
            with torch.no_grad():
                for param in self._model.parameters():
                    if param.device != device:
                        param_data = param.data.to(device)
                        param.data = param_data

                # Move model buffers one by one
                for buffer in self._model.buffers():
                    if buffer.device != device:
                        buffer_data = buffer.data.to(device)
                        buffer.data = buffer_data

        except Exception as e:
            print(f"Failed to move model parameters individually: {e}")
            # Fallback: try the standard .to() method
            try:
                self._model.to(device)
                print("Fallback: moved model using .to() method")
            except Exception as e2:
                print(f"Both methods failed: {e2}")
                return

        # Move optimizer states to device
        if hasattr(self, "_optimizer") and self._optimizer is not None:
            try:
                for param_group in self._optimizer.param_groups:
                    for param in param_group["params"]:
                        if param in self._optimizer.state:
                            state = self._optimizer.state[param]
                            for key, value in state.items():
                                if (
                                    isinstance(value, torch.Tensor)
                                    and value.device != device
                                ):
                                    state[key] = value.to(device)
            except Exception as e:
                print(f"Failed to move optimizer states: {e}")

        # Handle optimizer-in-backward case
        if hasattr(self, "_optim_ckpt_wrapper") and self._optimizer_in_bwd:
            try:
                for param, optimizer in self._optim_ckpt_wrapper.optim_map.items():
                    for param_group in optimizer.param_groups:
                        for p in param_group["params"]:
                            if p in optimizer.state:
                                state = optimizer.state[p]
                                for key, value in state.items():
                                    if (
                                        isinstance(value, torch.Tensor)
                                        and value.device != device
                                    ):
                                        state[key] = value.to(device)
            except Exception as e:
                print(f"Failed to move optimizer-in-backward states: {e}")

        gc_and_empty_cuda_cache()

        self._current_device = device

        move_time = time.perf_counter() - move_start
        utils.log_rank_zero(
            self._logger,
            f"Completed move to {device} in {move_time:.2f} seconds",
        )

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
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
    recipe_cfg = RecipeConfig(**cfg_dict)  # type: ignore

    recipe = FullFinetuneRecipeDistributed(cfg=recipe_cfg)  # type: ignore
    recipe.setup(cfg=recipe_cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    setproctitle.setproctitle("model-service")
    sys.exit(recipe_main())
