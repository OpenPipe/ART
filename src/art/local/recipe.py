# This is probably most similar to the recipe found in
# https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py

from functools import partial
import os
from omegaconf import DictConfig, ListConfig
import sys
import time
import torch
import torch.distributed
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.modules import TransformerDecoder
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing import Checkpointer
from torchtune.training.metric_logging import MetricLoggerInterface
from tqdm import tqdm
from typing import (
    Any,
    cast,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    overload,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

from .pack import PackedDataset, PackedTensors
from .grpo import GRPO, GRPOResult, shift_tensor

log = utils.get_logger("DEBUG")

T = TypeVar("T", covariant=True)
P = ParamSpec("P")


class ComponentConfig(DictConfig, Generic[T]):
    @overload
    def __init__(
        self,
        _component_: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...

    @overload
    def __init__(self, _component_: str, *args: Any, **kwargs: Any) -> None: ...

    def __init__(
        self, _component_: Union[Callable, str], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__({}, flags={"allow_objects": True})
        self._component_ = _component_
        if args:
            raise ValueError(
                "Positional arguments are not supported in ComponentConfig"
            )
        self.update(kwargs)

    def dict_config(self) -> DictConfig:
        return DictConfig(
            {
                "_component_": (
                    self._component_
                    if isinstance(self._component_, str)
                    else f"{self._component_.__module__}.{self._component_.__name__}"
                ),
                **{k: v for k, v in self.items() if k != "_component_"},
            }
        )


def instantiate_component(cfg: ComponentConfig[T], *args: Any, **kwargs: Any) -> T:
    if isinstance(cfg._component_, str):
        return config.instantiate(cfg, *args, **kwargs)
    _kwargs = {
        str(k): list(v) if isinstance(v, ListConfig) else v
        for k, v in cfg.items()
        if k != "_component_"
    }
    _kwargs.update(kwargs)
    return cfg._component_(*args, **_kwargs)


PLACEHOLDER: Any = None


class TuneRecipeConfig(DictConfig):
    def __init__(
        self,
        *,
        device: Optional[Union[str, torch.device]] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bf16",
        optimizer: ComponentConfig[Optimizer] = ComponentConfig(
            "torch.optim.AdamW", lr=2e-5, fused=True
        ),
        resume_from_checkpoint: bool = False,
        gradient_accumulation_steps: int = 1,
        checkpointer: ComponentConfig[Checkpointer] = PLACEHOLDER,
        seed: Optional[int] = None,
        epochs: int = 1,
        max_steps_per_epoch: Optional[int] = None,
        metric_logger: ComponentConfig[MetricLoggerInterface] = PLACEHOLDER,
        model: ComponentConfig[TransformerDecoder] = PLACEHOLDER,
        loss: ComponentConfig[GRPO] = ComponentConfig(GRPO),
        dataset: ComponentConfig[PackedDataset] = PLACEHOLDER,
        shuffle: bool = False,
        batch_size: int = 1,
        fsdp_cpu_offload: Optional[bool] = None,
        log_every_n_steps: Optional[int] = None,
        log_peak_memory_stats: Optional[bool] = None,
        log_grad_magnitude: Optional[bool] = None,
        optimizer_in_bwd: Optional[bool] = None,
        clip_grad_norm: Optional[Union[str, float]] = None,
        enable_activation_checkpointing: Optional[bool] = None,
        enable_activation_offloading: Optional[bool] = None,
        save_intermediate_checkpoints: Optional[bool] = None,
        reference_checkpointer: Optional[ComponentConfig[Checkpointer]] = None,
        compile: Optional[bool] = None,
        custom_sharded_layers: Optional[List[str]] = None,
        fsdp_reshard_after_forward: Optional[bool] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
        num_output_chunks: Optional[int] = None,
        profiler: Optional[ComponentConfig] = None,
    ) -> None:
        super().__init__({})
        self.device = device
        self.dtype = dtype
        self.optimizer = optimizer
        self.resume_from_checkpoint = resume_from_checkpoint
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpointer = checkpointer
        self.seed = seed
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.metric_logger = metric_logger
        self.model = model
        self.loss = loss
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if fsdp_cpu_offload is not None:
            self.fsdp_cpu_offload = fsdp_cpu_offload
        if log_every_n_steps is not None:
            self.log_every_n_steps = log_every_n_steps
        if log_peak_memory_stats is not None:
            self.log_peak_memory_stats = log_peak_memory_stats
        if log_grad_magnitude is not None:
            self.log_grad_magnitude = log_grad_magnitude
        if optimizer_in_bwd is not None:
            self.optimizer_in_bwd = optimizer_in_bwd
        if clip_grad_norm is not None:
            self.clip_grad_norm = clip_grad_norm
        if enable_activation_checkpointing is not None:
            self.enable_activation_checkpointing = enable_activation_checkpointing
        if enable_activation_offloading is not None:
            self.enable_activation_offloading = enable_activation_offloading
        if save_intermediate_checkpoints is not None:
            self.save_intermediate_checkpoints = save_intermediate_checkpoints
        if reference_checkpointer is not None:
            self.reference_checkpointer = reference_checkpointer
        if compile is not None:
            self.compile = compile
        if custom_sharded_layers is not None:
            self.custom_sharded_layers = custom_sharded_layers
        if fsdp_reshard_after_forward is not None:
            self.fsdp_reshard_after_forward = fsdp_reshard_after_forward
        if ac_mode is not None:
            self.ac_mode = ac_mode
        if ac_option is not None:
            self.ac_option = ac_option
        if num_output_chunks is not None:
            self.num_output_chunks = num_output_chunks
        if profiler is not None:
            self.profiler = profiler

    def dict_config(self) -> DictConfig:
        config = DictConfig({})
        for k, v in self.items():
            if isinstance(v, DictConfig) and "_component_" in v:
                v = v.copy()
                v["_component_"] = (
                    v["_component_"]
                    if isinstance(v["_component_"], str)
                    else f"{v['_component_'].__module__}.{v['_component_'].__name__}"
                )
                config[k] = v
            elif isinstance(v, torch.device):
                config[k] = str(v)
            elif isinstance(v, torch.dtype):
                config[k] = str(v)
            else:
                config[k] = v
        return config


class TypedDataLoader(DataLoader[T]):
    def __iter__(self) -> Iterator[T]:
        return super().__iter__()


class TuneRecipe(FTRecipeInterface):
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

    def __init__(self, cfg: TuneRecipeConfig) -> None:
        self._device = (
            cfg.device
            if isinstance(cfg.device, torch.device)
            else utils.get_device(device=cfg.device)
        )
        self._dtype = (
            cfg.dtype
            if isinstance(cfg.dtype, torch.dtype)
            else training.get_dtype(cfg.dtype, device=self._device)
        )

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError(
                "Using fused optimizer on CPU is only supported in PyTorch nightly."
            )

        # logging attributes
        self._log_every_n_steps: int = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats: bool = cfg.get("log_peak_memory_stats", False)
        self._log_grad_magnitude: bool = cfg.get("log_grad_magnitude", False)

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = training.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd: bool = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm: Optional[Union[str, float]] = cfg.get(
            "clip_grad_norm", None
        )

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
        self._enable_activation_checkpointing: bool = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading: bool = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type  # TODO: `model_type` type is not defined
            != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._save_intermediate_checkpoints = cfg.get(
            "save_intermediate_checkpoints", False
        )

    def load_checkpoint(
        self, cfg_checkpointer: ComponentConfig[Checkpointer]
    ) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = instantiate_component(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
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

    def setup(self, cfg: TuneRecipeConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = instantiate_component(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        if reference_checkpointer_cfg := cfg.get("reference_checkpointer", None):
            self.reference_model_state_dict = instantiate_component(
                reference_checkpointer_cfg
            ).load_checkpoint()[training.MODEL_KEY]
        else:
            self.reference_model_state_dict = None

        self._compile: bool = cfg.get("compile", False)
        if self._compile:
            torch.empty(1, device=self._device, requires_grad=True).backward()
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            reference_model_state_dict=self.reference_model_state_dict,
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )
        self._model.output_hidden_states = [len(self._model.layers) - 1]

        if self.reference_model_state_dict:
            # pin reference model state
            for value in self.reference_model_state_dict.values():
                if not isinstance(value, torch.distributed._tensor.DTensor):  # type: ignore
                    value.pin_memory()

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # initialize loss
        self._loss_fn = instantiate_component(cfg.loss)

        if self._compile:
            if self._is_rank_zero:
                log.info("Compiling loss with torch.compile...")
            self._loss_fn._forward_chunk = torch.compile(
                self._loss_fn._forward_chunk,
                backend=os.environ.get("TORCH_COMPILE_BACKEND", "inductor"),
            )

        if cfg.get("num_output_chunks", None) is not None:
            # set num_output_chunks for model
            self._model.set_num_output_chunks(cfg.num_output_chunks)

        if self._is_rank_zero:
            log.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
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

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        if self._is_rank_zero:
            log.info(f" Profiler config after instantiation: {profiler_cfg}")

            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: ComponentConfig[TransformerDecoder],
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        reference_model_state_dict: Optional[Dict[str, Any]] = None,
        custom_sharded_layers: Optional[List[str]] = None,
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

        if self._is_rank_zero:
            log.info(
                "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ..."
            )
            init_start = time.perf_counter()
        else:
            init_start = 0.0

        with (
            training.set_default_dtype(self._dtype),
            torch.device("meta") if training.is_distributed() else self._device,
        ):
            model = instantiate_component(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

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

        if training.is_distributed():
            # For FSDP sharding
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]
            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
            )

            with training.set_default_dtype(self._dtype), self._device:
                for m in model.modules():
                    # RoPE is not covered in state dict
                    if hasattr(m, "rope_init"):
                        m.rope_init()  # type: ignore

            # This method will convert the full model state dict into a sharded state
            # dict and load into the model
            training.load_from_full_model_state_dict(
                model,
                model_state_dict,
                self._device,
                self._is_rank_zero,
                strict=True,
                cpu_offload=fsdp_cpu_offload,
            )

            if reference_model_state_dict:
                # Temporarily patch model.load_state_dict to capture the sharded parameter tensors
                # for this rank when loading the reference model. This allows us to maintain a
                # reference copy of the sharded parameters that matches the FSDP sharding pattern,
                # which is needed for weight swapping during training.
                load_state_dict = model.load_state_dict

                def patch(
                    state_dict: Mapping[str, Any],
                    strict: bool = True,
                    assign: bool = False,
                ) -> Any:
                    reference_model_state_dict.clear()
                    reference_model_state_dict.update(state_dict)

                model.load_state_dict = patch

                training.load_from_full_model_state_dict(
                    model,
                    reference_model_state_dict,
                    self._device,
                    self._is_rank_zero,
                    strict=True,
                    cpu_offload=fsdp_cpu_offload,
                )

                model.load_state_dict = load_state_dict
        else:
            model.load_state_dict(model_state_dict)

            # Validate model was loaded in with the expected dtype.
            training.validate_expected_param_dtype(
                model.named_parameters(), dtype=self._dtype
            )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero:
            log.info(
                f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs"
            )
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        if training.is_distributed():
            # synchronize before training begins
            torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: ComponentConfig[Optimizer],
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: instantiate_component(cfg_optimizer, params=[param])
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
                            self._optim_ckpt_wrapper.state_dict()[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            if self._is_rank_zero:
                log.info("In-backward optimizers are set up.")
            return None
        else:
            optimizer = instantiate_component(
                cfg_optimizer, params=self._model.parameters()
            )
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            if self._is_rank_zero:
                log.info("Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self,
        cfg_dataset: ComponentConfig[PackedDataset],
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, TypedDataLoader[PackedTensors]]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = training.get_world_size_and_rank()

        ds = instantiate_component(cfg_dataset)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.seed or 0
        )
        dataloader = TypedDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=len(ds) > 1,
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the model weights and recipe state in
        different checkpoint files. To correctly resume training from an intermediate checkpoint,
        the model weights and recipe state must be provided.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        if intermediate_checkpoint and not self._save_intermediate_checkpoints:
            return

        if self._is_rank_zero:
            log.info(
                "Saving checkpoint. This may take some time. Retrieving full model state dict..."
            )
            start = time.perf_counter()
        else:
            start = 0.0

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        model_state_dict = (
            training.gather_cpu_state_dict(
                self._model.state_dict(),
                self._is_rank_zero,
                device=self._device,
            )
            if training.is_distributed()
            else self._model.state_dict()
        )

        if self._is_rank_zero:
            log.info(
                f"Getting full model state dict took {time.perf_counter() - start:.2f} secs"
            )

        if intermediate_checkpoint:
            start = time.perf_counter()
            if self._is_rank_zero:
                log.info("Getting optimizer state dict...")
            if self._optimizer:
                opt_state_dict = (
                    training.get_full_optimizer_state_dict(
                        self._optimizer,
                        self._is_rank_zero,
                        device=self._device,
                    )
                    if training.is_distributed()
                    else self._optimizer.state_dict()
                )
            else:
                opt_state_dict = {}
                for param, opt in self._optim_ckpt_wrapper.optim_map.items():
                    opt_state_dict[param] = (
                        training.get_full_optimizer_state_dict(
                            opt, self._is_rank_zero, device=self._device
                        )
                        if training.is_distributed()
                        else opt.state_dict()
                    )
            if self._is_rank_zero:
                log.info(
                    f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs"
                )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file

        if self._is_rank_zero:
            start = time.perf_counter()
            checkpoint_dict.update({training.MODEL_KEY: model_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        if training.is_distributed():
            torch.distributed.barrier()

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()
        torch.autograd.set_detect_anomaly(True)

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if self._optimizer:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_result = GRPOResult().to(self._device)

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
            grad_norm: torch.Tensor | None = None
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)  # type: ignore - `batch_to_device` expects a `dict`, not a `TypedDict`, but this should be fine

                # Assume the first token in the batch is the bos token
                bos_id = int(batch["tokens"].view(-1)[0].item())

                # Create grouped causal mask
                batch_size, seq_len = batch["tokens"].size()
                causal_mask = (
                    torch.tril(
                        torch.ones(
                            seq_len, seq_len, dtype=torch.bool, device=self._device
                        )
                    )
                    .unsqueeze(0)
                    .expand(batch_size, seq_len, seq_len)
                )
                group_mask = batch["group_ids"].unsqueeze(2) == batch[
                    "group_ids"
                ].unsqueeze(1)
                parent_mask = batch["parent_ids"].unsqueeze(2) == batch[
                    "group_ids"
                ].unsqueeze(1)
                mask = causal_mask & (group_mask | parent_mask)

                if self.reference_model_state_dict:
                    # Save current weights and load reference weights
                    model_state_dict = self._swap_state(self.reference_model_state_dict)

                    # Run reference model forward pass without affecting autograd
                    with torch.no_grad(), self.activations_handling_ctx:
                        hidden_states, logits = self._model(
                            tokens=batch["tokens"],
                            mask=mask,
                            input_pos=batch["input_pos"],
                        )
                        del hidden_states
                        if isinstance(logits, list):
                            reference_logprobs = torch.cat(
                                [
                                    torch.distributions.Categorical(
                                        logits=logits_chunk
                                    ).log_prob(
                                        shift_tensor(tokens, ignore_label=bos_id)
                                    )
                                    for logits_chunk, tokens in zip(
                                        logits,
                                        batch["tokens"].chunk(len(logits), dim=1),
                                    )
                                ],
                                dim=-1,
                            )
                        else:
                            reference_logprobs = cast(
                                torch.Tensor,
                                torch.distributions.Categorical(logits=logits).log_prob(
                                    shift_tensor(batch["tokens"], ignore_label=bos_id)
                                ),
                            )
                        del logits

                    # Restore original weights
                    self._swap_state(model_state_dict)
                else:
                    reference_logprobs = None

                with self.activations_handling_ctx:
                    hidden_states, logits = self._model(
                        tokens=batch["tokens"],
                        mask=mask,
                        input_pos=batch["input_pos"],
                    )
                del mask, batch["input_pos"], hidden_states  # type: ignore

                # Compute loss
                current_result = self._loss_fn.forward(
                    logits=logits,
                    tokens=batch["tokens"],
                    advantages=batch["advantages"],
                    logprobs=batch["logprobs"],
                    reference_logprobs=reference_logprobs,
                    mask=batch["assistant_mask"],
                    weights=batch["weights"],
                    bos_id=bos_id,
                )
                del logits, batch

                running_result += current_result

                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    if training.is_distributed():
                        for tensor in running_result.tensors():
                            torch.distributed.all_reduce(tensor)
                    current_loss = current_result.total_loss / current_result.num_tokens
                else:
                    current_loss = current_result.total_loss

                current_loss.backward()
                del current_loss

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if self._optimizer:
                        if training.is_distributed():
                            for tensor in running_result.tensors():
                                torch.distributed.all_reduce(tensor)
                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        training.scale_grads(self._model, 1 / running_result.num_tokens)

                        # Calculate gradient magnitude (L2 norm of all gradients)
                        grad_magnitude = None
                        if self._log_grad_magnitude:
                            grad_magnitude = torch.norm(
                                torch.stack(
                                    [
                                        torch.norm(p.grad.detach())
                                        for p in self._model.parameters()
                                        if p.grad is not None
                                    ]
                                )
                            )

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    per_token_result = running_result.per_token()
                    loss_to_log = per_token_result.total_loss.item()
                    policy_loss_to_log = per_token_result.policy_loss.item()
                    entropy_to_log = per_token_result.entropy.item()
                    kl_div_to_log = per_token_result.kl_div.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log:.4f}"
                    )
                    postfix = {
                        "loss": loss_to_log,
                        "policy": policy_loss_to_log,
                        "entropy": entropy_to_log,
                        "kl_div": kl_div_to_log,
                    }
                    if self._log_grad_magnitude and grad_magnitude is not None:
                        postfix["grad_magnitude"] = grad_magnitude.item()
                    pbar.set_postfix(postfix)

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "policy": policy_loss_to_log,
                            "entropy": entropy_to_log,
                            "kl_div": kl_div_to_log,
                            # "lr": get_lr(self._optimizer or self._optim_ckpt_wrapper),
                            "tokens_per_second_per_gpu": running_result.num_tokens
                            / (time_per_step * world_size),
                        }
                        if self._log_grad_magnitude and grad_magnitude is not None:
                            log_dict["grad_magnitude"] = grad_magnitude.item()
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
                    del running_result
                    running_result = GRPOResult().to(self._device)
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
                    ):
                        torch.cuda.memory._record_memory_history(
                            # Pylance infers the type of `enabled` as `str` though the function accepts `Literal[None, "state", "all"]`
                            enabled=None  # type: ignore
                        )

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        if training.is_distributed():
            torch.distributed.destroy_process_group()
        training.cleanup_before_training()

    def _swap_state(
        self, state_dict: Dict[str, Any], assign: bool = False
    ) -> Dict[str, Any]:
        """
        Swaps the current model state with the provided state dict.
        Manages GPU memory by moving states to CPU/device in the right order.

        Args:
            state_dict: Dictionary of state to load into model

        Returns:
            Original model state dict (moved to CPU)
        """
        # Save current model state and move to CPU
        current_state = {
            k: v.to("cpu", non_blocking=True)
            for k, v in self._model.state_dict().items()
        }

        # Move input state to device and load
        device_state = {
            k: v.to(self._device, non_blocking=True) for k, v in state_dict.items()
        }
        self._model.load_state_dict(device_state, assign=assign)

        return current_state


def recipe_main(cfg: TuneRecipeConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        log.debug(
            "Training is not distributed. If you want to train on multiple GPUs and are using the tune CLI, specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    elif not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo")

    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(
        recipe_name="FullFinetuneRecipe",
        cfg=cfg.dict_config() if isinstance(cfg, TuneRecipeConfig) else cfg,
    )

    recipe = TuneRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(config.parse(recipe_main)())  # type: ignore
