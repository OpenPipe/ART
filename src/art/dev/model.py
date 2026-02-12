from enum import Enum

from typing_extensions import Required, TypedDict

from .engine import EngineArgs


# Vendored from transformers.training_args.OptimizerNames
class OptimizerNames(str, Enum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    ADAMW_TORCH_4BIT = "adamw_torch_4bit"
    ADAMW_TORCH_8BIT = "adamw_torch_8bit"
    ADEMAMIX = "ademamix"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    ADEMAMIX_8BIT = "ademamix_8bit"
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADEMAMIX = "paged_ademamix_32bit"
    PAGED_ADEMAMIX_8BIT = "paged_ademamix_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"
    RMSPROP_BNB = "rmsprop_bnb"
    RMSPROP_8BIT = "rmsprop_bnb_8bit"
    RMSPROP_32BIT = "rmsprop_bnb_32bit"
    GALORE_ADAMW = "galore_adamw"
    GALORE_ADAMW_8BIT = "galore_adamw_8bit"
    GALORE_ADAFACTOR = "galore_adafactor"
    GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
    GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
    GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"
    LOMO = "lomo"
    ADALOMO = "adalomo"
    GROKADAMW = "grokadamw"
    SCHEDULE_FREE_RADAM = "schedule_free_radam"
    SCHEDULE_FREE_ADAMW = "schedule_free_adamw"
    SCHEDULE_FREE_SGD = "schedule_free_sgd"
    APOLLO_ADAMW = "apollo_adamw"
    APOLLO_ADAMW_LAYERWISE = "apollo_adamw_layerwise"


# Vendored from transformers.debug_utils.DebugOption
class DebugOption(str, Enum):
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"


# Vendored from transformers.trainer_utils.IntervalStrategy
class IntervalStrategy(str, Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


# Vendored from transformers.trainer_utils.SaveStrategy (which is an alias for IntervalStrategy)
SaveStrategy = IntervalStrategy


# Vendored from transformers.trainer_utils.HubStrategy
class HubStrategy(str, Enum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


# Vendored from transformers.trainer_utils.SchedulerType
class SchedulerType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"
    WORMHOLE = "wormhole"


# Vendored from transformers.trainer_utils.FSDPOption
class FSDPOption(str, Enum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"
    HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


class InternalModelConfig(TypedDict, total=False):
    """
    Model configuration.

    Args:
        init: Arguments for initializing an Unsloth FastLanguageModel.
        engine: Arguments for the vLLM engine.
        peft: Arguments for creating an Unsloth PEFT model wrapper.
        tinker: Arguments for the Tinker training client.
        trainer: Arguments for the GRPO trainer.
    """

    init_args: "InitArgs"
    engine_args: "EngineArgs"
    peft_args: "PeftArgs"
    tinker_args: "TinkerArgs | None"
    tinker_native_args: "TinkerNativeArgs | None"
    trainer_args: "TrainerArgs"


class TinkerArgs(TypedDict, total=False):
    renderer_name: Required[str]
    training_client_args: "TinkerTrainingClientArgs"


class TinkerNativeArgs(TypedDict, total=False):
    renderer_name: Required[str]
    training_client_args: "TinkerTrainingClientArgs"


class TinkerTrainingClientArgs(TypedDict, total=False):
    rank: int
    seed: int | None
    train_mlp: bool
    train_attn: bool
    train_unembed: bool
    user_metadata: dict[str, str] | None


class InitArgs(TypedDict, total=False):
    model_name: str
    max_seq_length: int
    dtype: str | None
    load_in_4bit: bool
    load_in_8bit: bool
    full_finetuning: bool
    token: str | None
    device_map: str
    rope_scaling: dict | None
    fix_tokenizer: bool
    trust_remote_code: bool
    use_gradient_checkpointing: str
    resize_model_vocab: int | None
    revision: str | None
    use_exact_model_name: bool
    fast_inference: bool
    gpu_memory_utilization: float
    float8_kv_cache: bool
    random_state: int
    max_lora_rank: int
    disable_log_stats: bool
    enable_prefix_caching: bool
    use_async: bool


class PeftArgs(TypedDict, total=False):
    r: int
    target_modules: list[str]
    lora_alpha: int
    lora_dropout: int
    bias: str
    layers_to_transform: list[int] | None
    layers_pattern: str | None
    use_gradient_checkpointing: str
    random_state: int
    max_seq_length: int
    use_rslora: bool
    modules_to_save: list[str] | None
    init_lora_weights: bool
    loftq_config: dict
    temporary_location: str


class TrainerArgs(TypedDict, total=False):
    output_dir: str | None
    overwrite_output_dir: bool
    do_train: bool
    do_eval: bool
    do_predict: bool
    eval_strategy: "IntervalStrategy | str"
    prediction_loss_only: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    per_gpu_train_batch_size: int | None
    per_gpu_eval_batch_size: int | None
    gradient_accumulation_steps: int
    eval_accumulation_steps: int | None
    eval_delay: float | None
    torch_empty_cache_steps: int | None
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    num_train_epochs: float
    max_steps: int
    lr_scheduler_type: "SchedulerType | str"
    lr_scheduler_kwargs: dict | str | None
    warmup_ratio: float
    warmup_steps: int
    log_level: str
    log_level_replica: str
    log_on_each_node: bool
    logging_dir: str | None
    logging_strategy: "IntervalStrategy | str"
    logging_first_step: bool
    logging_steps: float
    logging_nan_inf_filter: bool
    save_strategy: "SaveStrategy | str"
    save_steps: float
    save_total_limit: int | None
    save_safetensors: bool | None
    save_on_each_node: bool
    save_only_model: bool
    restore_callback_states_from_checkpoint: bool
    no_cuda: bool
    use_cpu: bool
    use_mps_device: bool
    seed: int
    data_seed: int | None
    jit_mode_eval: bool
    use_ipex: bool
    bf16: bool
    fp16: bool
    fp16_opt_level: str
    half_precision_backend: str
    bf16_full_eval: bool
    fp16_full_eval: bool
    tf32: bool | None
    local_rank: int
    ddp_backend: str | None
    tpu_num_cores: int | None
    tpu_metrics_debug: bool
    debug: str | list[DebugOption]
    dataloader_drop_last: bool
    eval_steps: float | None
    dataloader_num_workers: int
    dataloader_prefetch_factor: int | None
    past_index: int
    run_name: str | None
    disable_tqdm: bool | None
    remove_unused_columns: bool | None
    label_names: list[str] | None
    load_best_model_at_end: bool | None
    metric_for_best_model: str | None
    greater_is_better: bool | None
    ignore_data_skip: bool
    fsdp: list[FSDPOption] | str | None
    fsdp_min_num_params: int
    fsdp_config: dict | str | None
    fsdp_transformer_layer_cls_to_wrap: str | None
    accelerator_config: dict | str | None
    deepspeed: dict | str | None
    label_smoothing_factor: float
    optim: OptimizerNames | str
    optim_args: str | None
    adafactor: bool
    group_by_length: bool
    length_column_name: str | None
    report_to: str | list[str] | None
    ddp_find_unused_parameters: bool | None
    ddp_bucket_cap_mb: int | None
    ddp_broadcast_buffers: bool | None
    dataloader_pin_memory: bool
    dataloader_persistent_workers: bool
    skip_memory_metrics: bool
    use_legacy_prediction_loop: bool
    push_to_hub: bool
    resume_from_checkpoint: str | None
    hub_model_id: str | None
    hub_strategy: "HubStrategy | str"
    hub_token: str | None
    hub_private_repo: bool | None
    hub_always_push: bool
    gradient_checkpointing: bool
    gradient_checkpointing_kwargs: dict | str | None
    include_inputs_for_metrics: bool
    include_for_metrics: list[str]
    eval_do_concat_batches: bool
    fp16_backend: str
    push_to_hub_model_id: str | None
    push_to_hub_organization: str | None
    push_to_hub_token: str | None
    mp_parameters: str
    auto_find_batch_size: bool
    full_determinism: bool
    torchdynamo: str | None
    ray_scope: str | None
    ddp_timeout: int
    torch_compile: bool
    torch_compile_backend: str | None
    torch_compile_mode: str | None
    include_tokens_per_second: bool | None
    include_num_input_tokens_seen: bool | None
    neftune_noise_alpha: float | None
    optim_target_modules: str | list[str] | None
    batch_eval_metrics: bool
    eval_on_start: bool
    use_liger_kernel: bool | None
    eval_use_gather_object: bool | None
    average_tokens_across_devices: bool | None
    model_init_kwargs: dict | None
    max_prompt_length: int | None
    num_generations: int | None
    temperature: float
    max_completion_length: int | None
    ds3_gather_for_generation: bool
    beta: float
    reward_weights: list[float] | None
    sync_ref_model: bool
    ref_model_mixup_alpha: float
    ref_model_sync_steps: int
    log_completions: bool
