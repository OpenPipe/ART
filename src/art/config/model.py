import torch
from typing import Literal, TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from transformers.training_args import (
        DebugOption,
        FSDPOption,
        HubStrategy,
        IntervalStrategy,
        OptimizerNames,
        SaveStrategy,
        SchedulerType,
    )

    from .. import types


def get_model_config(
    base_model: "types.BaseModel", output_dir: str, config: "ModelConfig | None"
) -> "ModelConfig":
    from ..unsloth.checkpoints import get_last_iteration_dir

    if config is None:
        config = ModelConfig()
    enable_sleep_mode = config.get("init_args", {}).get("enable_sleep_mode", True)
    init_args = InitArgs(
        model_name=base_model,
        max_seq_length=32768,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_requests=True,
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(
            0.79 if enable_sleep_mode else 0.55
        ),  # Reduce if out of memory
        max_lora_rank=8,
        # Multi-step processing is not supported for the Xformers attention backend
        # which is the fallback for devices with compute capability < 8.0
        num_scheduler_steps=16 if torch.cuda.get_device_capability()[0] >= 8 else 1,
        enable_sleep_mode=enable_sleep_mode,
        use_async=True,
    )
    init_args.update(config.get("init_args", {}))
    if lora_path := get_last_iteration_dir(output_dir):
        init_args["model_name"] = lora_path
    peft_args = PeftArgs(
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=16,
        # Enable long context finetuning
        use_gradient_checkpointing="unsloth",  # type: ignore
        random_state=3407,
    )
    peft_args.update(config.get("peft_args", {}))
    train_args = TrainArgs(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        save_strategy="no",
        output_dir=output_dir,
        disable_tqdm=True,
        report_to="none",
    )
    train_args.update(config.get("train_args", {}))
    return ModelConfig(init_args=init_args, peft_args=peft_args, train_args=train_args)


class ModelConfig(TypedDict, total=False):
    """
    Model configuration.

    Args:
        init: Arguments for initializing an Unsloth FastLanguageModel.
        peft: Arguments for creating an Unsloth PEFT model wrapper.
        train: Arguments for the GRPO trainer.
    """

    init_args: "InitArgs"
    peft_args: "PeftArgs"
    train_args: "TrainArgs"


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
    disable_log_requests: bool
    disable_log_stats: bool
    enable_prefix_caching: bool
    enforce_eager: bool
    num_scheduler_steps: int
    enable_sleep_mode: bool
    use_async: bool


class PeftArgs(TypedDict, total=False):
    r: int
    target_modules: list[str]
    lora_alpha: int
    lora_dropout: int
    bias: str
    layers_to_transform: list[int] | None
    layers_pattern: str | None
    use_gradient_checkpointing: bool
    random_state: int
    max_seq_length: int  # not used anymore
    use_rslora: bool
    modules_to_save: list[str] | None
    init_lora_weights: bool
    loftq_config: dict
    temporary_location: str


class TrainArgs(TypedDict, total=False):
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
    log_level: str | None
    log_level_replica: str | None
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
    debug: str | list["DebugOption"]
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
    fsdp: list["FSDPOption"] | str | None
    fsdp_min_num_params: int
    fsdp_config: dict | str | None
    fsdp_transformer_layer_cls_to_wrap: str | None
    accelerator_config: dict | str | None
    deepspeed: dict | str | None
    label_smoothing_factor: float
    optim: "OptimizerNames | str"
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
    evaluation_strategy: "IntervalStrategy | str"
    push_to_hub_model_id: str | None
    push_to_hub_organization: str | None
    push_to_hub_token: str | None
    mp_parameters: str
    auto_find_batch_size: bool
    full_determinism: bool
    torchdynamo: str | None
    ray_scope: str | None
    ddp_timeout: int | None
    torch_compile: bool
    torch_compile_backend: str | None
    torch_compile_mode: str | None
    dispatch_batches: bool | None
    split_batches: bool | None
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
    temperature: float | None
    max_completion_length: int | None
    ds3_gather_for_generation: bool
    use_vllm: bool | None
    vllm_device: str | None
    vllm_gpu_memory_utilization: float
    vllm_dtype: str | None
    vllm_max_model_len: int | None
    beta: float
    reward_weights: list[float] | None
    sync_ref_model: bool
    ref_model_mixup_alpha: float
    ref_model_sync_steps: int
    log_completions: bool
