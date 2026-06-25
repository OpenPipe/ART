from ..megatron.model_support import default_target_modules_for_model
from .engine import EngineArgs
from .model import (
    PEFT_ARGS_MIGRATION_MESSAGE,
    BackendModelConfig,
    InitArgs,
    InternalModelConfig,
    LoRAConfig,
    TrainerArgs,
)
from .sequence_lengths import max_seq_length_from_model_config
from .validate import is_dedicated_mode


def default_target_modules(base_model: str) -> list[str]:
    return default_target_modules_for_model(base_model, allow_unvalidated_arch=True)


def get_model_config(
    base_model: str,
    output_dir: str,
    config: "InternalModelConfig | None",
    lora_config: "LoRAConfig | None" = None,
) -> "BackendModelConfig":
    from ..local.checkpoints import get_last_checkpoint_dir

    if config is None:
        config = InternalModelConfig()
    if "peft_args" in config:
        raise ValueError(PEFT_ARGS_MIGRATION_MESSAGE)

    dedicated = is_dedicated_mode(config)
    rollout_weights_mode = config.get("rollout_weights_mode", "lora")

    if dedicated:
        enable_sleep_mode = False
    else:
        enable_sleep_mode = config.get("engine_args", {}).get("enable_sleep_mode", True)

    configured_init_args = config.get("init_args", {})
    init_args = InitArgs(
        load_in_4bit=True,
        max_seq_length=max_seq_length_from_model_config(
            base_model,
            revision=configured_init_args.get("revision"),
            token=configured_init_args.get("token"),
        ),
        model_name=base_model,
    )
    engine_args = EngineArgs(
        allowed_local_media_path="/tmp",
        enable_sleep_mode=enable_sleep_mode,
        generation_config="vllm",
        model=base_model,
    )
    engine_args.update(config.get("engine_args", {}))
    init_args.update(configured_init_args)
    if last_checkpoint_dir := get_last_checkpoint_dir(output_dir):
        init_args["model_name"] = last_checkpoint_dir
    merged_lora_config = LoRAConfig(
        random_state=3407,
        target_modules=default_target_modules(base_model),
        use_gradient_checkpointing="unsloth",
    )
    if lora_config:
        merged_lora_config.update(lora_config)
    if rollout_weights_mode == "lora" and "lora_target_modules" not in config.get(
        "engine_args", {}
    ):
        engine_args["lora_target_modules"] = merged_lora_config["target_modules"]
    trainer_args = TrainerArgs(
        adam_beta1=0.9,
        adam_beta2=0.99,
        disable_tqdm=True,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        logging_steps=1,
        lr_scheduler_type="constant",
        max_grad_norm=0.1,
        num_generations=2,
        optim="paged_adamw_8bit",
        output_dir=output_dir,
        per_device_train_batch_size=2,
        report_to="none",
        save_strategy="no",
        weight_decay=0.1,
    )
    trainer_args.update(config.get("trainer_args", {}))
    result = BackendModelConfig(
        init_args=init_args,
        engine_args=engine_args,
        lora_config=merged_lora_config,
        rollout_weights_mode=rollout_weights_mode,
        tinker_args=config.get("tinker_args"),
        trainer_args=trainer_args,
    )
    if "allow_unvalidated_arch" in config:
        result["allow_unvalidated_arch"] = config["allow_unvalidated_arch"]
    if "trainer_gpu_ids" in config:
        result["trainer_gpu_ids"] = config["trainer_gpu_ids"]
    if "inference_gpu_ids" in config:
        result["inference_gpu_ids"] = config["inference_gpu_ids"]
    return result
