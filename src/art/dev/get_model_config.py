from .engine import EngineArgs
from .model import InitArgs, InternalModelConfig, PeftArgs, TrainerArgs


def get_model_config(
    base_model: str,
    output_dir: str,
    config: "InternalModelConfig | None",
) -> "InternalModelConfig":
    from ..local.checkpoints import get_last_checkpoint_dir

    if config is None:
        config = InternalModelConfig()

    enable_sleep_mode = config.get("engine_args", {}).get("enable_sleep_mode", True)
    init_args = InitArgs(
        fast_inference=False,
        load_in_4bit=True,
        max_seq_length=32768,
        model_name=base_model,
    )
    engine_args = EngineArgs(
        allowed_local_media_path="/tmp",
        enable_sleep_mode=enable_sleep_mode,
        generation_config="vllm",
        model=base_model,
    )
    engine_args.update(config.get("engine_args", {}))
    init_args.update(config.get("init_args", {}))
    if last_checkpoint_dir := get_last_checkpoint_dir(output_dir):
        init_args["model_name"] = last_checkpoint_dir
    peft_args = PeftArgs(
        lora_alpha=16,
        r=8,
        random_state=3407,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )
    peft_args.update(config.get("peft_args", {}))
    trainer_args = TrainerArgs(
        adam_beta1=0.9,
        adam_beta2=0.99,
        disable_tqdm=True,
        gradient_accumulation_steps=1,
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
    return InternalModelConfig(
        init_args=init_args,
        engine_args=engine_args,
        peft_args=peft_args,
        tinker_args=config.get("tinker_args"),
        trainer_args=trainer_args,
    )
