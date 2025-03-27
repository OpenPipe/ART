from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizerBase
from unsloth.models import FastLanguageModel  # type: ignore


def get_model_and_tokenizer(
    model_name: str,
    lora_rank: int = 32,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_requests=True,
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.62,  # Reduce if out of memory
        max_lora_rank=lora_rank,
        num_scheduler_steps=16,
        use_async=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        # Enable long context finetuning
        use_gradient_checkpointing="unsloth",  # type: ignore
        random_state=3407,
    )

    return model, tokenizer
