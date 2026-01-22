"""Model-specific configuration for chat templates and training defaults."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SFTDefaults:
    """Default SFT training parameters for a model."""

    batch_size: int
    learning_rate: float


@dataclass
class ModelConfig:
    """Configuration for a specific model's chat template."""

    instruction_part: str
    response_part: str
    sft_defaults: Optional[SFTDefaults] = None


# Model identifier -> configuration mapping
# These define the chat template markers used for "train on responses only"
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Qwen 2.5 models (ChatML format)
    "Qwen/Qwen2.5-0.5B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    "Qwen/Qwen2.5-14B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        sft_defaults=SFTDefaults(batch_size=2, learning_rate=1e-4),
    ),
    "Qwen/Qwen2.5-32B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    "Qwen/Qwen2.5-72B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    # Qwen 3 models (with thinking tokens)
    "Qwen/Qwen3-8B": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    "Qwen/Qwen3-14B": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    "Qwen/Qwen3-32B": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    "OpenPipe/Qwen3-14B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        sft_defaults=SFTDefaults(batch_size=2, learning_rate=1e-4),
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        sft_defaults=SFTDefaults(batch_size=2, learning_rate=1e-4),
    ),
    # Llama 3 models
    "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "meta-llama/Llama-3.2-1B-Instruct": ModelConfig(
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelConfig(
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    # Gemma models
    "google/gemma-2-2b-it": ModelConfig(
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    ),
    "google/gemma-2-9b-it": ModelConfig(
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    ),
    "google/gemma-2-27b-it": ModelConfig(
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    ),
}


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get the configuration for a given model.

    Args:
        model_id: The model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        ModelConfig if found, None otherwise
    """
    return MODEL_CONFIGS.get(model_id)


def detect_chat_template_parts(
    tokenizer_or_template: object,
) -> tuple[str, str]:
    """Detect instruction and response parts from a chat template string.

    This is a fallback when the model is not in MODEL_CONFIGS.

    Args:
        tokenizer_or_template: Either a tokenizer with chat_template attr,
                              or the chat template string directly

    Returns:
        Tuple of (instruction_part, response_part)
    """
    if hasattr(tokenizer_or_template, "chat_template"):
        template: str = getattr(tokenizer_or_template, "chat_template", "") or ""
    elif isinstance(tokenizer_or_template, str):
        template = tokenizer_or_template
    else:
        template = ""

    # ChatML format (Qwen, etc.)
    if "<|im_start|>" in template:
        return "<|im_start|>user\n", "<|im_start|>assistant\n"

    # Llama 3 format
    if "<|start_header_id|>" in template:
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    # Gemma format
    if "<start_of_turn>" in template:
        return "<start_of_turn>user\n", "<start_of_turn>model\n"

    # Mistral format
    if "[INST]" in template:
        return "[INST]", "[/INST]"

    # Default fallback to ChatML (most common)
    return "<|im_start|>user\n", "<|im_start|>assistant\n"


def get_instruction_response_parts(
    model_id: str,
    tokenizer: Optional[object] = None,
) -> tuple[str, str]:
    """Get instruction and response parts for a model.

    First checks MODEL_CONFIGS, then falls back to template detection.

    Args:
        model_id: The model identifier
        tokenizer: Optional tokenizer for fallback detection

    Returns:
        Tuple of (instruction_part, response_part)
    """
    # Check explicit config first
    config = get_model_config(model_id)
    if config is not None:
        return config.instruction_part, config.response_part

    # Fallback to detection
    if tokenizer is not None:
        return detect_chat_template_parts(tokenizer)

    # Ultimate fallback
    return detect_chat_template_parts("")
