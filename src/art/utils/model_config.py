"""Model-specific configuration for chat templates and training defaults."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model's chat template."""

    instruction_part: str
    response_part: str


# Explicit model configurations for models that can't be auto-detected.
# Models not listed here will fall back to auto-detection from the tokenizer's chat_template.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Qwen3 with thinking disabled - always includes empty <think> tags
    "OpenPipe/Qwen3-14B-Instruct": ModelConfig(
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
}


def detect_chat_template_parts(
    tokenizer: object,
) -> tuple[str, str]:
    """Detect instruction and response parts from a tokenizer's chat template.

    Args:
        tokenizer: A tokenizer with a chat_template attribute

    Returns:
        Tuple of (instruction_part, response_part)

    Raises:
        ValueError: If the tokenizer has no chat_template or the format is unrecognized
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or not isinstance(template, str):
        raise ValueError(
            "Cannot detect chat template parts: tokenizer has no chat_template attribute. "
            "Please specify instruction_part and response_part manually."
        )

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

    raise ValueError(
        f"Unrecognized chat template format. "
        f"Please specify instruction_part and response_part manually. "
        f"Template starts with: {template[:100]!r}..."
    )


def get_instruction_response_parts(
    model_id: str,
    tokenizer: object,
) -> tuple[str, str]:
    """Get instruction and response parts for a model.

    Checks for explicit model configuration first, then falls back to
    auto-detection from the tokenizer's chat template.

    Args:
        model_id: The model identifier
        tokenizer: Tokenizer with chat_template attribute

    Returns:
        Tuple of (instruction_part, response_part)

    Raises:
        ValueError: If chat template cannot be detected and model has no explicit config
    """
    # Check for explicit model configuration first
    if model_id in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_id]
        return config.instruction_part, config.response_part

    # Fall back to auto-detection
    try:
        return detect_chat_template_parts(tokenizer)
    except ValueError as e:
        raise ValueError(f"Failed to detect chat template for {model_id}: {e}") from e
