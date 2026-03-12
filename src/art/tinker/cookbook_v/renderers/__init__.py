"""
Renderers for converting message lists into training and sampling prompts.

Use viz_sft_dataset to visualize the output of different renderers. E.g.,
    python -m tinker_cookbook.supervised.viz_sft_dataset dataset_path=Tulu3Builder renderer_name=role_colon
"""

from collections.abc import Callable
from typing import Any

from ..image_processing_utils import ImageProcessor
from ..tokenizer_utils import Tokenizer

# Types and utilities used by external code
from .base import (
    # Content part types
    ContentPart,
    ImagePart,
    Message,
    # Streaming types
    MessageDelta,
    # Renderer base
    RenderContext,
    Renderer,
    Role,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    Utf8TokenDecoder,
    # Utility functions
    ensure_text,
    format_content_as_string,
    get_text_content,
    parse_content_blocks,
)

# Renderer classes used directly by tests
from .deepseek_v3 import DeepSeekV3ThinkingRenderer
from .gpt_oss import GptOssRenderer
from .qwen3 import Qwen3Renderer

# Global registry for custom renderer factories
_CUSTOM_RENDERER_REGISTRY: dict[str, Callable[[Tokenizer, Any], Renderer]] = {}


def register_renderer(
    name: str,
    factory: Callable[[Tokenizer, Any], Renderer],
) -> None:
    """Register a custom renderer factory.

    Args:
        name: The renderer name
        factory: A callable that takes (tokenizer, image_processor) and returns a Renderer.

    Example:
        def my_renderer_factory(tokenizer, image_processor=None):
            return MyCustomRenderer(tokenizer)

        register_renderer("Foo/foo_renderer", my_renderer_factory)
    """
    _CUSTOM_RENDERER_REGISTRY[name] = factory


def get_registered_renderer_names() -> list[str]:
    """Return a list of all registered custom renderer names."""
    return list(_CUSTOM_RENDERER_REGISTRY.keys())


def is_renderer_registered(name: str) -> bool:
    """Check if a renderer name is registered."""
    return name in _CUSTOM_RENDERER_REGISTRY


def unregister_renderer(name: str) -> bool:
    """Unregister a custom renderer factory.

    Args:
        name: The renderer name to unregister.

    Returns:
        True if the renderer was unregistered, False if it wasn't registered.
    """
    if name in _CUSTOM_RENDERER_REGISTRY:
        del _CUSTOM_RENDERER_REGISTRY[name]
        return True
    return False


def get_renderer(
    name: str,
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
    model_name: str | None = None,
) -> Renderer:
    """Factory function to create renderers by name.

    Args:
        name: Renderer name. Supported values:
            - "role_colon": Simple role:content format
            - "llama3": Llama 3 chat format
            - "qwen3": Qwen3 with thinking enabled
            - "qwen3_vl": Qwen3 vision-language with thinking
            - "qwen3_vl_instruct": Qwen3 vision-language instruct (no thinking)
            - "qwen3_disable_thinking": Qwen3 with thinking disabled
            - "qwen3_instruct": Qwen3 instruct 2507 (no thinking)
            - "qwen3_5": Qwen3.5 VL with thinking
            - "qwen3_5_disable_thinking": Qwen3.5 VL with thinking disabled
            - "deepseekv3": DeepSeek V3 (defaults to non-thinking mode)
            - "deepseekv3_disable_thinking": DeepSeek V3 non-thinking (alias)
            - "deepseekv3_thinking": DeepSeek V3 thinking mode
            - "kimi_k2": Kimi K2 Thinking format
            - "kimi_k25": Kimi K2.5 with thinking enabled
            - "kimi_k25_disable_thinking": Kimi K2.5 with thinking disabled
            - "gpt_oss_no_sysprompt": GPT-OSS without system prompt
            - "gpt_oss_low_reasoning": GPT-OSS with low reasoning
            - "gpt_oss_medium_reasoning": GPT-OSS with medium reasoning
            - "gpt_oss_high_reasoning": GPT-OSS with high reasoning
            - Custom renderers registered via register_renderer()
        tokenizer: The tokenizer to use.
        image_processor: Required for VL renderers.
        model_name: Model name for pickle metadata. If None, falls back to
            ``tokenizer.name_or_path``. Provide this explicitly when the tokenizer
            was loaded with a remapped name (e.g., Llama 3 models).

    Returns:
        A Renderer instance.

    Raises:
        ValueError: If the renderer name is unknown.
        AssertionError: If a VL renderer is requested without an image_processor.
    """

    def _stamp_pickle_metadata(renderer: Renderer) -> Renderer:
        """Stamp renderer with metadata needed for pickle support."""
        renderer._renderer_name = name
        renderer._model_name = (
            model_name if model_name is not None else tokenizer.name_or_path
        )
        renderer._has_image_processor = image_processor is not None
        return renderer

    # Check custom registry first
    if (factory := _CUSTOM_RENDERER_REGISTRY.get(name)) is not None:
        return _stamp_pickle_metadata(factory(tokenizer, image_processor))

    # Import renderer classes lazily to avoid circular imports and keep exports minimal
    from .deepseek_v3 import DeepSeekV3DisableThinkingRenderer
    from .gpt_oss import GptOssRenderer
    from .kimi_k2 import KimiK2Renderer
    from .kimi_k25 import KimiK25DisableThinkingRenderer, KimiK25Renderer
    from .llama3 import Llama3Renderer
    from .qwen3 import (
        Qwen3DisableThinkingRenderer,
        Qwen3InstructRenderer,
        Qwen3VLInstructRenderer,
        Qwen3VLRenderer,
    )
    from .qwen3_5 import Qwen3_5DisableThinkingRenderer, Qwen3_5Renderer
    from .role_colon import RoleColonRenderer

    renderer: Renderer
    if name == "role_colon":
        renderer = RoleColonRenderer(tokenizer)
    elif name == "llama3":
        renderer = Llama3Renderer(tokenizer)
    elif name == "qwen3":
        renderer = Qwen3Renderer(tokenizer)
    elif name == "qwen3_vl":
        assert image_processor is not None, (
            "qwen3_vl renderer requires an image_processor"
        )
        renderer = Qwen3VLRenderer(tokenizer, image_processor)
    elif name == "qwen3_vl_instruct":
        assert image_processor is not None, (
            "qwen3_vl_instruct renderer requires an image_processor"
        )
        renderer = Qwen3VLInstructRenderer(tokenizer, image_processor)
    elif name == "qwen3_disable_thinking":
        renderer = Qwen3DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_instruct":
        renderer = Qwen3InstructRenderer(tokenizer)
    elif name == "qwen3_5":
        renderer = Qwen3_5Renderer(tokenizer, image_processor=image_processor)
    elif name == "qwen3_5_disable_thinking":
        renderer = Qwen3_5DisableThinkingRenderer(
            tokenizer, image_processor=image_processor
        )
    elif name == "deepseekv3":
        # Default to non-thinking mode (matches HF template default behavior)
        renderer = DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_disable_thinking":
        # Alias for backward compatibility
        renderer = DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_thinking":
        renderer = DeepSeekV3ThinkingRenderer(tokenizer)
    elif name == "kimi_k2":
        renderer = KimiK2Renderer(tokenizer)
    elif name == "kimi_k25":
        renderer = KimiK25Renderer(tokenizer, image_processor=image_processor)
    elif name == "kimi_k25_disable_thinking":
        renderer = KimiK25DisableThinkingRenderer(
            tokenizer, image_processor=image_processor
        )
    elif name == "gpt_oss_no_sysprompt":
        renderer = GptOssRenderer(tokenizer, use_system_prompt=False)
    elif name == "gpt_oss_low_reasoning":
        renderer = GptOssRenderer(
            tokenizer, use_system_prompt=True, reasoning_effort="low"
        )
    elif name == "gpt_oss_medium_reasoning":
        renderer = GptOssRenderer(
            tokenizer, use_system_prompt=True, reasoning_effort="medium"
        )
    elif name == "gpt_oss_high_reasoning":
        renderer = GptOssRenderer(
            tokenizer, use_system_prompt=True, reasoning_effort="high"
        )
    else:
        raise ValueError(
            f"Unknown renderer: {name}. If this is a custom renderer, please register it via register_renderer()."
        )

    return _stamp_pickle_metadata(renderer)


__all__ = [
    # Types
    "ContentPart",
    "ImagePart",
    "Message",
    "Role",
    "TextPart",
    "ThinkingPart",
    "ToolCall",
    "ToolSpec",
    # Streaming types
    "MessageDelta",
    "StreamingMessageHeader",
    "StreamingTextDelta",
    "StreamingThinkingDelta",
    "Utf8TokenDecoder",
    # Renderer base
    "RenderContext",
    "Renderer",
    "TrainOnWhat",
    # Utility functions
    "ensure_text",
    "format_content_as_string",
    "get_text_content",
    "parse_content_blocks",
    # Registry
    "register_renderer",
    "unregister_renderer",
    "get_registered_renderer_names",
    "is_renderer_registered",
    # Factory
    "get_renderer",
    # Renderer classes (used by tests)
    "DeepSeekV3ThinkingRenderer",
    "GptOssRenderer",
    "Qwen3Renderer",
]
