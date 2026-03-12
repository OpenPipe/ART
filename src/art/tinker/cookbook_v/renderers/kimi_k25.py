"""Renderer for Moonshot AI's Kimi K2.5 models."""

from typing import cast

import tinker

from ..image_processing_utils import ImageProcessor
from ..tokenizer_utils import Tokenizer
from .base import (
    ContentPart,
    ImageProcessorProtocol,
    Message,
    Role,
    ToolSpec,
    image_to_chunk,
)
from .kimi_k2 import KimiK2Renderer
from .kimi_k2_5_tool_declaration_ts import encode_tools_to_typescript_style


class KimiK25Renderer(KimiK2Renderer):
    """
    Renderer for Kimi K2.5 with thinking enabled (default).

    Key differences from KimiK2Renderer:
    1. Generation prompt prefill: Appends `<think>` (open tag) to enable thinking mode
    2. Tool declarations: Uses TypeScript-style format instead of JSON

    Format:
        <|im_system|>system<|im_middle|>You are Kimi...<|im_end|>
        <|im_user|>user<|im_middle|>Hello<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>

    Historical assistant messages use empty <think></think> blocks (inherited from K2),
    while the generation prompt adds an open <think> tag to enable thinking.
    """

    image_processor: ImageProcessor | None

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor | None = None,
        strip_thinking_from_history: bool = True,
    ):
        super().__init__(
            tokenizer, strip_thinking_from_history=strip_thinking_from_history
        )
        self.image_processor = image_processor

    def _encode_multipart_content(
        self, content: list[ContentPart]
    ) -> list[tinker.ModelInputChunk]:
        chunks = []
        for part in content:
            if part["type"] == "text":
                chunks.append(
                    tinker.types.EncodedTextChunk(
                        tokens=self.tokenizer.encode(part["text"])
                    )
                )
            elif part["type"] == "image":
                assert self.image_processor is not None, (
                    "KimiK25Renderer must be initialized with an image processor in order to support image content parts"
                )
                chunks.append(
                    tinker.types.EncodedTextChunk(
                        tokens=self.tokenizer.encode(self._image_prefix)
                    )
                )
                chunks.append(
                    image_to_chunk(
                        part["image"],
                        cast(ImageProcessorProtocol, self.image_processor),
                    )
                )
                chunks.append(
                    tinker.types.EncodedTextChunk(
                        tokens=self.tokenizer.encode(self._image_suffix)
                    )
                )
            else:
                raise ValueError(f"Unsupported content type: {part['type']}")
        return chunks

    @property
    def _image_prefix(self) -> str:
        return "<|media_begin|>image<|media_content|>"

    @property
    def _image_suffix(self) -> str:
        return "<|media_end|>\n"

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """Build generation prompt with <think> prefill for thinking mode."""
        # If no prefill specified, use <think> to enable thinking
        if prefill is None:
            prefill = "<think>"
        return super().build_generation_prompt(messages, role=role, prefill=prefill)

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system messages with TypeScript-style tool specifications.

        Per the HuggingFace chat template, Kimi K2.5 uses TypeScript-style tool
        declarations instead of JSON format. The tool_declare message comes BEFORE
        the regular system message.

        Reference: kimi-k2.5-hf-tokenizer/chat_template.jinja
        """
        messages: list[Message] = []

        # Tool declaration message comes first (per HF chat template)
        if tools:
            tools_payload = [{"type": "function", "function": tool} for tool in tools]
            tools_ts_str = encode_tools_to_typescript_style(tools_payload)
            messages.append(Message(role="tool_declare", content=tools_ts_str))

        # Regular system message second (use default if none provided)
        actual_system_prompt = (
            system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
        )
        messages.append(Message(role="system", content=actual_system_prompt))

        return messages


class KimiK25DisableThinkingRenderer(KimiK25Renderer):
    """
    Renderer for Kimi K2.5 with thinking disabled.

    Uses `<think></think>` prefill instead of `<think>` to disable thinking mode.

    Format:
        <|im_system|>system<|im_middle|>You are Kimi...<|im_end|>
        <|im_user|>user<|im_middle|>Hello<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think></think>
    """

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """Build generation prompt with <think></think> prefill to disable thinking."""
        # If no prefill specified, use <think></think> to disable thinking
        if prefill is None:
            prefill = "<think></think>"
        return super(KimiK25Renderer, self).build_generation_prompt(
            messages, role=role, prefill=prefill
        )
