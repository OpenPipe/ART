"""
Qwen3 family renderers - text and vision-language models.

Includes:
- Qwen3Renderer: Base Qwen3 with thinking enabled
- Qwen3DisableThinkingRenderer: Qwen3 with thinking disabled
- Qwen3InstructRenderer: Qwen3 instruct 2507 models (no <think> tag)
- Qwen3VLRenderer: Vision-language Qwen3 with thinking
- Qwen3VLInstructRenderer: Vision-language instruct models
"""

import json
from typing import cast

import tinker

from ..image_processing_utils import ImageProcessor
from ..tokenizer_utils import Tokenizer
from .base import (
    ImagePart,
    ImageProcessorProtocol,
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    TextPart,
    ToolSpec,
    UnparsedToolCall,
    _tool_call_payload,
    image_to_chunk,
    parse_content_blocks,
    parse_response_for_stop_token,
    remove_thinking,
)


def _merge_consecutive_text_parts(
    chunks: list[ImagePart | TextPart],
) -> list[ImagePart | TextPart]:
    """Merge consecutive TextParts into single parts.

    This ensures text is tokenized as a single string, matching HuggingFace's
    apply_chat_template behavior which tokenizes the full rendered string at once.
    Without merging, tokenization boundaries between chunks can produce different
    token sequences (though they decode to identical strings).
    """
    if not chunks:
        return chunks

    merged: list[ImagePart | TextPart] = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk["type"] == "text" and merged[-1]["type"] == "text":
            merged[-1] = TextPart(type="text", text=merged[-1]["text"] + chunk["text"])
        else:
            merged.append(chunk)
    return merged


class Qwen3Renderer(Renderer):
    """
    Renderer for Qwen3 models with thinking enabled.

    This renderer is designed to match HuggingFace's Qwen3 chat template behavior
    (with enable_thinking=True, which is the default). This ensures compatibility
    with the OpenAI-compatible /chat/completions endpoint, which uses HF templates.

    Reference: https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json

    Format:
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>
        [reasoning content]
        </think>
        I can help you with...<|im_end|>

    The default strip_thinking_from_history=True matches HF behavior where thinking
    blocks are stripped from historical assistant messages in multi-turn conversations.
    Use strip_thinking_from_history=False for multi-turn RL to get the extension property.
    """

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips <think>...</think> blocks
                from assistant messages in multi-turn history. This matches HuggingFace's
                Qwen3 chat template behavior. Set to False to preserve thinking in history
                (useful for multi-turn RL where you need the extension property).

        Note: When strip_thinking_from_history=True, this renderer produces identical
        tokens to HuggingFace's apply_chat_template with enable_thinking=True.

        See /rl/sequence-extension in the docs for details on how strip_thinking_from_history
        affects multi-turn RL compute efficiency.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    @property
    def has_extension_property(self) -> bool:
        """Extension property depends on strip_thinking_from_history setting.

        When strip_thinking_from_history=False, thinking blocks are preserved in
        history, so each successive observation is a prefix extension of the previous.

        When strip_thinking_from_history=True (default), thinking blocks are stripped
        from historical messages, breaking the extension property.
        """
        return not self.strip_thinking_from_history

    def _get_qwen_role_for_message(self, message: Message) -> str:
        """Get the role to use for rendering a message in Qwen format.

        Per HuggingFace Qwen3 chat template, tool messages are rendered with role "user".
        """
        role = message["role"]
        if role == "tool":
            return "user"
        return role

    def _wrap_qwen_tool_response(self, content: str) -> str:
        """Wrap tool response content in Qwen's <tool_response> tags."""
        return f"<tool_response>\n{content}\n</tool_response>"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        content = message["content"]

        if isinstance(content, list):
            # Structured content - handle with list operations
            parts = content
            if (
                self.strip_thinking_from_history
                and message["role"] == "assistant"
                and not ctx.is_last
            ):
                # Remove thinking parts for historical messages
                parts = remove_thinking(parts)
            # Render parts in order, preserving interleaved thinking/text structure.
            # No separator needed - whitespace is preserved in TextPart for roundtrip identity.
            rendered_parts = []
            for p in parts:
                if p["type"] == "thinking":
                    rendered_parts.append(f"<think>{p['thinking']}</think>")
                elif p["type"] == "text":
                    rendered_parts.append(p["text"])
                # ToolCallPart handled via message's tool_calls field
            output_content = "".join(rendered_parts)
        else:
            # String content - pass through as-is.
            # Note: strip_thinking_from_history only works with list-based content.
            # For stripping to work on historical messages, use structured content
            # with ThinkingPart separated from text (as returned by parse_response).
            output_content = content

        # Handle tool response wrapping
        if message["role"] == "tool":
            output_content = self._wrap_qwen_tool_response(output_content)

        # Handle tool_calls field
        if "tool_calls" in message:
            # Add leading newline to match HF template behavior
            output_content += "\n" + "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        output_content += "<|im_end|>"
        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, (
            f"Expected single token for <|im_end|>, got {len(tokens)}"
        )
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Parse <think>...</think> and <tool_call>...</tool_call> blocks together
        # to preserve ordering. Tool calls use Qwen's format:
        # - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        # - https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py#L279-L282
        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Parse all blocks in one pass, preserving order
        parts = parse_content_blocks(content)

        if parts is not None:
            assistant_message["content"] = parts

            # Also populate tool_calls and unparsed_tool_calls fields for backward compatibility
            # TODO: Consider moving away from TypedDicts for part types - current approach
            # relies on runtime type checking (p["type"] == "tool_call") without static guarantees.
            tool_calls = [p["tool_call"] for p in parts if p["type"] == "tool_call"]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            unparsed = [
                UnparsedToolCall(raw_text=p["raw_text"], error=p["error"])
                for p in parts
                if p["type"] == "unparsed_tool_call"
            ]
            if unparsed:
                assistant_message["unparsed_tool_calls"] = unparsed
        else:
            # No special blocks found - keep as string for backward compatibility
            assistant_message["content"] = content

        return assistant_message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        Qwen3's HF template accepts either:
        - message['reasoning_content'] as a separate field
        - <think>...</think> embedded in content

        We use reasoning_content for cleaner separation.
        """
        result: dict = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            # Extract thinking into reasoning_content, keep text in content
            thinking_parts = []
            text_parts = []
            for p in content:
                if p["type"] == "thinking":
                    thinking_parts.append(p["thinking"])
                elif p["type"] == "text":
                    text_parts.append(p["text"])
                # Skip tool_call/unparsed_tool_call - handled via tool_calls field

            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        # Handle tool_calls
        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message["tool_calls"]
            ]

        # Handle tool response fields
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Qwen3 tool specifications.

        Qwen3 uses XML `<tools>` tags containing JSON tool definitions in OpenAI format,
        appended to the system message content.

        References:
        - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        - https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json
        """
        tools_text = ""
        if tools:
            # Each tool is wrapped in {"type": "function", "function": {...}} per OpenAI format
            # Use separators=(", ", ": ") to match HF's tojson filter output
            tool_lines = "\n".join(
                json.dumps(
                    {"type": "function", "function": tool}, separators=(", ", ": ")
                )
                for tool in tools
            )
            tools_text = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        # Add separator between system prompt and tools if system prompt exists
        if system_prompt:
            content = system_prompt + "\n\n" + tools_text
        else:
            content = tools_text

        return [Message(role="system", content=content)]


class Qwen3DisableThinkingRenderer(Qwen3Renderer):
    """
    Renderer for Qwen3 hybrid models with thinking disabled.

    This renderer matches HuggingFace's Qwen3 chat template behavior with
    enable_thinking=False (or thinking=False for apply_chat_template). It adds
    empty <think>\\n\\n</think>\\n\\n blocks to assistant messages, signaling to
    the model that it should respond directly without extended reasoning.

    Use this renderer when you want to train or sample from Qwen3 models in
    "non-thinking" mode while maintaining compatibility with the OpenAI endpoint.
    """

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        # Get the base rendered message
        rendered = super().render_message(message, ctx)

        # Add empty thinking block to header for last assistant message
        # This goes in header (weight=0) so observation matches generation prompt.
        if message["role"] == "assistant" and ctx.is_last:
            content = message.get("content", "")
            if isinstance(content, str):
                has_think = "<think>" in content
            else:
                has_think = any(p["type"] == "thinking" for p in content)

            if not has_think:
                empty_think_tokens = self.tokenizer.encode(
                    "<think>\n\n</think>\n\n", add_special_tokens=False
                )
                old_header_tokens = (
                    list(rendered.header.tokens) if rendered.header else []
                )
                new_header = tinker.EncodedTextChunk(
                    tokens=old_header_tokens + empty_think_tokens
                )
                rendered = RenderedMessage(
                    header=new_header,
                    output=rendered.output,
                    stop_overlap=rendered.stop_overlap,
                )

        return rendered


class Qwen3InstructRenderer(Qwen3Renderer):
    """
    Renderer for Qwen3 instruct 2507 models. Unlike the earlier Qwen3 models, these models do not
    use the <think> tag at all.

    Inherits from Qwen3Renderer. ThinkingPart in content is still handled (rendered as
    <think>...</think>) in case the conversation includes thinking.
    """

    @property
    def has_extension_property(self) -> bool:
        """Qwen3 Instruct always satisfies extension - no thinking to strip from history."""
        # NOTE: If callers include ThinkingPart in history, Qwen3Renderer may still strip it
        # when strip_thinking_from_history=True, so extension can break.
        # This is a rare case that'll only occur if we prompt the instruct model
        # with a conversation from a different model.
        return True


class Qwen3VLRenderer(Qwen3Renderer):
    """
    Vision-language renderer for Qwen3-VL models with thinking support.

    Format like this:
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>
        I can help you with...<|im_end|>

    The default strip_thinking_from_history=True matches the non-VL Qwen3Renderer behavior.
    """

    image_processor: ImageProcessor

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor,
        strip_thinking_from_history: bool = True,
        merge_text_chunks: bool = True,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.strip_thinking_from_history = strip_thinking_from_history
        self.merge_text_chunks = merge_text_chunks

    def _preprocess_message_parts(
        self, message: Message, *, strip_thinking: bool = False
    ) -> list[ImagePart | TextPart]:
        """Convert message content to list form for VL rendering.

        Converts ThinkingPart to <think>...</think> text (or strips if strip_thinking=True).
        Wraps images with vision tokens. ToolCallPart is not supported in VL content list
        (use message's tool_calls field instead).
        """
        content = message["content"]
        if isinstance(content, str):
            base_parts: list[ImagePart | TextPart] = [
                TextPart(type="text", text=content)
            ]
        else:
            # Convert structured content to ImagePart/TextPart list
            base_parts: list[ImagePart | TextPart] = []
            for p in content:
                if p["type"] == "text":
                    base_parts.append(cast(TextPart, p))
                elif p["type"] == "image":
                    base_parts.append(cast(ImagePart, p))
                elif p["type"] == "thinking":
                    if not strip_thinking:
                        # Render thinking as <think>...</think> text
                        base_parts.append(
                            TextPart(
                                type="text", text=f"<think>{p['thinking']}</think>"
                            )
                        )
                    # else: strip thinking by not appending
                # ToolCallPart and UnparsedToolCallPart are handled via message's tool_calls field

        # Wrap images with vision tokens
        chunks: list[ImagePart | TextPart] = []
        for content_chunk in base_parts:
            if content_chunk["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_start|>"))

            chunks.append(content_chunk)

            if content_chunk["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_end|>"))

        return chunks

    def _wrap_qwen_tool_response_chunks(
        self, chunks: list[ImagePart | TextPart]
    ) -> list[ImagePart | TextPart]:
        """Wrap content chunks in Qwen's <tool_response> tags for multimodal messages."""
        return (
            [TextPart(type="text", text="<tool_response>\n")]
            + chunks
            + [TextPart(type="text", text="\n</tool_response>")]
        )

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        # Strip thinking from history for non-last assistant messages (matching non-VL behavior)
        strip_thinking = (
            self.strip_thinking_from_history
            and message["role"] == "assistant"
            and not ctx.is_last
        )
        output_chunks = self._preprocess_message_parts(
            message, strip_thinking=strip_thinking
        )

        # Handle tool response wrapping
        if message["role"] == "tool":
            output_chunks = self._wrap_qwen_tool_response_chunks(output_chunks)

        if "tool_calls" in message:
            # Add leading newline to match HF template behavior
            output_chunks += [
                TextPart(
                    type="text",
                    text="\n"
                    + "\n".join(
                        [
                            f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                            for tool_call in message["tool_calls"]
                        ]
                    ),
                )
            ]
        output_chunks += [TextPart(type="text", text="<|im_end|>")]

        if self.merge_text_chunks:
            output_chunks = _merge_consecutive_text_parts(output_chunks)

        output_chunks_encoded: list[tinker.ModelInputChunk] = [
            image_to_chunk(
                image_or_str=x["image"],
                image_processor=cast(ImageProcessorProtocol, self.image_processor),
            )
            if x["type"] == "image"
            else tinker.EncodedTextChunk(
                tokens=self.tokenizer.encode(x["text"], add_special_tokens=False)
            )
            for x in output_chunks
        ]

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        return RenderedMessage(header=header, output=output_chunks_encoded)


class Qwen3VLInstructRenderer(Qwen3VLRenderer):
    """
    Renderer for Qwen3-VL Instruct models.

    Unlike the Qwen3-VL Thinking models, The Qwen3-VL Instruct models do not use the <think> tag.
    """

    pass
