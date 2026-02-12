"""
DeepSeek V3 family renderers.

Includes:
- DeepSeekV3ThinkingRenderer: V3 models in thinking mode
- DeepSeekV3DisableThinkingRenderer: V3 models with thinking disabled
"""

import json
import re

import tinker

from ..tokenizer_utils import Tokenizer
from .base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
)


class _DeepSeekV3BaseRenderer(Renderer):
    """
    Base renderer for DeepSeek V3 models with common rendering logic.

    This is a private base class. Use DeepSeekV3ThinkingRenderer or
    DeepSeekV3DisableThinkingRenderer instead.

    System messages at position 0 are rendered without role tokens (matching HF template).
    System messages at later positions require system_role_as_user=True to convert to user role.

    The default strip_thinking_from_history=True matches HF behavior where thinking
    traces are removed from historical assistant messages in multi-turn conversations.
    Use strip_thinking_from_history=False for multi-turn RL to get the extension property.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        system_role_as_user: bool = False,
        strip_thinking_from_history: bool = True,
    ):
        super().__init__(tokenizer)
        self.system_role_as_user = system_role_as_user
        self.strip_thinking_from_history = strip_thinking_from_history

    @property
    def has_extension_property(self) -> bool:
        """Extension property depends on strip_thinking_from_history setting.

        When strip_thinking_from_history=False, thinking traces are preserved in
        history, so each successive observation is a prefix extension of the previous.

        When strip_thinking_from_history=True (default), thinking traces are stripped
        from historical messages, breaking the extension property.
        """
        return not self.strip_thinking_from_history

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a single message to tokens.

        Args:
            message: The message to render.
            ctx: Context about the message's position, including:
                - idx: The index of this message (0-based)
                - is_last: Whether this is the last message (affects thinking stripping)
                - prev_message: The previous message, used to detect post-tool formatting
        """
        # Check if this assistant message follows a tool response
        follows_tool = (
            ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
        )

        content = message["content"]

        if message["role"] == "system":
            # HF template collects all system messages at the start without role tokens
            # We only support this for idx=0; later system messages need system_role_as_user=True
            content_str = ensure_text(content)
            if ctx.idx == 0:
                header_tokens: list[int] = []
                output_str = content_str
            elif self.system_role_as_user:
                # Convert later system messages to user role
                role_token = self._get_special_token("User")
                header_tokens = [role_token]
                output_str = content_str
            else:
                raise ValueError(
                    "DeepSeek only supports system message at start. "
                    "Use system_role_as_user=True to convert later system messages to user role."
                )
        elif message["role"] == "user":
            role_token = self._get_special_token("User")
            header_tokens = [role_token]
            output_str = ensure_text(content)
        elif message["role"] == "assistant":
            has_tool_calls = "tool_calls" in message and message["tool_calls"]

            # Determine if we should strip thinking content from this message
            should_strip_thinking = (
                self.strip_thinking_from_history
                and not has_tool_calls
                and not ctx.is_last
            )

            if isinstance(content, list):
                # Structured content - handle with list operations
                parts = content
                # Render parts in order, preserving interleaved thinking/text structure.
                # No separator needed - whitespace is preserved in TextPart for roundtrip identity.
                rendered_parts = []
                for p in parts:
                    if p["type"] == "thinking":
                        if should_strip_thinking:
                            # Skip thinking content entirely when stripping
                            # (header gets </think> added separately to match HF format)
                            pass
                        else:
                            rendered_parts.append(f"<think>{p['thinking']}</think>")
                    elif p["type"] == "text":
                        rendered_parts.append(p["text"])
                    # ToolCallPart handled via message's tool_calls field
                output_content = "".join(rendered_parts)
            else:
                # String content - pass through as-is.
                # Stripping only works with structured content (ThinkingPart).
                output_content = content

            if follows_tool:
                # Post-tool assistant: no role token, content flows directly after tool output
                header_tokens = []
                output_str = output_content
            else:
                # Normal assistant message
                role_token = self._get_special_token("Assistant")
                header_tokens = [role_token]
                output_str = output_content
        elif message["role"] == "tool":
            # Tool responses use special tool output tokens to match HF template
            header_tokens = self.tokenizer.encode(
                "<｜tool▁output▁begin｜>", add_special_tokens=False
            )
            output_str = ensure_text(content) + "<｜tool▁output▁end｜>"
        else:
            raise ValueError(f"Unsupported role: {message['role']}")

        # Handle tool calls in assistant messages
        # HF format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜><｜tool▁calls▁end｜>
        if "tool_calls" in message and message["tool_calls"]:
            output_str += "<｜tool▁calls▁begin｜>"
            for tool_call in message["tool_calls"]:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
                output_str += f"<｜tool▁call▁begin｜>{func_name}<｜tool▁sep｜>{args}<｜tool▁call▁end｜>"
            output_str += "<｜tool▁calls▁end｜>"

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)

        # Add end_of_sentence only for assistant messages with content
        # (not for empty generation prompt messages)
        if message["role"] == "assistant" and message["content"]:
            output_tokens.append(self._end_message_token)

        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=output_tokens)
        ]
        # Only include header if non-empty; tinker rejects empty token chunks with
        # "Chunk N has empty tokens list". This happens for system messages at idx=0.
        if header_tokens:
            return RenderedMessage(
                header=tinker.types.EncodedTextChunk(tokens=header_tokens),
                output=output,
            )
        else:
            return RenderedMessage(output=output)

    def _get_special_token(self, name: str) -> int:
        sep = chr(65372)
        s = f"<{sep}{name}{sep}>"
        res = self.tokenizer.encode(s, add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for {s}, got {res}"
        return res[0]

    @property
    def _bos_tokens(self) -> list[int]:
        return [self._get_special_token("begin▁of▁sentence")]

    @property
    def _end_message_token(self) -> int:
        return self._get_special_token("end▁of▁sentence")

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_deepseek_tool_calls(
        self, content: str
    ) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
        """Parse tool calls from DeepSeek V3.1 format.

        Expected format (per HuggingFace model card and chat template):
            <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>func_name<｜tool▁sep｜>{"arg":"value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>

        Multiple tool calls are chained directly without separators.

        References:
            - DeepSeek V3.1 Model Card: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
            - Chat Template: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/assets/chat_template.jinja
        """
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []

        calls_match = re.search(
            r"<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>", content, re.DOTALL
        )
        if not calls_match:
            return tool_calls, unparsed_tool_calls

        for match in re.finditer(
            r"<｜tool▁call▁begin｜>(\w+)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>",
            calls_match.group(1),
            re.DOTALL,
        ):
            raw_text = match.group(0)
            func_name, args_str = match.group(1), match.group(2).strip()

            try:
                json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name=func_name, arguments=args_str
                        )
                    )
                )
            except json.JSONDecodeError as e:
                unparsed_tool_calls.append(
                    UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
                )

        return tool_calls, unparsed_tool_calls

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Parse DeepSeek-specific tool calls
        tool_calls, unparsed_tool_calls = self._parse_deepseek_tool_calls(content)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        # Strip tool calls section from content (both parsed and unparsed)
        if tool_calls or unparsed_tool_calls:
            content = re.sub(
                r"\s*<｜tool▁calls▁begin｜>.*?<｜tool▁calls▁end｜>",
                "",
                content,
                flags=re.DOTALL,
            )
            content = content.strip()

        # Parse <think>...</think> blocks into ThinkingPart/TextPart list
        parts = parse_think_blocks(content)
        if parts is not None:
            assistant_message["content"] = parts
        else:
            assistant_message["content"] = content

        return assistant_message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        DeepSeek's API uses reasoning_content for thinking models, similar to OpenAI's o1.
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
        """Create system message with DeepSeek V3.1 tool specifications.

        DeepSeek V3.1 tool calling requires tools to be described in the system message
        using a specific format with ### headers and inline JSON parameters.

        Note: Tool calling is supported in non-thinking mode only.

        References:
            - DeepSeek V3.1 Model Card (ToolCall section): https://huggingface.co/deepseek-ai/DeepSeek-V3.1
            - DeepSeek V3.1 Chat Template: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/assets/chat_template.jinja
            - DeepSeek API Tool Calls Guide: https://api-docs.deepseek.com/guides/tool_calls
        """
        tools_text = ""
        if tools:
            # Format each tool with ### header, description, and parameters
            tool_blocks = []
            for tool in tools:
                tool_block = f"""### {tool["name"]}
Description: {tool["description"]}

Parameters: {json.dumps(tool["parameters"])}"""
                tool_blocks.append(tool_block)

            tools_text = f"""

## Tools
You have access to the following tools:

{chr(10).join(tool_blocks)}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>

Where:
- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces"""

        return [Message(role="system", content=system_prompt + tools_text)]


class DeepSeekV3ThinkingRenderer(_DeepSeekV3BaseRenderer):
    """
    Renderer for DeepSeek V3 models in THINKING mode.

    Format:
        <|begin_of_sentence|><|User|>question<|Assistant|><think>reasoning</think>answer<|end_of_sentence|>

    For non-thinking mode, use DeepSeekV3DisableThinkingRenderer instead.

    Generation prompts include <think> prefill to trigger thinking mode.
    Think tags in message content come from ThinkPart rendering.

    When strip_thinking_from_history=True (default), historical assistant messages
    get </think> added to header and thinking content stripped, matching HF behavior.
    """

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render message, adding </think> to header when stripping thinking from history.

        HF's thinking=True template uses </think> at the start of historical assistant
        messages to signal "we're past the thinking phase, here's the answer".
        """
        rendered = super().render_message(message, ctx)

        # Add </think> to header for historical assistant messages when stripping thinking.
        # This matches the base class's should_strip_thinking logic - only historical messages
        # (not the last one) get </think> added. The last message is the supervised target and
        # should preserve its format (including any ThinkingPart).
        follows_tool = (
            ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
        )
        should_add_think_close = (
            message["role"] == "assistant"
            and not follows_tool
            and self.strip_thinking_from_history
            and not ctx.is_last
        )

        if should_add_think_close:
            think_close_tokens = self.tokenizer.encode(
                "</think>", add_special_tokens=False
            )
            old_header_tokens = list(rendered.header.tokens) if rendered.header else []
            new_header = tinker.EncodedTextChunk(
                tokens=old_header_tokens + think_close_tokens
            )
            rendered = RenderedMessage(header=new_header, output=rendered.output)

        return rendered

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """Build generation prompt with <think> prefill to trigger thinking mode.

        Does NOT add <think> when the previous message is a tool response,
        as tool-use conversations stay in non-thinking mode (matching HF behavior).
        """
        # Don't add <think> prefill after tool responses - tool use is non-thinking mode
        if messages and messages[-1]["role"] == "tool":
            return super().build_generation_prompt(messages, role, prefill)

        # Add <think> prefill to trigger thinking, combined with any user-provided prefill
        think_prefill = "<think>" + (prefill or "")
        return super().build_generation_prompt(messages, role, think_prefill)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse response, prepending <think> since we prefill with it.

        When sampling with build_generation_prompt, the <think> tag is part of the
        prefill and not included in the sampled tokens. The response will be
        "reasoning</think>answer" so we prepend <think> if necessary.
        """
        think_prefix_token: int = self.tokenizer.convert_tokens_to_ids("<think>")  # type: ignore[assignment]
        think_suffix_token: int = self.tokenizer.convert_tokens_to_ids("</think>")  # type: ignore[assignment]

        # Only prepend <think> if the response doesn't already start with it and contains </think>
        starts_with_think = len(response) > 0 and response[0] == think_prefix_token
        if not starts_with_think and think_suffix_token in response:
            response = [think_prefix_token] + response

        return super().parse_response(response)


class DeepSeekV3DisableThinkingRenderer(_DeepSeekV3BaseRenderer):
    """
    Renderer for DeepSeek V3 models in NON-THINKING mode.

    Format:
        <|begin_of_sentence|><|User|>question<|Assistant|></think>answer<|end_of_sentence|>

    The </think> prefix signals to the model to skip reasoning and respond directly.
    Any <think>...</think> blocks in the content are stripped.

    For thinking mode, use DeepSeekV3ThinkingRenderer instead.
    """

    @property
    def has_extension_property(self) -> bool:
        """Non-thinking mode always satisfies extension - no thinking to strip from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render message in non-thinking mode.

        For assistant messages (not following tool):
        - Strip any ThinkingPart from structured content
        - Add </think> to header to signal non-thinking mode
        """
        # Check if this assistant message follows a tool response
        follows_tool = (
            ctx.prev_message is not None and ctx.prev_message["role"] == "tool"
        )

        if message["role"] == "assistant" and not follows_tool:
            content = message["content"]

            # Strip thinking from content
            if isinstance(content, list):
                # Remove ThinkingPart, keep only text
                text_content = "".join(
                    p["text"] for p in content if p["type"] == "text"
                )
            else:
                # Strip <think>...</think> blocks from string content
                text_content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                )

            message = message.copy()
            message["content"] = text_content

        # Call parent to get base rendering
        rendered = super().render_message(message, ctx)

        # Add </think> to header for assistant messages (not following tool)
        # This goes in header (weight=0) so observation matches generation prompt.
        if message["role"] == "assistant" and not follows_tool:
            think_close_tokens = self.tokenizer.encode(
                "</think>", add_special_tokens=False
            )
            old_header_tokens = list(rendered.header.tokens) if rendered.header else []
            new_header = tinker.EncodedTextChunk(
                tokens=old_header_tokens + think_close_tokens
            )
            rendered = RenderedMessage(header=new_header, output=rendered.output)

        return rendered
