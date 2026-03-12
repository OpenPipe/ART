"""Renderer for Moonshot AI's Kimi K2 models."""

from dataclasses import dataclass, field
import json
import re
from typing import Iterator
import warnings

import tinker
import torch

from ..tokenizer_utils import Tokenizer
from .base import (
    ContentPart,
    Message,
    MessageDelta,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    Utf8TokenDecoder,
    ensure_list,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
)

_TOOL_CALLS_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    r"|<\|tool_call_section_begin\|>(.*?)<\|tool_call_section_end\|>",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _split_tool_calls_section(content: str) -> tuple[str, str | None]:
    match = _TOOL_CALLS_SECTION_RE.search(content)
    if not match:
        return content, None
    tool_section = match.group(1) if match.group(1) is not None else match.group(2)
    return content[: match.start()], tool_section


def _extract_tool_name(tool_id: str) -> str:
    if not tool_id:
        return ""
    name_part = tool_id.split(":", 1)[0]
    if "." in name_part:
        _, name_part = name_part.split(".", 1)
    return name_part


def _parse_tool_calls_section(
    tool_section: str,
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []

    for match in _TOOL_CALL_RE.finditer(tool_section):
        raw_text = match.group(0)
        tool_id = match.group(1).strip()
        args_str = match.group(2).strip()
        func_name = _extract_tool_name(tool_id)

        try:
            json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    function=ToolCall.FunctionBody(name=func_name, arguments=args_str),
                    id=tool_id if tool_id else None,
                )
            )
        except json.JSONDecodeError as e:
            unparsed_tool_calls.append(
                UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
            )

    return tool_calls, unparsed_tool_calls


# =============================================================================
# Streaming Parser
# =============================================================================

# Tags we need to detect during streaming
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"


def _longest_matching_suffix_prefix(text: str, tag: str) -> int:
    """Find longest suffix of text that matches a prefix of tag.

    This is used during streaming to determine how many characters at the end
    of accumulated text might be the beginning of a tag, and thus shouldn't
    be emitted yet.

    Args:
        text: The accumulated text to check.
        tag: The tag we're looking for (e.g., "<think>").

    Returns:
        Length of the longest suffix of text that matches a prefix of tag.

    Examples:
        >>> _longest_matching_suffix_prefix("hello", "<think>")
        0  # no suffix matches any prefix
        >>> _longest_matching_suffix_prefix("hello<", "<think>")
        1  # "<" matches prefix "<"
        >>> _longest_matching_suffix_prefix("hello<th", "<think>")
        3  # "<th" matches prefix "<th"
        >>> _longest_matching_suffix_prefix("hello<thx", "<think>")
        0  # "<thx" doesn't match any prefix of "<think>"
    """
    max_check = min(
        len(text), len(tag) - 1
    )  # -1 because full tag would be found, not buffered
    for length in range(max_check, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


@dataclass
class KimiK2StreamingParser:
    """Stateful streaming parser for Kimi K2 model output.

    Parses tokens incrementally, yielding deltas as content becomes available.
    Handles UTF-8 token boundaries and <think>...</think> block transitions.

    Usage:
        parser = KimiK2StreamingParser(tokenizer, end_token)
        for token in response_tokens:
            for delta in parser.feed(token):
                # Handle delta (StreamingMessageHeader, StreamingTextDelta, etc.)
        for delta in parser.finish():
            # Handle final deltas including complete Message
    """

    tokenizer: Tokenizer
    end_message_token: int

    # Internal state
    _utf8_decoder: Utf8TokenDecoder = field(init=False)
    _accumulated_text: str = field(init=False, default="")
    _header_emitted: bool = field(init=False, default=False)
    _in_thinking: bool = field(init=False, default=False)
    _content_index: int = field(init=False, default=0)
    _last_emitted_pos: int = field(init=False, default=0)
    _finished: bool = field(init=False, default=False)
    _all_tokens: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._utf8_decoder = Utf8TokenDecoder(self.tokenizer)
        self._accumulated_text = ""
        self._header_emitted = False
        self._in_thinking = False
        self._content_index = 0
        self._last_emitted_pos = 0
        self._finished = False
        self._all_tokens = []

    def feed(self, token: int) -> Iterator[MessageDelta]:
        """Feed a single token and yield any resulting deltas.

        Args:
            token: A single token ID from the model output.

        Yields:
            MessageDelta objects as content becomes available.
        """
        if self._finished:
            return

        self._all_tokens.append(token)

        # Check for end token
        if token == self.end_message_token:
            self._finished = True
            return

        # Try to decode the token
        decoded = self._utf8_decoder.decode([token])
        if decoded is None:
            # Token was buffered (incomplete UTF-8 sequence)
            return

        self._accumulated_text += decoded

        # Emit header on first content
        if not self._header_emitted:
            self._header_emitted = True
            yield StreamingMessageHeader(role="assistant")

        # Process the new text for deltas
        yield from self._emit_deltas()

    def _emit_deltas(self) -> Iterator[MessageDelta]:
        """Emit deltas for any new content since last emission."""
        text = self._accumulated_text
        pos = self._last_emitted_pos

        while pos < len(text):
            if not self._in_thinking:
                # Look for <think> tag
                think_start = text.find(_THINK_OPEN_TAG, pos)
                if think_start == -1:
                    # No <think> tag found - emit text up to a safe point.
                    # Keep any trailing chars that could be the start of "<think>".
                    suffix_from_pos = text[pos:]
                    keep = _longest_matching_suffix_prefix(
                        suffix_from_pos, _THINK_OPEN_TAG
                    )
                    safe_end = len(text) - keep
                    if safe_end > pos:
                        new_text = text[pos:safe_end]
                        if new_text:
                            yield StreamingTextDelta(
                                text=new_text, content_index=self._content_index
                            )
                        self._last_emitted_pos = safe_end
                    break
                elif think_start > pos:
                    # Emit text before <think>
                    new_text = text[pos:think_start]
                    if new_text:
                        yield StreamingTextDelta(
                            text=new_text, content_index=self._content_index
                        )
                    pos = think_start

                if text[pos:].startswith(_THINK_OPEN_TAG):
                    # Enter thinking mode
                    self._in_thinking = True
                    self._content_index += 1
                    pos += len(_THINK_OPEN_TAG)
                    self._last_emitted_pos = pos
            else:
                # In thinking mode - look for </think>
                think_end = text.find(_THINK_CLOSE_TAG, pos)
                if think_end == -1:
                    # No </think> found - emit thinking up to safe point.
                    # Keep any trailing chars that could be the start of "</think>".
                    suffix_from_pos = text[pos:]
                    keep = _longest_matching_suffix_prefix(
                        suffix_from_pos, _THINK_CLOSE_TAG
                    )
                    safe_end = len(text) - keep
                    if safe_end > pos:
                        new_thinking = text[pos:safe_end]
                        if new_thinking:
                            yield StreamingThinkingDelta(
                                thinking=new_thinking, content_index=self._content_index
                            )
                        self._last_emitted_pos = safe_end
                    break
                else:
                    # Emit thinking before </think>
                    new_thinking = text[pos:think_end]
                    if new_thinking:
                        yield StreamingThinkingDelta(
                            thinking=new_thinking, content_index=self._content_index
                        )
                    # Exit thinking mode
                    self._in_thinking = False
                    self._content_index += 1
                    pos = think_end + len(_THINK_CLOSE_TAG)
                    self._last_emitted_pos = pos

    def finish(self) -> Iterator[MessageDelta]:
        """Finish parsing and yield any remaining content plus final Message.

        Call this after all tokens have been fed (either naturally when
        end_message_token is seen, or when the stream ends).

        Yields:
            Any remaining deltas and the complete Message.
        """
        # Flush any buffered UTF-8 tokens
        remaining = self._utf8_decoder.flush()
        if remaining:
            self._accumulated_text += remaining

        # Emit header if we haven't yet (empty response edge case)
        if not self._header_emitted:
            self._header_emitted = True
            yield StreamingMessageHeader(role="assistant")

        # Emit any remaining content
        text = self._accumulated_text
        pos = self._last_emitted_pos

        if pos < len(text):
            remaining_text = text[pos:]
            if self._in_thinking:
                # Unclosed thinking block - emit as thinking
                if remaining_text:
                    yield StreamingThinkingDelta(
                        thinking=remaining_text, content_index=self._content_index
                    )
            else:
                if remaining_text:
                    yield StreamingTextDelta(
                        text=remaining_text, content_index=self._content_index
                    )

        # Build and yield the final complete Message
        # Use the batch parser for consistency
        message, _success = parse_response_for_stop_token(
            self._all_tokens, self.tokenizer, self.end_message_token
        )

        content = message.get("content", "")
        if isinstance(content, str):
            # Handle tool calls if present
            text_content, tool_section = _split_tool_calls_section(content)
            if tool_section is not None:
                tool_calls, unparsed_tool_calls = _parse_tool_calls_section(
                    tool_section
                )
                if tool_calls:
                    message["tool_calls"] = tool_calls
                if unparsed_tool_calls:
                    message["unparsed_tool_calls"] = unparsed_tool_calls

            content_parts = parse_think_blocks(text_content)
            message["content"] = (
                content_parts if content_parts is not None else text_content
            )

        yield message

    def reset(self) -> None:
        """Reset parser state for reuse."""
        self._utf8_decoder.reset()
        self._accumulated_text = ""
        self._header_emitted = False
        self._in_thinking = False
        self._content_index = 0
        self._last_emitted_pos = 0
        self._finished = False
        self._all_tokens = []


class KimiK2Renderer(Renderer):
    """
    Format for moonshotai/Kimi-K2-Thinking:
        <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
        <|im_user|>user<|im_middle|>What can you help me with?<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>reasoning</think>I can help you with...<|im_end|>

    Historical assistant messages use empty <think></think> blocks, while the assistant messages after the
    last non-tool-call assistant message preserves reasoning_content in the thinking block.

    Note: Per the HuggingFace chat template, the default system message is automatically
    prepended if no system message is provided. This ensures train-eval consistency when
    using HF's apply_chat_template for inference.
    """

    DEFAULT_SYSTEM_PROMPT = "You are Kimi, an AI assistant created by Moonshot AI."

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Ensure a default system message is present if none exists.

        This matches the HuggingFace chat template behavior where a default system
        message is automatically added when none is provided.

        The default system message is inserted at the appropriate position:
        - If messages is empty: adds default system message
        - If starting with tool_declare: inserts default system after tool_declare (if no system message follows)
        - Otherwise: prepends default system message before first message (if first message isn't system)
        """
        if not messages:
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [default_system]

        # Accept both system and tool_declare as valid starting messages
        first_role = messages[0]["role"]
        if first_role == "tool_declare":
            # Check if a system message already exists after tool_declare
            if len(messages) >= 2 and messages[1]["role"] == "system":
                return messages
            # No system message, insert default after tool_declare
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [messages[0], default_system] + list(messages[1:])
        elif first_role != "system":
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [default_system] + list(messages)

        return messages

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """
        Render a message. For assistant messages, ctx.is_last controls whether thinking is preserved
        (True) or stripped to empty <think></think> (False).
        """
        role = message["role"]

        # Build role token based on role type
        if role == "user":
            header_str = f"<|im_user|>{role}<|im_middle|>"
        elif role == "assistant":
            header_str = f"<|im_assistant|>{role}<|im_middle|>"
        elif role == "system":
            header_str = f"<|im_system|>{role}<|im_middle|>"
        elif role == "tool_declare":
            # Tool declaration uses system token but with "tool_declare" as display name
            header_str = f"<|im_system|>{role}<|im_middle|>"
        elif role == "tool":
            # HF template uses message.name if present, otherwise role
            role_name = message.get("name")
            if not role_name:
                warnings.warn(
                    "Tool message missing 'name' field. Using 'tool' as fallback. "
                    "Consider setting 'name' to match the tool function name for better context.",
                    UserWarning,
                    stacklevel=3,
                )
                role_name = role
            header_str = f"<|im_system|>{role_name}<|im_middle|>"

            # Tool responses have special formatting - need tool_call_id to correlate with the call
            tool_call_id = message.get("tool_call_id", "")
            if not tool_call_id:
                warnings.warn(
                    "Tool message missing 'tool_call_id' field. KimiK2Renderer requires 'tool_call_id' "
                    "to render tool results correctly. The value should match ToolCall.id from the "
                    "assistant's tool_calls.",
                    UserWarning,
                    stacklevel=3,
                )
            header_str += f"## Return of {tool_call_id}\n"
        else:
            # Unknown roles default to system-style formatting
            header_str = f"<|im_system|>{role}<|im_middle|>"

        # Build output content
        content = message["content"]
        output: list[tinker.ModelInputChunk] = []
        if role == "assistant":
            output_str = ""
            # Extract thinking and text from content list
            parts = ensure_list(content)
            thinking_content = "".join(
                p["thinking"] for p in parts if p["type"] == "thinking"
            )
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")

            # Preserve thinking for the last assistant message, or for all messages
            # when strip_thinking_from_history is False.
            if (
                ctx.is_last or not self.strip_thinking_from_history
            ) and thinking_content:
                output_str = f"<think>{thinking_content}</think>"
            else:
                output_str = "<think></think>"
            output_str += text_content

            # Handle tool calls
            if "tool_calls" in message and message["tool_calls"]:
                output_str += "<|tool_calls_section_begin|>"
                for idx, tool_call in enumerate(message["tool_calls"]):
                    tool_id = tool_call.id
                    if not tool_id:
                        tool_id = f"functions.{tool_call.function.name}:{idx}"
                    args = tool_call.function.arguments
                    output_str += f"<|tool_call_begin|>{tool_id}<|tool_call_argument_begin|>{args}<|tool_call_end|>"
                output_str += "<|tool_calls_section_end|>"
            output_str += "<|im_end|>"
            output.append(
                tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_str))
            )
        elif isinstance(content, str) or (
            len(content) == 1 and content[0]["type"] == "text"
        ):
            # Single-part/text content
            output_str = ensure_text(content) + "<|im_end|>"
            output.append(
                tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_str))
            )
        else:
            # Mult-part content (e.g. text+image(s))
            assert isinstance(content, list), (
                f"Expected list of content parts, got {type(content)}"
            )
            output = self._encode_multipart_content(
                content + [TextPart(type="text", text="<|im_end|>")]
            )

        header = tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(header_str))

        return RenderedMessage(header=header, output=output)

    def _encode_multipart_content(
        self, content: list[ContentPart]
    ) -> list[tinker.ModelInputChunk]:
        raise NotImplementedError(
            "Multipart/Image content encoding is not supported for Kimi K2 renderer"
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        messages = self._ensure_system_message(messages)
        chunks: list[tinker.types.ModelInputChunk] = []

        # Find last assistant message without tool calls (matches hf template behavior).
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx]["role"] == "assistant" and not messages[idx].get(
                "tool_calls"
            ):
                last_assistant_idx = idx
                break

        for idx, message in enumerate(messages):
            is_assistant = message["role"] == "assistant"
            is_last_assistant = is_assistant and (
                last_assistant_idx == -1 or idx > last_assistant_idx
            )

            # We cannot simply set is_last=False since we might be generating a new assistant message following a tool response,
            # and we need to preserve the thinking that leads to the tool call.
            ctx = RenderContext(
                idx=idx,
                is_last=is_last_assistant,
                prev_message=messages[idx - 1] if idx > 0 else None,
            )
            rendered_message = self.render_message(message, ctx)
            header_chunk = rendered_message.header
            output_chunks = rendered_message.output
            if header_chunk:
                chunks.append(header_chunk)
            chunks.extend([x for x in output_chunks if x])

        # Add generation prompt for new assistant message
        gen_prompt = f"<|im_assistant|>{role}<|im_middle|>"
        chunks.append(
            tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(gen_prompt))
        )
        if prefill:
            chunks.append(
                tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(prefill))
            )
        return tinker.ModelInput(chunks=chunks)

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        """
        Build tokens and per-token weights for supervised fine-tuning. Since Kimi K2 renderer does not satisfy the extension property, this method is provided to return multiple examples in case we want to train on multiple assistant messages, potentially across multiple turns of user-assistant conversation.
        """

        if (
            train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE
            or train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN
        ):
            return [
                self.build_supervised_example(messages, train_on_what=train_on_what)
            ]

        # split the messages into turns by user messages
        user_message_idxs = [
            idx for idx, message in enumerate(messages) if message["role"] == "user"
        ]

        supervised_examples: list[tuple[tinker.ModelInput, torch.Tensor]] = []

        if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            warnings.warn(
                "WARNING: Using train_on_what=ALL_MESSAGES/ALL_TOKENS/ALL_USER_AND_SYSTEM_MESSAGES/CUSTOMIZED with a renderer that "
                "does not satisfy the extension property (has_extension_property=False). "
                "The behavior is we apply the same `train_on_what` to all turns. This may not be the desired behavior.",
                UserWarning,
                stacklevel=3,
            )

        # We separate the turns by user messages. The first turn is the messages before the second user message.
        for user_message_idx in [*user_message_idxs[1:], len(messages)]:
            current_messages = messages[:user_message_idx]
            if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                supervised_examples.append(
                    self.build_supervised_example(
                        current_messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
                    )
                )
            else:
                supervised_examples.append(
                    self.build_supervised_example(
                        current_messages, train_on_what=train_on_what
                    )
                )

        return supervised_examples

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Override to properly handle thinking preservation for the last assistant message.
        Also ensures default system message is prepended if none is present.
        """
        messages = self._ensure_system_message(messages)

        # Kimi K2 hf template preserves the thinking of the assistant messages after the last non-tool-call assistant message.
        # We do the same in general. However, we intentionally skip the last message (which differs from HF template behavior) since for a complete conversation,
        # we would want to preserve the thinking of the last round of conversation between the user and the assistant (which could include multiple assistant messages and tool calls).
        # This is because the trajectory would then be taken for SFT without losing all the thinking content.
        last_assistant_idx = -1
        for idx in range(len(messages) - 2, -1, -1):
            if messages[idx]["role"] == "assistant" and not messages[idx].get(
                "tool_calls"
            ):
                last_assistant_idx = idx
                break

        model_input_chunks_weights: list[
            tuple[tinker.types.ModelInputChunk, float]
        ] = []

        for idx, message in enumerate(messages):
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                assert "trainable" in message, (
                    "When using CUSTOMIZED train_on_what, each message must have a trainable field"
                )
            else:
                assert "trainable" not in message, (
                    "When using non-CUSTOMIZED train_on_what, each message must not have a trainable field"
                )

            is_assistant = message["role"] == "assistant"
            is_last_message = idx == len(messages) - 1
            is_user_or_system = message["role"] in ["user", "system"]

            # For Kimi K2, preserve thinking only for the suffix after the last non-tool-call assistant.
            # If no such assistant exists, the suffix is the entire message list.
            # Preserve thinking only for assistants after the last non-tool-call assistant.
            is_last_assistant_turn = is_assistant and (
                last_assistant_idx == -1 or idx > last_assistant_idx
            )

            is_last_assistant = is_assistant and is_last_message
            ctx = RenderContext(
                idx=idx,
                is_last=is_last_assistant_turn,
                prev_message=messages[idx - 1] if idx > 0 else None,
            )
            rendered_message = self.render_message(message, ctx)

            header_part = rendered_message.header
            output_parts = rendered_message.output

            header_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
            if header_part:
                model_input_chunks_weights += [(header_part, header_weight)]

            # We include all assistant messages in the last round of assistant-tool interactions as the last assistant message.
            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    output_has_weight = is_last_assistant
                case TrainOnWhat.LAST_ASSISTANT_TURN:
                    output_has_weight = is_last_assistant_turn
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    output_has_weight = is_assistant
                case TrainOnWhat.ALL_MESSAGES:
                    output_has_weight = True
                case TrainOnWhat.ALL_TOKENS:
                    output_has_weight = True
                case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                    output_has_weight = is_user_or_system
                case TrainOnWhat.CUSTOMIZED:
                    output_has_weight = message.get("trainable", False)
                case _:
                    raise ValueError(f"Unknown train_on_what: {train_on_what}")

            model_input_chunks_weights += [
                (output_part, int(output_has_weight))
                for output_part in output_parts
                if output_part
            ]

        weights_data = [
            w for chunk, w in model_input_chunks_weights for _ in range(chunk.length)
        ]
        weights_tensor = torch.tensor(weights_data)

        model_input_chunks = [chunk for chunk, _ in model_input_chunks_weights]
        return tinker.ModelInput(chunks=model_input_chunks), weights_tensor

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>")
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

        content = assistant_message["content"]
        assert isinstance(content, str)

        # Handle tool calls if present
        text_content, tool_section = _split_tool_calls_section(content)
        if tool_section is not None:
            tool_calls, unparsed_tool_calls = _parse_tool_calls_section(tool_section)
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if unparsed_tool_calls:
                assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        content_parts = parse_think_blocks(text_content)
        assistant_message["content"] = (
            content_parts if content_parts is not None else text_content
        )

        return assistant_message, True

    def parse_response_streaming(self, response: list[int]) -> Iterator[MessageDelta]:
        """Parse response tokens with streaming, yielding incremental deltas.

        This method enables real-time display of model output by yielding
        partial content as it becomes available, rather than waiting for
        the complete response.

        Args:
            response: Token IDs from the model.

        Yields:
            StreamingMessageHeader: Once at the start of the message.
            StreamingTextDelta: Incremental text content.
            StreamingThinkingDelta: Incremental thinking content.
            Message: The complete parsed message at the end.

        Example:
            for delta in renderer.parse_response_streaming(tokens):
                if isinstance(delta, StreamingMessageHeader):
                    print(f"New message from {delta.role}")
                elif isinstance(delta, StreamingThinkingDelta):
                    print(f"[thinking] {delta.thinking}", end="")
                elif isinstance(delta, StreamingTextDelta):
                    print(delta.text, end="")
                elif isinstance(delta, Message):
                    print(f"\\nComplete: {delta}")
        """
        parser = KimiK2StreamingParser(
            tokenizer=self.tokenizer,
            end_message_token=self._end_message_token,
        )

        for token in response:
            yield from parser.feed(token)

        yield from parser.finish()

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        Kimi K2's HF template explicitly expects reasoning_content as a separate field.
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
        """Create system messages with Kimi K2 tool specifications.

        Per the HuggingFace chat template, Kimi K2 places the tool_declare message
        BEFORE the regular system message. The tool_declare payload expects the
        OpenAI-style tool schema ({"type":"function","function":{...}}).
        If no system_prompt is provided, uses the default system prompt to match
        HuggingFace chat template behavior.

        Reference: https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/chat_template.jinja
        """
        messages: list[Message] = []

        # Tool declaration message comes first (per HF chat template)
        if tools:
            tools_payload = [{"type": "function", "function": tool} for tool in tools]
            # Use sort_keys=True since Kimi K2 sorts keys alphabetically with its own custom apply_chat_template function
            tools_json = json.dumps(
                tools_payload, separators=(",", ":"), sort_keys=True
            )
            messages.append(Message(role="tool_declare", content=tools_json))

        # Regular system message second (use default if none provided)
        actual_system_prompt = (
            system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
        )
        messages.append(Message(role="system", content=actual_system_prompt))

        return messages
