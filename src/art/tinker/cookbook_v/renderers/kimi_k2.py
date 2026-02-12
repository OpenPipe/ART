"""Renderer for Moonshot AI's Kimi K2 models."""

import json
import re
import warnings

import tinker
import torch

from .base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
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


class KimiK2Renderer(Renderer):
    """
    Format for moonshotai/Kimi-K2-Thinking:
        <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
        <|im_user|>user<|im_middle|>What can you help me with?<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>reasoning</think>I can help you with...<|im_end|>

    Historical assistant messages use empty <think></think> blocks, while the final assistant
    response preserves reasoning_content in the thinking block.

    Note: Per the HuggingFace chat template, the default system message is automatically
    prepended if no system message is provided. This ensures train-eval consistency when
    using HF's apply_chat_template for inference.
    """

    DEFAULT_SYSTEM_PROMPT = "You are Kimi, an AI assistant created by Moonshot AI."

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
        output_str = ""
        if role == "assistant":
            # Extract thinking and text from content list
            parts = ensure_list(message["content"])
            thinking_content = "".join(
                p["thinking"] for p in parts if p["type"] == "thinking"
            )
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")

            # For the last assistant message (is_last=True), preserve thinking; otherwise use empty think block
            if ctx.is_last and thinking_content:
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
        else:
            output_str = ensure_text(message["content"])

        output_str += "<|im_end|>"

        header = tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(header_str))
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_str))
        ]
        return RenderedMessage(header=header, output=output)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        messages = self._ensure_system_message(messages)
        chunks: list[tinker.types.ModelInputChunk] = []

        for idx, message in enumerate(messages):
            # For generation prompt, no message is "last assistant" since we're generating new response
            ctx = RenderContext(
                idx=idx,
                is_last=False,
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

        # Find last non-tool-call assistant message index
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if (
                messages[idx]["role"] == "assistant"
                and "tool_calls" not in messages[idx]
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

            is_last_message = idx == len(messages) - 1
            is_assistant = message["role"] == "assistant"
            is_user_or_system = message["role"] in ["user", "system"]

            # For Kimi K2, preserve thinking only for the suffix after the last non-tool-call assistant.
            is_last_assistant = (
                is_assistant and last_assistant_idx != -1 and idx >= last_assistant_idx
            )
            ctx = RenderContext(
                idx=idx,
                is_last=is_last_assistant,
                prev_message=messages[idx - 1] if idx > 0 else None,
            )
            rendered_message = self.render_message(message, ctx)

            header_part = rendered_message.header
            output_parts = rendered_message.output

            header_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
            if header_part:
                model_input_chunks_weights += [(header_part, header_weight)]

            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    output_has_weight = is_last_message and is_assistant
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
