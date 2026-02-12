"""GptOssRenderer - OpenAI's open source model format (Harmony)."""

from datetime import datetime
import json
import re
import warnings

import tinker
import torch

from ..tokenizer_utils import Tokenizer
from .base import (
    ContentPart,
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    ensure_list,
    ensure_text,
)

# =============================================================================
# TypeScript formatting utilities (stateless, used for Harmony tool definitions)
# =============================================================================


def _json_type_to_typescript(schema: dict) -> str:
    """Convert a single JSON schema type to TypeScript."""
    if "oneOf" in schema:
        return " | ".join(_json_type_to_typescript(s) for s in schema["oneOf"])
    if "anyOf" in schema:
        return " | ".join(_json_type_to_typescript(s) for s in schema["anyOf"])

    json_type = schema.get("type", "any")

    if isinstance(json_type, list):
        return " | ".join(_json_type_to_typescript({"type": t}) for t in json_type)

    if json_type == "string":
        if "enum" in schema:
            return " | ".join(json.dumps(v) for v in schema["enum"])
        base_type = "string"
    elif json_type == "number" or json_type == "integer":
        base_type = "number"
    elif json_type == "boolean":
        base_type = "boolean"
    elif json_type == "array":
        items_type = _json_type_to_typescript(schema.get("items", {}))
        base_type = f"{items_type}[]"
    elif json_type == "object":
        base_type = _json_schema_to_typescript(schema)
    else:
        base_type = "any"

    if schema.get("nullable"):
        return f"{base_type} | null"
    return base_type


def _json_schema_to_typescript(schema: dict) -> str:
    """Convert JSON schema to an inline TypeScript-ish type string."""
    if schema.get("type") != "object":
        return "any"

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_parts = []
    for prop_name, prop_schema in properties.items():
        prop_type = _json_type_to_typescript(prop_schema)
        optional = "" if prop_name in required else "?"
        type_parts.append(f"{prop_name}{optional}: {prop_type}")

    return "{ " + ", ".join(type_parts) + " }"


def _schema_comments(schema: dict) -> list[str]:
    """Extract comments from schema (title, description, examples)."""
    comments: list[str] = []
    title = schema.get("title")
    if title:
        comments.append(str(title))
        comments.append("")
    description = schema.get("description")
    if description:
        comments.append(str(description))
    examples = schema.get("examples")
    if examples:
        comments.append("Examples:")
        for example in examples:
            comments.append(f"- {json.dumps(example)}")
    return comments


def _format_parameters_block(schema: dict) -> str:
    """Format function parameters as a TypeScript-style block."""
    if schema.get("type") != "object" or not schema.get("properties"):
        return "()"

    lines = []
    header = "(_:"
    schema_description = schema.get("description")
    if schema_description:
        header += f" // {schema_description}"
    lines.append(header)
    lines.append("{")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for prop_name, prop_schema in properties.items():
        for comment in _schema_comments(prop_schema):
            lines.append(f"// {comment}")
        prop_type = _json_type_to_typescript(prop_schema)
        optional = "" if prop_name in required else "?"
        default_comment = ""
        if "default" in prop_schema:
            default_comment = f" // default: {json.dumps(prop_schema['default'])}"
        lines.append(f"{prop_name}{optional}: {prop_type},{default_comment}")

    lines.append("})")
    return "\n".join(lines)


def _format_tool_definition(tool: ToolSpec) -> str:
    """Format a single tool as a Harmony TypeScript-style definition."""
    lines = []
    if tool.get("description"):
        lines.append(f"// {tool['description']}")

    params = tool.get("parameters") or {}
    params_block = _format_parameters_block(params)
    lines.append(f"type {tool['name']} = {params_block} => any;")
    return "\n".join(lines)


class GptOssRenderer(Renderer):
    """
    Renderer for OpenAI's open source models using the Harmony format.

    Wire format: <|start|>role<|channel|>channel<|message|>content<|end|>
    No newlines between messages. Last assistant message ends with <|return|>;
    historical assistant messages end with <|end|>.

    Harmony Channels
    ----------------
    Each assistant message specifies a "channel" that controls how the content is
    interpreted and displayed. An assistant turn can have multiple channel segments
    (rendered as separate <|start|>assistant... blocks):

    - analysis: Chain-of-thought reasoning (hidden from end users, like <think> blocks)
    - commentary: Tool calls to developer-defined functions, or user-visible "preambles"
      before tool calls. Uses `to=functions.name` to route to specific tools.
    - final: The user-facing response text

    A typical assistant turn with thinking + tool call + final answer would render as:
        <|start|>assistant<|channel|>analysis<|message|>{thinking}<|end|>
        <|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{args}<|call|>
        ... (tool result) ...
        <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

    Tool Calling
    ------------
    - Tool definitions: Go in developer message with TypeScript-style syntax
    - Tool calls: <|start|>assistant to=functions.name<|channel|>commentary <|constrain|>json<|message|>{args}<|call|>
    - Tool results: <|start|>functions.name to=assistant<|channel|>commentary<|message|>{result}<|end|>

    Reference: https://raw.githubusercontent.com/openai/openai-cookbook/main/articles/openai-harmony.md
    """

    # System prompt content (without rendering tokens). Tool channel instructions are NOT
    # included here; they are only added when tools are defined in the developer message.
    system_prompt_content = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        "Current date: {current_date}\n\n"
        "Reasoning: {reasoning_effort}\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    use_system_prompt: bool = False
    reasoning_effort: str | None = None
    current_date: str | None = (
        None  # If use_system_prompt=True, will use the current date if this is None. Set this to a fixed date for deterministic system prompt.
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_system_prompt: bool = False,
        reasoning_effort: str | None = None,
        current_date: str | None = None,
    ):
        super().__init__(tokenizer)
        self.use_system_prompt = use_system_prompt
        self.reasoning_effort = reasoning_effort
        self.current_date = current_date
        assert use_system_prompt == (reasoning_effort is not None), (
            "Reasoning effort must be set iff using system prompt"
        )

    # Internal role for OpenAI's system prompt (bypasses system->developer mapping)
    _INTERNAL_SYSTEM_ROLE = "_gptoss_internal_system"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]

        # Handle tool result messages (role="tool")
        if role == "tool":
            return self._render_tool_result_message(message, ctx)

        # Internal system role renders as actual "system" without transformation
        if role == self._INTERNAL_SYSTEM_ROLE:
            role = "system"
        # User-provided "system" messages map to "developer" (per HF template)
        elif role == "system":
            role = "developer"

        header_str = f"<|start|>{role}"
        output_str = ""
        tool_calls: list[ToolCall] = []

        if message["role"] == "assistant":
            # Assistant channels. See https://cookbook.openai.com/articles/openai-harmony
            # Extract text and thinking from content list
            parts = ensure_list(message["content"])
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")
            thinking_content = "".join(
                p["thinking"] for p in parts if p["type"] == "thinking"
            )
            tool_calls = message.get("tool_calls") or []

            # Analysis channel (CoT) - only if there's thinking content
            if thinking_content:
                output_str += f"<|channel|>analysis<|message|>{thinking_content}<|end|><|start|>assistant"

            # Handle tool calls (goes in commentary channel)
            if tool_calls:
                # If there's text content with tool calls, render as commentary preamble first
                if text_content:
                    output_str += f"<|channel|>commentary<|message|>{text_content}<|end|><|start|>assistant"
                output_str += self._render_tool_calls(tool_calls)
            else:
                # Final channel (Response Content)
                output_str += f"<|channel|>final<|message|>{text_content}"
        elif message["role"] == "system":
            # User-provided system messages get "# Instructions" wrapper (rendered as developer)
            output_str += (
                f"<|message|># Instructions\n\n{ensure_text(message['content'])}\n\n"
            )
        else:
            # user, developer, internal system, and other roles: plain content
            output_str += f"<|message|>{ensure_text(message['content'])}"

        # End token logic:
        # - Tool calls: each tool call already includes <|call|> via _render_tool_calls, no end token needed
        # - Assistant (no tool calls): <|return|> if last message, <|end|> otherwise
        # - All other roles: <|end|>
        if message["role"] == "assistant":
            if not tool_calls:
                if ctx.is_last:
                    output_str += "<|return|>"
                else:
                    output_str += "<|end|>"
            # Note: tool_calls case needs no end token here - _render_tool_calls adds <|call|>
        else:
            output_str += "<|end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _render_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        """Render tool calls in Harmony commentary channel format.

        Each tool call becomes a separate commentary message:
        to=functions.name<|channel|>commentary <|constrain|>json<|message|>{args}

        Multiple tool calls are separated by <|call|><|start|>assistant.
        """
        result_parts = []
        for i, tc in enumerate(tool_calls):
            # Format: to=functions.name<|channel|>commentary <|constrain|>json<|message|>{args}
            result_parts.append(
                f" to=functions.{tc.function.name}<|channel|>commentary <|constrain|>json<|message|>"
                f"{tc.function.arguments}<|call|>"
            )
            # If not the last tool call, close message and start new assistant message
            if i < len(tool_calls) - 1:
                result_parts.append("<|start|>assistant")
        return "".join(result_parts)

    def _render_tool_result_message(
        self, message: Message, ctx: RenderContext
    ) -> RenderedMessage:
        """Render a tool result message.

        Format: <|start|>functions.name to=assistant<|channel|>commentary<|message|>{result}<|end|>

        IMPORTANT: The tool name MUST be provided in the message's "name" field.
        The renderer is stateless and cannot track tool_call_id -> name mappings.
        When constructing tool result messages, always include the "name" field:

            {"role": "tool", "name": "get_weather", "content": "72 degrees", "tool_call_id": "..."}

        If "name" is missing, this will produce "functions.unknown" which is incorrect.
        """
        # Get the tool name from the "name" field
        tool_name = message.get("name", "")
        if not tool_name:
            warnings.warn(
                "Tool message missing 'name' field. GptOssRenderer requires the 'name' field "
                "to render tool results correctly. Add 'name' to your tool messages: "
                "{'role': 'tool', 'name': 'function_name', 'content': '...', 'tool_call_id': '...'}",
                UserWarning,
                stacklevel=3,
            )
            tool_name = "unknown"

        # Ensure qualified with "functions." prefix
        if not tool_name.startswith("functions."):
            tool_name = f"functions.{tool_name}"

        # Build the header with tool name as role and to=assistant
        header_str = f"<|start|>{tool_name} to=assistant"

        # Tool results go in commentary channel
        content = ensure_text(message["content"])
        output_str = f"<|channel|>commentary<|message|>{content}<|end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _get_system_message(self) -> Message | None:
        """Return system message if configured, else None.

        Uses internal role to render as actual 'system' (not mapped to 'developer').
        """
        if not self.use_system_prompt:
            return None
        current_date = (
            self.current_date
            if self.current_date is not None
            else datetime.now().strftime("%Y-%m-%d")
        )
        content = self.system_prompt_content.format(
            current_date=current_date,
            reasoning_effort=self.reasoning_effort,
        )
        return Message(role=self._INTERNAL_SYSTEM_ROLE, content=content)

    @property
    def _bos_tokens(self) -> list[int]:
        # GptOss has no BOS token. System prompt is prepended as a message.
        return []

    def _warn_if_user_system_message(self, messages: list[Message]) -> None:
        """Warn if user provides system message when use_system_prompt=True."""
        if self.use_system_prompt and messages and messages[0]["role"] == "system":
            warnings.warn(
                "use_system_prompt=True but messages already start with a system message. "
                "The built-in system prompt will be prepended, resulting in two system messages. "
                "Either set use_system_prompt=False or remove the system message from your messages.",
                UserWarning,
                stacklevel=3,
            )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """Build generation prompt, prepending system message if configured."""
        self._warn_if_user_system_message(messages)
        system_msg = self._get_system_message()
        if system_msg:
            messages = [system_msg] + list(messages)
        return super().build_generation_prompt(messages, role, prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build supervised example, prepending system message if configured."""
        self._warn_if_user_system_message(messages)
        system_msg = self._get_system_message()
        if system_msg:
            messages = [system_msg] + list(messages)
        return super().build_supervised_example(messages, train_on_what)

    @property
    def _return_token(self) -> int:
        res = self.tokenizer.encode("<|return|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|return|>, got {len(res)}"
        return res[0]

    @property
    def _call_token(self) -> int:
        res = self.tokenizer.encode("<|call|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|call|>, got {len(res)}"
        return res[0]

    def get_stop_sequences(self) -> list[int]:
        # Both <|return|> and <|call|> are stop tokens
        # <|return|> for normal completion, <|call|> for tool calls
        return [self._return_token, self._call_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        call_count = response.count(self._call_token)
        return_count = response.count(self._return_token)
        if call_count == 0 and return_count == 0:
            str_response = self.tokenizer.decode(response)
            return Message(role="assistant", content=str_response), False
        if call_count > 1:
            raise ValueError(
                f"When parsing response, expected at most 1 <|call|> token, but got {call_count}. "
                "You probably are using the wrong stop tokens when sampling"
            )
        if return_count > 1:
            raise ValueError(
                f"When parsing response, expected at most 1 <|return|> token, but got {return_count}. "
                "You probably are using the wrong stop tokens when sampling"
            )

        stop_idx = response.index(self._return_token) if return_count else None
        if call_count:
            call_idx = response.index(self._call_token)
            if stop_idx is None or call_idx < stop_idx:
                stop_idx = call_idx

        assert stop_idx is not None
        str_response = self.tokenizer.decode(response[:stop_idx])
        parts, tool_calls, unparsed = self._parse_harmony_output(str_response)
        content: list[ContentPart] | str = parts if parts else str_response

        message: Message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed

        return message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        GptOss uses the analysis channel for thinking, which maps to reasoning_content
        in OpenAI's API format.
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

    def _parse_harmony_output(
        self, content: str
    ) -> tuple[list[ContentPart], list[ToolCall], list[UnparsedToolCall]]:
        messages = self._parse_harmony_messages(content)
        parts: list[ContentPart] = []
        tool_calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []

        for msg in messages:
            msg_content = msg["content"] or ""
            msg_raw_text = msg["raw_text"] or ""
            if not msg_content.strip():
                continue

            recipient = msg["recipient"]
            if recipient and recipient.startswith("functions."):
                tool_name = recipient.split("functions.", 1)[1]
                try:
                    json.loads(msg_content)
                    tool_calls.append(
                        ToolCall(
                            function=ToolCall.FunctionBody(
                                name=tool_name, arguments=msg_content.strip()
                            ),
                            id=None,  # Harmony format doesn't include tool call IDs
                        )
                    )
                except json.JSONDecodeError as e:
                    unparsed.append(
                        UnparsedToolCall(
                            raw_text=msg_raw_text, error=f"Invalid JSON: {e}"
                        )
                    )
                continue

            channel = msg["channel"]
            if channel == "analysis":
                parts.append(ThinkingPart(type="thinking", thinking=msg_content))
            elif channel == "final":
                parts.append(TextPart(type="text", text=msg_content))
            elif channel == "commentary":
                parts.append(TextPart(type="text", text=msg_content))

        return parts, tool_calls, unparsed

    def _parse_harmony_messages(self, content: str) -> list[dict[str, str | None]]:
        """Parse Harmony format content into a list of message dicts.

        Uses manual string parsing (find/rfind) rather than regex. This approach
        is intentional: it will continue to work if we move away from using
        stringified tokens, which would be preferable for robustness.
        """
        messages: list[dict[str, str | None]] = []
        idx = 0
        message_token = "<|message|>"
        end_tokens = ("<|end|>", "<|call|>", "<|return|>")

        while True:
            message_idx = content.find(message_token, idx)
            if message_idx == -1:
                break

            header_start = content.rfind("<|start|>", idx, message_idx)
            if header_start == -1:
                header_start = idx
            header = content[header_start:message_idx]

            content_start = message_idx + len(message_token)
            end_idx = len(content)
            end_token = ""
            for token in end_tokens:
                token_idx = content.find(token, content_start)
                if token_idx != -1 and token_idx < end_idx:
                    end_idx = token_idx
                    end_token = token

            body = content[content_start:end_idx]

            channel = None
            channel_match = re.search(r"<\|channel\|>([^<\s]+)", header)
            if channel_match:
                channel = channel_match.group(1)

            recipient = None
            recipient_match = re.search(r"to=([^\s<]+)", header)
            if recipient_match:
                recipient = recipient_match.group(1)

            content_type = None
            content_type_match = re.search(r"<\|constrain\|>\s*([^\s<]+)", header)
            if content_type_match:
                content_type = content_type_match.group(1)

            messages.append(
                {
                    "channel": channel,
                    "recipient": recipient,
                    "content_type": content_type,
                    "content": body,
                    "raw_text": content[header_start : end_idx + len(end_token)]
                    if end_token
                    else content[header_start:],
                }
            )

            idx = end_idx + len(end_token)

        return messages

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create conversation prefix with tools in Harmony format.

        Returns a list of messages to prepend to conversations:
        1. If tools present: A system message with tool routing instruction
        2. A developer message with user instructions and tool definitions

        Tools are defined using TypeScript-ish syntax in a `functions` namespace,
        following the OpenAI Harmony spec.

        Note: When using this with tools, you typically don't need use_system_prompt=True
        since this method provides the necessary system setup for tool routing.

        Reference: https://raw.githubusercontent.com/openai/openai-cookbook/main/articles/openai-harmony.md
        """
        messages: list[Message] = []

        # Tool routing instruction goes in system message (per Harmony spec)
        if tools:
            messages.append(
                Message(
                    role=self._INTERNAL_SYSTEM_ROLE,
                    content="Calls to these tools must go to the commentary channel: 'functions'.",
                )
            )

        # User instructions and tool definitions go in developer message
        content_parts: list[str] = []
        if system_prompt:
            content_parts.append(f"# Instructions\n\n{system_prompt}")

        if tools:
            tool_defs = [_format_tool_definition(tool) for tool in tools]
            tools_text = "\n\n".join(tool_defs)
            content_parts.append(
                "# Tools\n\n## functions\n\nnamespace functions {\n\n"
                f"{tools_text}\n\n"
                "} // namespace functions"
            )

        if content_parts:
            content = "\n\n".join(content_parts)
            messages.append(Message(role="developer", content=content))

        return messages
