"""
Qwen3.5 family renderer.

Qwen3.5 models are VL models with the same basic
chat format as Qwen3-VL (im_start/im_end, thinking, vision tokens) but with a
different tool calling format.

Tool calling differences from Qwen3:
- Qwen3: JSON format  {"name": ..., "arguments": ...}
- Qwen3.5: XML format  <function=name><parameter=param>value</parameter></function>

Unlike Qwen3, the Qwen3.5 HF template:
- Always adds <think>...</think> blocks to assistant messages after the last user
  message (empty if no reasoning content).
- Always adds <think>\\n to the generation prompt.

Reference: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/tokenizer_config.json
"""

import json
import re

from .base import (
    ImagePart,
    Message,
    RenderContext,
    Role,
    TextPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
)
from .qwen3 import Qwen3VLRenderer

_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>\s*$",
    re.DOTALL,
)
_PARAM_BLOCK_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


class Qwen3_5Renderer(Qwen3VLRenderer):
    """
    Renderer for Qwen3.5 models.

    Subclasses Qwen3VLRenderer since Qwen3.5 models are VL models sharing the same
    basic chat format. Overrides tool calling to use Qwen3.5's XML parameter format.

    The Qwen3.5 HF template adds empty <think> blocks to assistant messages after
    the last user message. This is handled via ctx.last_user_index, which is
    populated by the base build_generation_prompt/build_supervised_example.
    """

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Override to produce the full generation suffix directly.

        Builds the header tokens manually and appends <think>\\n. This matches
        the Qwen3.5 template's add_generation_prompt behavior for thinking mode.
        """
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n<think>\n"
        return self.tokenizer.encode(header_str, add_special_tokens=False)

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Insert empty think block for assistant messages after the last user query."""
        if ctx.idx <= ctx.last_user_index:
            return ""

        content = message.get("content", "")
        has_think = False
        if isinstance(content, list):
            has_think = any(p["type"] == "thinking" for p in content)
        elif isinstance(content, str):
            has_think = "<think>" in content

        return "" if has_think else "<think>\n\n</think>\n\n"

    def _format_thinking_text(self, thinking: str) -> str:
        """Qwen3.5 uses newline-padded think blocks."""
        return f"<think>\n{thinking}\n</think>\n\n"

    def _to_openai_tool_arguments(self, arguments: str) -> str | dict:
        """Qwen3.5 chat template expects arguments as a mapping for |items."""
        return json.loads(arguments)

    def _parse_qwen3_5_tool_call_xml(
        self, raw_text: str
    ) -> ToolCall | UnparsedToolCall:
        """Parse Qwen3.5 XML-style tool calls from a raw <tool_call> block."""
        match = _FUNCTION_BLOCK_RE.match(raw_text)
        if not match:
            return UnparsedToolCall(
                raw_text=raw_text, error="Malformed Qwen3.5 tool call XML"
            )

        function_name = match.group("name").strip()
        body = match.group("body")
        if not function_name:
            return UnparsedToolCall(raw_text=raw_text, error="Missing function name")

        arguments: dict[str, object] = {}
        pos = 0
        for param in _PARAM_BLOCK_RE.finditer(body):
            if body[pos : param.start()].strip():
                return UnparsedToolCall(
                    raw_text=raw_text,
                    error="Unexpected non-parameter content inside <function> block",
                )

            param_name = param.group("name").strip()
            param_value_text = param.group("value").strip("\n")
            if not param_name:
                return UnparsedToolCall(raw_text=raw_text, error="Empty parameter name")

            try:
                param_value: object = json.loads(param_value_text)
            except json.JSONDecodeError:
                param_value = param_value_text

            arguments[param_name] = param_value
            pos = param.end()

        if body[pos:].strip():
            return UnparsedToolCall(
                raw_text=raw_text,
                error="Unexpected trailing content inside <function> block",
            )

        return ToolCall(
            function=ToolCall.FunctionBody(
                name=function_name,
                arguments=json.dumps(arguments),
            )
        )

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse response, prepending <think>\\n since the generation prompt prefills it.

        When sampling with build_generation_prompt, <think>\\n is part of the generation
        suffix and not included in the sampled tokens. The response will be
        "reasoning\\n</think>\\n\\nanswer" so we prepend <think>\\n if necessary.

        Also strips leading/trailing whitespace from thinking content to match the
        HF template behavior (which applies |trim to reasoning_content).
        """
        think_prefix_tokens = self.tokenizer.encode(
            "<think>\n", add_special_tokens=False
        )
        think_suffix_token = self.tokenizer.encode("</think>", add_special_tokens=False)
        assert len(think_suffix_token) == 1

        starts_with_think = (
            len(response) >= len(think_prefix_tokens)
            and response[: len(think_prefix_tokens)] == think_prefix_tokens
        )
        if not starts_with_think and think_suffix_token[0] in response:
            response = think_prefix_tokens + response

        message, success = super().parse_response(response)
        if not success:
            return message, success

        # Strip whitespace from thinking content (matches HF template |trim behavior)
        content = message.get("content")
        if isinstance(content, list):
            first_text_after_thinking: TextPart | None = None
            seen_thinking = False
            for p in content:
                if p["type"] == "thinking":
                    p["thinking"] = p["thinking"].strip()
                    seen_thinking = True
                elif seen_thinking and p["type"] == "text":
                    first_text_after_thinking = p
                    break

            # Template inserts exactly two separator newlines between </think> and text.
            if first_text_after_thinking is not None and first_text_after_thinking[
                "text"
            ].startswith("\n\n"):
                first_text_after_thinking["text"] = first_text_after_thinking["text"][
                    2:
                ]

        # Qwen3 parent parser assumes JSON inside <tool_call>; convert XML blocks here.
        converted_xml_calls: list[ToolCall] = []
        remaining_unparsed: list[UnparsedToolCall] = []
        for unparsed in message.get("unparsed_tool_calls", []):
            if "<function=" not in unparsed.raw_text:
                remaining_unparsed.append(unparsed)
                continue
            parsed = self._parse_qwen3_5_tool_call_xml(unparsed.raw_text)
            if isinstance(parsed, ToolCall):
                converted_xml_calls.append(parsed)
            else:
                remaining_unparsed.append(parsed)

        if converted_xml_calls:
            message["tool_calls"] = message.get("tool_calls", []) + converted_xml_calls
        if remaining_unparsed:
            message["unparsed_tool_calls"] = remaining_unparsed
        else:
            message.pop("unparsed_tool_calls", None)

        return message, success

    def _format_tool_call_xml(self, tool_call: ToolCall) -> str:
        """Format a single tool call in Qwen3.5's XML parameter format."""
        args = (
            json.loads(tool_call.function.arguments)
            if tool_call.function.arguments
            else {}
        )
        lines = [f"<tool_call>\n<function={tool_call.function.name}>"]
        for param_name, param_value in args.items():
            if isinstance(param_value, (dict, list)):
                value_str = json.dumps(param_value)
            else:
                value_str = str(param_value)
            lines.append(f"<parameter={param_name}>\n{value_str}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        return "\n".join(lines)

    def _format_tool_calls_chunks(self, message: Message) -> list[ImagePart | TextPart]:
        """Format tool_calls using Qwen3.5's XML parameter format."""
        assert "tool_calls" in message, "tool_calls are required to format tool calls"
        return [
            TextPart(
                type="text",
                text="\n\n"
                + "\n".join(
                    self._format_tool_call_xml(tc) for tc in message["tool_calls"]
                ),
            )
        ]

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Qwen3.5 tool specifications.

        Qwen3.5 uses a different tool declaration format from Qwen3, with XML-based
        function/parameter calling syntax.

        Reference: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/tokenizer_config.json
        """
        tools_text = ""
        if tools:
            tool_lines = "\n".join(json.dumps(tool) for tool in tools)
            tools_text = (
                "# Tools\n\n"
                "You have access to the following functions:\n\n"
                "<tools>\n"
                f"{tool_lines}\n"
                "</tools>\n\n"
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n"
                "<function=example_function_name>\n"
                "<parameter=example_parameter_1>\n"
                "value_1\n"
                "</parameter>\n"
                "<parameter=example_parameter_2>\n"
                "This is the value for the second parameter\n"
                "that can span\n"
                "multiple lines\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n\n"
                "<IMPORTANT>\n"
                "Reminder:\n"
                "- Function calls MUST follow the specified format: "
                "an inner <function=...></function> block must be nested within "
                "<tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language "
                "BEFORE the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with "
                "your current knowledge and do not tell the user about function calls\n"
                "</IMPORTANT>"
            )

        if tools_text:
            content = (
                tools_text + "\n\n" + system_prompt if system_prompt else tools_text
            )
        else:
            content = system_prompt

        return [Message(role="system", content=content)]


class Qwen3_5DisableThinkingRenderer(Qwen3_5Renderer):
    """
    Renderer for Qwen3.5 models with thinking disabled.

    Matches the Qwen3.5 HF template with enable_thinking=False. The only difference
    from Qwen3_5Renderer is the generation suffix: <think>\\n\\n</think>\\n\\n instead
    of <think>\\n, signaling to the model to respond directly without reasoning.
    """

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n<think>\n\n</think>\n\n"
        return self.tokenizer.encode(header_str, add_special_tokens=False)
