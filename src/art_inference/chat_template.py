import json
import re
from typing import Any

THINKING_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {
    "enable_thinking": False,
    "preserve_thinking": True,
}
TOOL_CALL_ARGUMENTS_AS_MAPPING_ATTR = "_art_tool_call_arguments_as_mapping"
_QWEN_DROP_PRIOR_THINKING = "{%- if loop.index0 > ns.last_query_index %}"
_QWEN_PRESERVE_PRIOR_THINKING = (
    "{%- if (preserve_thinking is defined and preserve_thinking is true) or "
    "(loop.index0 > ns.last_query_index) %}"
)
_GEMMA_DROP_PRIOR_THINKING = (
    "thinking_text and loop.index0 > ns_turn.last_user_idx and "
    "message.get('tool_calls')"
)
_GEMMA_PRESERVE_PRIOR_THINKING = (
    "thinking_text and ((preserve_thinking is defined and preserve_thinking is true) "
    "or (loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')))"
)
_MINIMAX_DROP_PRIOR_THINKING = "reasoning_content and loop.index0 > ns.last_user_index"
_MINIMAX_PRESERVE_PRIOR_THINKING = (
    "reasoning_content and ((preserve_thinking is defined and preserve_thinking is "
    "true) or loop.index0 > ns.last_user_index)"
)
# These operations infer reasoning from arbitrary assistant content and can
# discard everything before the last <think> or between repeated </think> tags.
# Match the operations, not a model revision or the text of a particular answer.
_QWEN_INLINE_REASONING = re.compile(
    r"\s*".join(
        r"\{%[-+]?\s*" + re.escape(statement) + r"\s*[-+]?%\}"
        for statement in (
            "if '</think>' in content",
            "set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "set content = content.split('</think>')[-1].lstrip('\\n')",
            "endif",
        )
    )
)


def _without_inline_reasoning_parser(template: str) -> str:
    matches = list(_QWEN_INLINE_REASONING.finditer(template))
    if not matches:
        return template
    from jinja2 import Environment, TemplateSyntaxError

    # Only executable block tokens may be edited. The same spelling inside a
    # quoted expression, raw block or comment is literal template data.
    # Jinja lexes normalized newlines; map its positions to the original source.
    normalized = re.sub(r"\r\n?", "\n", template)
    offsets = [
        i
        for i, char in enumerate(template)
        if not (char == "\n" and i and template[i - 1] == "\r")
    ]
    starts: set[int] = set()
    cursor = 0
    try:
        for _, kind, value in Environment().lex(template):
            start = normalized.find(value, cursor)
            if start < 0 or normalized[cursor:start].strip():
                return template  # Lexer normalization could not be source-joined.
            if kind == "block_begin":
                starts.add(offsets[start])
            cursor = start + len(value)
    except TemplateSyntaxError:
        return template  # Leave invalid templates to their existing renderer.
    edits = {
        (match.start(), match.end()): "" for match in matches if match.start() in starts
    }
    if not edits:
        return template
    # Dropping structured reasoning must not trim the visible assistant body.
    for content in (
        "render_content(message.content, true)|trim",
        "(render_content(message.content, true) if preserve_thinking and message.role == 'assistant' else render_content(message.content, true)|trim)",
    ):
        statement = "{%- set content = " + content + " %}"
        for match in re.finditer(re.escape(statement), template):
            if match.start() in starts:
                edits[match.span()] = (
                    "{%- set content = (render_content(message.content, true) if message.role == 'assistant' else render_content(message.content, true)|trim) %}"
                )
    for (start, end), replacement in sorted(edits.items(), reverse=True):
        template = template[:start] + replacement + template[end:]
    return template


def chat_template_with_preserved_thinking(chat_template: object) -> object:
    """Preserve structured reasoning without interpreting tags in plain content."""
    if isinstance(chat_template, dict):
        return {
            name: chat_template_with_preserved_thinking(template)
            for name, template in chat_template.items()
        }
    if not isinstance(chat_template, str):
        return chat_template
    chat_template = _without_inline_reasoning_parser(chat_template)
    replacements = (
        (
            _QWEN_DROP_PRIOR_THINKING,
            _QWEN_PRESERVE_PRIOR_THINKING,
            "enable_thinking" in chat_template,
        ),
        (
            _GEMMA_DROP_PRIOR_THINKING,
            _GEMMA_PRESERVE_PRIOR_THINKING,
            True,
        ),
        (
            _MINIMAX_DROP_PRIOR_THINKING,
            _MINIMAX_PRESERVE_PRIOR_THINKING,
            True,
        ),
        (
            "(loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls'))",
            "preserve_thinking or (loop.index0 > ns_turn.last_user_idx)",
            True,
        ),
        *(
            (
                f"message.{field} and not future_final_message.found",
                f"message.{field} and ((preserve_thinking is defined and preserve_thinking is true) or not future_final_message.found)",
                True,
            )
            for field in ("content", "thinking")
        ),
        (
            "{#- CoT is dropped during all previous turns, so we never render it for inference #}",
            "{%- if preserve_thinking is defined and preserve_thinking is true and message.thinking is defined %}<|start|>assistant<|channel|>analysis<|message|>{{ message.thinking }}<|end|>{%- endif %}",
            True,
        ),
        (
            '"<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>"',
            '"<|start|>assistant<|channel|>final<|message|>" + message.content + ("<|return|>" if preserve_thinking else "<|end|>")',
            True,
        ),
    )
    for old, new, supported in replacements:
        if supported and chat_template.count(old) == 1:
            chat_template = chat_template.replace(old, new)
    # Kimi 2.5 splits historical turns into a branch that blanks reasoning.
    # Preserve them using its ordinary assistant-turn branch instead.
    if (
        "set hist_msgs = messages[:ns.last_non_tool_call_assistant_msg+1]"
        in chat_template
    ):
        chat_template = (
            chat_template.replace(
                "set hist_msgs = messages[:ns.last_non_tool_call_assistant_msg+1]",
                "set hist_msgs = [] if preserve_thinking else messages[:ns.last_non_tool_call_assistant_msg+1]",
            )
            .replace(
                "set suffix_msgs = messages[ns.last_non_tool_call_assistant_msg+1:]",
                "set suffix_msgs = messages if preserve_thinking else messages[ns.last_non_tool_call_assistant_msg+1:]",
            )
            .replace(
                "{%- if thinking is defined and thinking is false -%}",
                "{%- if thinking is defined and thinking is false and not preserve_thinking -%}",
                1,
            )
        )
    # Qwen's native reasoning parser returns the sampled whitespace. The stock
    # template trims it, then invents a newline before </think>. Preserve the
    # parsed field verbatim; keep the original non-thinking/legacy fallback.
    if (
        "<think>" in chat_template
        and "reasoning_content|trim" in chat_template
        and "if not preserve_thinking or message.reasoning_content" not in chat_template
    ):
        chat_template = chat_template.replace(
            "{%- set reasoning_content = reasoning_content|trim %}",
            "{%- if not preserve_thinking or message.reasoning_content is not string %}"
            "{%- set reasoning_content = reasoning_content|trim %}{%- endif %}",
        )
        chat_template = chat_template.replace(
            "reasoning_content + '\\n</think>\\n\\n'",
            "reasoning_content + ('</think>\\n\\n' if preserve_thinking and message.reasoning_content is string and reasoning_content else '\\n</think>\\n\\n')",
        )
        chat_template = chat_template.replace(
            "set content = render_content(message.content, true)|trim",
            "set content = (render_content(message.content, true) if preserve_thinking and message.role == 'assistant' else render_content(message.content, true)|trim)",
        )
    if "clear_thinking" in chat_template:
        chat_template = chat_template.replace(
            "{{ content.strip() }}",
            "{{ content if not clear_thinking else content.strip() }}",
        ).replace(
            "{%- if content.strip() -%}",
            "{%- if (content if not clear_thinking else content.strip()) -%}",
        )
    if "ns_turn.last_user_idx" in chat_template:
        for value in ("message['content']", "item['text']"):
            chat_template = chat_template.replace(
                f"{{{{- {value} | trim -}}}}",
                f"{{{{- {value} if preserve_thinking and message.role == 'assistant' else {value} | trim -}}}}",
            )
    # Embed preservation defaults so direct tokenizer use and inference engines
    # agree with ART. Passing a value explicitly still takes precedence.
    for name, value in (("preserve_thinking", "true"), ("clear_thinking", "false")):
        default = f"{{%- set {name} = {name} | default({value}) -%}}"
        if name in chat_template and default not in chat_template:
            chat_template = default + chat_template
    return chat_template


def configure_preserved_thinking_chat_template(tokenizer: object) -> object:
    chat_template = getattr(tokenizer, "chat_template", None)
    configured = chat_template_with_preserved_thinking(chat_template)
    if configured != chat_template:
        setattr(tokenizer, "chat_template", configured)
    return tokenizer


def default_chat_template_kwargs_for_template(
    chat_template: object,
) -> dict[str, Any]:
    kwargs = default_preservation_kwargs_for_template(chat_template)
    if not isinstance(chat_template, str):
        return kwargs
    if "enable_thinking" in chat_template:
        kwargs["enable_thinking"] = False
    return kwargs


def default_preservation_kwargs_for_template(chat_template: object) -> dict[str, Any]:
    """History defaults independent of whether the next turn should think."""
    kwargs: dict[str, Any] = {}
    if not isinstance(chat_template, str):
        return kwargs
    if "preserve_thinking" in chat_template:
        kwargs["preserve_thinking"] = True
    if "clear_thinking" in chat_template:
        kwargs["clear_thinking"] = False
    if "deepseek_v4_python_encoder" in chat_template:
        kwargs["drop_thinking"] = False
    return kwargs


def default_chat_template_kwargs_for_tokenizer(tokenizer: object) -> dict[str, Any]:
    return default_chat_template_kwargs_for_template(
        getattr(tokenizer, "chat_template", None)
    )


def merge_chat_template_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**(defaults or {}), **(overrides or {})}


def _template_requires_structured_tool_arguments(chat_template: object) -> bool:
    if not isinstance(chat_template, str):
        return False
    arguments_access = (
        r"(?:(?<![.\w])(?:tool_call|tc|function)\s*(?:\.\s*"
        r"(?:function\s*\.\s*)?arguments\b|\[\s*['\"]arguments['\"]\s*\])"
        r"|(?<![.\w])arguments\b)"
    )
    if re.search(rf"{arguments_access}\s*\|\s*items\b", chat_template):
        return True
    if re.search(rf"{arguments_access}\s*\.\s*items\s*\(", chat_template):
        return True
    if re.search(rf"{arguments_access}\s+is\s+mapping\b", chat_template):
        return True
    aliases = re.findall(
        r"{%[-+]?\s*set\s+([A-Za-z_]\w*)\s*=\s*([^%]*)[-+]?%}",
        chat_template,
    )
    return any(
        re.search(arguments_access, expression)
        and re.search(rf"\b{re.escape(alias)}\s*\.\s*items\s*\(", chat_template)
        for alias, expression in aliases
    )


def normalize_tool_call_arguments_for_chat_template(
    messages: list[dict[str, Any]],
    chat_template: object,
    *,
    require_mapping: bool = False,
) -> list[dict[str, Any]]:
    """Give chat templates the structured tool arguments they require.

    Templates that interpolate the raw JSON string must keep string arguments,
    so only templates that iterate structured arguments trigger normalization.
    """
    if not require_mapping and not _template_requires_structured_tool_arguments(
        chat_template
    ):
        return messages
    normalized: list[dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            normalized.append(message)
            continue
        normalized_calls = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = (
                function.get("arguments") if isinstance(function, dict) else None
            )
            if isinstance(arguments, str):
                assert isinstance(function, dict)
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "tool-call arguments are not valid JSON"
                    ) from error
                if not isinstance(arguments, dict):
                    raise ValueError("tool-call arguments must decode to a JSON object")
                call = {**call, "function": {**function, "arguments": arguments}}
            normalized_calls.append(call)
        normalized.append({**message, "tool_calls": normalized_calls})
    return normalized
