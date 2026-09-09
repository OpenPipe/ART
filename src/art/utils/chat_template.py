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
    "or loop.index0 > ns_turn.last_user_idx) and message.get('tool_calls')"
)
_MINIMAX_DROP_PRIOR_THINKING = "reasoning_content and loop.index0 > ns.last_user_index"
_MINIMAX_PRESERVE_PRIOR_THINKING = (
    "reasoning_content and ((preserve_thinking is defined and preserve_thinking is "
    "true) or loop.index0 > ns.last_user_index)"
)


def chat_template_with_preserved_thinking(chat_template: object) -> object:
    """Add opt-in prior-turn reasoning gates to supported templates."""
    if not isinstance(chat_template, str):
        return chat_template
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
    )
    for old, new, supported in replacements:
        if supported and chat_template.count(old) == 1:
            chat_template = chat_template.replace(old, new)
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
    kwargs: dict[str, Any] = {}
    if not isinstance(chat_template, str):
        return kwargs
    if "enable_thinking" in chat_template:
        kwargs["enable_thinking"] = False
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
