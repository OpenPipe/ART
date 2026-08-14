import json
from typing import Any

THINKING_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {
    "enable_thinking": False,
    "preserve_thinking": True,
}
_QWEN_DROP_PRIOR_THINKING = "{%- if loop.index0 > ns.last_query_index %}"
_QWEN_PRESERVE_PRIOR_THINKING = (
    "{%- if (preserve_thinking is defined and preserve_thinking is true) or "
    "(loop.index0 > ns.last_query_index) %}"
)


def chat_template_with_preserved_thinking(chat_template: object) -> object:
    """Add Qwen's newer opt-in prior-turn reasoning gate to older templates."""
    if (
        not isinstance(chat_template, str)
        or "enable_thinking" not in chat_template
        or chat_template.count(_QWEN_DROP_PRIOR_THINKING) != 1
    ):
        return chat_template
    return chat_template.replace(
        _QWEN_DROP_PRIOR_THINKING, _QWEN_PRESERVE_PRIOR_THINKING
    )


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
    if "arguments|items" in chat_template.replace(" ", ""):
        return True
    return "arguments" in chat_template and ".items(" in chat_template


def normalize_tool_call_arguments_for_chat_template(
    messages: list[dict[str, Any]],
    chat_template: object,
) -> list[dict[str, Any]]:
    """Give chat templates the structured tool arguments they require.

    Templates that interpolate the raw JSON string must keep string arguments,
    so only templates that iterate structured arguments trigger normalization.
    """
    if not _template_requires_structured_tool_arguments(chat_template):
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
                    arguments = json.loads(arguments)
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
