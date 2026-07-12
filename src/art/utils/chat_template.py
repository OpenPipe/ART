from typing import Any

THINKING_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {
    "enable_thinking": False,
    "preserve_thinking": True,
}
TOOL_CALL_ARGUMENTS_AS_MAPPING_ATTR = "_art_tool_call_arguments_as_mapping"


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
