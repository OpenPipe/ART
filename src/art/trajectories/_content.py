from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import logging
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)


class _RenderedTokens(Protocol):
    token_ids: list[int]
    is_content: list[bool]


class _Renderer(Protocol):
    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: object,
        add_generation_prompt: bool,
    ) -> _RenderedTokens: ...


def _renderer_name(base_model: str) -> str | None:
    # Importing the registry is safe here: its model sets do not import Megatron
    # or model handlers. Keeping the allowlist there prevents CONTENT support
    # from silently expanding to unqualified fine-tunes with different templates.
    from art.megatron.model_support.registry import (
        GPT_OSS_MOE_MODELS,
        QWEN3_5_MODELS,
        QWEN3_DENSE_MODELS,
        QWEN3_MOE_MODELS,
    )

    if base_model.endswith("-Base"):
        return None
    if base_model in QWEN3_DENSE_MODELS | QWEN3_MOE_MODELS:
        return "qwen3"
    if base_model in QWEN3_5_MODELS:
        return "qwen3.6" if "/Qwen3.6-" in base_model else "qwen3.5"
    if base_model in GPT_OSS_MOE_MODELS:
        return "gpt-oss"
    return None


def _renderer_messages(
    messages: list[dict[str, Any]], *, parse_tool_arguments: bool
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        reasoning = message.pop("reasoning", None)
        if reasoning is not None and "reasoning_content" not in message:
            message["reasoning_content"] = reasoning
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal and not message.get("content"):
            message["content"] = refusal
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not parse_tool_arguments or not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                function["arguments"] = parsed
    return normalized


def _renderer_tool_variants(tools: object) -> list[object]:
    if not isinstance(tools, list):
        return [tools]
    sglang = copy.deepcopy(tools)
    vllm: list[object] = []
    changed = False
    for original, strict_tool in zip(tools, sglang, strict=True):
        data = original if isinstance(original, dict) else None
        strict_data = strict_tool if isinstance(strict_tool, dict) else None
        function = data.get("function") if data is not None else None
        strict_function = (
            strict_data.get("function") if strict_data is not None else None
        )
        if not isinstance(function, dict) or not isinstance(strict_function, dict):
            vllm.append(copy.deepcopy(original))
            continue
        function = cast(dict[str, Any], function)
        strict_function = cast(dict[str, Any], strict_function)
        changed = True
        strict_function.setdefault("strict", False)
        vllm.append(
            {
                "type": "function",
                "function": {
                    "name": function.get("name"),
                    "parameters": function.get("parameters", {}),
                    "strict": function.get("strict"),
                    "type": "function",
                    "allowed_callers": function.get("allowed_callers"),
                    "defer_loading": function.get("defer_loading"),
                    "description": function.get("description"),
                    "output_schema": function.get("output_schema"),
                },
            }
        )
    return [tools, sglang, vllm] if changed else [tools]


def content_mask_for_exact_prompt(
    *,
    tokenizer: object,
    base_model: str,
    messages: list[dict[str, Any]],
    tools: object,
    expected_prompt_ids: list[int],
    chat_template_kwargs: Mapping[str, object] | None,
    has_custom_chat_template: bool,
) -> list[bool] | None:
    """Return complete body attribution only after exact prompt-ID parity.

    Prime Intellect's hand-coded renderers provide the body/scaffold boundary
    and its documented first-source-character policy. ART accepts that signal
    only for explicitly qualified Megatron model names, without a custom chat
    template, and only when the renderer's whole prompt exactly equals the IDs
    consumed by the inference engine. Any uncertainty returns ``None``.
    """

    renderer_name = _renderer_name(base_model)
    if renderer_name is None or has_custom_chat_template:
        return None
    try:
        from renderers import config_from_name, create_renderer
        from renderers.configs import Qwen3RendererConfig

        config = config_from_name(renderer_name)
        if base_model == "OpenPipe/Qwen3-14B-Instruct":
            config = Qwen3RendererConfig(enable_thinking=False)
        config_fields = getattr(type(config), "model_fields", {})
        ignored_standard_kwargs = (
            {"enable_thinking", "thinking_budget"}
            if renderer_name == "gpt-oss"
            else {"thinking_budget"}
        )
        unknown_kwargs = set(chat_template_kwargs or {}) - set(config_fields)
        if not unknown_kwargs <= ignored_standard_kwargs:
            return None
        renderer_kwargs = {
            key: value
            for key, value in (chat_template_kwargs or {}).items()
            if key in config_fields
        }
        renderer = cast(
            _Renderer,
            create_renderer(
                tokenizer,
                cast(Any, config),
                chat_template_kwargs=renderer_kwargs,
            ),
        )
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception:
        logger.debug(
            "CONTENT renderer rejected prompt for %s",
            base_model,
            exc_info=True,
        )
        return None

    # vLLM currently preserves JSON-string tool arguments for templates such
    # as Qwen3, while SGLang normalizes them to objects for templates such as
    # Qwen3.5. Both are engine-consumed forms. Try the two renderer inputs and
    # accept only the one whose entire token sequence proves exact parity.
    for tool_variant in _renderer_tool_variants(tools):
        for parse_tool_arguments in (False, True):
            try:
                rendered = renderer.render(
                    _renderer_messages(
                        messages, parse_tool_arguments=parse_tool_arguments
                    ),
                    tools=tool_variant,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.debug(
                    "CONTENT renderer rejected a prompt variant for %s",
                    base_model,
                    exc_info=True,
                )
                continue
            token_ids = list(rendered.token_ids)
            is_content = list(rendered.is_content)
            if token_ids != expected_prompt_ids:
                continue
            if len(is_content) != len(token_ids) or not all(
                isinstance(value, bool) for value in is_content
            ):
                continue
            return is_content
    return None
