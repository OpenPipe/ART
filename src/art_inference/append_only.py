"""Relate an engine's rendered assistant message to its original sampled IDs.

Only inference may establish these mappings, using its own parsed response and
renderer. Training tokenization must continue to use the actual served tokens.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
import json
import sys
from typing import Any

from .token_prefix import PrefixEdit, prefix_edits

PrefixObservation = tuple[list[int], list[int], tuple[PrefixEdit, ...]]


def _rendering_edits(tokenizer, rendered, raw):
    # Stable protocol/multimodal markers separate independent text edits. Keep
    # them outside replacements so native image offsets remain translatable.
    special = set(getattr(tokenizer, "all_special_ids", ()))
    source = [(i, token) for i, token in enumerate(rendered) if token in special]
    target = [(i, token) for i, token in enumerate(raw) if token in special]
    if not source or [t for _, t in source] != [t for _, t in target]:
        return prefix_edits(rendered, raw)
    edits = []
    start = raw_start = 0
    for (stop, _), (raw_stop, _) in zip(
        [*source, (len(rendered), None)], [*target, (len(raw), None)], strict=True
    ):
        edits.extend(
            PrefixEdit(start + edit.start, start + edit.stop, edit.replacement)
            for edit in prefix_edits(rendered[start:stop], raw[raw_start:raw_stop])
        )
        start, raw_start = stop + 1, raw_stop + 1
    return tuple(edits)


def patch_deepseek_renderer(
    module: Any, preserves: Callable[[], bool], *, prefix_only: bool = False
) -> None:
    """Render historical DeepSeek turns in their original reasoning mode."""
    render = module.render_message

    @wraps(render)
    def render_message(index, messages, thinking_mode, *args, **kwargs):
        if preserves():
            # A generation setting controls the next answer. Parsed historical
            # answers already specify whether that turn included reasoning.
            for message in messages[index:]:
                if message.get("role") == "assistant":
                    thinking_mode = (
                        "thinking"
                        if any(
                            message.get(name) is not None
                            for name in ("reasoning", "reasoning_content", "thinking")
                        )
                        else "chat"
                    )
                    break
                if message is not messages[index] and message.get("role") in (
                    "user",
                    "developer",
                ):
                    break
            # V3.2 also gates reasoning and the preceding <think> on the last
            # user in the full chat, even with drop_thinking=False.
            if prefix_only:
                messages = messages[: index + 1]
                message = messages[index]
                if (
                    message.get("role") == "assistant"
                    and thinking_mode == "thinking"
                    and not any(
                        message.get(name)
                        for name in (
                            "reasoning",
                            "reasoning_content",
                            "thinking",
                            "tool_calls",
                        )
                    )
                ):
                    # V3.2 rejects an empty reasoning field, although sampling
                    # </think> immediately is valid. Preserve that exact block.
                    return module.thinking_end_token + render(
                        index, messages, "chat", *args, **kwargs
                    )
        elif not prefix_only:
            # V4's encoder overrides drop_thinking when tools are present. An
            # explicit request to remove history still controls each rendering.
            kwargs["drop_thinking"] = True
        return render(index, messages, thinking_mode, *args, **kwargs)

    module.render_message = render_message


def patch_harmony(module: Any, preserves: Callable[[], bool], namespace: str) -> None:
    """Keep Harmony analysis and final-turn stops, including imported aliases."""
    original = module.render_for_completion

    @wraps(original)
    def render(messages):
        if not preserves():
            return original(messages)
        from openai_harmony import Conversation, RenderConversationConfig, Role

        encoding = module.get_encoding()
        config = RenderConversationConfig(auto_drop_analysis=False)
        # Harmony's whole-conversation renderer changes prior <|return|> to
        # <|end|>. Rendering each completed message keeps its original framing.
        tokens = [
            token
            for message in messages
            for token in encoding.render_conversation_for_training(
                Conversation.from_messages([message]), config=config
            )
        ]
        return tokens + encoding.render_conversation_for_completion(
            Conversation.from_messages([]), Role.ASSISTANT, config=config
        )

    replacements = {original: render}
    drop = getattr(module, "auto_drop_analysis_messages", None)
    if drop is not None:
        replacements[drop] = lambda messages: (
            messages if preserves() else drop(messages)
        )
    for name, loaded in tuple(sys.modules.items()):
        if loaded is not None and name.startswith(namespace):
            for attribute, value in tuple(vars(loaded).items()):
                for old, new in replacements.items():
                    if value is old:
                        setattr(loaded, attribute, new)
    module.render_for_completion = render
    if drop is not None:
        module.auto_drop_analysis_messages = replacements[drop]


def shifted_span(start: int, length: int, edits: Sequence[PrefixEdit]) -> int:
    """Move a multimodal span after text edits without changing its contents."""
    shift = 0
    for edit in edits:
        if edit.stop <= start:
            shift += len(edit.replacement) - (edit.stop - edit.start)
        elif edit.start < start + length:
            raise ValueError("History token edit intersects a multimodal placeholder")
    return start + shift


def aligned_values(
    values: Any, edits: Sequence[PrefixEdit], *, positions=False, fill=None
):
    """Translate per-token metadata through edits confined to text tokens."""
    if positions:
        import torch

        parts, start, shift = [], 0, 0
        for edit in edits:
            parts.append(values[..., start : edit.start] + shift)
            origin = (
                values[..., edit.start : edit.start + 1]
                if edit.start < values.shape[-1]
                else values[..., -1:] + 1
            )
            parts.append(
                origin + shift + values.new_tensor(list(range(len(edit.replacement))))
            )
            shift += len(edit.replacement) - (edit.stop - edit.start)
            start = edit.stop
        return torch.cat([*parts, values[..., start:] + shift], dim=-1)
    result = list(values)
    for edit in reversed(edits):
        replaced = result[edit.start : edit.stop]
        value = (
            fill
            if fill is not None
            else (result[min(edit.start, len(result) - 1)] if result else 0)
        )
        if any(item != value for item in replaced):
            raise ValueError("History token edit crosses a per-token metadata boundary")
        result[edit.start : edit.stop] = [value] * len(edit.replacement)
    return values.new_tensor(result) if hasattr(values, "new_tensor") else result


def output_prefix_observations(
    tokenizer: Any,
    rendered_prompt: Sequence[int],
    raw_prompt: Sequence[int],
    raw_output: Sequence[int],
    *,
    prompt_edits: Sequence[PrefixEdit] | None = None,
) -> list[PrefixObservation]:
    """Preserve native IDs for the response and its last reasoning boundary.

    In particular, reasoning can survive a later action edit, for streaming and
    non-streaming generations in any protocol. A decode/encode pass must retain
    the same special-token sequence before we use an intermediate boundary.
    Keep at most two entries so tool-heavy outputs do not multiply prompt storage.
    """
    output = list(raw_output)
    if not output:
        return []
    rendered = list(
        tokenizer.encode(_decode(tokenizer, output), add_special_tokens=False)
    )
    if not rendered:
        return []
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    raw_boundaries = [
        (i + 1, token) for i, token in enumerate(output) if token in special_ids
    ]
    rendered_boundaries = [
        (i + 1, token) for i, token in enumerate(rendered) if token in special_ids
    ]
    boundaries = []
    if [token for _, token in raw_boundaries] == [
        token for _, token in rendered_boundaries
    ]:
        boundaries = [
            (r[0], n[0])
            for r, n in zip(rendered_boundaries, raw_boundaries, strict=True)
            if _decode(tokenizer, [r[1]]) in {"</think>", "<channel|>", "<|end|>"}
        ][-1:]
    if not boundaries or boundaries[-1] != (len(rendered), len(output)):
        boundaries.append((len(rendered), len(output)))
    entries = []
    prompt_edits = (
        tuple(prompt_edits)
        if prompt_edits is not None
        else _rendering_edits(tokenizer, rendered_prompt, raw_prompt)
    )
    offset = len(rendered_prompt)
    for rendered_end, raw_end in boundaries:
        edits = tuple(
            PrefixEdit(offset + edit.start, offset + edit.stop, edit.replacement)
            for edit in _rendering_edits(
                tokenizer, rendered[:rendered_end], output[:raw_end]
            )
        )
        entries.append(
            (
                [*rendered_prompt, *rendered[:rendered_end]],
                [*raw_prompt, *output[:raw_end]],
                (*prompt_edits, *edits),
            )
        )
    return entries


def merge_chat_delta(message: dict[str, Any], delta: dict[str, Any]) -> None:
    for name, value in delta.items():
        if name == "tool_calls" and isinstance(value, list):
            calls = message.setdefault("tool_calls", [])
            for part in value:
                index = part["index"]
                while len(calls) <= index:
                    calls.append({})
                merge_chat_delta(
                    calls[index], {k: v for k, v in part.items() if k != "index"}
                )
        elif isinstance(value, dict):
            merge_chat_delta(message.setdefault(name, {}), value)
        elif isinstance(value, str):
            message[name] = (
                value
                if name in {"role", "type", "id"}
                else message.get(name, "") + value
            )


def preserves_history(kwargs: Mapping[str, Any] | None) -> bool:
    options = kwargs or {}
    return not (
        options.get("preserve_thinking") is False
        or options.get("clear_thinking") is True
        or options.get("drop_thinking") is True
    )


def has_renderable_tool_arguments(message: Mapping[str, Any]) -> bool:
    functions = [call.get("function", {}) for call in message.get("tool_calls") or []]
    if message.get("function_call"):
        functions.append(message["function_call"])
    try:
        return all(
            not isinstance(f.get("arguments"), str)
            or isinstance(json.loads(f["arguments"] or "{}"), dict)
            for f in functions
        )
    except json.JSONDecodeError:
        return False


def _decode(tokenizer: Any, tokens: Sequence[int]) -> str:
    return tokenizer.decode(list(tokens), skip_special_tokens=False)


def _whitespace_prefix(tokenizer: Any, rendered: list[int], raw: list[int]) -> int:
    """Find an exact raw-token boundary for a whitespace-normalized prefix."""
    target = _decode(tokenizer, rendered)
    source = _decode(tokenizer, raw)
    i = j = 0
    while i < len(target):
        if target[i].isspace():
            i += 1
        elif j < len(source) and source[j].isspace():
            j += 1
        elif j < len(source) and target[i] == source[j]:
            i += 1
            j += 1
        else:
            return 0
    if target and target[-1].isspace():
        while j < len(source) and source[j].isspace():
            j += 1
    # Decoding keeps the native IDs, including non-canonical tokenizations. A
    # boundary inside a token cannot be preserved independently of the action.
    low, high = 0, len(raw)
    while low < high:
        middle = (low + high) // 2
        if len(_decode(tokenizer, raw[:middle])) < j:
            low = middle + 1
        else:
            high = middle
    return low if _decode(tokenizer, raw[:low]) == source[:j] else 0


def chat_prefix_observations(
    tokenizer: Any,
    rendered_prompt: Sequence[int],
    completed_prompt: Sequence[int],
    raw_prompt: Sequence[int],
    raw_output: Sequence[int],
    *,
    reasoning_prompt: Sequence[int] | None = None,
    complete: bool = True,
) -> list[PrefixObservation]:
    """Record a native response and, separately, its unchanged reasoning.

    ``completed_prompt`` must come from rendering the original request plus the
    engine's parsed response. ``reasoning_prompt`` renders the same request with
    only that response's reasoning. Keeping the latter mapping lets callers edit
    the action without losing the exact reasoning prefix. Explicit requests to
    drop thinking must bypass observation and lookup.

    Truncated responses may contribute a reasoning prefix, but cannot replace a
    completed turn's framing. No sampled IDs or log probabilities are invented.
    """
    prompt, completed = list(rendered_prompt), list(completed_prompt)
    output = list(raw_output)
    if not output or completed[: len(prompt)] != prompt:
        return []
    entries: list[PrefixObservation] = []
    if reasoning_prompt is not None:
        boundary = next(
            (i for i, (a, b) in enumerate(zip(completed, reasoning_prompt)) if a != b),
            min(len(completed), len(reasoning_prompt)),
        )
        if boundary > len(prompt):
            sampled_boundary = _whitespace_prefix(
                tokenizer, completed[len(prompt) : boundary], output
            )
            if sampled_boundary:
                rendered = completed[:boundary]
                raw = [*raw_prompt, *output[:sampled_boundary]]
                entries.append(
                    (rendered, raw, _rendering_edits(tokenizer, rendered, raw))
                )
    if complete and len(completed) > len(prompt):
        # A template's separator after the sampled stop belongs to the next
        # turn. Leave it in the rendered suffix instead of swallowing it.
        while (
            len(completed) > len(prompt)
            and _decode(tokenizer, completed[-1:]).isspace()
        ):
            completed.pop()
        # User stop strings (and APIs omitting the sampled stop ID) do not
        # prove the template's terminal framing. Never delete that framing.
        if completed[-1] != output[-1]:
            return entries
        raw = [*raw_prompt, *output]
        entries.append((completed, raw, _rendering_edits(tokenizer, completed, raw)))
    return entries


async def chat_response_prefixes(
    tokenizer: Any,
    request: Any,
    raw_prompt: Sequence[int],
    choices: Sequence[tuple[Mapping[str, Any], Sequence[int], bool]],
    render: Callable[[Any], Awaitable[list[int] | None]],
) -> list[PrefixObservation]:
    """Observe parsed native chat choices using that engine's request renderer."""
    payload = request.model_dump(mode="python")
    if not preserves_history(payload.get("chat_template_kwargs")):
        return []
    rendered = await render(request)
    messages = payload.get("messages")
    if rendered is None or not isinstance(messages, list):
        return []
    fields = type(request).model_fields
    payload.update(
        (name, value)
        for name, value in (
            ("add_generation_prompt", False),
            ("continue_final_message", False),
            ("input_ids", None),
            ("n", 1),
            ("stream", False),
            ("stream_options", None),
        )
        if name in fields
    )

    async def complete(message: Mapping[str, Any]) -> list[int] | None:
        return await render(
            type(request).model_validate({**payload, "messages": [*messages, message]})
        )

    entries: list[PrefixObservation] = []
    for message, output, finished in choices:
        # An invalid sampled call must reach the caller unchanged. It cannot be
        # rendered as a completed tool turn; native special-token observations
        # still preserve its reasoning prefix.
        if not has_renderable_tool_arguments(message):
            continue
        completed = await complete(message)
        if completed is None:
            continue
        reasoning = {
            key: message[key]
            for key in ("reasoning", "reasoning_content", "thinking")
            if message.get(key) is not None
        }
        reasoning_prompt = (
            await complete({"role": "assistant", "content": "", **reasoning})
            if reasoning
            else None
        )
        entries.extend(
            chat_prefix_observations(
                tokenizer,
                rendered,
                completed,
                raw_prompt,
                output,
                reasoning_prompt=reasoning_prompt,
                complete=finished,
            )
        )
    return entries
