"""Preserve sampled history before an ART-owned vLLM engine serves the next turn."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
import copy
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import importlib
import inspect
import json
from types import MethodType
from typing import Any

from .append_only import (
    aligned_values,
    chat_response_prefixes,
    merge_chat_delta,
    output_prefix_observations,
    patch_deepseek_renderer,
    patch_harmony,
    preserves_history,
    shifted_span,
)
from .chat_template import (
    chat_template_with_preserved_thinking,
    configure_preserved_thinking_chat_template,
)
from .token_prefix import TokenPrefixStore, prefix_edits


@dataclass
class _Turn:
    scope: str
    tokenizer: Any
    rendered: list[int] = field(default_factory=list)
    prompt: list[int] = field(default_factory=list)
    outputs: dict[int, tuple[list[int], bool]] = field(default_factory=dict)
    external_request: Any = None
    external_observer: Callable[[Any, list[Any]], Awaitable[None]] | None = None
    enabled: bool = True


_CURRENT: ContextVar[_Turn | None] = ContextVar("art_history", default=None)
_PREFIXES = TokenPrefixStore()
_DROP_THINKING: ContextVar[bool] = ContextVar("art_vllm_drop_thinking", default=False)


def _configure_native_tokenizer(tokenizer, importer):
    apply = getattr(tokenizer, "apply_chat_template", None)
    if (
        not callable(apply)
        or getattr(apply, "__module__", None) != "vllm.tokenizers.deepseek_v32"
    ):
        return
    module = importer("vllm.tokenizers.deepseek_v32")
    if not getattr(module, "_art_append_only", False):
        encode = module.encode_messages
        patch_deepseek_renderer(
            importer("vllm.tokenizers.deepseek_v32_encoding"),
            lambda: not _DROP_THINKING.get(),
            prefix_only=True,
        )

        @wraps(encode)
        def encode_messages(*args, **kwargs):
            kwargs["drop_thinking"] = _DROP_THINKING.get()
            return encode(*args, **kwargs)

        module.encode_messages = encode_messages
        module._art_append_only = True

    @wraps(apply)
    def apply_template(self, *args, **kwargs):
        token = _DROP_THINKING.set(not preserves_history(kwargs))
        try:
            return apply(*args, **kwargs)
        finally:
            _DROP_THINKING.reset(token)

    tokenizer.apply_chat_template = MethodType(apply_template, tokenizer)


def set_external_history_observer(
    observer: Callable[[Any, list[Any]], Awaitable[None]],
    importer=importlib.import_module,
) -> None:
    """Use a hosting service's transactional/distributed store for its requests."""
    base = importer("vllm.renderers.base").BaseRenderer
    base._art_history_external_observer = staticmethod(observer)


async def _observe(turn: _Turn, entries) -> None:
    if turn.external_request is not None:
        assert turn.external_observer is not None
        await turn.external_observer(turn.external_request, entries)
    else:
        for rendered, raw, edits in entries:
            _PREFIXES.insert(turn.scope, rendered, raw, "content", edits)


def _tokens(prompt: Any) -> list[int] | None:
    if isinstance(prompt, dict):
        tokens = prompt.get("prompt_token_ids")
        if isinstance(tokens, (list, tuple)):
            return list(tokens)
    return None


def replace_prompt_tokens(prompt, tokens, edits=None):
    """Keep native multimodal offsets and per-token metadata aligned."""
    result = {**prompt, "prompt_token_ids": tokens}
    if tokens == prompt["prompt_token_ids"]:
        return result
    edits = prefix_edits(prompt["prompt_token_ids"], tokens) if edits is None else edits
    if prompt.get("mm_placeholders"):
        result["mm_placeholders"] = {}
        for modality, spans in prompt["mm_placeholders"].items():
            result["mm_placeholders"][modality] = []
            for span in spans:
                shifted = copy.copy(span)
                shifted.offset = shifted_span(span.offset, span.length, edits)
                result["mm_placeholders"][modality].append(shifted)
    for name in (
        "is_token_ids",
        "prompt_is_token_ids",
        "token_type_ids",
        "assistant_tokens_mask",
    ):
        if prompt.get(name) is not None:
            fill = (
                True
                if name in ("is_token_ids", "prompt_is_token_ids")
                else (0 if name == "token_type_ids" else None)
            )
            result[name] = aligned_values(prompt[name], edits, fill=fill)
    # Text offsets refer to the normalized string, which is no longer the
    # source of these exact IDs. They are optional engine metadata.
    result.pop("prompt_token_offsets", None)
    return result


def patch_history(importer=importlib.import_module) -> None:
    base = importer("vllm.renderers.base").BaseRenderer
    if getattr(base, "_art_append_only", False):
        return
    init = base.__init__

    @wraps(init)
    def initialize(self, config, tokenizer):
        if tokenizer is not None:
            configure_preserved_thinking_chat_template(tokenizer)
            _configure_native_tokenizer(tokenizer, importer)
        init(self, config, tokenizer)

    base.__init__ = initialize
    base._art_append_only = True
    patch_harmony(
        importer("vllm.entrypoints.openai.parser.harmony_utils"),
        lambda: (turn := _CURRENT.get()) is None or turn.enabled,
        "vllm.",
    )

    params = importer("vllm.renderers.params").ChatParams
    apply_kwargs = params.get_apply_chat_template_kwargs

    @wraps(apply_kwargs)
    def template_kwargs(self):
        kwargs = {
            "preserve_thinking": True,
            "clear_thinking": False,
            "drop_thinking": False,
            **apply_kwargs(self),
        }
        if kwargs.get("chat_template") is not None:
            kwargs["chat_template"] = chat_template_with_preserved_thinking(
                kwargs["chat_template"]
            )
        return kwargs

    params.get_apply_chat_template_kwargs = template_kwargs

    responses_utils = importer("vllm.entrypoints.openai.responses.utils")
    responses_serving = importer("vllm.entrypoints.openai.responses.serving")
    construct = responses_utils.construct_input_messages

    @wraps(construct)
    def input_messages(*, prev_response_output=None, request_input, **kwargs):
        # vLLM's previous_response_id path otherwise drops reasoning and tool
        # calls, although its ordinary input path already knows how to retain
        # both. Keep the protocol's existing instructions replacement semantics.
        if prev_response_output:
            request_input = [
                *prev_response_output,
                *(
                    [{"role": "user", "content": request_input}]
                    if isinstance(request_input, str)
                    else request_input
                ),
            ]
        return construct(request_input=request_input, **kwargs)

    setattr(responses_utils, "construct_input_messages", input_messages)
    setattr(responses_serving, "construct_input_messages", input_messages)

    engine = importer("vllm.v1.engine.async_llm").AsyncLLM
    generate = engine.generate
    signature = inspect.signature(generate)

    @wraps(generate)
    def serve(self, prompt, *args, **kwargs):
        turn = _CURRENT.get()
        tokens = _tokens(prompt)
        if turn is None or not turn.enabled or tokens is None:
            return generate(self, prompt, *args, **kwargs)
        turn.rendered = tokens
        # Batched completions run several generators concurrently in one API
        # request. Each generator must publish only its own prompt and outputs.
        outputs = {}
        turn.outputs = outputs
        sampling = signature.bind(self, prompt, *args, **kwargs).arguments.get(
            "sampling_params"
        )
        delta = getattr(getattr(sampling, "output_kind", None), "name", None) == "DELTA"
        match = (
            _PREFIXES.lookup(turn.scope, tokens, None)
            if turn.external_request is None
            else None
        )
        raw_prompt = (
            list(match.raw_prefix) + tokens[match.rendered_length :]
            if match
            else tokens
        )
        turn.prompt = raw_prompt
        prompt = replace_prompt_tokens(prompt, raw_prompt, match.edits if match else ())

        async def stream():
            async for output in generate(self, prompt, *args, **kwargs):
                for choice in output.outputs:
                    generated_ids = list(choice.token_ids)
                    if delta:
                        generated_ids = (
                            outputs.get(choice.index, ([], False))[0] + generated_ids
                        )
                    outputs[choice.index] = (
                        generated_ids,
                        choice.finish_reason not in (None, "length", "abort"),
                    )
                yield output
            # Decode/re-encode differences matter for completions and for chat
            # renderers that already preserve the sampled text exactly.
            for raw, _ in outputs.values():
                if turn.external_request is None:
                    await _observe(
                        turn,
                        output_prefix_observations(
                            turn.tokenizer, tokens, raw_prompt, raw
                        ),
                    )

        return stream()

    engine.generate = serve
    chat = importer("vllm.entrypoints.openai.chat_completion.serving").OpenAIServingChat
    completion = importer(
        "vllm.entrypoints.openai.completion.serving"
    ).OpenAIServingCompletion

    def install(owner, method, *, observe_chat=False, observe_responses=False):
        original = getattr(owner, method)

        @wraps(original)
        async def create(self, request, raw_request=None):
            headers = getattr(raw_request, "headers", {}) or {}
            # Caladan supplies a distributed store and its own engine adapter.
            # Its observer uses the same shared ART implementation.
            external = bool(headers.get("x-caladan-prefix-scope"))
            external_observer = getattr(base, "_art_history_external_observer", None)
            if external and external_observer is None:
                return await original(self, request, raw_request)
            tokenizer = getattr(self.renderer, "tokenizer", None)
            if tokenizer is None:
                return await original(self, request, raw_request)
            material = [
                request.model,
                getattr(request, "chat_template", None),
                getattr(request, "chat_template_kwargs", None),
                headers.get("authorization", ""),
            ]
            scope = hashlib.sha256(
                json.dumps(material, sort_keys=True).encode()
            ).hexdigest()
            turn = _Turn(
                scope,
                tokenizer,
                external_request=raw_request if external else None,
                external_observer=external_observer,
                enabled=preserves_history(
                    getattr(request, "chat_template_kwargs", None)
                ),
            )

            async def observe(messages):
                if not turn.enabled or not observe_chat or not turn.prompt:
                    return

                async def render(value):
                    result = await self.render_chat_request(value)
                    if not isinstance(result, tuple) or len(result[1]) != 1:
                        return None
                    return _tokens(result[1][0])

                choices = [
                    (message, *turn.outputs[index])
                    for index, message in messages.items()
                    if index in turn.outputs
                ]
                entries = await chat_response_prefixes(
                    tokenizer, request, turn.prompt, choices, render
                )
                await _observe(turn, entries)

            async def observe_response(response):
                if (
                    not observe_responses
                    or not turn.enabled
                    or getattr(self, "use_harmony", False)
                    or not getattr(response, "output", None)
                    or not turn.prompt
                ):
                    return
                messages = responses_utils.construct_chat_messages_with_tool_call(
                    response.output
                )
                # Built-in tool execution can contain several generations. The
                # engine-level observations cover those individual turns.
                if len(messages) != 1 or messages[0].get("role") != "assistant":
                    return
                previous = None
                if request.previous_response_id:
                    async with self.response_store_lock:
                        previous = self.response_store.get(request.previous_response_id)
                conversation, inputs = await self._make_request(request, previous)
                if len(inputs) != 1:
                    return
                protocol = importer("vllm.entrypoints.openai.chat_completion.protocol")
                tools = responses_utils.construct_tool_dicts(
                    request.tools, request.tool_choice
                )
                view = protocol.ChatCompletionRequest(
                    model=request.model,
                    messages=conversation,
                    chat_template_kwargs=self._effective_chat_template_kwargs(request),
                )

                async def render(value):
                    online = getattr(self, "online_renderer", None)
                    if online is None:  # vLLM 0.24
                        online = self.openai_serving_render
                    _, inputs = await online.preprocess_chat(
                        value,
                        value.messages,
                        default_template=self.chat_template,
                        default_template_content_format=self.chat_template_content_format,
                        default_template_kwargs=self._effective_chat_template_kwargs(
                            request
                        ),
                        tool_dicts=tools,
                        parser=self.parser,
                    )
                    return _tokens(inputs[0]) if len(inputs) == 1 else None

                if await render(view) != _tokens(inputs[0]):
                    return
                if 0 in turn.outputs:
                    await _observe(
                        turn,
                        await chat_response_prefixes(
                            tokenizer,
                            view,
                            turn.prompt,
                            [(messages[0], *turn.outputs[0])],
                            render,
                        ),
                    )

            token = _CURRENT.set(turn)
            try:
                result = await original(self, request, raw_request)
                if not isinstance(result, AsyncIterator):
                    choices = getattr(result, "choices", ())
                    await observe(
                        {
                            c.index: c.message.model_dump(exclude_none=True)
                            for c in choices
                            if hasattr(c, "message")
                        }
                    )
                    await observe_response(result)
                    return result
            finally:
                _CURRENT.reset(token)

            async def stream():
                token = _CURRENT.set(turn)
                messages = {}
                try:
                    async for event in result:
                        if (
                            observe_chat
                            and isinstance(event, str)
                            and event.startswith("data: ")
                        ):
                            data = event[6:].strip()
                            if data == "[DONE]":
                                await observe(messages)
                            else:
                                for choice in json.loads(data).get("choices", []):
                                    message = messages.setdefault(
                                        choice["index"], {"role": "assistant"}
                                    )
                                    merge_chat_delta(message, choice.get("delta") or {})
                        if observe_responses and getattr(event, "type", None) in (
                            "response.completed",
                            "response.incomplete",
                        ):
                            await observe_response(event.response)
                        yield event
                finally:
                    _CURRENT.reset(token)

            return stream()

        setattr(owner, method, create)

    install(chat, "create_chat_completion", observe_chat=True)
    install(completion, "create_completion", observe_chat=False)
    install(
        responses_serving.OpenAIServingResponses,
        "create_responses",
        observe_responses=True,
    )
