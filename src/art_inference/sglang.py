"""History preservation for SGLang's native chat, Responses, and Messages APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
import copy
from functools import wraps
import importlib
import json
from typing import Any

from .append_only import (
    aligned_values,
    chat_response_prefixes,
    merge_chat_delta,
    patch_deepseek_renderer,
    patch_harmony,
    preserves_history,
    shifted_span,
)
from .chat_template import (
    chat_template_with_preserved_thinking,
    configure_preserved_thinking_chat_template,
)
from .token_prefix import apply_prefix_edits

_DROP_THINKING: ContextVar[bool] = ContextVar("art_sglang_drop_thinking", default=False)


def replace_prompt_tokens(tokenized, edits):
    """Translate expanded multimodal positions with the preserved text prefix."""
    if not edits:
        return
    mm = getattr(tokenized, "mm_inputs", None)
    if mm is not None:
        mm = copy.copy(mm)
        mm.mm_items = [copy.copy(item) for item in mm.mm_items]
        for item in mm.mm_items:
            if item.offsets is not None:
                item.offsets = [
                    (
                        shifted_span(start, end - start + 1, edits),
                        shifted_span(start, end - start + 1, edits) + end - start,
                    )
                    for start, end in item.offsets
                ]
        for name in ("input_ids", "padded_input_ids"):
            if getattr(mm, name, None) is not None:
                setattr(mm, name, apply_prefix_edits(list(getattr(mm, name)), edits))
        for name in ("mrope_positions", "token_type_ids", "visible_frame_counts"):
            if getattr(mm, name, None) is not None:
                setattr(
                    mm,
                    name,
                    aligned_values(
                        getattr(mm, name),
                        edits,
                        positions=name == "mrope_positions",
                        fill=0 if name == "token_type_ids" else None,
                    ),
                )
        tokenized.mm_inputs = mm
    if getattr(tokenized, "token_type_ids", None) is not None:
        tokenized.token_type_ids = aligned_values(
            tokenized.token_type_ids, edits, fill=0
        )


class _TokenizerView:
    def __init__(self, tokenizer: Any, generation: bool):
        self._tokenizer = tokenizer
        self._generation = generation

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["add_generation_prompt"] = self._generation
        return self._tokenizer.apply_chat_template(*args, **kwargs)


def patch_history(
    observe: Callable[[Any, list[Any]], Awaitable[None]],
    importer=importlib.import_module,
) -> None:
    chat = importer("sglang.srt.entrypoints.openai.serving_chat").OpenAIServingChat
    responses = importer(
        "sglang.srt.entrypoints.openai.serving_responses"
    ).OpenAIServingResponses
    if getattr(chat, "_art_append_only", False):
        return
    patch_harmony(
        importer("sglang.srt.entrypoints.harmony_utils"),
        lambda: not _DROP_THINKING.get(),
        "sglang.",
    )
    process = chat._process_messages

    def patch_encoder(encoding):
        encode = encoding.encode_messages

        @wraps(encode)
        def encode_messages(*args, **kwargs):
            kwargs.setdefault("drop_thinking", _DROP_THINKING.get())
            return encode(*args, **kwargs)

        encoding.encode_messages = encode_messages

    for name in ("dsv4", "dsv32"):
        encoding = importer(f"sglang.srt.entrypoints.openai.encoding_{name}")
        patch_encoder(encoding)
        patch_deepseek_renderer(
            encoding, lambda: not _DROP_THINKING.get(), prefix_only=name == "dsv32"
        )

    @wraps(process)
    def process_messages(self, request, is_multimodal):
        configure_preserved_thinking_chat_template(self.tokenizer_manager.tokenizer)
        if getattr(request, "chat_template", None) is not None:
            request.chat_template = chat_template_with_preserved_thinking(
                request.chat_template
            )
        # Templates already embed their defaults. Adding them to this request
        # would shadow a Responses adapter's outer rendering options. Likewise,
        # retain its encoder context unless this chat request overrides it.
        options = getattr(request, "chat_template_kwargs", None) or {}
        token = _DROP_THINKING.set(
            not preserves_history(options)
            if any(
                k in options
                for k in ("preserve_thinking", "clear_thinking", "drop_thinking")
            )
            else _DROP_THINKING.get()
        )
        try:
            return process(self, request, is_multimodal)
        finally:
            _DROP_THINKING.reset(token)

    chat._process_messages = process_messages

    async def record(self, request, raw_request, prompt, choices):
        if prompt is None or not preserves_history(
            getattr(request, "chat_template_kwargs", None)
        ):
            return

        async def render(value):
            view = copy.copy(self)
            generation = len(value.messages) == len(request.messages)
            if not generation:
                # Native request rendering converts a trailing assistant into
                # a user turn. This view renders an already completed response.
                view._handle_last_assistant_message = lambda messages, _: (
                    messages,
                    None,
                )
            view.tokenizer_manager = copy.copy(self.tokenizer_manager)
            view.tokenizer_manager.tokenizer = _TokenizerView(
                self.tokenizer_manager.tokenizer,
                generation,
            )
            result = view._process_messages(
                value,
                is_multimodal=self.tokenizer_manager.model_config.is_multimodal,
            ).prompt_ids
            return result if isinstance(result, list) and result else None

        entries = await chat_response_prefixes(
            self.tokenizer_manager.tokenizer, request, prompt, choices, render
        )
        if entries:
            await observe(raw_request, entries)

    full = chat._handle_non_streaming_request

    @wraps(full)
    async def complete(self, adapted_request, request, raw_request):
        result = await full(self, adapted_request, request, raw_request)
        payload = (
            result.model_dump(exclude_none=True)
            if hasattr(result, "model_dump")
            else json.loads(result.body)
        )
        prompt = payload.get("prompt_token_ids")
        choices = []
        for choice in payload.get("choices", []):
            prompt = prompt or choice.get("prompt_token_ids")
            ids = choice.get("token_ids") or (choice.get("logprobs") or {}).get(
                "token_ids"
            )
            if ids is not None:
                choices.append(
                    (
                        choice["message"],
                        ids,
                        choice.get("finish_reason") not in ("length", "abort"),
                    )
                )
        if choices:
            await record(self, request, raw_request, prompt, choices)
        return result

    chat._handle_non_streaming_request = complete
    generate = chat._generate_chat_stream

    @wraps(generate)
    async def stream(self, adapted_request, request, raw_request):
        messages, tokens, finished = {}, {}, {}
        prompt = None
        async for event in generate(self, adapted_request, request, raw_request):
            if isinstance(event, str) and event.startswith("data: "):
                data = event[6:].strip()
                if data == "[DONE]":
                    await record(
                        self,
                        request,
                        raw_request,
                        prompt,
                        [
                            (message, tokens[index], finished.get(index, False))
                            for index, message in messages.items()
                            if index in tokens
                        ],
                    )
                else:
                    payload = json.loads(data)
                    prompt = prompt or payload.get("prompt_token_ids")
                    for choice in payload.get("choices", []):
                        index = choice["index"]
                        merge_chat_delta(
                            messages.setdefault(index, {"role": "assistant"}),
                            choice.get("delta") or {},
                        )
                        if "token_ids" in choice:
                            tokens.setdefault(index, []).extend(choice["token_ids"])
                        if choice.get("finish_reason"):
                            finished[index] = choice["finish_reason"] not in (
                                "length",
                                "abort",
                            )
            yield event

    chat._generate_chat_stream = stream

    normalize = responses._normalize_response_message_for_chat

    @classmethod
    def normalize_message(cls, message):
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if isinstance(message, dict) and message.get("type") == "reasoning":
            # A summary may differ from the sampled reasoning. Joining content
            # parts with newlines would also alter the original response.
            parts = message.get("content") or message.get("summary") or []
            text = "".join(part.get("text", "") for part in parts)
            return {"role": "assistant", "reasoning_content": text} if text else None
        return normalize(message)

    responses._normalize_response_message_for_chat = normalize_message
    construct = responses._construct_input_messages

    @wraps(construct)
    def input_messages(self, request, prev_response=None):
        if prev_response is not None:
            inputs = (
                [{"role": "user", "content": request.input}]
                if isinstance(request.input, str)
                else request.input
            )
            request = request.model_copy(
                update={"input": [*prev_response.output, *inputs]}
            )
            prev_response = prev_response.model_copy(update={"output": []})
        return construct(self, request, prev_response)

    responses._construct_input_messages = input_messages
    harmony = responses._construct_input_messages_with_harmony

    @wraps(harmony)
    def harmony_messages(self, request, prev_response):
        if prev_response is None:
            return harmony(self, request, prev_response)
        # SGLang deletes reasoning from the stored list in place. Give it a
        # private list, and retain the original prefix unless rewriting was
        # explicitly requested for this call.
        previous = list(self.msg_store[prev_response.id])
        view = copy.copy(self)
        view.msg_store = {**self.msg_store, prev_response.id: previous.copy()}
        result = harmony(view, request, prev_response)
        retained = view.msg_store[prev_response.id]
        if preserves_history(getattr(request, "chat_template_kwargs", None)):
            if result[: len(retained)] != retained:
                raise RuntimeError("SGLang's stored-history rendering contract changed")
            return [*previous, *result[len(retained) :]]
        return result

    responses._construct_input_messages_with_harmony = harmony_messages

    create = responses.create_responses

    @wraps(create)
    async def create_responses(self, request, raw_request=None):
        async def record_response(payload):
            generations = payload.get("token_generations") or []
            if getattr(self, "use_harmony", False) or len(generations) != 1:
                return
            generation = generations[0]
            messages = self._merge_consecutive_assistant_messages(
                [
                    message
                    for item in payload.get("output", [])
                    if (message := self._normalize_response_message_for_chat(item))
                    is not None
                ]
            )
            if len(messages) != 1 or messages[0].get("role") != "assistant":
                return
            previous = (
                self.response_store.get(request.previous_response_id)
                if request.previous_response_id
                else None
            )
            protocol = importer("sglang.srt.entrypoints.openai.protocol")
            view = protocol.ChatCompletionRequest(
                model=request.model,
                messages=self._construct_input_messages(request, previous),
                tools=self._response_tools_to_chat_tools(request) or None,
                chat_template=getattr(request, "chat_template", None),
                chat_template_kwargs=getattr(request, "chat_template_kwargs", None),
            )
            await record(
                self,
                view,
                raw_request,
                generation["prompt_token_ids"],
                [
                    (
                        messages[0],
                        [token["token_id"] for token in generation["output_tokens"]],
                        payload.get("status") == "completed",
                    )
                ],
            )

        token = _DROP_THINKING.set(
            not preserves_history(getattr(request, "chat_template_kwargs", None))
        )
        try:
            result = await create(self, request, raw_request)
        finally:
            _DROP_THINKING.reset(token)
        if not isinstance(result, AsyncIterator):
            payload = (
                result.model_dump(exclude_none=True)
                if hasattr(result, "model_dump")
                else json.loads(result.body)
            )
            await record_response(payload)
            return result

        async def stream_response():
            token = _DROP_THINKING.set(
                not preserves_history(getattr(request, "chat_template_kwargs", None))
            )
            try:
                async for event in result:
                    if isinstance(event, str) and event.startswith(
                        ("event: response.completed\n", "event: response.incomplete\n")
                    ):
                        await record_response(
                            json.loads(event.split("data: ", 1)[1])["response"]
                        )
                    yield event
            finally:
                _DROP_THINKING.reset(token)

        return stream_response()

    responses.create_responses = create_responses
    chat._art_append_only = True
