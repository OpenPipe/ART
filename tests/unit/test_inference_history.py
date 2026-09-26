import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

from pydantic import BaseModel, model_validator
import pytest

from art_inference import vllm
from art_inference.append_only import chat_prefix_scope
from art_inference.token_prefix import TokenPrefixStore


class Request(BaseModel):
    model: str = "model"
    messages: list[dict]
    stream: bool = False
    stream_options: dict | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    truncate_prompt_tokens: int | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template_kwargs: dict = {}
    previous_response_id: str | None = None
    tools: list = []
    tool_choice: str = "auto"
    parallel_tool_calls: bool | None = None

    @model_validator(mode="after")
    def validate_stream(self):
        if self.stream_options and not self.stream:
            raise ValueError("Stream options can only be defined when stream=True")
        return self


class Message(BaseModel):
    role: str = "assistant"
    reasoning_content: str
    content: str
    tool_calls: list[dict] = []


@pytest.fixture
def serving(monkeypatch):
    monkeypatch.setattr(vllm, "_PREFIXES", TokenPrefixStore())

    class Tokenizer:
        all_special_ids = [ord("#"), ord("~")]

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode()

    class Renderer:
        def __init__(self, config, tokenizer):
            self.tokenizer = tokenizer

    class Params:
        def get_apply_chat_template_kwargs(self):
            return {}

    class Engine:
        def __init__(self):
            self.prompts = []
            self.sampled = None

        async def generate(self, prompt, sampling_params=None):
            self.prompts.append(prompt["prompt_token_ids"])
            chunks = (
                [b"\nthought\n#", b"action~"]
                if sampling_params
                else [b"\nthought\n#action~"]
            )
            if self.sampled is not None:
                chunks = [self.sampled]
            for index, tokens in enumerate(chunks):
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=list(tokens),
                            finish_reason="stop" if index == len(chunks) - 1 else None,
                        )
                    ]
                )

    class Serving:
        def __init__(self):
            self.renderer = Renderer(None, Tokenizer())
            self.engine = Engine()
            self.online_renderer = SimpleNamespace(preprocess_chat=self.preprocess_chat)
            self.chat_template = None
            self.chat_template_content_format = "string"
            self.parser = None
            self.message = Message(reasoning_content="\nthought\n", content="action")

        async def preprocess_chat(self, request, messages, **kwargs):
            return await self.render_chat_request(request)

        def _effective_chat_template_kwargs(self, request):
            return request.chat_template_kwargs

        async def _make_request(self, request, previous):
            return request.messages, (await self.render_chat_request(request))[1]

        async def render_chat_request(self, request):
            text = ""
            for message in request.messages:
                if message["role"] == "assistant":
                    text += (
                        "A"
                        + message.get("reasoning_content", "").strip()
                        + "#"
                        + (message.get("content") or "")
                        + "".join(
                            call["function"]["name"]
                            for call in message.get("tool_calls") or []
                        )
                        + "~\n"
                    )
                else:
                    text += "U" + message["content"] + ";"
            if request.add_generation_prompt:
                text += "A"
            return [], [{"prompt_token_ids": list(text.encode())}]

        async def create_chat_completion(self, request, raw_request=None):
            _, inputs = await self.render_chat_request(request)
            message = self.message
            samples = {
                "token_ids": list(self.engine.sampled or b"\nthought\n#action~"),
                "logprobs": [-0.25]
                * len(self.engine.sampled or b"\nthought\n#action~"),
            }
            usage = {"completion_tokens": len(samples["token_ids"])}

            async def stream():
                params = SimpleNamespace(output_kind=SimpleNamespace(name="DELTA"))
                async for _ in self.engine.generate(inputs[0], params):
                    pass
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {"index": 0, "delta": message.model_dump(), **samples}
                            ],
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            if request.stream:
                return stream()
            async for _ in self.engine.generate(inputs[0]):
                pass
            return SimpleNamespace(
                choices=[SimpleNamespace(index=0, message=message, **samples)],
                usage=usage,
            )

        async def create_completion(self, request, raw_request=None):
            return await self.create_chat_completion(request, raw_request)

        async def create_responses(self, request, raw_request=None):
            _, inputs = await self.render_chat_request(request)
            async for _ in self.engine.generate(inputs[0]):
                pass
            response = SimpleNamespace(
                output=[
                    Message(
                        reasoning_content="\nthought\n", content="action"
                    ).model_dump()
                ]
            )
            if not request.stream:
                return response

            async def events():
                yield SimpleNamespace(type="response.completed", response=response)

            return events()

    def construct_input_messages(**kwargs):
        return kwargs

    utils = SimpleNamespace(
        construct_input_messages=construct_input_messages,
        construct_chat_messages_with_tool_call=lambda items: items,
        construct_tool_dicts=lambda *_: None,
    )
    modules = {
        "vllm.renderers.base": SimpleNamespace(BaseRenderer=Renderer),
        "vllm.renderers.params": SimpleNamespace(ChatParams=Params),
        "vllm.v1.engine.async_llm": SimpleNamespace(AsyncLLM=Engine),
        "vllm.entrypoints.openai.chat_completion.serving": SimpleNamespace(
            OpenAIServingChat=Serving
        ),
        "vllm.entrypoints.openai.completion.serving": SimpleNamespace(
            OpenAIServingCompletion=Serving
        ),
        "vllm.entrypoints.openai.responses.serving": SimpleNamespace(
            OpenAIServingResponses=Serving
        ),
        "vllm.entrypoints.openai.responses.utils": utils,
        "vllm.entrypoints.openai.parser.harmony_utils": SimpleNamespace(
            render_for_completion=lambda messages: messages
        ),
        "vllm.entrypoints.openai.chat_completion.protocol": SimpleNamespace(
            ChatCompletionRequest=Request
        ),
    }
    vllm.patch_history(modules.__getitem__)
    return Serving(), modules


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("action", ["action", "edited action"])
def test_served_history_survives_renderer_normalization(serving, stream, action):
    server, _ = serving

    async def run():
        user = {"role": "user", "content": "question"}
        first = Request(
            messages=[user],
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
        )
        response = await server.create_chat_completion(first)
        if stream:
            assert (await anext(response)).startswith("data: ")
            assert await anext(response) == "data: [DONE]\n\n"
            await response.aclose()
        prior = server.engine.prompts[0] + list(b"\nthought\n#")
        assistant = Message(
            reasoning_content="\nthought\n", content=action
        ).model_dump()
        await server.create_chat_completion(
            Request(messages=[user, assistant, {"role": "user", "content": "next"}])
        )
        assert server.engine.prompts[-1][: len(prior)] == prior
        assert ("#" + action + "~\nU").encode() in bytes(server.engine.prompts[-1])

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget_field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("truncate", [None, 11])
def test_completed_observation_has_no_second_generation_budget_or_truncation(
    serving, stream, budget_field, truncate
):
    server, _ = serving
    render = server.render_chat_request
    completed = []

    async def validated_render(request):
        result = await render(request)
        prompt = result[1][0]
        if request.truncate_prompt_tokens is not None:
            prompt["prompt_token_ids"] = prompt["prompt_token_ids"][
                -request.truncate_prompt_tokens :
            ]
        output_budget = request.max_completion_tokens or request.max_tokens or 0
        assert len(prompt["prompt_token_ids"]) + output_budget <= 48
        if not request.add_generation_prompt:
            completed.append(bytes(prompt["prompt_token_ids"]))
        return result

    server.render_chat_request = validated_render
    request = Request(
        messages=[{"role": "user", "content": "question"}],
        stream=stream,
        truncate_prompt_tokens=truncate,
        max_tokens=32 if budget_field == "max_tokens" else None,
        max_completion_tokens=32 if budget_field == "max_completion_tokens" else None,
    )

    async def run():
        response = await server.create_chat_completion(request)
        if stream:
            assert [event async for event in response][-1] == "data: [DONE]\n\n"
        else:
            assert response.choices[0].message.content == "action"

    asyncio.run(run())
    assert b"Uquestion;Athought#action~\n" in completed
    assert getattr(request, budget_field) == 32
    assert request.truncate_prompt_tokens == truncate


def test_external_observations_use_the_host_store_without_local_publication(serving):
    server, modules = serving
    observations = []

    async def observe(request, entries):
        observations.extend(entries)

    vllm.set_external_history_observer(observe, modules.__getitem__)

    async def run():
        await server.create_chat_completion(
            Request(messages=[{"role": "user", "content": "question"}]),
            SimpleNamespace(headers={"x-caladan-prefix-scope": "scope"}),
        )

    asyncio.run(run())
    assert observations
    assert not vllm._PREFIXES._scopes


@pytest.mark.parametrize("stream", [False, True])
def test_serial_filtered_response_retains_all_samples_but_not_full_certificate(
    serving, monkeypatch, stream
):
    server, modules = serving
    sampled = b"\nthought\n#firstsecond~"
    server.engine.sampled = sampled
    server.message = Message(
        reasoning_content="\nthought\n",
        content="",
        tool_calls=[
            {
                "index": 0,
                "type": "function",
                "function": {"name": "first", "arguments": "{}"},
            }
        ],
    )
    entries, raw_choices = [], []
    observer = vllm.chat_response_prefixes

    async def observe_choices(tokenizer, request, prompt, choices, render):
        raw_choices.extend(copy.deepcopy(choices))
        return await observer(tokenizer, request, prompt, choices, render)

    async def observe(raw_request, values):
        assert raw_request.headers == {"x-caladan-prefix-scope": "base"}
        entries.extend(values)

    monkeypatch.setattr(vllm, "chat_response_prefixes", observe_choices)
    vllm.set_external_history_observer(observe, modules.__getitem__)

    async def run():
        request = Request(
            messages=[{"role": "user", "content": "question"}],
            parallel_tool_calls=False,
            tools=[{"type": "function"}],
            stream=stream,
        )
        result = await server.create_chat_completion(
            request, SimpleNamespace(headers={"x-caladan-prefix-scope": "base"})
        )
        if stream:
            chunks = [chunk async for chunk in result]
            payload = json.loads(chunks[0][len("data: ") :])
            choice = payload["choices"][0]
            message = choice["delta"]
            usage = payload["usage"]
            assert chunks[-1] == "data: [DONE]\n\n"
        else:
            message = result.choices[0].message.model_dump()
            choice = vars(result.choices[0])
            usage = result.usage
        assert message == server.message.model_dump()
        assert choice["token_ids"] == list(sampled)
        assert choice["logprobs"] == [-0.25] * len(sampled)
        assert usage == {"completion_tokens": len(sampled)}

    asyncio.run(run())
    assert len(raw_choices) == 1
    assert raw_choices[0][1:] == (list(sampled), True)
    assert entries
    assert all(b"first" not in bytes(raw) for _, raw, _ in entries)


def test_native_scope_does_not_reuse_old_local_certificate(serving):
    server, _ = serving
    base = hashlib.sha256(
        json.dumps(["model", None, {}, ""], sort_keys=True).encode()
    ).hexdigest()
    prompt = list(b"Uquestion;A")
    vllm._PREFIXES.insert(base, prompt, list(b"unsafe"), "content")
    asyncio.run(
        server.create_chat_completion(
            Request(messages=[{"role": "user", "content": "question"}])
        )
    )
    assert server.engine.prompts == [prompt]
    assert chat_prefix_scope(base) in vllm._PREFIXES._scopes


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True, None, "absent"])
def test_responses_serial_policy_reaches_chat_observer(
    serving, monkeypatch, stream, parallel
):
    server, _ = serving
    observed = []
    original = vllm.chat_response_prefixes

    async def observe(tokenizer, request, prompt, choices, render):
        observed.append(request.parallel_tool_calls)
        return await original(tokenizer, request, prompt, choices, render)

    monkeypatch.setattr(vllm, "chat_response_prefixes", observe)

    async def run():
        request = Request(
            messages=[{"role": "user", "content": "question"}],
            parallel_tool_calls=parallel if isinstance(parallel, bool) else None,
            stream=stream,
        )
        if parallel == "absent":
            del request.parallel_tool_calls
        result = await server.create_responses(request)
        if stream:
            async for _ in result:
                pass

    asyncio.run(run())
    assert observed == [parallel is not False]


def test_previous_responses_keep_reasoning_and_tool_calls(serving):
    _, modules = serving
    previous = [{"type": "reasoning"}, {"type": "function_call"}]
    utils = modules["vllm.entrypoints.openai.responses.utils"]
    assert utils.construct_input_messages(
        prev_response_output=previous, request_input="next"
    )["request_input"] == [*previous, {"role": "user", "content": "next"}]


def test_batched_generations_keep_separate_prompt_histories(serving):
    server, _ = serving
    turn = vllm._Turn("scope", server.renderer.tokenizer)

    async def run():
        token = vllm._CURRENT.set(turn)
        try:
            first = server.engine.generate({"prompt_token_ids": list(b"first")})
            second = server.engine.generate({"prompt_token_ids": list(b"second")})
            await anext(first)
            await anext(second)
            assert [item async for item in first] == []
            assert [item async for item in second] == []
        finally:
            vllm._CURRENT.reset(token)

    asyncio.run(run())
    for prompt in (b"first", b"second"):
        tokens = list(prompt + b"\nthought\n#action~")
        match = vllm._PREFIXES.lookup("scope", tokens, None)
        assert match is not None
        assert list(match.raw_prefix) == tokens


def test_vllm_multimodal_offsets_follow_preserved_text():
    from art_inference.token_prefix import PrefixEdit

    span = SimpleNamespace(offset=4, length=2)
    original = {
        "prompt_token_ids": [1, 2, 3, 4, 9, 9],
        "mm_placeholders": {"image": [span]},
        "token_type_ids": [0, 0, 0, 0, 1, 1],
    }
    edited = vllm.replace_prompt_tokens(
        original, [1, 2, 2, 3, 4, 9, 9], [PrefixEdit(1, 2, (2, 2))]
    )
    assert edited["mm_placeholders"]["image"][0].offset == 5
    assert span.offset == 4
    assert edited["token_type_ids"] == [0, 0, 0, 0, 0, 1, 1]
    with pytest.raises(ValueError, match="multimodal placeholder"):
        vllm.replace_prompt_tokens(original, [1], [PrefixEdit(4, 6, ())])


def test_native_deepseek_v32_respects_preservation_options():
    encoding = SimpleNamespace(
        encode_messages=lambda **kwargs: kwargs,
        render_message=lambda index, messages, **kwargs: messages[index],
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return encoding.encode_messages(
                drop_thinking=messages[-1]["role"] == "user"
            )

    Tokenizer.apply_chat_template.__module__ = "vllm.tokenizers.deepseek_v32"
    tokenizer = Tokenizer()
    vllm._configure_native_tokenizer(tokenizer, lambda _: encoding)
    messages = [{"role": "user"}]
    assert tokenizer.apply_chat_template(messages)["drop_thinking"] is False
    assert (
        tokenizer.apply_chat_template(messages, drop_thinking=True)["drop_thinking"]
        is True
    )
    assert (
        tokenizer.apply_chat_template(messages, preserve_thinking=False)[
            "drop_thinking"
        ]
        is True
    )
    assert tokenizer.apply_chat_template(messages)["drop_thinking"] is False


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_responses_use_the_native_renderer_and_preserve_ids(serving, legacy, stream):
    server, _ = serving
    if legacy:
        server.openai_serving_render = server.online_renderer
        del server.online_renderer

    async def run():
        request = Request(
            messages=[{"role": "user", "content": "question"}], stream=stream
        )
        response = await server.create_responses(request)
        if stream:
            assert (await anext(response)).type == "response.completed"
            await response.aclose()
        previous = Request(
            messages=[
                *request.messages,
                Message(reasoning_content="\nthought\n", content="action").model_dump(),
                {"role": "user", "content": "next"},
            ]
        )
        await server.create_responses(previous)
        assert bytes(server.engine.prompts[-1]).startswith(
            b"Uquestion;A\nthought\n#action~"
        )

    asyncio.run(run())
