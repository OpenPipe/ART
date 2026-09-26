import asyncio
import json
from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from art_inference import sglang
from art_inference.token_prefix import TokenPrefixStore


class Request(BaseModel):
    model: str = "model"
    messages: list[dict] = []
    input: str | list = "next"
    chat_template_kwargs: dict = {}
    add_generation_prompt: bool = True
    skip_special_tokens: bool = True
    previous_response_id: str | None = None
    stream: bool = False
    continue_final_message: bool = False
    parallel_tool_calls: bool | None = None


class Response(BaseModel):
    id: str = "previous"
    output: list = []


class ChatRequest(Request):
    parallel_tool_calls: bool = True


@pytest.fixture
def serving():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode()

        def apply_chat_template(
            self, messages, *, add_generation_prompt=True, **kwargs
        ):
            text = "".join(
                "A"
                + m.get("reasoning_content", "").strip()
                + "#"
                + m.get("content", "")
                + "~\n"
                if m["role"] == "assistant"
                else "U" + m["content"] + ";"
                for m in messages
            )
            return self.encode(text + ("A" if add_generation_prompt else ""))

    encoding = SimpleNamespace(
        encode_messages=lambda **kwargs: kwargs,
        render_message=lambda index, messages, thinking_mode: messages[index],
    )

    class Serving:
        def __init__(self):
            self.tokenizer_manager = SimpleNamespace(
                tokenizer=Tokenizer(), model_config=SimpleNamespace(is_multimodal=False)
            )
            self.msg_store = {}
            self.response_store = {}

        def _process_messages(self, request, is_multimodal):
            request.skip_special_tokens = False
            self.options = request.chat_template_kwargs
            self.dsv4_options = encoding.encode_messages()
            messages, _ = self._handle_last_assistant_message(
                [dict(message) for message in request.messages], request
            )
            return SimpleNamespace(
                prompt_ids=self.tokenizer_manager.tokenizer.apply_chat_template(
                    messages
                )
            )

        def _handle_last_assistant_message(self, messages, request):
            # SGLang 0.5.15 request rendering rewrites a trailing assistant.
            if messages and messages[-1]["role"] == "assistant":
                if request.continue_final_message:
                    return messages[:-1], messages[-1]["content"]
                messages[-1] = {"role": "user", "content": messages[-1]["content"]}
            return messages, None

        async def _handle_non_streaming_request(
            self, adapted_request, request, raw_request
        ):
            prompt = self._process_messages(request, False).prompt_ids
            choice = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "\nthought\n",
                    "content": "action",
                },
                "token_ids": list(b"\nthought\n#action~"),
                "finish_reason": "stop",
            }
            return SimpleNamespace(
                body=json.dumps(
                    {"prompt_token_ids": prompt, "choices": [choice]}
                ).encode()
            )

        async def _generate_chat_stream(self, adapted_request, request, raw_request):
            result = json.loads(
                (
                    await self._handle_non_streaming_request(
                        adapted_request, request, raw_request
                    )
                ).body
            )
            choice = result["choices"][0]
            choice["delta"] = choice.pop("message")
            yield "data: " + json.dumps(result) + "\n\n"
            yield "data: [DONE]\n\n"

        async def create_responses(self, request, raw_request=None):
            prompt = self._process_messages(
                Request(messages=request.input), False
            ).prompt_ids
            payload = {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "content": [{"text": "\nthought\n"}]},
                    {"role": "assistant", "content": "action"},
                ],
                "token_generations": [
                    {
                        "prompt_token_ids": prompt,
                        "output_tokens": [
                            {"token_id": token} for token in b"\nthought\n#action~"
                        ],
                    }
                ],
            }
            if not request.stream:
                return SimpleNamespace(body=json.dumps(payload).encode())

            async def events():
                yield (
                    "event: response.completed\ndata: "
                    + json.dumps({"response": payload})
                    + "\n\n"
                )

            return events()

        @staticmethod
        def _merge_consecutive_assistant_messages(messages):
            combined = {}
            for message in messages:
                combined.update(message)
            return [combined]

        def _response_tools_to_chat_tools(self, request):
            return []

        @classmethod
        def _normalize_response_message_for_chat(cls, message):
            return message

        def _construct_input_messages(self, request, prev_response=None):
            return (
                [*prev_response.output, *request.input]
                if prev_response
                else request.input
            )

        def _construct_input_messages_with_harmony(self, request, prev_response):
            previous = self.msg_store[prev_response.id]
            previous[:] = [m for m in previous if m.channel != "analysis"]
            return [*previous, request.input]

    entries = []

    async def observe(request, observations):
        entries.extend(observations)

    modules = {
        "sglang.srt.entrypoints.openai.serving_chat": SimpleNamespace(
            OpenAIServingChat=Serving
        ),
        "sglang.srt.entrypoints.openai.serving_responses": SimpleNamespace(
            OpenAIServingResponses=Serving
        ),
        "sglang.srt.entrypoints.openai.encoding_dsv4": encoding,
        "sglang.srt.entrypoints.openai.protocol": SimpleNamespace(
            ChatCompletionRequest=ChatRequest
        ),
        "sglang.srt.entrypoints.harmony_utils": SimpleNamespace(
            render_for_completion=lambda messages: messages
        ),
        "sglang.srt.entrypoints.openai.encoding_dsv32": SimpleNamespace(
            encode_messages=lambda **kwargs: kwargs,
            render_message=lambda index, messages, **kwargs: messages[index],
        ),
    }
    sglang.patch_history(observe, modules.__getitem__)
    return Serving(), entries


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True, None, "absent"])
def test_responses_serial_policy_reaches_chat_observer(
    serving, monkeypatch, stream, parallel
):
    server, _ = serving
    observed = []
    original = sglang.chat_response_prefixes

    async def observe(tokenizer, request, prompt, choices, render):
        observed.append(request.parallel_tool_calls)
        return await original(tokenizer, request, prompt, choices, render)

    monkeypatch.setattr(sglang, "chat_response_prefixes", observe)

    async def run():
        request = Request(
            input=[{"role": "user", "content": "question"}],
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


@pytest.mark.parametrize("stream", [False, True])
def test_sglang_observes_normalized_history_and_edited_actions(serving, stream):
    server, entries = serving
    request = Request(messages=[{"role": "user", "content": "question"}])

    async def run():
        if stream:
            async for _ in server._generate_chat_stream(None, request, None):
                pass
        else:
            await server._handle_non_streaming_request(None, request, None)

    asyncio.run(run())
    store = TokenPrefixStore()
    for rendered, raw, edits in entries:
        store.insert("scope", rendered, raw, "lineage", edits)
    prompt = list(b"Uquestion;Athought#edited~\nUnext;A")
    match = store.lookup("scope", prompt, "lineage")
    assert match is not None
    actual = list(match.raw_prefix) + prompt[match.rendered_length :]
    assert bytes(actual) == b"Uquestion;A\nthought\n#edited~\nUnext;A"
    assert server.options == {}
    assert request.skip_special_tokens is False
    assert server.dsv4_options["drop_thinking"] is False


@pytest.mark.parametrize("preserve", [False, True])
def test_harmony_opt_out_does_not_mutate_stored_history(serving, preserve):
    server, _ = serving
    thought, final = (
        SimpleNamespace(channel="analysis"),
        SimpleNamespace(channel="final"),
    )
    server.msg_store["previous"] = [thought, final]
    result = server._construct_input_messages_with_harmony(
        Request(chat_template_kwargs={"preserve_thinking": preserve}), Response()
    )
    assert result == ([thought, final, "next"] if preserve else [final, "next"])
    assert server.msg_store["previous"] == [thought, final]


def test_responses_use_full_reasoning_and_retain_output_items(serving):
    server, _ = serving
    assert server._normalize_response_message_for_chat(
        {
            "type": "reasoning",
            "content": [{"text": "full "}, {"text": "reasoning"}],
            "summary": [{"text": "short summary"}],
        }
    ) == {"role": "assistant", "reasoning_content": "full reasoning"}
    items = [{"type": "reasoning"}, {"type": "function_call"}]
    assert server._construct_input_messages(Request(), Response(output=items)) == [
        *items,
        {"role": "user", "content": "next"},
    ]


def test_explicit_dsv4_opt_out_is_scoped_to_one_request(serving):
    server, _ = serving
    server._process_messages(
        Request(chat_template_kwargs={"drop_thinking": True}), False
    )
    assert server.dsv4_options["drop_thinking"] is True
    server._process_messages(Request(), False)
    assert server.dsv4_options["drop_thinking"] is False


@pytest.mark.parametrize("stream", [False, True])
def test_responses_history_opt_out_reaches_nested_chat_rendering(serving, stream):
    server, entries = serving

    async def run():
        result = await server.create_responses(
            Request(
                input=[{"role": "user", "content": "question"}],
                chat_template_kwargs={"preserve_thinking": False},
                stream=stream,
            )
        )
        if stream:
            async for _ in result:
                pass

    asyncio.run(run())
    assert server.dsv4_options["drop_thinking"] is True
    assert not entries
    server._process_messages(Request(), False)
    assert server.dsv4_options["drop_thinking"] is False


def test_multimodal_offsets_and_rope_positions_follow_text_edits():
    import torch

    from art_inference.token_prefix import PrefixEdit

    mm = SimpleNamespace(
        mm_items=[SimpleNamespace(offsets=[(4, 5)])],
        input_ids=[1, 2, 3, 4, 9, 9],
        padded_input_ids=[1, 2, 3, 4, -1, -1],
        mrope_positions=torch.tensor(
            [[0, 1, 2, 3, 4, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 4]]
        ),
        mrope_position_delta=torch.tensor([0]),
        token_type_ids=torch.tensor([0, 0, 0, 0, 1, 1]),
    )
    tokenized = SimpleNamespace(mm_inputs=mm)
    sglang.replace_prompt_tokens(tokenized, [PrefixEdit(1, 2, (2, 2))])
    actual = tokenized.mm_inputs
    assert actual.mm_items[0].offsets == [(5, 6)]
    assert mm.mm_items[0].offsets == [(4, 5)]
    assert actual.padded_input_ids == [1, 2, 2, 3, 4, -1, -1]
    assert actual.mrope_positions.tolist() == [
        [0, 1, 2, 3, 4, 5, 5],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 5],
    ]
    assert actual.token_type_ids.tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert actual.mrope_position_delta is mm.mrope_position_delta


@pytest.mark.parametrize("stream", [False, True])
def test_responses_observe_the_complete_native_generation(serving, stream):
    server, entries = serving

    async def run():
        result = await server.create_responses(
            Request(input=[{"role": "user", "content": "question"}], stream=stream)
        )
        if stream:
            assert (await anext(result)).startswith("event: response.completed")
            await result.aclose()

    asyncio.run(run())
    assert entries
    assert bytes(entries[-1][1]) == b"Uquestion;A\nthought\n#action~"
