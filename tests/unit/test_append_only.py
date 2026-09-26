import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from art.token_prefix import TokenPrefixCache, apply_prefix_edits
from art.utils.append_only import chat_prefix_observations, chat_response_prefixes
from art_inference.append_only import (
    chat_prefix_scope,
    output_prefix_observations,
    patch_deepseek_renderer,
)


@pytest.mark.parametrize("reasoning", [None, "", "\nthought\n"])
def test_dsv32_earlier_turn_framing_survives_a_new_user_turn(reasoning):
    def render(index, messages, thinking_mode):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        thinking = thinking_mode == "thinking"
        if messages[index]["role"] == "user":
            return "USER" + (
                "<think>" if thinking and index == last_user else "</think>"
            )
        if thinking and index > last_user:
            assert messages[index]["reasoning_content"], "Missing reasoning"
            return messages[index]["reasoning_content"] + "END"
        return "END"

    encoding = SimpleNamespace(render_message=render, thinking_end_token="</think>")
    preserve = True
    patch_deepseek_renderer(encoding, lambda: preserve, prefix_only=True)
    messages = [
        {"role": "user"},
        {"role": "assistant", "reasoning_content": reasoning},
    ]

    def encode(messages):
        return "".join(
            encoding.render_message(i, messages, "thinking")
            for i in range(len(messages))
        )

    completed = encode(messages)
    assert encode([*messages, {"role": "user"}]).startswith(completed)
    if reasoning:
        preserve = False
        assert not encode([*messages, {"role": "user"}]).startswith(completed)


class Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens, *, skip_special_tokens=False):
        return bytes(tokens).decode()


@pytest.mark.parametrize("parallel", [False, True, None])
@pytest.mark.parametrize("visible_calls", [0, 1, 2])
@pytest.mark.parametrize("reasoning", [False, True])
def test_serial_projection_cannot_certify_hidden_sampled_actions(
    parallel, visible_calls, reasoning
):
    class Request(BaseModel):
        messages: list[dict]
        parallel_tool_calls: bool | None = None
        tools: list[dict] = [{"type": "function"}]

    tokenizer = Tokenizer()
    encode = tokenizer.encode
    request = Request(messages=[{"role": "user"}], parallel_tool_calls=parallel)
    calls = [
        {"type": "function", "function": {"name": name, "arguments": "{}"}}
        for name in ("first", "second")
    ]
    message = {"role": "assistant", "tool_calls": calls[:visible_calls]}
    if reasoning:
        message["reasoning_content"] = "thought"

    async def render(value):
        if len(value.messages) == 1:
            return encode("prompt:")
        assistant = value.messages[-1]
        text = "thought#" if assistant.get("reasoning_content") else ""
        text += "".join(
            call["function"]["name"] for call in assistant.get("tool_calls", [])
        )
        return encode("prompt:" + text + "END")

    sampled = encode(("\nthought#" if reasoning else "") + "firstsecondEND")
    choices = [(message, sampled, True), (message, sampled, False)]
    before = deepcopy((request.model_dump(), choices))
    entries = asyncio.run(
        chat_response_prefixes(tokenizer, request, encode("prompt:"), choices, render)
    )
    full = [entry for entry in entries if entry[1] == encode("prompt:") + sampled]
    assert len(full) == (0 if parallel is False else 1)
    if parallel is False:
        assert all(b"first" not in bytes(entry[1]) for entry in entries)
        # A distinct, aligned reasoning boundary survives only if the parsed
        # action makes it distinguishable from the reasoning-only rendering.
        assert bool(entries) == bool(reasoning and visible_calls)
    assert (request.model_dump(), choices) == before
    for rendered, raw, edits in entries:
        assert apply_prefix_edits(rendered, edits) == raw


def test_serial_non_tool_turn_retains_full_certificate():
    class Request(BaseModel):
        messages: list[dict] = []
        parallel_tool_calls: bool = False

    async def render(value):
        return [1, 2] if value.messages else [1]

    entries = asyncio.run(
        chat_response_prefixes(
            Tokenizer(),
            Request(),
            [1],
            [
                (
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "first", "arguments": "{}"}}
                        ],
                    },
                    [3, 2],
                    True,
                ),
                ({"role": "assistant"}, [2], True),
            ],
            render,
        )
    )
    assert len(entries) == 1 and entries[0][1] == [1, 2]


def test_policy_scope_keeps_old_certificates_in_a_separate_namespace():
    from art_inference.token_prefix import TokenPrefixStore

    cache = TokenPrefixStore()
    base = "a" * 64
    current = chat_prefix_scope(base)
    assert current != base
    assert len(current) == 64
    assert current == chat_prefix_scope(base)
    for canonical in ([1, 2], [1, 2, 3]):
        cache.insert(base, canonical, [9], "lineage")
    assert cache.lookup(current, [1, 2, 3, 4], "lineage") is None
    cache.insert(current, [1, 2], [8], "lineage")
    match = cache.lookup(current, [1, 2, 3, 4], "lineage")
    assert match is not None and match.raw_prefix == (8,)


def test_many_protocol_markers_do_not_multiply_long_prompt_storage():
    tokenizer = SimpleNamespace(
        all_special_ids=[1, 2, 3],
        encode=lambda text, **_: [ord(char) for char in text],
        decode=lambda tokens, **_: "".join(chr(token) for token in tokens),
    )
    prompt = [100] * 128_000
    output = [101, 1, 102, 2, 103, 3] * 100
    entries = output_prefix_observations(tokenizer, prompt, prompt, output)
    assert len(entries) <= 2
    assert entries[-1][1] == prompt + output


@pytest.mark.parametrize("action", ["pass", '<tool>{"x": 1}</tool>'])
def test_normalized_reasoning_survives_an_edited_action(action):
    tokenizer = Tokenizer()
    encode = tokenizer.encode
    prompt = encode("USER question ASSISTANT <think>\n")
    rendered = prompt + encode(f"reasoning\n</think>\n\n{action}END\n")
    sampled = encode(f"\nreasoning\n</think>\n\n{action}END")
    reasoning = prompt + encode("reasoning\n</think>\n\nEND\n")
    observations = chat_prefix_observations(
        tokenizer, prompt, rendered, prompt, sampled, reasoning_prompt=reasoning
    )
    assert len(observations) == 2
    cache = TokenPrefixCache()
    for rendered_prefix, raw_prefix, edits in observations:
        assert apply_prefix_edits(rendered_prefix, edits) == raw_prefix
        cache.insert(rendered_prefix, raw_prefix, "rollout", edits)

    for next_action in (action, "corrected action"):
        next_prompt = prompt + encode(
            f"reasoning\n</think>\n\n{next_action}END\nUSER next ASSISTANT"
        )
        match = cache.lookup(next_prompt, "rollout")
        assert match is not None
        served = list(match.raw_prefix) + next_prompt[match.rendered_length :]
        assert served == prompt + encode(
            f"\nreasoning\n</think>\n\n{next_action}END\nUSER next ASSISTANT"
        )


def test_truncation_does_not_replace_completed_turn_framing():
    tokenizer = Tokenizer()
    encode = tokenizer.encode
    assert not chat_prefix_observations(
        tokenizer,
        encode("prompt"),
        encode("prompt partial END"),
        encode("prompt"),
        encode(" partial"),
        complete=False,
    )


def test_changed_prompt_cannot_be_registered_as_a_continuation():
    assert not chat_prefix_observations(Tokenizer(), [1], [2, 3], [4], [5])


@pytest.mark.parametrize(
    "options",
    [{"preserve_thinking": False}, {"clear_thinking": True}, {"drop_thinking": True}],
)
def test_explicit_rewriting_bypasses_observation(options):
    class Request(BaseModel):
        messages: list[dict]
        chat_template_kwargs: dict

    async def render(_):
        pytest.fail("Explicit rewriting must not create preservation mappings")

    assert (
        asyncio.run(
            chat_response_prefixes(
                Tokenizer(),
                Request(messages=[], chat_template_kwargs=options),
                [],
                [],
                render,
            )
        )
        == []
    )


class StrictFunction(BaseModel):
    name: str
    arguments: str


class StrictToolCall(BaseModel):
    id: str
    type: str
    function: StrictFunction


class StrictMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[StrictToolCall] | None = None
    function_call: StrictFunction | None = None


class StrictRequest(BaseModel):
    messages: list[StrictMessage]


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
@pytest.mark.parametrize(
    ("arguments", "legacy", "historical"),
    [
        ({"id": 3}, False, False),
        ('{  "id": 3  }', False, False),
        ({"id": 3}, True, False),
        ({"previous": True}, False, True),
    ],
    ids=("mapping", "string", "legacy-mapping", "historical-mapping"),
)
def test_response_observation_accepts_structured_tool_arguments(
    arguments, legacy, historical
):
    tokenizer = Tokenizer()
    messages = [StrictMessage(role="user", content="question")]
    if historical:
        function = StrictFunction(name="previous", arguments="{}")
        object.__setattr__(function, "arguments", arguments)
        messages.extend(
            [
                StrictMessage(
                    role="assistant",
                    tool_calls=[
                        StrictToolCall(
                            id="previous", type="function", function=function
                        )
                    ],
                ),
                StrictMessage(role="user", content="next"),
            ]
        )
    request = StrictRequest(messages=messages)
    message = {"role": "assistant", "content": "answer"}
    if not historical:
        function = {"name": "lookup", "arguments": arguments}
        if legacy:
            message["function_call"] = function
        else:
            message["tool_calls"] = [
                {"id": "call", "type": "function", "function": function}
            ]
    original_request = request.model_dump(mode="python", warnings=False)
    original_message = deepcopy(message)
    rendered_arguments: list[list[str]] = []

    async def render(value: StrictRequest) -> list[int]:
        if len(value.messages) == len(request.messages):
            return tokenizer.encode("prompt:")
        completed: list[str] = []
        for rendered_message in value.messages:
            completed.extend(
                call.function.arguments for call in rendered_message.tool_calls or []
            )
            if rendered_message.function_call:
                completed.append(rendered_message.function_call.arguments)
        rendered_arguments.append(completed)
        return tokenizer.encode("prompt:answerEND")

    observations = asyncio.run(
        chat_response_prefixes(
            tokenizer,
            request,
            tokenizer.encode("prompt:"),
            [(message, tokenizer.encode("answerEND"), True)],
            render,
        )
    )

    expected = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    assert rendered_arguments == [[expected]]
    assert observations
    assert request.model_dump(mode="python", warnings=False) == original_request
    assert message == original_message


def test_custom_stop_does_not_delete_the_template_terminator():
    encode = Tokenizer().encode
    assert (
        chat_prefix_observations(
            Tokenizer(),
            encode("prompt:"),
            encode("prompt:answerEND"),
            encode("prompt:"),
            encode("answer"),
            complete=True,
        )
        == []
    )


def test_reasoning_boundary_does_not_absorb_whitespace_merged_into_action():
    from art_inference.append_only import _whitespace_prefix

    tokenizer = Tokenizer()
    raw = tokenizer.encode("thought</think>\n\naction")
    boundary = _whitespace_prefix(tokenizer, tokenizer.encode("thought</think>"), raw)
    assert tokenizer.decode(raw[:boundary]) == "thought</think>"


def test_missing_lineage_never_selects_an_ambiguous_tokenization():
    cache = TokenPrefixCache()
    cache.insert([1, 2], [1, 3], "first")
    match = cache.lookup([1, 2], None)
    assert match is not None and match.raw_prefix == (1, 3)
    cache.insert([1, 2], [1, 4], "second")
    assert cache.lookup([1, 2], None) is None
    first = cache.lookup([1, 2], "first")
    second = cache.lookup([1, 2], "second")
    assert first is not None and first.raw_prefix == (1, 3)
    assert second is not None and second.raw_prefix == (1, 4)


def test_history_edits_leave_multimodal_markers_outside_replacements():
    from types import SimpleNamespace

    from art_inference.append_only import _rendering_edits
    from art_inference.vllm import replace_prompt_tokens

    rendered = [1, 2, 9, 9, 3, 4]
    raw = [1, 1, 2, 9, 9, 3, 4, 4]
    edits = _rendering_edits(SimpleNamespace(all_special_ids=[9]), rendered, raw)
    assert apply_prefix_edits(rendered, edits) == raw
    original = {
        "prompt_token_ids": rendered,
        "mm_placeholders": {"image": [SimpleNamespace(offset=2, length=2)]},
    }
    updated = replace_prompt_tokens(original, raw, edits)
    assert updated["mm_placeholders"]["image"][0].offset == 3
