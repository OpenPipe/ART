from __future__ import annotations

import builtins
from datetime import datetime, timedelta
import math
import random
import re
from statistics import median
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from typing import Any

from anthropic.types import ImageBlockParam, Message, MessageParam
from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputParam,
    ResponseOutputMessageParam,
)
import pytest

import art
from art.trajectories import (
    ChatCompletionsExchange,
    ChatCompletionsMessageSource,
    ChatCompletionsRequest,
    CompletionsExchange,
    CompletionsRequest,
    MessagesExchange,
    MessagesRequest,
    ResponsesExchange,
    ResponsesItemSource,
    ResponsesRequest,
    TrajectoryExchanges,
)


def _chat_exchange(
    prompt: list[int],
    output: list[int],
    *,
    model: str = "test/model",
    offset: int = 0,
) -> ChatCompletionsExchange:
    messages: list[ChatCompletionMessageParam] = []
    for turn in range(offset + 1):
        messages.append({"role": "user", "content": f"turn {turn}"})
        if turn < offset:
            messages.append({"role": "assistant", "content": "answer"})
    response = ChatCompletion.model_validate(
        {
            "id": f"chat-{offset}",
            "object": "chat.completion",
            "created": offset,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                    "prompt_token_ids": prompt,
                    "token_ids": output,
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{token}",
                                "logprob": -token / 10,
                                "bytes": [],
                                "top_logprobs": [],
                            }
                            for token in output
                        ]
                    },
                }
            ],
        }
    )
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return ChatCompletionsExchange(
        request=ChatCompletionsRequest(
            model=model,
            messages=messages,
        ),
        response=response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def _completion_exchange(
    *,
    prompt: str | list[str] | list[int] | list[list[int]] = "question",
    echo: bool = False,
) -> CompletionsExchange:
    response = Completion.model_validate(
        {
            "id": "completion-1",
            "object": "text_completion",
            "created": 0,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "text": f"{'question' if echo else ''}answer",
                    "prompt_token_ids": [1],
                    "token_ids": [2],
                    "logprobs": {
                        "tokens": ["token_id:2"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [{}],
                        "text_offset": [0],
                    },
                }
            ],
        }
    )
    request = CompletionsRequest(model="test/model", prompt="question", echo=echo)
    request["prompt"] = prompt
    start = datetime(2026, 1, 1)
    return CompletionsExchange(
        request=request,
        response=response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def _message_exchange(
    request: MessagesRequest,
    *,
    identifier: str = "message-1",
    content: list[dict[str, object]] | None = None,
    duration: timedelta = timedelta(milliseconds=1),
    offset: int = 0,
    response_model: str = "test/model",
    **response_extra: object,
) -> MessagesExchange:
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    response = Message.model_validate(
        {
            "id": identifier,
            "type": "message",
            "role": "assistant",
            "model": response_model,
            "content": content or [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            **response_extra,
        }
    )
    return MessagesExchange(
        request=request,
        response=response,
        start_time=start,
        end_time=start + duration,
    )


def test_exact_tokens_form_one_append_only_history_without_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "wandb-artifact:///entity/project/run:step0"
    empty = _chat_exchange([1, 2, 3, 4], [], model=model, offset=2)
    empty.response.choices[0].message.content = ""
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2], model=model, offset=0),
                _chat_exchange([1, 2, 3], [4], model=model, offset=1),
                empty,
            ]
        )
    )
    real_import = builtins.__import__

    def import_without_tokenizer_dependencies(name: str, *args: Any, **kwargs: Any):
        if name.partition(".")[0] in {"transformers", "wandb"}:
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_tokenizer_dependencies)

    tokenized = trajectory.tokenize()

    assert tokenized.token_ids == [1, 2, 3, 4]
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]
    assert math.isnan(tokenized.logprobs[0])
    assert tokenized.logprobs[1] == -0.2
    assert math.isnan(tokenized.logprobs[2])
    assert tokenized.logprobs[3] == -0.4


def test_empty_tool_calls_normalization_preserves_exact_continuation() -> None:
    first = _chat_exchange([1], [2])
    first_data = first.response.model_dump(mode="python")
    first_data["choices"][0]["message"]["tool_calls"] = []
    first.response = ChatCompletion.model_validate(first_data)
    second = _chat_exchange([1, 2, 3], [4], offset=1)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).tokenize()

    assert tokenized.token_ids == [1, 2, 3, 4]
    assert tokenized.logprobs[1::2] == [-0.2, -0.4]


def test_messages_exact_prompt_and_output_do_not_load_a_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            messages=[
                _message_exchange(
                    MessagesRequest(
                        model="test/model",
                        messages=[{"role": "user", "content": "question"}],
                        max_tokens=16,
                    ),
                    identifier="message-exact",
                    prompt_token_ids=[1, 2],
                    token_ids=[3],
                    logprobs=[-0.3],
                )
            ]
        )
    )
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer",
        lambda _config: pytest.fail("exact Messages evidence loaded a tokenizer"),
    )

    tokenized = trajectory.tokenize()

    assert tokenized.token_ids == [1, 2, 3]
    assert all(math.isnan(value) for value in tokenized.logprobs[:2])
    assert tokenized.logprobs[2] == -0.3
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_anthropic_cache_control_change_starts_new_source_lineage() -> None:
    def exchange(
        *,
        identifier: str,
        messages: list[MessageParam],
        prompt_token_ids: list[int],
        output_token_id: int,
        offset: int,
    ) -> MessagesExchange:
        return _message_exchange(
            MessagesRequest(
                model="test/model",
                messages=messages,
                max_tokens=16,
            ),
            identifier=identifier,
            content=[{"type": "text", "text": f"answer {offset}"}],
            offset=offset,
            prompt_token_ids=prompt_token_ids,
            token_ids=[output_token_id],
            logprobs=[-0.1],
        )

    first = exchange(
        identifier="message-1",
        messages=[{"role": "user", "content": [{"type": "text", "text": "question"}]}],
        prompt_token_ids=[1],
        output_token_id=2,
        offset=0,
    )
    second = exchange(
        identifier="message-2",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "question",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer 0"}],
            },
            {"role": "user", "content": "follow up"},
        ],
        prompt_token_ids=[1, 2, 3],
        output_token_id=4,
        offset=1,
    )

    histories = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[first, second])
    ).anthropic_messages_histories()

    assert len(histories) == 2
    updated = histories[1]
    source = updated.message_sources[0]
    assert source is not None
    assert source.exchange is second
    assert source.request_index == 0
    assert updated.tokenize().token_ids == [1, 2, 3, 4]


def test_converted_anthropic_system_history_preserves_exact_assistant_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _message_exchange(
        MessagesRequest(
            model="test/model",
            system="system",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=16,
        ),
        identifier="message-system-exact",
        prompt_token_ids=[10, 11],
        token_ids=[12],
        logprobs=[-0.12],
    )
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(messages=[exchange]))
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer",
        lambda _config: pytest.fail("exact converted history loaded a tokenizer"),
    )

    history = trajectory.anthropic_messages_history().as_chat_completions_history()
    assert all(
        source is None or source.exchange is exchange
        for source in history.message_sources
    )
    tokenized = history.tokenize()

    assert tokenized.token_ids == [10, 11, 12]
    assert tokenized.logprobs[-1] == pytest.approx(-0.12)
    assert tokenized.flags[-1] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_converted_anthropic_system_source_rejects_sampled_response_mutation() -> None:
    exchange = _message_exchange(
        MessagesRequest(
            model="test/model",
            system="system",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=16,
        ),
        identifier="message-system-source",
        prompt_token_ids=[10, 11],
        token_ids=[12],
        logprobs=[-0.12],
    )
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(messages=[exchange]))
        .anthropic_messages_history()
        .as_chat_completions_history()
    )
    history.messages[0] = {"role": "assistant", "content": "answer"}

    with pytest.raises(ValueError, match="no longer matches its source exchange"):
        history.tokenize()


def test_converted_responses_history_tokenizes_without_native_chat_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange(
        "response-converted-exact", 12, prompt_token_ids=[10, 11]
    )
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer",
        lambda _config: pytest.fail("exact converted history loaded a tokenizer"),
    )

    history = trajectory.responses_history().as_chat_completions_history()
    assert all(
        source is None or source.exchange is exchange
        for source in history.message_sources
    )
    tokenized = history.tokenize()

    assert tokenized.token_ids == [10, 11, 12]
    assert tokenized.logprobs[-1] == pytest.approx(-0.1)
    assert tokenized.flags[-1] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_malformed_explicit_exact_token_metadata_fails_closed() -> None:
    chat = _chat_exchange([1], [2])
    chat_extra = chat.response.choices[0].model_extra
    assert chat_extra is not None
    chat_extra["prompt_token_ids"] = [1, "invalid"]

    completion = _completion_exchange()
    completion_extra = completion.response.choices[0].model_extra
    assert completion_extra is not None
    completion_extra["token_ids"] = [2, "invalid"]

    response = _response_exchange("response-invalid", 2)
    response_extra = response.response.model_extra
    assert response_extra is not None
    response_extra["token_generations"][0]["output_tokens"] = [{"token_id": "invalid"}]

    message = _message_exchange(
        MessagesRequest(
            model="test/model",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=16,
        ),
        identifier="message-invalid",
        token_ids=[2, "invalid"],
    )

    trajectories = [
        art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=[chat])),
        art.Trajectory(exchanges=TrajectoryExchanges(completions=[completion])),
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[response])),
        art.Trajectory(exchanges=TrajectoryExchanges(messages=[message])),
    ]
    for trajectory in trajectories:
        with pytest.raises(ValueError, match="exact token"):
            trajectory.tokenize(base_model="base/model")


@pytest.mark.parametrize("token_id", [-1, True])
def test_exact_token_metadata_rejects_negative_ids_and_booleans(
    token_id: object,
) -> None:
    exchange = _response_exchange("response-invalid-token", 2)
    extra = exchange.response.model_extra
    assert extra is not None
    extra["token_generations"][0]["output_tokens"] = [{"token_id": token_id}]

    with pytest.raises(ValueError, match="invalid exact token ID"):
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange])).tokenize()


@pytest.mark.parametrize(
    "exchange",
    [
        _completion_exchange(prompt=["batched"]),
        _completion_exchange(prompt=[[1, 2]]),
        _completion_exchange(echo=True),
    ],
)
def test_completions_support_single_item_batches_and_echo(
    exchange: CompletionsExchange,
) -> None:
    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize()
    assert tokenized.token_ids == [1, 2]


def test_completions_echo_preserves_prompt_logprobs_without_sampling_them() -> None:
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["token_ids"] = [2]
    payload["choices"][0]["logprobs"] = {
        "tokens": ["token_id:1", "token_id:2"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": [{}, {}],
        "text_offset": [0, 8],
    }
    exchange.response = Completion.model_validate(payload)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 2]
    assert tokenized.logprobs == [-0.1, -0.2]
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_completions_echo_does_not_strip_repeated_prompt_token_from_completion() -> (
    None
):
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["text"] = "questionquestionanswer"
    payload["choices"][0]["token_ids"] = [1, 2]
    payload["choices"][0]["logprobs"] = {
        "tokens": ["token_id:1", "token_id:2"],
        "token_logprobs": [-0.2, -0.3],
        "top_logprobs": [{}, {}],
        "text_offset": [8, 16],
    }
    exchange.response = Completion.model_validate(payload)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"question": [1], "questionanswer": [1, 2]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, object]], **kwargs: object
        ) -> list[int]:
            raise AssertionError(
                "Completions tokenization must not render chat messages"
            )

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 1, 2]
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.SAMPLED,
        art.TokenFlag.SAMPLED,
    ]
    assert all(math.isnan(logprob) for logprob in tokenized.logprobs)


def test_completions_echo_strips_prompt_from_proven_combined_token_carrier() -> None:
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["text"] = "questionquestionanswer"
    payload["choices"][0]["token_ids"] = [1, 1, 2]
    payload["choices"][0]["logprobs"] = {
        "tokens": ["token_id:1", "token_id:2"],
        "token_logprobs": [-0.2, -0.3],
        "top_logprobs": [{}, {}],
        "text_offset": [8, 16],
    }
    exchange.response = Completion.model_validate(payload)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 1, 2]
    assert tokenized.logprobs[1:] == pytest.approx([-0.2, -0.3])
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_completions_echo_strips_prompt_from_proven_textual_carrier() -> None:
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["token_ids"] = [1, 2]
    payload["choices"][0]["logprobs"] = {
        "tokens": ["question", "answer"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": [{}, {}],
        "text_offset": [0, 8],
    }
    exchange.response = Completion.model_validate(payload)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 2]
    assert tokenized.logprobs == pytest.approx([-0.1, -0.2])
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_completions_echo_prefers_full_logprob_carrier_to_id_prefix_heuristic() -> None:
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["token_ids"] = [1, 2]
    payload["choices"][0]["logprobs"] = {
        "tokens": ["token_id:1", "token_id:1", "token_id:2"],
        "token_logprobs": [-0.1, -0.2, -0.3],
        "top_logprobs": [{}, {}, {}],
        "text_offset": [0, 8, 9],
    }
    exchange.response = Completion.model_validate(payload)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 1, 2]
    assert tokenized.logprobs == [-0.1, -0.2, -0.3]
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_completions_echo_without_prompt_ids_falls_back_without_sampling_prompt() -> (
    None
):
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0].pop("prompt_token_ids")
    payload["choices"][0].pop("token_ids")
    payload["choices"][0]["logprobs"] = {
        "tokens": ["token_id:1", "token_id:2"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": [{}, {}],
        "text_offset": [0, 8],
    }
    exchange.response = Completion.model_validate(payload)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"question": [1], "answer": [2]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            raise AssertionError((messages, kwargs))

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2]
    assert tokenized.flags == [art.TokenFlag(0), art.TokenFlag.SAMPLED]
    assert all(math.isnan(logprob) for logprob in tokenized.logprobs)


def test_batched_completions_echo_uses_selected_prompt_boundary() -> None:
    exchange = _completion_exchange(prompt=["p0", "p1"], echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"] = [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": "p0a",
            "logprobs": {
                "tokens": ["p0", "a"],
                "token_logprobs": [-9.0, -0.1],
                "top_logprobs": [{}, {}],
                "text_offset": [0, 2],
            },
        },
        {
            "index": 1,
            "finish_reason": "stop",
            "text": "p1b",
            "logprobs": {
                "tokens": ["p1", "b"],
                "token_logprobs": [-9.0, -0.2],
                "top_logprobs": [{}, {}],
                "text_offset": [0, 2],
            },
        },
    ]
    exchange.request["n"] = 1
    exchange.response = Completion.model_validate(payload)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"p0": [1], "p1": [2], "a": [3], "b": [4]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            raise AssertionError((messages, kwargs))

    histories = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).completions_string_histories()

    assert [
        history.tokenize(tokenizer=Tokenizer()).token_ids for history in histories
    ] == [
        [1, 3],
        [2, 4],
    ]
    assert [
        history.tokenize(tokenizer=Tokenizer()).logprobs[-1] for history in histories
    ] == pytest.approx([-0.1, -0.2])


def test_completions_exact_ids_accept_textual_logprobs() -> None:
    exchange = _completion_exchange()
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["logprobs"]["tokens"] = ["answer"]
    payload["choices"][0]["logprobs"]["token_logprobs"] = [-0.75]
    exchange.response = Completion.model_validate(payload)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 2]
    assert tokenized.logprobs[1] == -0.75


def test_mutated_completions_token_prompt_drops_stale_exact_evidence() -> None:
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[_completion_exchange()])
    ).completions_token_history()
    history.prompt[0] = 99

    tokenized = history.tokenize()

    assert tokenized.token_ids == [99, 2]
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_mutated_completions_token_choice_is_rejected() -> None:
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[_completion_exchange()])
    ).completions_token_history()
    history.prompt[-1] = 99

    with pytest.raises(ValueError, match="sampled output"):
        history.tokenize()


def test_completions_history_rejects_model_and_sampled_span_mutation() -> None:
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[_completion_exchange()])
    ).completions_token_history()
    history.model = "other/model"
    with pytest.raises(ValueError, match="model no longer matches"):
        history.tokenize()

    history.model = "test/model"
    history.sampled_spans = [(0, len(history.prompt))]
    with pytest.raises(ValueError, match="exactly match choice-backed"):
        history.tokenize()


@pytest.mark.parametrize(
    "protocol",
    ["chat_completions", "messages", "responses", "completions"],
)
def test_source_backed_histories_reject_model_mutation(protocol: str) -> None:
    if protocol == "chat_completions":
        history = art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])])
        ).chat_completions_history()
    elif protocol == "messages":
        history = art.Trajectory(
            exchanges=TrajectoryExchanges(
                messages=[
                    _message_exchange(
                        MessagesRequest(
                            model="test/model",
                            messages=[{"role": "user", "content": "question"}],
                            max_tokens=16,
                        ),
                        prompt_token_ids=[1],
                        token_ids=[2],
                        logprobs=[-0.2],
                    )
                ]
            )
        ).anthropic_messages_history()
    elif protocol == "responses":
        history = art.Trajectory(
            exchanges=TrajectoryExchanges(
                responses=[_response_exchange("response-model", 2)]
            )
        ).responses_history()
    else:
        history = art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[_completion_exchange()])
        ).completions_token_history()

    history.model = "other/model"

    with pytest.raises(ValueError, match="model no longer matches"):
        history.tokenize()


def test_completions_string_history_preserves_textual_logprobs() -> None:
    exchange = _completion_exchange()
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0].pop("prompt_token_ids", None)
    payload["choices"][0].pop("token_ids", None)
    payload["choices"][0]["logprobs"]["tokens"] = ["answer"]
    payload["choices"][0]["logprobs"]["token_logprobs"] = [-0.8]
    exchange.response = Completion.model_validate(payload)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).completions_string_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"question": [1], "answer": [2]}[text]

        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            raise AssertionError("Completions tokenization does not render chat")

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2]
    assert tokenized.logprobs[1] == -0.8
    assert tokenized.flags[1] == art.TokenFlag.SAMPLED


def test_mutated_completions_string_prompt_retokens_without_stale_exact() -> None:
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[_completion_exchange()])
    ).completions_string_history()
    history.prompt = "changed" + history.prompt[len("question") :]
    first = history.prompt_sources[0]
    history.prompt_sources[0] = type(first)(
        start=0,
        end=len("changed"),
        source=first.source,
    )
    shift = len("changed") - len("question")
    second = history.prompt_sources[1]
    history.prompt_sources[1] = type(second)(
        start=second.start + shift,
        end=second.end + shift,
        source=second.source,
    )
    history.sampled_spans = [
        (start + shift, end + shift) for start, end in history.sampled_spans
    ]

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"changed": [9], "answer": [2]}[text]

        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            raise AssertionError("Completions tokenization does not render chat")

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [9, 2]
    assert tokenized.flags[0] == art.TokenFlag(0)


def test_batched_completions_map_each_choice_to_its_prompt() -> None:
    exchange = _completion_exchange(prompt=["first", "second"])
    payload = exchange.response.model_dump(mode="python")
    payload["choices"] = [
        {
            **payload["choices"][0],
            "index": 0,
            "text": "one",
            "prompt_token_ids": [10],
            "token_ids": [11],
            "logprobs": {
                "tokens": ["token_id:11"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{}],
                "text_offset": [0],
            },
        },
        {
            **payload["choices"][0],
            "index": 1,
            "text": "two",
            "prompt_token_ids": [20],
            "token_ids": [21],
            "logprobs": {
                "tokens": ["token_id:21"],
                "token_logprobs": [-0.2],
                "top_logprobs": [{}],
                "text_offset": [0],
            },
        },
    ]
    exchange.response = Completion.model_validate(payload)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).tokenize(multi_history=True)

    assert [history.token_ids for history in tokenized.histories] == [
        [10, 11],
        [20, 21],
    ]
    assert [history.flags for history in tokenized.histories] == [
        [art.TokenFlag.EXACT, art.TokenFlag.EXACT | art.TokenFlag.SAMPLED],
        [art.TokenFlag.EXACT, art.TokenFlag.EXACT | art.TokenFlag.SAMPLED],
    ]


def test_history_tokenization_rejects_negative_source_indices() -> None:
    exchange = _response_exchange("response-negative-source", 2)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).responses_history()
    history.input_sources[-1] = ResponsesItemSource(
        exchange=exchange,
        output_index=-1,
        generation_index=0,
    )

    with pytest.raises(ValueError, match="out of bounds"):
        history.tokenize()


def test_batched_completions_reject_ambiguous_choice_indices() -> None:
    exchange = _completion_exchange(prompt=["first", "second"])
    payload = exchange.response.model_dump(mode="python")
    payload["choices"] = [
        {**payload["choices"][0], "index": 0},
        {**payload["choices"][0], "index": 2},
    ]
    exchange.response = Completion.model_validate(payload)

    with pytest.raises(ValueError, match="Ambiguous"):
        art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange])).tokenize(
            multi_history=True
        )


def test_randomized_completions_projection_preserves_every_choice_once() -> None:
    from art.trajectories._tokenize import _tokenize_trajectory_with_trace

    rng = random.Random(0)
    for case in range(20):
        prompt_count = rng.randint(1, 5)
        choices_per_prompt = rng.randint(1, 4)
        exchange = _completion_exchange(
            prompt=[f"prompt-{case}-{index}" for index in range(prompt_count)]
        )
        exchange.request["n"] = choices_per_prompt
        template = exchange.response.model_dump(mode="python")["choices"][0]
        choices: list[dict[str, object]] = []
        expected: list[list[int]] = []
        for prompt_index in range(prompt_count):
            prompt_id = 10_000 + case * 100 + prompt_index
            for local_choice in range(choices_per_prompt):
                choice_index = prompt_index * choices_per_prompt + local_choice
                output_id = 20_000 + case * 100 + choice_index
                expected.append([prompt_id, output_id])
                choices.append(
                    {
                        **template,
                        "index": choice_index,
                        "text": f"answer-{output_id}",
                        "prompt_token_ids": [prompt_id],
                        "token_ids": [output_id],
                        "logprobs": {
                            "tokens": [f"token_id:{output_id}"],
                            "token_logprobs": [-0.1],
                            "top_logprobs": [{}],
                            "text_offset": [0],
                        },
                    }
                )
        rng.shuffle(choices)
        response = exchange.response.model_dump(mode="python")
        response["choices"] = choices
        exchange.response = Completion.model_validate(response)

        trajectory = art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[exchange])
        )
        tokenized = trajectory.tokenize(multi_history=True)

        assert [history.token_ids for history in tokenized.histories] == expected
        assert all(
            history.flags
            == [art.TokenFlag.EXACT, art.TokenFlag.EXACT | art.TokenFlag.SAMPLED]
            for history in tokenized.histories
        )
        traced, traces = _tokenize_trajectory_with_trace(trajectory)
        assert [history.token_ids for history in traced.histories] == expected
        assert [
            next(key for key in trace.source_keys if key is not None).prompt_index
            for trace in traces
        ] == [
            prompt_index
            for prompt_index in range(prompt_count)
            for _ in range(choices_per_prompt)
        ]
        assert all(len(trace.sources) == 1 for trace in traces)


def test_branching_and_multiple_models_require_explicit_resolution() -> None:
    alternate = _chat_exchange([9], [3], offset=1)
    alternate.request["messages"] = [{"role": "user", "content": "alternate"}]
    branching = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2], offset=0),
                alternate,
            ]
        )
    )
    with pytest.raises(ValueError, match="exactly one history"):
        branching.tokenize()
    assert len(branching.tokenize(multi_history=True).histories) == 2

    mixed = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2], model="one", offset=0),
                _chat_exchange([3], [4], model="two", offset=1),
            ]
        )
    )
    with pytest.raises(ValueError, match="exactly one model"):
        mixed.tokenize()
    assert mixed.tokenize(model="two").token_ids == [3, 4]
    assert [
        history.model for history in mixed.tokenize(multi_history=True).histories
    ] == ["one", "two"]


def test_model_selection_prefers_exact_identity_over_glob_interpretation() -> None:
    literal_model = "org/model[1]"
    wildcard_match = _chat_exchange([3], [4], model="org/model1", offset=1)
    exact_match = _chat_exchange([1], [2], model=literal_model)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[wildcard_match, exact_match])
    )

    tokenized = trajectory.tokenize(model=literal_model)

    assert tokenized.model == literal_model
    assert tokenized.token_ids == [1, 2]


def test_legacy_additional_histories_require_multi_history_and_model() -> None:
    first = _chat_exchange([1], [2]).response.choices[0]
    second = _chat_exchange([3], [4]).response.choices[0]
    trajectory = art.Trajectory(
        messages_and_choices=[first],
        additional_histories=[art.LegacyHistory(messages_and_choices=[second])],
    )

    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize(model="test/model")
    with pytest.raises(ValueError, match="requires model="):
        trajectory.tokenize(multi_history=True)

    tokenized = trajectory.tokenize(multi_history=True, model="test/model")

    assert [history.token_ids for history in tokenized.histories] == [[1, 2], [3, 4]]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [11]

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: object
    ) -> list[int]:
        self.calls.append(kwargs)
        return [10, 11] if messages[-1]["role"] == "assistant" else [10]


def test_fallback_uses_template_overrides_and_nan_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _message_exchange(
        MessagesRequest(
            model="wandb-artifact:///entity/project/run:step0",
            messages=[{"role": "user", "content": "question"}],
            chat_template="request-template",
            chat_template_kwargs={"request": True},
            thinking={"type": "enabled", "budget_tokens": 128},
        ),
        duration=timedelta(seconds=1),
    )
    tokenizer = _FakeTokenizer()
    loaded_base_models: list[str] = []
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer",
        lambda config: loaded_base_models.append(config.base_model) or tokenizer,
    )
    monkeypatch.setattr(
        "art.trajectories._tokenize._artifact_config",
        lambda _model: pytest.fail("explicit base_model should bypass W&B"),
    )

    result = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[exchange])
    ).tokenize(
        base_model="base/model",
        chat_template="explicit-template",
        chat_template_kwargs={"explicit": True},
    )

    assert result.token_ids == [10, 11]
    assert loaded_base_models == ["base/model"]
    assert result.flags == [art.TokenFlag(0), art.TokenFlag.SAMPLED]
    assert math.isnan(result.logprobs[1])
    assert len(tokenizer.calls) == 3
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [
        False,
        False,
        True,
    ]
    assert [call["tokenize"] for call in tokenizer.calls] == [True, False, True]
    assert all(
        {
            key: value
            for key, value in call.items()
            if key not in {"add_generation_prompt", "tokenize"}
        }
        == {
            "tools": None,
            "chat_template": "explicit-template",
            "request": True,
            "explicit": True,
            "enable_thinking": True,
            "thinking_budget": 128,
        }
        for call in tokenizer.calls
    )


@pytest.mark.parametrize(
    ("model", "artifact_name"),
    [
        ("wandb-artifact:///entity/project/run", "entity/project/run:latest"),
        ("wandb-artifact:///entity/project/run:step0", "entity/project/run:step0"),
    ],
)
def test_checkpoint_fallback_preserves_artifact_version_and_renderer(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    artifact_name: str,
) -> None:
    artifact_names: list[str] = []

    class Api:
        def artifact(self, name: str) -> SimpleNamespace:
            artifact_names.append(name)
            return SimpleNamespace(
                metadata={
                    "wandb.base_model": "base/model",
                    "renderer": {
                        "tokenizer_revision": "revision",
                        "chat_template": "template",
                        "chat_template_kwargs": {"thinking": True},
                    },
                }
            )

    wandb = ModuleType("wandb")
    apis = ModuleType("wandb.apis")
    public = ModuleType("wandb.apis.public")
    setattr(public, "Api", Api)
    setattr(apis, "public", public)
    setattr(wandb, "apis", apis)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setitem(sys.modules, "wandb.apis", apis)
    monkeypatch.setitem(sys.modules, "wandb.apis.public", public)
    exchange = _chat_exchange([], [], model=model)
    extra = exchange.response.choices[0].model_extra
    assert extra is not None
    extra.pop("prompt_token_ids")
    extra.pop("token_ids")
    exchange.response.choices[0].logprobs = None
    tokenizer = _FakeTokenizer()
    configs = []
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer",
        lambda config: configs.append(config) or tokenizer,
    )

    art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize()
    config = configs[0]

    assert artifact_names == [artifact_name]
    assert config.base_model == "base/model"
    assert config.revision == "revision"
    assert config.chat_template == "template"
    assert config.chat_template_kwargs == {"thinking": True}
    assert tokenizer.calls[0]["chat_template"] == "template"
    assert tokenizer.calls[0]["thinking"] is True


def test_loaded_tokenizers_are_cached_by_model_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.trajectories._tokenize import (
        _cached_tokenizer,
        _load_tokenizer,
        _TokenizerConfig,
    )

    loaded: list[tuple[str, str | None]] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, *, revision: str | None) -> object:
            loaded.append((model, revision))
            return object()

    transformers = ModuleType("transformers")
    setattr(transformers, "AutoTokenizer", AutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    _cached_tokenizer.cache_clear()
    try:
        config = _TokenizerConfig("test/model", revision="revision")
        assert _load_tokenizer(config) is _load_tokenizer(config)
        assert loaded == [("test/model", "revision")]
    finally:
        _cached_tokenizer.cache_clear()


def test_deepseek_v4_uses_arts_protocol_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.trajectories._tokenize import _cached_tokenizer

    raw = object()
    wrapped = object()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, *, revision: str | None) -> object:
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert revision is None
            return raw

    transformers = ModuleType("transformers")
    setattr(transformers, "AutoTokenizer", AutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "art.megatron.dsv4.tokenizer.get_dsv4_tokenizer",
        lambda tokenizer: (
            wrapped if tokenizer is raw else pytest.fail("wrong tokenizer")
        ),
    )
    _cached_tokenizer.cache_clear()
    try:
        assert _cached_tokenizer("deepseek-ai/DeepSeek-V4-Flash", None) is wrapped
    finally:
        _cached_tokenizer.cache_clear()


def test_anthropic_fallback_preserves_thinking_and_tool_history() -> None:
    from art.trajectories._tokenize import _anthropic_messages

    messages = _anthropic_messages(
        {
            "system": [{"type": "text", "text": "system"}],
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reason"},
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "lookup",
                            "input": {"key": "value"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "result"}],
                        },
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
        }
    )

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "calling",
            "reasoning": "reason",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key": "value"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "user", "content": "continue"},
    ]


@pytest.mark.parametrize("top_level_only", [False, True])
def test_reasoning_stripped_messages_history_preserves_exact_tokens(
    top_level_only: bool,
) -> None:
    def exchange(
        offset: int,
        request_messages: list[MessageParam],
        answer: str,
        prompt_token_ids: list[int],
        token_ids: list[int],
    ) -> MessagesExchange:
        start = datetime(2026, 1, 1) + timedelta(seconds=offset)
        thinking: dict[str, Any] = {
            "type": "thinking",
            "thinking": f"thought-{offset}",
            "signature": "signature",
        }
        text: dict[str, Any] = {"type": "text", "text": answer}
        response_extra: dict[str, Any] = {}
        if top_level_only:
            response_extra = {
                "prompt_token_ids": prompt_token_ids,
                "token_ids": [90 + offset, *token_ids],
                "logprobs": [
                    -9.0 - offset,
                    *[-0.1 * token for token in token_ids],
                ],
            }
        else:
            thinking.update({"token_ids": [90 + offset], "logprobs": [-9.0 - offset]})
            text.update(
                {
                    "token_ids": token_ids,
                    "logprobs": [-0.1 * token for token in token_ids],
                }
            )
        return MessagesExchange(
            request=MessagesRequest(
                model="test/model",
                messages=request_messages,
                max_tokens=16,
                thinking={"type": "enabled", "budget_tokens": 8},
            ),
            response=Message.model_validate(
                {
                    "id": f"message-{offset}",
                    "type": "message",
                    "role": "assistant",
                    "model": "test/model",
                    "content": [thinking, text],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": len(token_ids)},
                    **response_extra,
                }
            ),
            start_time=start,
            end_time=start + timedelta(milliseconds=1),
        )

    first = exchange(
        0,
        [{"role": "user", "content": "one"}],
        "first",
        [10],
        [101, 102],
    )
    second = exchange(
        1,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ],
        "second",
        [10, 101, 102, 11],
        [201],
    )

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {
                "one": [10],
                "first": [50],
                "second": [60],
                "thought-1": [70],
                "two": [11],
            }[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            by_length = {
                1: [10],
                2: [10, 50],
                3: [10, 50, 11],
                4: [10, 50, 11, 70, 60],
            }
            return by_length[len(messages)]

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[first, second])
    ).anthropic_messages_histories()[1]
    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [10, 101, 102, 11, 91, 201]
    assert tokenized.logprobs[1:3] == pytest.approx([-10.1, -10.2])
    assert tokenized.logprobs[-2] == pytest.approx(-10.0)
    assert tokenized.logprobs[-1] == pytest.approx(-20.1)
    assert tokenized.flags[1:3] == [
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]
    assert tokenized.flags[-2:] == [
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_choice_logprobs_survive_tokenizer_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([], [])
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="answer",
                    logprob=-0.7,
                    bytes=list(b"answer"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] != "assistant":
                return [10]
            if str(messages[-1]["content"]).startswith("ART_TRAJECTORY_"):
                return [10, 99, 12]
            return [10, 11, 12]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                input_ids={"turn 0": [10], "answer": [11], "turn 1": [20]}[text]
            )

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize(
        base_model="base/model",
    )
    assert result.token_ids == [10, 11, 12]
    assert result.logprobs[1] == -0.7
    assert math.isnan(result.logprobs[2])


def test_chat_fallback_rejects_unique_text_match_in_generation_scaffold() -> None:
    exchange = _chat_exchange([], [])
    exchange.request["messages"] = [{"role": "user", "content": "question"}]
    choice = exchange.response.choices[0]
    assert choice.model_extra is not None
    choice.model_extra.pop("prompt_token_ids", None)
    choice.model_extra.pop("token_ids", None)
    assert choice.logprobs is not None
    choice.logprobs = choice.logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="answer",
                    logprob=-0.7,
                    bytes=list(b"answer"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant":
                return [1, 11, 12]
            return [1, 11] if add_generation_prompt else [1]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(input_ids={"question": [1], "answer": [11]}[text])

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    with pytest.raises(ValueError, match="uniquely locate"):
        history.tokenize(tokenizer=Tokenizer())


def test_chat_exact_ids_reject_unique_match_in_generation_scaffold() -> None:
    exchange = _chat_exchange([], [11])
    exchange.request["messages"] = [{"role": "user", "content": "question"}]

    class Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant":
                return [1, 11, 12]
            return [1, 11] if add_generation_prompt else [1]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(input_ids={"question": [1], "answer": [11]}[text])

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    with pytest.raises(ValueError, match="uniquely locate"):
        history.tokenize(tokenizer=Tokenizer())


def test_chat_prompt_ids_do_not_bind_output_to_trailing_scaffold() -> None:
    exchange = _chat_exchange([1, 2], [])
    choice = exchange.response.choices[0]
    assert choice.logprobs is not None
    choice.logprobs = choice.logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="answer",
                    logprob=-0.7,
                    bytes=list(b"answer"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [1, 2, 12, 11] if messages[-1]["role"] == "assistant" else [1, 2]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                input_ids={"turn 0": [10], "answer": [11], "turn 1": [20]}[text]
            )

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    with pytest.raises(ValueError, match="content boundary"):
        history.tokenize(tokenizer=Tokenizer())


def test_chat_exact_hidden_suffix_preserves_rendered_trailing_scaffold() -> None:
    exchange = _chat_exchange([1, 99], [7, 8])
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] != "assistant":
                return [1, 99]
            if str(messages[-1]["content"]).startswith("ART_TRAJECTORY_"):
                return [1, 99, 999, 9]
            return [1, 99, 7, 9]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del text, kwargs
            return SimpleNamespace(input_ids=[7])

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 99, 7, 8, 9]
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
    ]


def test_each_chat_choice_preserves_its_visible_fallback_logprobs() -> None:
    exchange = _chat_exchange([], [])
    exchange.request["messages"] = [{"role": "user", "content": "question"}]
    data = exchange.response.model_dump(mode="python")
    choices = []
    for index, (text, logprob) in enumerate((("left", -0.1), ("right", -0.2))):
        choice = data["choices"][0].copy()
        choice.pop("prompt_token_ids", None)
        choice.pop("token_ids", None)
        choice["index"] = index
        choice["message"] = {"role": "assistant", "content": text}
        choice["logprobs"] = {
            "content": [
                {
                    "token": text,
                    "logprob": logprob,
                    "bytes": list(text.encode()),
                    "top_logprobs": [],
                }
            ]
        }
        choices.append(choice)
    data["choices"] = choices
    exchange.response = ChatCompletion.model_validate(data)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"question": [1], "left": [2], "right": [3]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [
                token for message in messages for token in self(str(message["content"]))
            ]

    histories = trajectory.chat_completions_histories()
    direct = [history.tokenize(tokenizer=Tokenizer()) for history in histories]
    tokenized = trajectory.tokenize(multi_history=True, tokenizer=Tokenizer())

    assert [history.logprobs[-1] for history in direct] == [-0.1, -0.2]
    assert [history.logprobs[-1] for history in tokenized.histories] == [-0.1, -0.2]

    from art.trajectories._tokenize import _tokenize_trajectory_with_trace

    traced, traces = _tokenize_trajectory_with_trace(trajectory, tokenizer=Tokenizer())
    assert [history.logprobs[-1] for history in traced.histories] == [-0.1, -0.2]
    assert [
        {key.index for key in trace.source_keys if key is not None} for trace in traces
    ] == [{0}, {1}]


def test_chat_fallback_anchors_sampled_text_away_from_equal_user_text() -> None:
    first = _chat_exchange([], [], offset=0)
    first.request["messages"] = [{"role": "user", "content": "q"}]
    first_data = first.response.model_dump(mode="python")
    first_choice = first_data["choices"][0]
    first_choice.pop("prompt_token_ids", None)
    first_choice.pop("token_ids", None)
    first_choice["message"]["content"] = "same"
    first_choice["logprobs"]["content"] = [
        {
            "token": "same",
            "logprob": -0.1,
            "bytes": list(b"same"),
            "top_logprobs": [],
        }
    ]
    first.response = ChatCompletion.model_validate(first_data)

    second = _chat_exchange([], [], offset=1)
    second.request["messages"] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "same"},
        {"role": "user", "content": "same"},
    ]
    second_data = second.response.model_dump(mode="python")
    second_choice = second_data["choices"][0]
    second_choice.pop("prompt_token_ids", None)
    second_choice.pop("token_ids", None)
    second_choice["message"]["content"] = "other"
    second_choice["logprobs"]["content"] = [
        {
            "token": "other",
            "logprob": -0.2,
            "bytes": list(b"other"),
            "top_logprobs": [],
        }
    ]
    second.response = ChatCompletion.model_validate(second_data)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"q": [1], "same": [2], "other": [3]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [
                token for message in messages for token in self(str(message["content"]))
            ]

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).chat_completions_history()
    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2, 2, 3]
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
    ]
    assert tokenized.logprobs[1] == -0.1
    assert math.isnan(tokenized.logprobs[2])
    assert tokenized.logprobs[3] == -0.2


def test_exact_chat_ids_accept_ordinary_positional_logprobs() -> None:
    exchange = _chat_exchange([1], [2])
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None and logprobs.content
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                logprobs.content[0].model_copy(
                    update={"token": "answer", "logprob": -0.75}
                )
            ]
        }
    )

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 2]
    assert tokenized.logprobs[1] == -0.75


def test_chat_view_preserves_two_turn_textual_logprobs_without_exact_ids() -> None:
    exchanges = [_chat_exchange([], [], offset=index) for index in range(2)]
    for index, exchange in enumerate(exchanges):
        choice = exchange.response.choices[0]
        assert choice.model_extra is not None
        choice.model_extra.pop("prompt_token_ids", None)
        choice.model_extra.pop("token_ids", None)
        assert choice.logprobs is not None
        choice.logprobs = choice.logprobs.model_copy(
            update={
                "content": [
                    ChatCompletionTokenLogprob(
                        token="answer",
                        logprob=-0.4 - index / 10,
                        bytes=list(b"answer"),
                        top_logprobs=[],
                    )
                ]
            }
        )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            if len(messages) == 4:
                return [10, 11, 20, 11]
            if messages[-1]["role"] == "assistant":
                return [10, 11]
            return [10]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                input_ids={"turn 0": [10], "answer": [11], "turn 1": [20]}[text]
            )

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=exchanges)
    ).chat_completions_history()
    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [10, 11, 20, 11]
    assert tokenized.logprobs[1] == pytest.approx(-0.4)
    assert tokenized.logprobs[3] == pytest.approx(-0.5)
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
    ]


def test_chat_view_uses_later_exact_prompt_when_first_is_missing() -> None:
    first = _chat_exchange([], [11])
    second = _chat_exchange([10, 11, 20], [], offset=1)
    second.response.choices[0].message.content = "final"
    choice = second.response.choices[0]
    assert choice.model_extra is not None
    choice.model_extra.pop("token_ids", None)
    assert choice.logprobs is not None
    choice.logprobs = choice.logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="final",
                    logprob=-0.5,
                    bytes=list(b"final"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            token_ids = {
                "turn 0": [10],
                "answer": [11],
                "turn 1": [20],
                "final": [21],
            }
            return [
                token
                for message in messages
                for token in token_ids[str(message["content"])]
            ]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                input_ids={
                    "turn 0": [10],
                    "answer": [11],
                    "turn 1": [20],
                    "final": [21],
                }[text]
            )

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).chat_completions_history()
    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [10, 11, 20, 21]
    assert tokenized.logprobs[-1] == pytest.approx(-0.5)
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT,
        art.TokenFlag.SAMPLED,
    ]


def test_chat_content_and_refusal_logprobs_are_combined_in_protocol_order() -> None:
    exchange = _chat_exchange([1], [2, 3])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice["message"]["refusal"] = "refusal"
    choice["logprobs"] = {
        "content": [
            {
                "token": "token_id:2",
                "logprob": -0.2,
                "bytes": [],
                "top_logprobs": [],
            }
        ],
        "refusal": [
            {
                "token": "token_id:3",
                "logprob": -0.3,
                "bytes": [],
                "top_logprobs": [],
            }
        ],
    }
    exchange.response = ChatCompletion.model_validate(data)

    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize()

    assert tokenized.token_ids == [1, 2, 3]
    assert tokenized.logprobs[1:] == pytest.approx([-0.2, -0.3])


def test_chat_content_and_refusal_logprobs_must_match_exact_ids() -> None:
    exchange = _chat_exchange([1], [2, 3])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice["message"]["refusal"] = "refusal"
    choice["logprobs"] = {
        "content": [
            {
                "token": "token_id:2",
                "logprob": -0.2,
                "bytes": [],
                "top_logprobs": [],
            }
        ],
        "refusal": [
            {
                "token": "token_id:4",
                "logprob": -0.4,
                "bytes": [],
                "top_logprobs": [],
            }
        ],
    }
    exchange.response = ChatCompletion.model_validate(data)

    with pytest.raises(ValueError, match="disagree with choice logprobs"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[exchange])
        ).tokenize()


def test_chat_visible_logprobs_include_content_and_refusal() -> None:
    from art.trajectories._tokenize import _visible_logprobs

    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice["message"]["refusal"] = "refusal"
    choice["logprobs"] = {
        "content": [
            {
                "token": "answer",
                "logprob": -0.2,
                "bytes": list(b"answer"),
                "top_logprobs": [],
            }
        ],
        "refusal": [
            {
                "token": "refusal",
                "logprob": -0.3,
                "bytes": list(b"refusal"),
                "top_logprobs": [],
            }
        ],
    }
    exchange.response = ChatCompletion.model_validate(data)

    assert _visible_logprobs(exchange) == [
        ("answer", -0.2),
        ("refusal", -0.3),
    ]


def test_empty_chat_prompt_ids_are_missing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([], [2])

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 20] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return [20]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize(base_model="base/model")

    assert tokenized.token_ids == [10, 2]
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_missing_completion_renders_only_missing_region_when_prompt_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([99], [])
    extra = exchange.response.choices[0].model_extra
    assert extra is not None
    extra.pop("token_ids", None)
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="answer",
                    logprob=-0.5,
                    bytes=list(b"answer"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 20] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return [20]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    tokenized = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize(base_model="base/model")

    assert tokenized.token_ids == [99, 20]
    assert tokenized.logprobs[1] == -0.5
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.SAMPLED,
    ]


def test_ambiguous_visible_logprobs_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([], [])
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="answer",
                    logprob=-0.7,
                    bytes=list(b"answer"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 11, 12, 11] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del text, kwargs
            return SimpleNamespace(input_ids=[11])

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    with pytest.raises(ValueError, match="uniquely locate"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[exchange])
        ).tokenize(
            base_model="base/model",
        )


def test_legacy_token_and_logprob_length_mismatch_raises() -> None:
    exchange = _chat_exchange([1], [2, 3])
    choice = exchange.response.choices[0]
    assert choice.logprobs is not None
    content = choice.logprobs.content
    assert content
    choice.logprobs = choice.logprobs.model_copy(
        update={
            "content": [
                content[0].model_copy(
                    update={"token": "answer", "bytes": list(b"answer")}
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="differ in length"):
        art.Trajectory(messages_and_choices=[choice]).tokenize(model="test/model")


def test_anthropic_fallback_rejects_unknown_content_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image: ImageBlockParam = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "...",
        },
    }
    message: MessageParam = {"role": "user", "content": [image]}
    exchange = _message_exchange(
        MessagesRequest(
            model="test/model",
            messages=[message],
        ),
        duration=timedelta(seconds=1),
    )
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: _FakeTokenizer()
    )

    with pytest.raises(ValueError, match="Unsupported Anthropic content block"):
        art.Trajectory(exchanges=TrajectoryExchanges(messages=[exchange])).tokenize(
            base_model="base/model",
        )


def test_undecodable_visible_token_bytes_fall_back_to_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([], [])
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="ordinary-token",
                    logprob=-0.7,
                    bytes=[0xF0],
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 11] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return [11]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize(
        base_model="base/model",
    )

    assert result.token_ids == [10, 11]
    assert math.isnan(result.logprobs[1])


def test_json_round_trip_preserves_exchange_types() -> None:
    exchange = _chat_exchange([1], [2])
    request: dict[str, Any] = {
        "model": "test/model",
        "messages": [
            {"role": "assistant", "content": "answer", "reasoning": "thinking"}
        ],
    }
    exchange.request = ChatCompletionsRequest(**request)
    original = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    )
    dumped = original.model_dump(mode="json", warnings="error")
    assert dumped["exchanges"]["chat_completions"][0]["request"] == request
    restored = art.Trajectory.model_validate_json(original.model_dump_json())
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")
    assert isinstance(restored.exchanges.chat_completions[0].response, ChatCompletion)


def _response_exchange(
    response_id: str,
    output_id: int,
    *,
    previous_response_id: str | None = None,
    offset: int = 0,
    prompt_token_ids: list[int] | None = None,
) -> ResponsesExchange:
    response = Response.model_validate(
        {
            "id": response_id,
            "created_at": float(offset),
            "model": "test/model",
            "object": "response",
            "output": [
                {
                    "id": f"message-{response_id}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "answer",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "token_generations": [
                {
                    "prompt_token_ids": prompt_token_ids or [10],
                    "output_tokens": [{"token_id": output_id, "logprob": -0.1}],
                    "output_indices": [0],
                }
            ],
        }
    )
    request = ResponsesRequest(model="test/model", input=f"turn {offset}")
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return ResponsesExchange(
        request=request,
        response=response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def test_cross_exchange_responses_reasoning_split_uses_later_prompt_backbone() -> None:
    first = _response_exchange("response-1", 3, prompt_token_ids=[1])
    first_data = first.response.model_dump(mode="python")
    first_data["output"] = [
        {
            "id": "reasoning-response-1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "think"}],
        },
        first_data["output"][0],
    ]
    first_data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [
                {"token_id": 2, "logprob": -0.2},
                {"token_id": 3, "logprob": -0.3},
            ],
            "output_indices": [0, 1],
        }
    ]
    first.response = Response.model_validate(first_data)
    second = _response_exchange(
        "response-2",
        5,
        previous_response_id="response-1",
        offset=1,
        prompt_token_ids=[1, 3, 4],
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first, second])
    )

    tokenized = trajectory.tokenize(multi_history=True)

    assert [history.token_ids for history in tokenized.histories] == [
        [1, 2, 3],
        [1, 3, 4, 5],
    ]
    assert math.isnan(tokenized.histories[1].logprobs[0])
    assert tokenized.histories[1].logprobs[1] == -0.3
    assert math.isnan(tokenized.histories[1].logprobs[2])
    assert tokenized.histories[1].logprobs[3] == -0.1
    assert tokenized.histories[1].flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def _response_with_content_logprobs(*, exact_second: bool) -> ResponsesExchange:
    exchange = _response_exchange("response-content-logprobs", 0)
    data = exchange.response.model_dump(mode="python")
    data.pop("token_generations", None)

    def entry(token: str, token_id: int | None, logprob: float) -> dict[str, Any]:
        return {
            "token": token,
            "logprob": logprob,
            "bytes": list(("a" if token_id == 11 else "b").encode()),
            "top_logprobs": [],
            **({"token_id": token_id} if token_id is not None else {}),
        }

    data["output"][0]["content"] = [
        {
            "type": "output_text",
            "text": "a",
            "annotations": [],
            "logprobs": [entry("token_id:11", 11, -0.1)],
        },
        {
            "type": "output_text",
            "text": "b",
            "annotations": [],
            "logprobs": [
                entry(
                    "token_id:12" if exact_second else "b",
                    12 if exact_second else None,
                    -0.2,
                )
            ],
        },
    ]
    exchange.response = Response.model_validate(data)
    return exchange


def test_responses_aggregates_complete_exact_pairs_across_content_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [10]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.Trajectory(
        exchanges=TrajectoryExchanges(
            responses=[_response_with_content_logprobs(exact_second=True)]
        )
    ).tokenize(
        base_model="base/model",
    )

    assert result.token_ids == [10, 11, 12]
    assert result.logprobs[1:] == [-0.1, -0.2]
    assert result.flags[1:] == [
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_responses_tool_output_source_is_not_sampled_without_generation() -> None:
    from art.trajectories._tokenize import (
        _responses_output_is_sampled,
        _source_is_sampled,
    )

    exchange = _response_exchange("response-tool-output", 0)
    data = exchange.response.model_dump(mode="python")
    data.pop("token_generations", None)
    data["output"] = [
        {
            "type": "function_call_output",
            "id": "output-1",
            "call_id": "call-1",
            "output": "result",
            "status": "completed",
        }
    ]
    exchange.response = Response.model_validate(data)

    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )
    assert history.messages[-1]["role"] == "tool"
    source = history.message_sources[-1]
    assert source is not None
    assert not _source_is_sampled(source)
    assert not _responses_output_is_sampled({"type": "function_call_output"})


def test_responses_source_rejects_boolean_generation_index() -> None:
    from art.trajectories._tokenize import _responses_source_generation

    exchange = _response_exchange("response-bool-generation", 2)
    source = ChatCompletionsMessageSource(
        exchange=exchange,
        output_indices=(0,),
        generation_index=True,
    )

    with pytest.raises(ValueError, match="generation index is invalid"):
        _responses_source_generation(source)


def test_responses_missing_token_generations_falls_back_for_visible_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange("response-empty-raw", 0)
    data = exchange.response.model_dump(mode="python")
    data.pop("token_generations", None)
    exchange.response = Response.model_validate(data)
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: _FakeTokenizer()
    )

    result = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).tokenize(
        base_model="base/model",
        chat_template="template",
        chat_template_kwargs={},
    )

    assert result.token_ids == [10, 11]


def test_responses_does_not_use_partial_exact_content_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 11, 12] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                input_ids=[11 if text in {"a", "token_id:11"} else 12]
            )

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.Trajectory(
        exchanges=TrajectoryExchanges(
            responses=[_response_with_content_logprobs(exact_second=False)]
        )
    ).tokenize(
        base_model="base/model",
    )

    assert result.token_ids == [10, 11, 12]
    assert result.logprobs[1:] == [-0.1, -0.2]


def test_responses_rejects_only_unrenderable_prompt_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            assistant_count = sum(
                message["role"] == "assistant" for message in messages
            )
            return [10, *range(2, 2 + assistant_count)]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    request_reasoning = _response_exchange("request-reasoning", 2)
    request_data = request_reasoning.response.model_dump(mode="python")
    request_data.pop("token_generations", None)
    request_reasoning.response = Response.model_validate(request_data)
    request_reasoning.request["input"] = [
        {
            "id": "reasoning-1",
            "summary": [{"type": "summary_text", "text": "request thought"}],
            "type": "reasoning",
        }
    ]

    response_reasoning = _response_exchange("response-reasoning", 2)
    data = response_reasoning.response.model_dump(mode="python")
    data["output"] = [
        {
            "id": "reasoning-2",
            "summary": [{"type": "summary_text", "text": "response thought"}],
            "type": "reasoning",
        }
    ]
    data.pop("token_generations", None)
    response_reasoning.response = Response.model_validate(data)

    art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[request_reasoning])
    ).tokenize(
        base_model="base/model",
    )

    single = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[response_reasoning])
    )
    assert single.tokenize(base_model="base/model").token_ids == [
        10,
        2,
    ]

    continuation = _response_exchange(
        "continuation",
        3,
        previous_response_id=response_reasoning.response.id,
        offset=1,
    )
    continuation_data = continuation.response.model_dump(mode="python")
    continuation_data.pop("token_generations", None)
    continuation.response = Response.model_validate(continuation_data)
    assert art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[response_reasoning, continuation])
    ).tokenize(
        base_model="base/model",
    ).token_ids == [10, 2, 3]


def test_responses_opaque_reasoning_requires_exact_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange("opaque-reasoning", 2)
    response = exchange.response.model_dump(mode="python")
    response["output"] = [
        {
            "id": "reasoning-1",
            "encrypted_content": "opaque",
            "summary": [],
            "type": "reasoning",
        }
    ]
    exchange.response = Response.model_validate(response)
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: _FakeTokenizer()
    )

    assert art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange])).tokenize(
        base_model="base/model",
    ).token_ids == [10, 2]

    response = exchange.response.model_dump(mode="python")
    response.pop("token_generations", None)
    exchange.response = Response.model_validate(response)
    with pytest.raises(ValueError, match="no renderable text"):
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange])).tokenize(
            base_model="base/model",
        )


def test_responses_parallel_function_calls_form_one_assistant_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange("parallel-tools", 2)
    response_data = exchange.response.model_dump(mode="python")
    response_data.pop("token_generations", None)
    exchange.response = Response.model_validate(response_data)
    exchange.request["input"] = [
        {
            "id": "reasoning-1",
            "summary": [{"type": "summary_text", "text": "think"}],
            "type": "reasoning",
        },
        {"type": "function_call", "call_id": "one", "name": "first", "arguments": "{}"},
        {
            "type": "function_call",
            "call_id": "two",
            "name": "second",
            "arguments": "{}",
        },
    ]
    seen: list[list[dict[str, Any]]] = []

    class Tokenizer(_FakeTokenizer):
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            seen.append(messages)
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange])).tokenize(
        base_model="base/model",
    )

    assistant = seen[0][0]
    assert assistant["reasoning"] == "think"
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == [
        "first",
        "second",
    ]


def test_responses_token_generations_preserve_every_generation() -> None:
    exchange = _response_exchange("multi-generation", 2)
    data = exchange.response.model_dump(mode="python")
    data["output"].append(
        {
            "id": "message-second",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "second",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [1, 2, 3],
            "output_tokens": [{"token_id": 4, "logprob": -0.4}],
            "output_indices": [1],
        },
    ]
    exchange.response = Response.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).responses_history()

    tokenized = history.tokenize()

    assert tokenized.token_ids == [1, 2, 3, 4]
    assert tokenized.logprobs[1] == -0.2
    assert tokenized.logprobs[3] == -0.4
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_responses_prompt_disagreement_splits_unless_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange("retokenized-generation", 101)
    data = exchange.response.model_dump(mode="python")
    data["output"].append(
        {
            "id": "message-second",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "dog",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 101, "logprob": -0.1, "text": "cat"}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [1, 500, 3],
            "output_tokens": [{"token_id": 4, "logprob": -0.4, "text": "dog"}],
            "output_indices": [1],
        },
    ]
    exchange.response = Response.model_validate(data)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
    histories = trajectory.responses_histories()
    assert len(histories) == 2
    assert [history.tokenize().token_ids for history in histories] == [
        [1, 101],
        [1, 500, 3, 4],
    ]
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.responses_history()
    history = trajectory.responses_history(reconcile_text_equivalent_tokenizations=True)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"cat": [500], "dog": [4]}[text]

        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            raise AssertionError("Exact Responses tokenization does not render chat")

    monkeypatch.setattr(
        "art.trajectories._tokenize._WARNED_PREFIX_RETOKENIZATION", False
    )
    with pytest.warns(UserWarning, match="preserved the original sampled token IDs"):
        tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 101, 3, 4]
    assert tokenized.logprobs[1] == -0.1


@pytest.mark.parametrize(
    "token_generations, match",
    [
        ([], "must be omitted"),
        (
            [
                {
                    "output_tokens": [{"token_id": 2}],
                    "output_indices": [0],
                }
            ],
            "prompt_token_ids",
        ),
        (
            [
                {
                    "prompt_token_ids": [1],
                    "output_tokens": [{"token_id": True}],
                    "output_indices": [0],
                }
            ],
            "exact token ID",
        ),
        (
            [
                {
                    "prompt_token_ids": [1],
                    "output_tokens": [{"token_id": 2}],
                    "output_indices": [True],
                }
            ],
            "integers",
        ),
        (
            [
                {
                    "prompt_token_ids": [1],
                    "output_tokens": [{"token_id": 2}],
                    "output_indices": [1],
                }
            ],
            "out of bounds",
        ),
    ],
)
def test_responses_token_generations_fail_closed(
    token_generations: list[dict[str, Any]], match: str
) -> None:
    exchange = _response_exchange("invalid-generation", 2)
    data = exchange.response.model_dump(mode="python")
    data["token_generations"] = token_generations
    exchange.response = Response.model_validate(data)

    with pytest.raises(ValueError, match=match):
        art.Trajectory(
            exchanges=TrajectoryExchanges(responses=[exchange])
        ).responses_history().tokenize()


def test_responses_terminal_generation_without_output_items_is_tokenized() -> None:
    exchange = _response_exchange("terminal-eos", 2)
    data = exchange.response.model_dump(mode="python")
    data["output"] = []
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [],
        }
    ]
    exchange.response = Response.model_validate(data)

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).responses_history()
    tokenized = history.tokenize()

    assert history.input[-1] == {"role": "assistant", "content": ""}
    assert history.input_sources[-1] == ResponsesItemSource(
        exchange=exchange, generation_index=0
    )
    assert tokenized.token_ids == [1, 2]
    assert math.isnan(tokenized.logprobs[0])
    assert tokenized.logprobs[1] == pytest.approx(-0.2)
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]
    assert history.as_chat_completions_history().messages[-1] == {
        "role": "assistant",
        "content": "",
    }


def test_responses_terminal_generation_without_output_items_survives_rerender() -> None:
    exchange = _response_exchange("terminal-eos-rerender", 2)
    data = exchange.response.model_dump(mode="python")
    data["output"] = []
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [],
        }
    ]
    exchange.response = Response.model_validate(data)
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return [1]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [1]

    tokenized = history.tokenize(tokenizer=Tokenizer(), chat_template="custom")

    assert tokenized.token_ids == [1, 2]
    assert math.isnan(tokenized.logprobs[0])
    assert tokenized.logprobs[1] == pytest.approx(-0.2)
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_responses_empty_chat_source_requires_outputless_generation() -> None:
    from art.trajectories._tokenize import _responses_source_generation

    exchange = _response_exchange("nonempty-generation-empty-source", 2)
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )
    assistant_index = next(
        index
        for index, source in enumerate(history.message_sources)
        if source is not None and source.output_indices is not None
    )
    history.messages[assistant_index] = {"role": "assistant", "content": ""}
    invalid_source = ChatCompletionsMessageSource(
        exchange=exchange,
        output_indices=(),
        generation_index=0,
    )
    history.message_sources[assistant_index] = invalid_source

    with pytest.raises(ValueError, match="empty output source"):
        _responses_source_generation(invalid_source)
    with pytest.raises(ValueError, match="empty output source"):
        history.tokenize()


def test_responses_request_composite_source_validation_uses_contiguous_items() -> None:
    from art.trajectories._tokenize import _validate_history_sources

    exchange = _response_exchange("request-composite", 2)
    exchange.request["input"] = [
        {
            "type": "function_call",
            "call_id": f"call-{index}",
            "name": f"tool_{index}",
            "arguments": "{}",
        }
        for index in range(2)
    ]
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    assert len(history.messages[0].get("tool_calls", [])) == 2
    _validate_history_sources(history)


def test_responses_nonterminal_generation_without_output_items_raises() -> None:
    exchange = _response_exchange("hidden-control", 2)
    data = exchange.response.model_dump(mode="python")
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [],
        },
        {
            "prompt_token_ids": [1, 2],
            "output_tokens": [{"token_id": 3, "logprob": -0.3}],
            "output_indices": [0],
        },
    ]
    exchange.response = Response.model_validate(data)

    with pytest.raises(ValueError, match="nonterminal"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(responses=[exchange])
        ).responses_history()


def test_tokenization_rejects_mutated_mixed_representation() -> None:
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "user", "content": "hi"}]
    )
    trajectory.exchanges.chat_completions.append(_chat_exchange([1], [2]))

    with pytest.raises(ValueError, match="both exchanges and legacy histories"):
        trajectory.tokenize()


def test_responses_previous_response_id_resolves_local_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10] if len(messages) == 1 else [10, 20, 11]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    first = _response_exchange("resp-1", 20, prompt_token_ids=[10])
    second = _response_exchange(
        "resp-2",
        30,
        previous_response_id="resp-1",
        offset=1,
        prompt_token_ids=[10, 20, 11],
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first, second])
    )

    assert trajectory.tokenize(base_model="base/model").token_ids == [
        10,
        20,
        11,
        30,
    ]

    second.request["previous_response_id"] = "missing"
    assert len(trajectory.responses_histories()) == 2
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize(base_model="base/model")
    assert trajectory.responses_histories()[1].tokenize(
        base_model="base/model"
    ).token_ids == [10, 20, 11, 30]


def test_chat_prefix_retokenization_splits_unless_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _chat_exchange([1], [101, 102])
    first.response.choices[0].message.content = "cat"
    second = _chat_exchange([1, 500, 3], [4], offset=1)
    second.request["messages"] = [
        {"role": "user", "content": "turn 0"},
        {"role": "assistant", "content": "cat"},
        {"role": "user", "content": "turn 1"},
    ]

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {"cat": [500], "answer": [4], "turn 0": [1], "turn 1": [3]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            rendered: list[int] = []
            for message in messages:
                rendered.extend(self(str(message["content"])))
            return rendered

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    monkeypatch.setattr(
        "art.trajectories._tokenize._WARNED_PREFIX_RETOKENIZATION", False
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )
    histories = trajectory.chat_completions_histories()
    assert len(histories) == 2
    assert [history.tokenize().token_ids for history in histories] == [
        [1, 101, 102],
        [1, 500, 3, 4],
    ]
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.history()
    history = trajectory.chat_completions_history(
        reconcile_text_equivalent_tokenizations=True
    )

    with pytest.warns(UserWarning, match="preserved the original sampled token IDs"):
        tokenized = history.tokenize(base_model="base/model")

    assert tokenized.token_ids == [1, 101, 102, 3, 4]
    assert tokenized.logprobs[1:3] == [-10.1, -10.2]
    assert all(tokenized.flags[index] & art.TokenFlag.EXACT for index in (1, 2, 4))


def test_template_change_rerenders_scaffold_but_preserves_sampled_output() -> None:
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])])
    ).chat_completions_history()
    history.chat_template = "custom"

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {"turn 0": [10], "answer": [20]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: object,
            tokenize: bool,
            add_generation_prompt: bool,
            chat_template: str | None = None,
            **kwargs: object,
        ) -> list[int]:
            del tools, tokenize, add_generation_prompt, kwargs
            assert chat_template == "custom"
            if len(messages) == 1:
                return [10]
            if str(messages[-1]["content"]).startswith("ART_TRAJECTORY_"):
                return [10, 999, 30]
            return [10, 20, 30]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [10, 2, 30]
    assert tokenized.logprobs[1] == -0.2
    assert tokenized.flags[1] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_template_change_preserves_complete_exact_sampled_suffix() -> None:
    exchange = _chat_exchange([1], [2, 3])
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "custom"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [1], "answer": [2]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] != "assistant":
                return [1]
            if str(messages[-1]["content"]).startswith("ART_TRAJECTORY_"):
                return [1, 999, 9]
            return [1, 2, 9]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2, 3, 9]
    assert tokenized.logprobs[1:3] == pytest.approx([-0.2, -0.3])
    assert math.isnan(tokenized.logprobs[3])
    assert tokenized.flags[1:] == [
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
    ]


def test_responses_generation_evidence_is_atomic_and_partial_edits_do_not_replay() -> (
    None
):
    exchange = _response_exchange("reasoning-and-answer", 0)
    data = exchange.response.model_dump(mode="python")
    data["output"] = [
        {
            "id": "reasoning",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "think"}],
        },
        {
            "id": "message",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "answer",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
    ]
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [
                {"token_id": 2, "logprob": -0.2, "text": "think"},
                {"token_id": 3, "logprob": -0.3, "text": "answer"},
            ],
            "output_indices": [0, 1],
        }
    ]
    exchange.response = Response.model_validate(data)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                answer_start = text.index("answer")
                answer_end = answer_start + len("answer")
                offsets: list[tuple[int, int]]
                token_ids = [1]
                if "think" in text:
                    think_start = text.index("think")
                    think_end = think_start + len("think")
                    offsets = [
                        (0, think_start),
                        (think_start, think_end),
                        (answer_start, answer_end),
                        (answer_end, len(text)),
                    ]
                    token_ids.extend([20, 30, 9])
                else:
                    offsets = [
                        (0, answer_start),
                        (answer_start, answer_end),
                        (answer_end, len(text)),
                    ]
                    token_ids.extend([30, 9])
                return {"input_ids": token_ids, "offset_mapping": offsets}
            return {"turn 0": [1], "think": [20], "answer": [30]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> object:
            tokenize = kwargs.pop("tokenize")
            del kwargs
            if messages[-1]["role"] != "assistant":
                return [1]
            assistant = messages[-1]
            rendered = (
                f"<user>{messages[0]['content']}</user><assistant>"
                + (
                    f"<reasoning>{assistant['reasoning']}</reasoning>"
                    if assistant.get("reasoning")
                    else ""
                )
                + f"<content>{assistant['content']}</content></assistant>"
            )
            if not tokenize:
                return rendered
            if assistant.get("reasoning"):
                return [1, 20, 30, 9]
            return [1, 30, 9]

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).responses_history()
    exact = history.tokenize(tokenizer=Tokenizer(), chat_template="custom")
    assert exact.token_ids == [1, 2, 3, 9]
    assert exact.logprobs[1:3] == pytest.approx([-0.2, -0.3])
    assert math.isnan(exact.logprobs[3])

    del history.input[1]
    del history.input_sources[1]
    partial = history.tokenize(tokenizer=Tokenizer(), chat_template="custom")
    assert partial.token_ids == [1, 30, 9]
    assert 2 not in partial.token_ids
    assert partial.flags[1] == art.TokenFlag.SAMPLED
    assert math.isnan(partial.logprobs[1])


def test_responses_generation_source_rejects_content_from_another_generation() -> None:
    exchange = _response_exchange("generation-provenance", 2)
    data = exchange.response.model_dump(mode="python")
    data["output"].append(
        {
            "id": "message-second-generation",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "second",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [1, 2, 3],
            "output_tokens": [{"token_id": 4, "logprob": -0.4}],
            "output_indices": [1],
        },
    ]
    exchange.response = Response.model_validate(data)
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )
    first_generation = next(
        index
        for index, source in enumerate(history.message_sources)
        if source is not None
        and source.generation_index == 0
        and history.messages[index].get("role") == "assistant"
    )
    history.messages[first_generation] = {
        "role": "assistant",
        "content": "second",
    }

    with pytest.raises(ValueError, match="no longer matches its source exchange"):
        history.tokenize()


def test_responses_chat_rerender_preserves_equal_length_generation_evidence() -> None:
    exchange = _response_exchange("equal-length-generations", 2)
    data = exchange.response.model_dump(mode="python")
    data["output"].append(
        {
            "id": "message-second-generation",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "second",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [1, 2, 3],
            "output_tokens": [{"token_id": 4, "logprob": -0.4}],
            "output_indices": [1],
        },
    ]
    exchange.response = Response.model_validate(data)
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            mapping = {"turn 0": 10, "answer": 20, "second": 40}
            if not kwargs.get("return_offsets_mapping"):
                return [mapping[text]]
            token_ids: list[int] = []
            offsets: list[tuple[int, int]] = []
            for match in re.finditer(r"<m>(.*?)</m>|<s>", text):
                if match.group(1) is None:
                    token_ids.append(30)
                    offsets.append(match.span())
                else:
                    token_ids.append(mapping[match.group(1)])
                    offsets.append(match.span(1))
            return {"input_ids": token_ids, "offset_mapping": offsets}

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
            **kwargs: object,
        ) -> object:
            del kwargs
            result = ""
            for index, message in enumerate(messages):
                result += f"<m>{message['content']}</m>"
                if index == 1 and (len(messages) > 2 or add_generation_prompt):
                    result += "<s>"
            return self(result, return_offsets_mapping=True) if tokenize else result

    tokenized = history.tokenize(tokenizer=Tokenizer(), chat_template="custom")

    assert tokenized.token_ids == [10, 2, 30, 4]
    assert 20 not in tokenized.token_ids
    assert 40 not in tokenized.token_ids
    assert tokenized.logprobs[1::2] == pytest.approx([-0.2, -0.4])
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_responses_chat_sampled_source_requires_generation_identity() -> None:
    exchange = _response_exchange("missing-generation-identity", 2)
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )
    assistant_index = next(
        index
        for index, source in enumerate(history.message_sources)
        if source is not None and source.output_indices is not None
    )
    source = history.message_sources[assistant_index]
    assert source is not None
    history.message_sources[assistant_index] = ChatCompletionsMessageSource(
        exchange=source.exchange,
        output_indices=source.output_indices,
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [1], "answer": [2]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [
                token for message in messages for token in self(str(message["content"]))
            ]

    with pytest.raises(ValueError, match="no generation identity"):
        history.tokenize(tokenizer=Tokenizer())


def test_responses_chat_output_indices_reuse_one_complete_generation() -> None:
    exchange = _response_exchange(
        "response-output-indices",
        2,
        prompt_token_ids=[1],
    )
    history = art.ChatCompletionsHistory(
        model="test/model",
        messages=[
            {"role": "user", "content": "turn 0"},
            {"role": "assistant", "content": "answer"},
        ],
        message_sources=[
            ChatCompletionsMessageSource(exchange=exchange, request_index=0),
            ChatCompletionsMessageSource(
                exchange=exchange,
                output_indices=(0,),
                generation_index=0,
            ),
        ],
    )

    tokenized = history.tokenize()

    assert tokenized.token_ids == [1, 2]
    assert tokenized.logprobs[-1] == pytest.approx(-0.1)
    assert tokenized.flags[-1] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_responses_chat_output_indices_are_bounds_checked() -> None:
    exchange = _response_exchange("response-output-indices-invalid", 2)
    history = art.ChatCompletionsHistory(
        model="test/model",
        messages=[{"role": "assistant", "content": "answer"}],
        message_sources=[
            ChatCompletionsMessageSource(
                exchange=exchange,
                output_indices=(1,),
                generation_index=0,
            )
        ],
    )

    with pytest.raises(ValueError, match="out of bounds"):
        history.tokenize()


def _multi_output_responses_chat_history() -> art.ChatCompletionsHistory:
    exchange = _response_exchange("multi-output-generation", 2)
    data = exchange.response.model_dump(mode="python")
    data["output"][0]["content"][0]["text"] = "first"
    data["output"].append(
        {
            "id": "message-second-output",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "second",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [
                {"token_id": 2, "logprob": -0.2},
                {"token_id": 3, "logprob": -0.3},
            ],
            "output_indices": [0, 1],
        }
    ]
    exchange.response = Response.model_validate(data)
    return (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )


def test_responses_output_source_rejects_sibling_generation_output() -> None:
    history = _multi_output_responses_chat_history()
    first_output = next(
        index
        for index, source in enumerate(history.message_sources)
        if source is not None and source.output_indices == (0,)
    )
    history.messages[first_output] = {
        "role": "assistant",
        "content": "second",
    }

    with pytest.raises(ValueError, match="no longer matches its source exchange"):
        history.tokenize()


def test_responses_multi_output_generation_is_rendered_without_duplicate_evidence() -> (
    None
):
    history = _multi_output_responses_chat_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [1], "first": [20], "second": [30]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            result: list[int] = []
            for message in messages:
                result.extend(self(str(message["content"])))
            return result

    tokenized = history.tokenize(tokenizer=Tokenizer(), chat_template="custom")

    assert tokenized.token_ids == [1, 20, 30]
    assert all(math.isnan(value) for value in tokenized.logprobs)
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag.SAMPLED,
    ]


def test_responses_multi_output_chat_conversion_preserves_item_logprobs() -> None:
    projected = _multi_output_responses_chat_history()
    exchange = next(
        source.exchange
        for source in projected.message_sources
        if source is not None and source.output_indices == (0,)
    )
    assert isinstance(exchange, ResponsesExchange)
    data = exchange.response.model_dump(mode="python")
    data.pop("token_generations")
    for output, (text, logprob) in zip(
        data["output"], (("first", -0.1), ("second", -0.2)), strict=True
    ):
        output["content"][0]["logprobs"] = [
            {
                "token": text,
                "logprob": logprob,
                "bytes": list(text.encode()),
                "top_logprobs": [],
            }
        ]
    exchange.response = Response.model_validate(data)
    history = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [1], "first": [2], "second": [3]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [
                token for message in messages for token in self(str(message["content"]))
            ]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2, 3]
    assert math.isnan(tokenized.logprobs[0])
    assert tokenized.logprobs[1:] == [-0.1, -0.2]
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag.SAMPLED,
    ]


def test_mutable_chat_history_is_authoritative_and_does_not_replay_removed_turns() -> (
    None
):
    first = _chat_exchange([10], [20])
    first.request["messages"] = [{"role": "user", "content": "first"}]
    second = _chat_exchange([10, 20, 30], [40], offset=1)
    second.request["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).chat_completions_history()
    del history.messages[:2]
    del history.message_sources[:2]

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"second": [31], "answer": [41]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            contents = [message["content"] for message in messages]
            if contents == ["second", "answer"]:
                return [30, 41, 50]
            if len(contents) == 2 and str(contents[-1]).startswith("ART_TRAJECTORY_"):
                return [30, 99, 50]
            assert contents == ["second"]
            return [30] if add_generation_prompt else [30]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [30, 40, 50]
    assert 20 not in tokenized.token_ids
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
    ]


def test_request_assistant_messages_are_not_marked_sampled() -> None:
    exchange = _chat_exchange([10], [40])
    exchange.request["messages"] = [
        {"role": "assistant", "content": "seed"},
        {"role": "user", "content": "question"},
    ]
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"seed": [20], "question": [30], "answer": [41]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant" and len(messages) == 3:
                if str(messages[-1]["content"]).startswith("ART_TRAJECTORY_"):
                    return [5, 20, 6, 30, 7, 99, 8]
                return [5, 20, 6, 30, 7, 41, 8]
            assert add_generation_prompt
            return [5, 20, 6, 30, 7]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [5, 20, 6, 30, 7, 40, 8]
    assert not tokenized.flags[1] & art.TokenFlag.SAMPLED
    assert tokenized.flags[5] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_rerender_constrains_exact_output_to_its_message_region() -> None:
    exchange = _chat_exchange([7], [7])
    exchange.request["messages"] = [{"role": "user", "content": "same"}]
    exchange.response.choices[0].message.content = "same"
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            assert text == "same"
            return [7]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant":
                return [7, 99, 7]
            assert add_generation_prompt
            return [7, 99]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag(0),
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]
    assert math.isnan(tokenized.logprobs[0])
    assert tokenized.logprobs[2] == -0.7


def test_rerender_does_not_bind_sampled_ids_to_token_equivalent_user_text() -> None:
    exchange = _chat_exchange([7], [7])
    exchange.request["messages"] = [{"role": "user", "content": "cat"}]
    exchange.response.choices[0].message.content = "dog"
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return [7]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [7, 99, 7]

    with pytest.raises(ValueError, match="uniquely locate"):
        history.tokenize(tokenizer=Tokenizer())


def test_rerender_does_not_bind_unique_exact_id_outside_sampled_message() -> None:
    exchange = _chat_exchange([42], [7])
    exchange.request["messages"] = [{"role": "user", "content": "cat"}]
    exchange.response.choices[0].message.content = "dog"
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"cat": [7], "dog": [500]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant":
                return [42, 7, 99, 500]
            return [42, 7, 99] if add_generation_prompt else [42, 7]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [42, 7, 99, 7]
    assert not tokenized.flags[1] & art.TokenFlag.SAMPLED
    assert tokenized.flags[-1] == (art.TokenFlag.EXACT | art.TokenFlag.SAMPLED)
    assert math.isnan(tokenized.logprobs[1])
    assert tokenized.logprobs[-1] == -0.7


def test_rerender_rejects_sampled_text_ambiguous_with_trailing_scaffold() -> None:
    exchange = _chat_exchange([42], [7])
    exchange.request["messages"] = [{"role": "user", "content": "cat"}]
    exchange.response.choices[0].message.content = "dog"
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"cat": [1], "dog": [500]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [100, 500, 99, 500]

    with pytest.raises(ValueError, match="uniquely locate"):
        history.tokenize(tokenizer=Tokenizer())


def test_rerender_does_not_duplicate_sampled_trailing_eos() -> None:
    exchange = _chat_exchange([1], [7, 2])
    exchange.request["messages"] = [{"role": "user", "content": "question"}]
    exchange.response.choices[0].message.content = "answer"
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"question": [1], "answer": [7]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant":
                return [1, 99, 7, 2]
            assert add_generation_prompt
            return [1, 99]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 99, 7, 2]
    assert tokenized.token_ids.count(2) == 1
    assert tokenized.flags[-2:] == [
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]
    assert tokenized.logprobs[-2:] == [-0.7, -0.2]


def test_chat_view_preserves_initial_prompt_and_ignores_later_disagreement() -> None:
    first = _chat_exchange([1], [2])
    second = _chat_exchange([9, 8, 7], [3], offset=1)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [100], "answer": [2], "turn 1": [7]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [
                token for message in messages for token in self(str(message["content"]))
            ]

    history = trajectory.chat_completions_history(
        reconcile_text_equivalent_tokenizations=True
    )
    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2, 7, 3]


def test_reasoning_stripped_chat_histories_tokenize_authoritative_views() -> None:
    first = _chat_exchange([1], [2, 101, 102, 9])
    first.request["messages"] = [{"role": "user", "content": "one"}]
    first_data = first.response.model_dump(mode="python")
    first_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": "first",
        "reasoning": "thought-one",
    }
    first.response = ChatCompletion.model_validate(first_data)

    second = _chat_exchange([1, 101, 102, 9, 4], [5, 6], offset=1)
    second.request["messages"] = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
    ]
    second_data = second.response.model_dump(mode="python")
    second_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": "second",
        "reasoning": "thought-two",
    }
    second.response = ChatCompletion.model_validate(second_data)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )

    class Tokenizer:
        name_or_path = "test/model"

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {
                "one": [1],
                "first": [500],
                "two": [4],
                "thought-two": [5],
                "second": [6],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            assert [message.get("content") for message in messages] == [
                "one",
                "first",
                "two",
                "second",
            ]
            return [1, 500, 9, 4, 5, 6]

    tokenized = trajectory.tokenize(multi_history=True, tokenizer=Tokenizer())

    assert [history.token_ids for history in tokenized.histories] == [
        [1, 2, 101, 102, 9],
        [1, 101, 102, 9, 4, 5, 6],
    ]
    assert tokenized.histories[1].flags[1] & art.TokenFlag.SAMPLED
    assert tokenized.histories[1].flags[1] & art.TokenFlag.EXACT
    assert tokenized.histories[1].logprobs[1:3] == [-10.1, -10.2]
    assert tokenized.histories[1].flags[3] == (
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED
    )
    assert tokenized.histories[1].logprobs[3] == -0.9
    assert 2 not in tokenized.histories[1].token_ids
    assert 500 not in tokenized.histories[1].token_ids


def test_reasoning_stripped_histories_remain_trainable_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _chat_exchange([1], [2, 101, 102, 9])
    first.request["messages"] = [{"role": "user", "content": "one"}]
    first_data = first.response.model_dump(mode="python")
    first_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": "first",
        "reasoning": "thought-one",
    }
    first.response = ChatCompletion.model_validate(first_data)
    second = _chat_exchange([1, 101, 102, 9, 4], [5, 6], offset=1)
    second.request["messages"] = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
    ]
    second_data = second.response.model_dump(mode="python")
    second_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": "second",
        "reasoning": "thought-two",
    }
    second.response = ChatCompletion.model_validate(second_data)

    class Tokenizer:
        name_or_path = "test/model"

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {
                "one": [1],
                "first": [500],
                "two": [4],
                "thought-two": [5],
                "second": [6],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            content = [message.get("content") for message in messages]
            return (
                [1, 2, 101, 102, 9]
                if content == ["one", "first"]
                else [1, 500, 9, 4, 5, 6]
            )

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _: Tokenizer()
    )
    trajectories = [
        art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[first, second]),
            reward=reward,
        )
        for reward in (1.0, 0.0)
    ]
    group = art.TrajectoryGroup(trajectories=trajectories)

    from art.preprocessing.tokenize import tokenize_trajectory_groups
    from art.tinker_native.data import trajectory_groups_to_datums

    preprocessing = list(
        tokenize_trajectory_groups(
            Tokenizer(),  # type: ignore[arg-type, ty:invalid-argument-type]
            [group],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            shuffle_group_trajectories=False,
            drop_zero_advantage_trajectories=False,
        )
    )
    datums = trajectory_groups_to_datums(
        [group],
        renderer=None,
        tokenizer=None,
        normalize_advantages=False,
    )

    assert len(preprocessing) == 4
    assert len(datums) == 4
    assert all(
        not math.isnan(logprob)
        for result in preprocessing
        for logprob, sampled in zip(result.logprobs, result.assistant_mask, strict=True)
        if sampled
    )


def test_reasoning_stripped_tool_call_keeps_exact_evidence_for_strict_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _chat_exchange([1], [2, 7, 8])
    first.request["messages"] = [{"role": "user", "content": "one"}]
    first_data = first.response.model_dump(mode="python")
    first_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "reasoning": "thought",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    first.response = ChatCompletion.model_validate(first_data)
    second = _chat_exchange([1, 7, 8, 4], [5], offset=1)
    second.request["messages"] = [
        {"role": "user", "content": "one"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": first_data["choices"][0]["message"]["tool_calls"],
        },
        {"role": "user", "content": "two"},
    ]

    class Tokenizer:
        name_or_path = "test/model"

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"lookup": [7], "{}": [8]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            assert len(messages) == 4
            return [1, 7, 8, 4, 5]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _: Tokenizer()
    )
    trajectories = [
        art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[first, second]),
            reward=reward,
        )
        for reward in (1.0, 0.0)
    ]
    group = art.TrajectoryGroup(trajectories=trajectories)

    from art.preprocessing.tokenize import tokenize_trajectory_groups
    from art.tinker_native.data import trajectory_groups_to_datums

    tokenized = trajectories[0].tokenize(
        multi_history=True,
        tokenizer=Tokenizer(),
    )
    second_history = tokenized.histories[1]
    assert second_history.token_ids == [1, 7, 8, 4, 5]
    assert second_history.logprobs[1:3] == [-0.7, -0.8]
    assert second_history.flags[1:3] == [
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]

    preprocessing = list(
        tokenize_trajectory_groups(
            Tokenizer(),  # type: ignore[arg-type, ty:invalid-argument-type]
            [group],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            shuffle_group_trajectories=False,
            drop_zero_advantage_trajectories=False,
        )
    )
    datums = trajectory_groups_to_datums(
        [group],
        renderer=None,
        tokenizer=None,
        normalize_advantages=False,
    )

    assert len(preprocessing) == 4
    assert len(datums) == 4


def test_responses_prompt_repair_opt_in_uses_native_text_and_source_position() -> None:
    exchange = _response_exchange("repeated-retokenization", 101)
    data = exchange.response.model_dump(mode="python")
    data["output"].append(
        {
            "id": "message-second",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "dog",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [500],
            "output_tokens": [{"token_id": 101, "logprob": -0.1}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [500, 500, 3],
            "output_tokens": [{"token_id": 4, "logprob": -0.4}],
            "output_indices": [1],
        },
    ]
    exchange.response = Response.model_validate(data)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
    assert len(trajectory.responses_histories()) == 2
    history = trajectory.responses_history(reconcile_text_equivalent_tokenizations=True)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"answer": [500], "dog": [4]}[text]

        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            raise AssertionError("Exact Responses tokenization must not render chat")

    with pytest.warns(UserWarning, match="preserved the original sampled token IDs"):
        tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [500, 101, 3, 4]
    assert tokenized.logprobs[1] == -0.1


def test_rerender_marks_tool_call_only_generated_region_sampled() -> None:
    exchange = _chat_exchange([1], [2])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("token_ids")
    choice["logprobs"] = None
    choice["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"x":1}'},
            }
        ],
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                name_start = text.index("lookup")
                name_end = name_start + len("lookup")
                args_start = text.index('{"x":1}')
                args_end = args_start + len('{"x":1}')
                midpoint = args_start + 3
                return {
                    "input_ids": [1, 10, 20, 25, 30, 31, 26],
                    "offset_mapping": [
                        (0, text.index("<assistant>")),
                        (text.index("<assistant>"), name_start),
                        (name_start, name_end),
                        (name_end, args_start),
                        (args_start, midpoint),
                        (midpoint, args_end),
                        (args_end, len(text)),
                    ],
                }
            return {
                "turn 0": [1],
                "lookup": [20],
                '{"x":1}': [30, 31],
            }[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> object:
            tokenize = kwargs.pop("tokenize")
            del kwargs
            if messages[-1]["role"] == "assistant":
                function = messages[-1]["tool_calls"][0]["function"]
                rendered = (
                    f"<user>{messages[0]['content']}</user>"
                    f"<assistant><name>{function['name']}</name>"
                    f"<args>{function['arguments']}</args></assistant>"
                )
                if not tokenize:
                    return rendered
                return [1, 10, 20, 25, 30, 31, 26]
            assert add_generation_prompt
            return [1, 10]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 10, 20, 25, 30, 31, 26]
    assert tokenized.flags[2:6] == [art.TokenFlag.SAMPLED] * 4
    assert all(math.isnan(value) for value in tokenized.logprobs[2:6])
    assert not tokenized.flags[0] & art.TokenFlag.SAMPLED


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("lookup", "{}"),
        ("art_trajectory_probe_0", '{"art_trajectory_probe":true}'),
    ],
)
def test_tool_call_probe_handles_contextual_tokenization(
    name: str, arguments: str
) -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["logprobs"] = None
    choice["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {
                "turn 0": [1],
                name: [90],
                arguments: [91],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] != "assistant":
                return [1, 10]
            function = messages[-1]["tool_calls"][0]["function"]
            if function["name"] != name:
                return [1, 10, 98, 25, 99, 26]
            return [1, 10, 20, 25, 30, 26]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 10, 20, 25, 30, 26]
    assert tokenized.flags[2:5] == [art.TokenFlag.SAMPLED] * 3
    assert all(math.isnan(value) for value in tokenized.logprobs[2:5])


def test_rerender_proves_each_reasoning_and_content_part_separately() -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["message"] = {
        "role": "assistant",
        "reasoning": "think",
        "content": "answer",
    }
    choice["logprobs"] = {
        "content": [
            {
                "token": "answer",
                "logprob": -0.7,
                "bytes": list(b"answer"),
                "top_logprobs": [],
            }
        ],
        "refusal": None,
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                think_start = text.index("think")
                think_end = think_start + len("think")
                answer_start = text.index("answer")
                answer_end = answer_start + len("answer")
                return {
                    "input_ids": [1, 10, 11, 12, 13, 14],
                    "offset_mapping": [
                        (0, text.index("<assistant>")),
                        (text.index("<assistant>"), think_start),
                        (think_start, think_end),
                        (think_end, answer_start),
                        (answer_start, answer_end),
                        (answer_end, len(text)),
                    ],
                }
            return {"turn 0": [1], "think": [11], "answer": [13]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> object:
            tokenize = kwargs.pop("tokenize")
            del kwargs
            assistant = messages[-1]
            reasoning = assistant.get("reasoning") or assistant.get("reasoning_content")
            rendered = (
                f"<user>{messages[0]['content']}</user><assistant>"
                + (f"<reasoning>{reasoning}</reasoning>" if reasoning else "")
                + f"<content>{assistant['content']}</content></assistant>"
            )
            if not tokenize:
                return rendered
            if not reasoning:
                return [1, 10, 13, 14]
            return [1, 10, 11, 12, 13, 14]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag(0),
    ]
    assert tokenized.logprobs[4] == -0.7


def test_rerender_rejects_unproved_multi_part_boundaries() -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    data["choices"][0].pop("prompt_token_ids")
    data["choices"][0].pop("token_ids")
    data["choices"][0]["logprobs"] = None
    data["choices"][0]["message"] = {
        "role": "assistant",
        "reasoning": "think",
        "content": "answer",
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    history.chat_template = "rerender"

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [1], "think": [11], "answer": [12]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [1, 10, 11, 12, 13, 14]

    with pytest.raises(ValueError, match="sampled content boundary"):
        history.tokenize(tokenizer=Tokenizer())


def test_empty_sampled_messages_need_no_content_boundary() -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    data["choices"][0].pop("prompt_token_ids")
    data["choices"][0].pop("token_ids")
    data["choices"][0]["logprobs"] = None
    data["choices"][0]["message"]["content"] = ""
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return [1]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [1, 2]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2]
    assert tokenized.flags == [art.TokenFlag(0), art.TokenFlag(0)]


def test_empty_sampled_message_inserts_exact_control_token() -> None:
    exchange = _chat_exchange([], [2])
    exchange.response.choices[0].message.content = ""
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del text, kwargs
            return []

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            if messages[-1]["role"] == "assistant":
                return [1, 99, 9]
            assert add_generation_prompt
            return [1, 99]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 99, 2, 9]
    assert tokenized.logprobs[2] == -0.2
    assert tokenized.flags[2] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_renderer_ignored_refusal_is_appended_for_tokenization() -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["message"] = {
        "role": "assistant",
        "content": "answer",
        "refusal": "declined",
    }
    choice["logprobs"] = {
        "content": [
            {
                "token": "answer",
                "logprob": -0.4,
                "bytes": list(b"answer"),
                "top_logprobs": [],
            }
        ],
        "refusal": [
            {
                "token": "declined",
                "logprob": -0.5,
                "bytes": list(b"declined"),
                "top_logprobs": [],
            }
        ],
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                answer_start = text.index("answer")
                declined_start = text.index("declined")
                declined_end = declined_start + len("declined")
                return {
                    "input_ids": [1, 4, 5],
                    "offset_mapping": [
                        (0, answer_start),
                        (answer_start, declined_start),
                        (declined_start, declined_end),
                    ],
                }
            return {
                "turn 0": [1],
                "answer": [4],
                "declined": [5],
                "answerdeclined": [4, 5],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> object:
            tokenize = kwargs.pop("tokenize")
            del kwargs
            if not tokenize:
                return "".join(
                    f"<message>{message.get('content') or ''}</message>"
                    for message in messages
                )
            result: list[int] = []
            for message in messages:
                encoded = self(str(message.get("content") or ""))
                assert isinstance(encoded, list)
                for token in encoded:
                    assert isinstance(token, int)
                    result.append(token)
            return result

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 4, 5]
    assert tokenized.logprobs[1:] == [-0.4, -0.5]
    assert tokenized.flags[1:] == [art.TokenFlag.SAMPLED] * 2


def test_renderer_reasoning_content_alias_preserves_reasoning() -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["logprobs"] = None
    choice["message"] = {
        "role": "assistant",
        "reasoning": "think",
        "content": "answer",
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {"turn 0": [1], "think": [2], "answer": [3]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            result: list[int] = []
            for message in messages:
                if reasoning := message.get("reasoning_content"):
                    result.extend(self(str(reasoning)))
                if content := message.get("content"):
                    result.extend(self(str(content)))
            return result

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 2, 3]
    assert tokenized.flags == [
        art.TokenFlag(0),
        art.TokenFlag.SAMPLED,
        art.TokenFlag.SAMPLED,
    ]


def test_trimmed_render_preserves_authoritative_textual_logprob_tokens() -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["message"]["content"] = " helloworld "
    choice["logprobs"] = {
        "content": [
            {
                "token": " hello",
                "logprob": -0.4,
                "bytes": list(b" hello"),
                "top_logprobs": [],
            },
            {
                "token": "world",
                "logprob": -0.45,
                "bytes": list(b"world"),
                "top_logprobs": [],
            },
            {
                "token": " ",
                "logprob": -0.5,
                "bytes": [32],
                "top_logprobs": [],
            },
        ],
        "refusal": None,
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                if text == " helloworld ":
                    return {
                        "input_ids": [1118, 2222, 220],
                        "offset_mapping": [(0, 6), (6, 11), (11, 12)],
                    }
                content_start = text.index("helloworld")
                content_end = content_start + len("helloworld")
                return {
                    "input_ids": [1, 3765],
                    "offset_mapping": [
                        (0, content_start),
                        (content_start, content_end),
                    ],
                }
            return {
                "turn 0": [1],
                " helloworld ": [1118, 2222, 220],
                " hello": [1118],
                "world": [3333],
                " ": [220],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> object:
            tokenize = kwargs.pop("tokenize")
            del kwargs
            rendered = (
                f"<user>{messages[0]['content']}</user>"
                f"<assistant>{str(messages[-1]['content']).strip()}</assistant>"
            )
            if not tokenize:
                return rendered
            return [1, 3765]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [1, 1118, 2222, 220]
    assert tokenized.logprobs[1:] == [-0.4, -0.45, -0.5]
    assert tokenized.flags[1:] == [art.TokenFlag.SAMPLED] * 3


def test_textual_logprobs_reconstruct_split_utf8_bytes() -> None:
    from art.trajectories._tokenize import _visible_token_evidence

    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["message"]["content"] = "😊"
    choice["logprobs"]["content"] = [
        {
            "token": "�",
            "logprob": -0.4,
            "bytes": [240, 159, 152],
            "top_logprobs": [],
        },
        {
            "token": "�",
            "logprob": -0.5,
            "bytes": [138],
            "top_logprobs": [],
        },
    ]
    exchange.response = ChatCompletion.model_validate(data)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            assert text == "😊"
            return [11, 12]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: object,
            tokenize: bool,
            add_generation_prompt: bool,
            chat_template: str | None = None,
            **kwargs: object,
        ) -> object:
            del messages, tools, tokenize, add_generation_prompt, chat_template, kwargs
            raise AssertionError("template rendering is not expected")

    assert _visible_token_evidence(Tokenizer(), exchange, sampled_text="😊") == (
        [11, 12],
        [-0.4, -0.5],
    )


def test_textual_logprobs_reject_shifted_contextual_token_boundaries() -> None:
    from art.trajectories._tokenize import _visible_token_evidence

    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["message"]["content"] = " penalates"
    choice["logprobs"]["content"] = [
        {
            "token": " pena",
            "logprob": -0.4,
            "bytes": list(b" pena"),
            "top_logprobs": [],
        },
        {
            "token": "lates",
            "logprob": -0.5,
            "bytes": list(b"lates"),
            "top_logprobs": [],
        },
    ]
    exchange.response = ChatCompletion.model_validate(data)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                return {
                    "input_ids": [30, 31],
                    "offset_mapping": [(0, 6), (6, 10)],
                }
            return {" pena": [11], "lates": [12]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: object,
            tokenize: bool,
            add_generation_prompt: bool,
            chat_template: str | None = None,
            **kwargs: object,
        ) -> object:
            del messages, tools, tokenize, add_generation_prompt, chat_template, kwargs
            raise AssertionError("template rendering is not expected")

    assert _visible_token_evidence(
        Tokenizer(), exchange, sampled_text=" penalates"
    ) == ([11, 12], [-0.4, -0.5])


@pytest.mark.parametrize(
    ("content", "token_id", "trim", "adjacent_scaffold", "reject_empty"),
    [
        (" ", 220, True, False, False),
        ("\n", 198, True, False, False),
        (" ", 220, False, False, False),
        (" ", 220, False, True, False),
        (" ", 220, False, False, True),
    ],
)
def test_trimmed_whitespace_output_inserts_authoritative_logprob_token(
    content: str,
    token_id: int,
    trim: bool,
    adjacent_scaffold: bool,
    reject_empty: bool,
) -> None:
    exchange = _chat_exchange([], [])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["message"]["content"] = content
    choice["logprobs"] = {
        "content": [
            {
                "token": content,
                "logprob": -0.5,
                "bytes": list(content.encode()),
                "top_logprobs": [],
            }
        ],
        "refusal": None,
    }
    exchange.response = ChatCompletion.model_validate(data)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            if kwargs.get("return_offsets_mapping"):
                start = text.index("<assistant>") + len("<assistant>")
                end = text.index("</assistant>")
                if start != end:
                    scaffold_start = end - int(adjacent_scaffold)
                    return {
                        "input_ids": [
                            1,
                            token_id,
                            *([token_id] if adjacent_scaffold else []),
                            9,
                        ],
                        "offset_mapping": [
                            (0, start),
                            (start, scaffold_start),
                            *([(scaffold_start, end)] if adjacent_scaffold else []),
                            (end, len(text)),
                        ],
                    }
                return {
                    "input_ids": [1, 9],
                    "offset_mapping": [(0, start), (start, len(text))],
                }
            return {"turn 0": [1], content: [token_id]}[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> object:
            tokenize = kwargs.pop("tokenize")
            del kwargs
            content_value = str(messages[-1]["content"])
            if trim:
                content_value = content_value.strip()
            scaffold = " " if adjacent_scaffold else ""
            rendered = (
                f"<user>{messages[0]['content']}</user>"
                f"<assistant>{content_value}{scaffold}</assistant>"
            )
            if not tokenize:
                return rendered
            if reject_empty and not content_value:
                raise RuntimeError("template rejects empty assistant content")
            rendered_id = 777 if "ART_TRAJECTORY" in content_value else token_id
            return [
                1,
                *([rendered_id] if content_value else []),
                *([token_id] if adjacent_scaffold else []),
                9,
            ]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [
        1,
        token_id,
        *([token_id] if adjacent_scaffold else []),
        9,
    ]
    assert tokenized.logprobs[1] == -0.5
    assert tokenized.flags[1] == art.TokenFlag.SAMPLED
    if adjacent_scaffold:
        assert tokenized.flags[2] == art.TokenFlag(0)


def _repeated_text_rerender_history(turn_count: int) -> art.ChatCompletionsHistory:
    exchanges: list[ChatCompletionsExchange] = []
    prompt: list[int] = []
    messages: list[ChatCompletionMessageParam] = []
    for index in range(turn_count):
        prompt.extend([index * 2])
        messages.append({"role": "user", "content": f"u{index}"})
        exchange = _chat_exchange(list(prompt), [index * 2 + 1], offset=index)
        exchange.request["messages"] = list(messages)
        exchanges.append(exchange)
        prompt.append(index * 2 + 1)
        messages.append({"role": "assistant", "content": "answer"})
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=exchanges)
    ).chat_completions_history()
    history.chat_template = "rerender"
    return history


class _RepeatedTextTokenizer:
    def __init__(self) -> None:
        self.apply_calls = 0

    def __call__(self, text: str, **kwargs: object) -> object:
        if not kwargs.get("return_offsets_mapping"):
            return [1000] if text == "answer" else [2000 + int(text[1:])]
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in re.finditer(r"<([ua])>(.*?)</\1>", text):
            content_start, content_end = match.span(2)
            token_ids.extend(
                [
                    3000 if match.group(1) == "u" else 3001,
                    1000
                    if match.group(2) == "answer"
                    else 2000 + int(match.group(2)[1:]),
                    3002,
                ]
            )
            offsets.extend(
                [
                    (match.start(), content_start),
                    (content_start, content_end),
                    (content_end, match.end()),
                ]
            )
        return {"input_ids": token_ids, "offset_mapping": offsets}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        **kwargs: object,
    ) -> object:
        del kwargs
        self.apply_calls += 1
        rendered = "".join(
            f"<{'a' if message['role'] == 'assistant' else 'u'}>"
            f"{message['content']}</{'a' if message['role'] == 'assistant' else 'u'}>"
            for message in messages
        )
        return self(rendered, return_offsets_mapping=True) if tokenize else rendered


def test_rerender_calls_chat_template_once_for_many_turns() -> None:
    history = _repeated_text_rerender_history(32)

    tokenizer = _RepeatedTextTokenizer()
    history.tokenize(tokenizer=tokenizer)

    assert tokenizer.apply_calls == 2


def test_repeated_text_rerender_scaling_is_near_linear() -> None:
    medians: list[float] = []
    for turn_count in (32, 64, 128):
        history = _repeated_text_rerender_history(turn_count)
        samples: list[float] = []
        for _ in range(5):
            tokenizer = _RepeatedTextTokenizer()
            started = perf_counter()
            history.tokenize(tokenizer=tokenizer)
            samples.append(perf_counter() - started)
            assert tokenizer.apply_calls == 2
        medians.append(median(samples))

    assert medians[1] < medians[0] * 3
    assert medians[2] < medians[1] * 3


def test_reasoning_split_trajectory_reuses_prevalidated_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchanges: list[ChatCompletionsExchange] = []
    request_messages: list[ChatCompletionMessageParam] = []
    prompt: list[int] = []
    for index in range(40):
        request_messages.append({"role": "user", "content": f"u{index}"})
        prompt.append(3000 + index)
        exchange = _chat_exchange(
            list(prompt), [1000 + index, 2000 + index], offset=index
        )
        exchange.request["messages"] = list(request_messages)
        payload = exchange.response.model_dump(mode="python")
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "reasoning": f"r{index}",
            "content": f"a{index}",
        }
        exchange.response = ChatCompletion.model_validate(payload)
        exchanges.append(exchange)
        request_messages.append({"role": "assistant", "content": f"a{index}"})
        prompt.append(2000 + index)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=exchanges)
    )
    projected = trajectory.chat_completions_histories()
    assert len(projected) == 40
    for history in projected:
        for source in history.message_sources:
            if source is not None:
                assert any(source.exchange is exchange for exchange in exchanges)
    assert all(
        any(
            source is not None
            and source.choice_index == 0
            and source.exchange is exchanges[0]
            for source in history.message_sources
        )
        for history in projected
    )

    from art.trajectories import _tokenize

    original = _tokenize._history_matches_projection
    calls = 0

    def counted(history: art.History) -> bool:
        nonlocal calls
        calls += 1
        return original(history)

    monkeypatch.setattr(_tokenize, "_history_matches_projection", counted)

    tokenized = trajectory.tokenize(multi_history=True)
    _, traces = _tokenize._tokenize_trajectory_with_trace(trajectory)
    first_key = next(key for key in traces[0].source_keys if key is not None)

    assert len(tokenized.histories) == 40
    assert all(first_key in trace.sources for trace in traces)
    assert calls == 0


def test_explicit_template_override_rerenders_exact_exchange_scaffold() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])])
    )

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {"turn 0": [10], "answer": [20]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            chat_template: str | None = None,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            assert chat_template == "custom"
            if messages[-1]["role"] == "assistant":
                if str(messages[-1]["content"]).startswith("ART_TRAJECTORY_"):
                    return [10, 999, 30]
                return [10, 20, 30]
            assert add_generation_prompt
            return [10]

    tokenized = trajectory.tokenize(
        tokenizer=Tokenizer(),
        chat_template="custom",
    )

    assert tokenized.token_ids == [10, 2, 30]
    assert tokenized.logprobs[1] == -0.2


def test_responses_external_context_requires_or_uses_exact_prompt_tokens() -> None:
    exchange = _response_exchange(
        "external", 2, previous_response_id="outside-trajectory"
    )
    response = exchange.response.model_dump(mode="python")
    response.pop("token_generations", None)
    exchange.response = Response.model_validate(response)
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).responses_history()
    with pytest.raises(ValueError, match="without exact prompt tokens"):
        history.tokenize(base_model="base/model")

    response = exchange.response.model_dump(mode="python")
    response["token_generations"] = [
        {
            "prompt_token_ids": [7, 8],
            "output_tokens": [{"token_id": 2, "logprob": -0.1}],
            "output_indices": [0],
        }
    ]
    exchange.response = Response.model_validate(response)
    tokenized = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .tokenize()
    )

    assert tokenized.token_ids == [7, 8, 2]
    assert tokenized.flags == [
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT,
        art.TokenFlag.EXACT | art.TokenFlag.SAMPLED,
    ]


def test_responses_conversation_requires_exact_prompt_tokens() -> None:
    exchange = _response_exchange("conversation", 2)
    exchange.request["conversation"] = "conversation-1"
    response = exchange.response.model_dump(mode="python")
    response.pop("token_generations", None)
    exchange.response = Response.model_validate(response)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))

    with pytest.raises(ValueError, match="conversation history requires exact"):
        trajectory.tokenize(base_model="base/model")

    response = exchange.response.model_dump(mode="python")
    response["token_generations"] = [
        {
            "prompt_token_ids": [5],
            "output_tokens": [{"token_id": 2, "logprob": -0.1}],
            "output_indices": [0],
        }
    ]
    exchange.response = Response.model_validate(response)
    assert trajectory.tokenize().token_ids == [5, 2]


def test_tokenized_results_materialize_metadata_and_group_shape() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])]),
        reward=0.75,
        metrics={"correct": True},
        metadata={"source": {"name": "unit"}},
    )
    group = art.TrajectoryGroup(
        [trajectory], metrics={"batch": 1}, metadata={"split": "test"}
    )

    tokenized = group.tokenize()

    assert tokenized.trajectories[0].model == "test/model"
    assert tokenized.trajectories[0].reward == 0.75
    assert tokenized.trajectories[0].metadata == {"source": {"name": "unit"}}
    assert tokenized.metrics == {"batch": 1}
    assert tokenized.metadata == {"split": "test"}
    assert "underlying" not in tokenized.model_dump()


def test_private_trace_covers_sampled_tokens_for_every_protocol() -> None:
    from art.trajectories._tokenize import _tokenize_trajectory_with_trace

    trajectories = [
        art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])])
        ),
        art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[_completion_exchange()])
        ),
        art.Trajectory(
            exchanges=TrajectoryExchanges(
                responses=[_response_exchange("response-trace", 2)]
            )
        ),
        art.Trajectory(
            exchanges=TrajectoryExchanges(
                messages=[
                    _message_exchange(
                        MessagesRequest(
                            model="test/model",
                            messages=[{"role": "user", "content": "question"}],
                            max_tokens=16,
                        ),
                        identifier="message-trace",
                        prompt_token_ids=[1],
                        token_ids=[2],
                        logprobs=[-0.2],
                    )
                ]
            )
        ),
    ]

    for trajectory in trajectories:
        tokenized, traces = _tokenize_trajectory_with_trace(trajectory)

        assert len(tokenized.histories) == len(traces) == 1
        history = tokenized.histories[0]
        trace = traces[0]
        trace.validate(history)
        assert sum(key is not None for key in trace.source_keys) == sum(
            bool(flag & art.TokenFlag.SAMPLED) for flag in history.flags
        )
        assert len(trace.sources) == 1


def test_private_trace_keys_do_not_collide_for_repeated_empty_response_ids() -> None:
    from art.trajectories._tokenize import _tokenize_trajectory_with_trace

    first = _chat_exchange([1], [2])
    second = _chat_exchange([1, 2, 3], [4], offset=1)
    first.response.id = second.response.id = ""
    second.start_time = first.start_time
    second.end_time = first.end_time
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )

    tokenized, [trace] = _tokenize_trajectory_with_trace(trajectory)
    sampled_keys = [key for key in trace.source_keys if key is not None]

    assert tokenized.histories[0].token_ids == [1, 2, 3, 4]
    assert len(set(sampled_keys)) == 2
    assert len(trace.sources) == 2


def test_responses_fallback_trace_does_not_retrain_echoed_output_items() -> None:
    from art.trajectories._tokenize import (
        _first_introduction_mask,
        _tokenize_trajectory_with_trace,
    )

    def output(item_id: str, text: str) -> ResponseOutputMessageParam:
        return {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }

    def response(
        response_id: str, items: list[ResponseOutputMessageParam], offset: int
    ) -> Response:
        return Response.model_validate(
            {
                "id": response_id,
                "created_at": float(offset),
                "model": "test/model",
                "object": "response",
                "output": items,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )

    user = EasyInputMessageParam(role="user", content="question")
    first_outputs = [output("one", "first"), output("two", "second")]
    first_response = response(
        "response-fallback",
        first_outputs,
        0,
    )
    first_input: ResponseInputParam = [user]
    first_request = ResponsesRequest(model="test/model", input=first_input)
    first_request["chat_template_kwargs"] = {"enable_thinking": False}
    first = ResponsesExchange(
        request=first_request,
        response=first_response,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1, 0, 0, 0, 1000),
    )
    echoed: ResponseInputParam = [*first_outputs]
    second_input: ResponseInputParam = [
        user,
        *echoed,
        EasyInputMessageParam(role="user", content="continue"),
    ]
    second = ResponsesExchange(
        request=ResponsesRequest(
            model="test/model",
            input=second_input,
        ),
        response=response("response-final", [output("final", "final")], 1),
        start_time=datetime(2026, 1, 1, 0, 0, 1),
        end_time=datetime(2026, 1, 1, 0, 0, 1, 1000),
    )

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> list[int]:
            del kwargs
            return {
                "question": [1],
                "first": [2],
                "second": [3],
                "firstsecond": [2, 3],
                "continue": [4],
                "final": [5],
            }[text]

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [
                token for message in messages for token in self(str(message["content"]))
            ]

    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first, second])
    )
    tokenized, traces = _tokenize_trajectory_with_trace(
        trajectory,
        tokenizer=Tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )

    assert len(tokenized.histories) == len(traces) == 2
    seen: set[object] = set()
    trained_first_response = 0
    first_response_indices: set[int] = set()
    for trace in traces:
        trainable = _first_introduction_mask(trace.source_keys, seen)
        for selected, key in zip(trainable, trace.source_keys, strict=True):
            if key is not None and key.response_id == first_response.id:
                first_response_indices.add(key.index)
                trained_first_response += selected

    assert first_response_indices == {0}
    assert trained_first_response == 2


def test_completions_history_requires_exhaustive_source_spans() -> None:
    history = art.CompletionsTokenHistory(
        model="test/model",
        prompt=[1],
        prompt_sources=[],
        sampled_spans=[],
    )

    with pytest.raises(ValueError, match="exhaustively cover"):
        history.tokenize()


def test_exchange_training_rejects_locally_tokenized_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.preprocessing.tokenize import tokenize_trajectory_groups

    model = "wandb-artifact:///entity/project/run:step0"
    fallback = _chat_exchange([], [], model=model)
    fallback_extra = fallback.response.choices[0].model_extra
    assert fallback_extra is not None
    fallback_extra.pop("prompt_token_ids")
    fallback_extra.pop("token_ids")
    fallback.response.choices[0].logprobs = None

    class Tokenizer:
        name_or_path = model

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            self.calls.append(kwargs)
            return [1, 2] if messages[-1]["role"] == "assistant" else [1]

        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [2]

        def decode(self, token_id: int) -> str:
            return str(token_id)

    group = art.TrajectoryGroup(
        [
            art.Trajectory(
                exchanges=TrajectoryExchanges(chat_completions=[fallback]),
                reward=1,
            ),
            art.Trajectory(
                exchanges=TrajectoryExchanges(
                    chat_completions=[_chat_exchange([1], [3], model=model)]
                ),
                reward=0,
            ),
        ]
    )
    monkeypatch.setattr(
        "art.trajectories._tokenize._artifact_config",
        lambda _model: pytest.fail("supplied tokenizer should bypass W&B"),
    )
    tokenizer = Tokenizer()

    with pytest.raises(
        RuntimeError,
        match="requires exact inference-provided token IDs",
    ):
        list(
            tokenize_trajectory_groups(
                tokenizer,  # type: ignore[arg-type, ty:invalid-argument-type]
                [group],
                allow_training_without_logprobs=True,
                scale_rewards=False,
                shuffle_group_trajectories=False,
                chat_template_kwargs={"serverless": True},
            )
        )
    assert all(call["serverless"] is True for call in tokenizer.calls)


def test_exchange_training_accepts_exact_split_tokenizations() -> None:
    from art.preprocessing.tokenize import tokenize_trajectory_groups

    def trajectory(reward: float) -> art.Trajectory:
        first = _chat_exchange([1], [101])
        first.response.choices[0].message.content = "cat"
        second = _chat_exchange([1, 500, 3], [4], offset=1)
        second.request["messages"] = [
            {"role": "user", "content": "turn 0"},
            {"role": "assistant", "content": "cat"},
            {"role": "user", "content": "turn 1"},
        ]
        return art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[first, second]),
            reward=reward,
        )

    class Tokenizer:
        name_or_path = "test/model"

        def decode(self, token_id: int) -> str:
            return str(token_id)

    results = list(
        tokenize_trajectory_groups(
            Tokenizer(),  # type: ignore[arg-type, ty:invalid-argument-type]
            [art.TrajectoryGroup([trajectory(1.0), trajectory(0.0)])],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            shuffle_group_trajectories=False,
        )
    )

    assert [result.token_ids for result in results] == [
        [1, 101],
        [1, 500, 3, 4],
        [1, 101],
        [1, 500, 3, 4],
    ]
    assert [result.assistant_mask for result in results] == [
        [0, 1],
        [0, 0, 0, 1],
        [0, 1],
        [0, 0, 0, 1],
    ]


def test_exchange_training_requires_logprobs_unless_allowed() -> None:
    from art.preprocessing.tokenize import TokenizedResult, tokenize_trajectory_groups

    class Tokenizer:
        name_or_path = "test/model"

        def decode(self, token_id: int) -> str:
            return str(token_id)

    missing = _chat_exchange([1], [2])
    missing.response.choices[0].logprobs = None
    group = art.TrajectoryGroup(
        [
            art.Trajectory(
                exchanges=TrajectoryExchanges(chat_completions=[missing]), reward=1
            ),
            art.Trajectory(
                exchanges=TrajectoryExchanges(
                    chat_completions=[_chat_exchange([1], [3])]
                ),
                reward=0,
            ),
        ]
    )

    def tokenize(*, allow_missing: bool) -> list[TokenizedResult]:
        return list(
            tokenize_trajectory_groups(
                # This exact-token path only calls decode.
                Tokenizer(),  # type: ignore[arg-type, ty:invalid-argument-type]
                [group],
                allow_training_without_logprobs=allow_missing,
                scale_rewards=False,
                shuffle_group_trajectories=False,
            )
        )

    with pytest.raises(RuntimeError, match="missing logprobs"):
        tokenize(allow_missing=False)
    assert len(tokenize(allow_missing=True)) == 2
