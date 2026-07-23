from __future__ import annotations

import builtins
from datetime import datetime, timedelta
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from anthropic.types import ImageBlockParam, Message, MessageParam
from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.responses import Response
import pytest

import art
from art.trajectories import (
    ChatCompletionsExchange,
    ChatCompletionsRequest,
    CompletionsExchange,
    CompletionsRequest,
    MessagesExchange,
    MessagesRequest,
    ResponsesExchange,
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


def test_messages_exact_prompt_and_output_do_not_load_a_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Message.model_validate(
        {
            "id": "message-exact",
            "type": "message",
            "role": "assistant",
            "model": "test/model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "prompt_token_ids": [1, 2],
            "token_ids": [3],
            "logprobs": [-0.3],
        }
    )
    start = datetime(2026, 1, 1)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            messages=[
                MessagesExchange(
                    request=MessagesRequest(
                        model="test/model",
                        messages=[{"role": "user", "content": "question"}],
                        max_tokens=16,
                    ),
                    response=response,
                    start_time=start,
                    end_time=start + timedelta(milliseconds=1),
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
    response_extra["raw_output_tokens"] = [{"token_id": "invalid"}]

    message_response = Message.model_validate(
        {
            "id": "message-invalid",
            "type": "message",
            "role": "assistant",
            "model": "test/model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "token_ids": [2, "invalid"],
        }
    )
    start = datetime(2026, 1, 1)
    message = MessagesExchange(
        request=MessagesRequest(
            model="test/model",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=16,
        ),
        response=message_response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
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


@pytest.mark.parametrize("response_token_ids", [[1, 2], [2]])
def test_completions_echo_preserves_prompt_logprobs_without_sampling_them(
    response_token_ids: list[int],
) -> None:
    exchange = _completion_exchange(echo=True)
    payload = exchange.response.model_dump(mode="python")
    payload["choices"][0]["token_ids"] = response_token_ids
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
    response = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test/model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    start = datetime(2026, 1, 1)
    exchange = MessagesExchange(
        request=MessagesRequest(
            model="wandb-artifact:///entity/project/run:step0",
            messages=[{"role": "user", "content": "question"}],
            chat_template="request-template",
            chat_template_kwargs={"request": True},
            thinking={"type": "enabled", "budget_tokens": 128},
        ),
        response=response,
        start_time=start,
        end_time=start + timedelta(seconds=1),
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
    assert tokenizer.calls == [
        {
            "tools": None,
            "tokenize": True,
            "add_generation_prompt": False,
            "chat_template": "explicit-template",
            "request": True,
            "explicit": True,
            "enable_thinking": True,
            "thinking_budget": 128,
        },
    ]


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


def test_reasoning_stripped_history_uses_stream_block_tokens() -> None:
    def exchange(
        offset: int,
        request_messages: list[MessageParam],
        answer: str,
        token_ids: list[int],
    ) -> MessagesExchange:
        start = datetime(2026, 1, 1) + timedelta(seconds=offset)
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
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": f"thought-{offset}",
                            "signature": "signature",
                            "token_ids": [90 + offset],
                            "logprobs": [-9.0 - offset],
                        },
                        {
                            "type": "text",
                            "text": answer,
                            "token_ids": token_ids,
                            "logprobs": [-0.1 * token for token in token_ids],
                        },
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": len(token_ids)},
                }
            ),
            start_time=start,
            end_time=start + timedelta(milliseconds=1),
        )

    first = exchange(
        0,
        [{"role": "user", "content": "one"}],
        "first",
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
        [201],
    )

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {
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
            return [10, 11, 12] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del text, kwargs
            return SimpleNamespace(input_ids=[11])

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


def test_ambiguous_visible_logprobs_fail_closed(
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
    result = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    ).tokenize(
        base_model="base/model",
    )

    assert result.token_ids == [10, 11, 12, 11]
    assert all(math.isnan(logprob) for logprob in result.logprobs[1:])


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
    response = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test/model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    start = datetime(2026, 1, 1)
    image: ImageBlockParam = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "...",
        },
    }
    message: MessageParam = {"role": "user", "content": [image]}
    exchange = MessagesExchange(
        request=MessagesRequest(
            model="test/model",
            messages=[message],
        ),
        response=response,
        start_time=start,
        end_time=start + timedelta(seconds=1),
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
            "raw_output_tokens": [{"token_id": output_id, "logprob": -0.1}],
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


def _response_with_content_logprobs(*, exact_second: bool) -> ResponsesExchange:
    exchange = _response_exchange("response-content-logprobs", 0)
    data = exchange.response.model_dump(mode="python")
    data.pop("raw_output_tokens", None)

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


def test_responses_empty_raw_tokens_fall_back_for_visible_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange("response-empty-raw", 0)
    data = exchange.response.model_dump(mode="python")
    data["raw_output_tokens"] = []
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
    data.pop("raw_output_tokens", None)
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
    response.pop("raw_output_tokens", None)
    exchange.response = Response.model_validate(response)
    with pytest.raises(ValueError, match="no renderable text"):
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange])).tokenize(
            base_model="base/model",
        )


def test_responses_parallel_function_calls_form_one_assistant_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _response_exchange("parallel-tools", 2)
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
    first = _response_exchange("resp-1", 20)
    second = _response_exchange("resp-2", 30, previous_response_id="resp-1", offset=1)
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
    with pytest.raises(ValueError, match="outside this trajectory"):
        trajectory.responses_histories()[1].tokenize(base_model="base/model")


def test_prefix_retokenization_preserves_sampled_ids_and_logprobs(
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
            assert text == "cat"
            assert not add_special_tokens
            return [500]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    monkeypatch.setattr(
        "art.trajectories._tokenize._WARNED_PREFIX_RETOKENIZATION", False
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )

    with pytest.warns(UserWarning, match="preserved the original sampled token IDs"):
        tokenized = trajectory.tokenize(base_model="base/model")

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
            assert text == "answer"
            assert not add_special_tokens
            return [20]

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
            return [10] if len(messages) == 1 else [10, 20, 30]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [10, 2, 30]
    assert tokenized.logprobs[1] == -0.2
    assert tokenized.flags[1] == art.TokenFlag.EXACT | art.TokenFlag.SAMPLED


def test_explicit_template_override_rerenders_exact_exchange_scaffold() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])])
    )

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert text == "answer"
            assert not add_special_tokens
            return [20]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            chat_template: str | None = None,
            **kwargs: object,
        ) -> list[int]:
            del kwargs
            assert chat_template == "custom"
            return [10, 20, 30]

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
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[exchange])
    ).responses_history()
    with pytest.raises(ValueError, match="without exact prompt tokens"):
        history.tokenize(base_model="base/model")

    response = exchange.response.model_dump(mode="python")
    response["prompt_token_ids"] = [7, 8]
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
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))

    with pytest.raises(ValueError, match="conversation history requires exact"):
        trajectory.tokenize(base_model="base/model")

    response = exchange.response.model_dump(mode="python")
    response["prompt_token_ids"] = [5]
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


def test_completions_history_requires_exhaustive_source_spans() -> None:
    history = art.CompletionsTokenHistory(
        model="test/model",
        prompt=[1],
        prompt_sources=[],
        sampled_spans=[],
    )

    with pytest.raises(ValueError, match="exhaustively cover"):
        history.tokenize()


def test_exchange_trajectories_feed_existing_training_tokenizer(
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

    results = list(
        tokenize_trajectory_groups(
            tokenizer,  # type: ignore[arg-type, ty:invalid-argument-type]
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
            shuffle_group_trajectories=False,
            chat_template_kwargs={"serverless": True},
        )
    )

    assert [result.token_ids for result in results] == [[1, 2], [1, 3]]
    assert [result.assistant_mask for result in results] == [[0, 1], [0, 1]]
    assert all(call["serverless"] is True for call in tokenizer.calls)


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
