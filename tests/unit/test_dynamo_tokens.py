from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import pytest

from art import TrainableModel
from art.model import _OpenAIChatCompletionsProxy
from art.openai import consume_chat_completion_stream, finalize_chat_completion
from art.preprocessing.dynamo_tokens import (
    COMPLETION_LOGPROBS_KEY,
    attach_dynamo_token_metadata,
    choice_completion_logprobs,
)
from art.preprocessing.tokenize import assemble_vllm_training_sequences
from art.trajectories import (
    ChatCompletionsExchange,
    LegacyHistory,
    Trajectory,
    TrajectoryExchanges,
)
from art.trajectories._protocols import _chat_response
from art.trajectories._tokenize import _sampled_evidence_fingerprint


@pytest.fixture
def payload() -> dict[str, Any]:
    return {
        "id": "dynamo-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "yes",
                    "reasoning_content": "think",
                },
                "logprobs": {
                    "content": [{"token": "yes", "logprob": -0.2, "top_logprobs": []}]
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        "nvext": {
            "engine_data": {
                "finished": True,
                "prompt_token_ids": [10, 11, 12],
                "completion_token_ids": [20, 21, 22],
                "completion_logprobs": [-0.1, -0.2, -0.3],
            }
        },
    }


def exchange(response: ChatCompletion) -> ChatCompletionsExchange:
    now = datetime.now(timezone.utc)
    return ChatCompletionsExchange(
        request={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        response=response,
        start_time=now,
        end_time=now,
    )


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_exact_training_tokens_preserve_text_response(
    payload: dict[str, Any], finish_reason: str
) -> None:
    payload["choices"][0]["finish_reason"] = finish_reason
    response = finalize_chat_completion(ChatCompletion.model_validate(payload))
    choice = response.choices[0]
    assert (
        choice.message.model_dump(exclude_none=True) == payload["choices"][0]["message"]
    )
    assert choice.logprobs is not None
    assert choice.logprobs.content is not None
    assert len(choice.logprobs.content) == 1
    assert choice.logprobs.content[0].token == "yes"
    assert choice.finish_reason == finish_reason
    seq = assemble_vllm_training_sequences(
        tokenizer=cast(Any, None),
        histories=[LegacyHistory(messages_and_choices=[choice])],
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=Trajectory(messages_and_choices=[choice], reward=1),
    )[0]
    assert seq.token_ids == [10, 11, 12, 20, 21, 22]
    assert seq.logprobs[3:] == [-0.1, -0.2, -0.3]
    assert seq.assistant_mask == [0, 0, 0, 1, 1, 1]


def test_raw_exchange_and_serialization_use_exact_engine_sequence(
    payload: dict[str, Any],
) -> None:
    captured = exchange(ChatCompletion.model_validate(payload))
    restored = ChatCompletionsExchange.model_validate_json(captured.model_dump_json())
    for item in [captured, restored]:
        tokenized = Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[item]), reward=1
        ).tokenize()
        assert tokenized.tokens == [10, 11, 12, 20, 21, 22]
        assert tokenized.logprobs[3:] == [-0.1, -0.2, -0.3]


def test_numeric_logprobs_are_part_of_source_identity(payload: dict[str, Any]) -> None:
    captured = exchange(ChatCompletion.model_validate(payload))
    before = _sampled_evidence_fingerprint(
        captured, protocol="chat_completions", index=0
    )
    assert captured.response.choices[0].model_extra is not None
    captured.response.choices[0].model_extra[COMPLETION_LOGPROBS_KEY][0] = -1.0
    assert (
        _sampled_evidence_fingerprint(captured, protocol="chat_completions", index=0)
        != before
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_token_ids", []),
        ("prompt_token_ids", [True]),
        ("completion_token_ids", [1, -1]),
        ("completion_token_ids", [1.0]),
        ("completion_logprobs", [-0.1]),
        ("completion_logprobs", [-0.1, float("inf"), -0.3]),
        ("completion_logprobs", [-0.1, float("nan"), -0.3]),
        ("completion_logprobs", [-0.1, True, -0.3]),
        ("finished", False),
    ],
)
def test_invalid_metadata_rejected_without_partial_mutation(
    payload: dict[str, Any], field: str, value: Any
) -> None:
    payload["nvext"]["engine_data"][field] = value
    response = ChatCompletion.model_validate(payload)
    original = deepcopy(response.model_extra)
    with pytest.raises(ValueError):
        attach_dynamo_token_metadata(response)
    assert response.choices[0].model_extra == {}
    assert response.model_extra == original


def test_native_vllm_and_multi_choice_are_unchanged(payload: dict[str, Any]) -> None:
    for native in [True, False]:
        data = deepcopy(payload)
        data["choices"] *= 2
        if native:
            data["prompt_token_ids"] = [1, 2]
            for choice in data["choices"]:
                choice["token_ids"] = [3]
        response = ChatCompletion.model_validate(data)
        before = response.model_dump()
        attach_dynamo_token_metadata(response)
        assert response.model_dump() == before


def test_partial_native_conflict_is_rejected(payload: dict[str, Any]) -> None:
    payload["prompt_token_ids"] = [999]
    with pytest.raises(ValueError, match="conflicts"):
        attach_dynamo_token_metadata(ChatCompletion.model_validate(payload))


def test_missing_logprobs_are_not_invented(payload: dict[str, Any]) -> None:
    payload["nvext"]["engine_data"].pop("completion_logprobs")
    payload["choices"][0]["logprobs"] = None
    response = finalize_chat_completion(ChatCompletion.model_validate(payload))
    assert choice_completion_logprobs(response.choices[0]) is None
    assert response.choices[0].logprobs is None


def sse(payload: dict[str, Any], *, include_engine: bool = True) -> bytes:
    common = {
        "id": "dynamo-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
    }
    chunks = [
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "yes",
                        "reasoning_content": "think",
                    },
                }
            ],
        },
        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **common,
            "choices": [],
            "usage": payload["usage"],
            **({"nvext": payload["nvext"]} if include_engine else {}),
        },
    ]
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    ).encode()


@pytest.mark.asyncio
async def test_sdk_stream_and_captured_stream_keep_final_metadata(
    payload: dict[str, Any],
) -> None:
    body = sse(payload)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        stream = await client.chat.completions.create(
            model="test", messages=[], stream=True
        )
        response = await consume_chat_completion_stream(stream)
    captured = _chat_response(body, stream=True)
    for item in [response, captured]:
        assert item.choices[0].model_extra is not None
        assert item.choices[0].model_extra["token_ids"] == [20, 21, 22]
        assert choice_completion_logprobs(item.choices[0]) == [-0.1, -0.2, -0.3]
        assert Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[exchange(item)]), reward=1
        ).tokenize().tokens == [10, 11, 12, 20, 21, 22]


@pytest.mark.asyncio
async def test_managed_client_requests_and_attaches_engine_metadata(
    payload: dict[str, Any],
) -> None:
    class Completions:
        async def create(self, **kwargs: Any) -> ChatCompletion:
            assert kwargs["extra_body"]["nvext"] == {
                "extra_fields": ["engine_data", "timing"],
                "cache_salt": "tenant",
            }
            return ChatCompletion.model_validate(payload)

    model = TrainableModel(
        run_name="test", name="test", project="test", base_model="test"
    )
    proxy = _OpenAIChatCompletionsProxy(
        Completions(), lambda _: None, model._default_chat_completion_extra_body()
    )
    response = await proxy.create(
        model="test",
        messages=[],
        extra_body={"nvext": {"extra_fields": ["timing"], "cache_salt": "tenant"}},
    )
    assert choice_completion_logprobs(response.choices[0]) == [-0.1, -0.2, -0.3]


@pytest.mark.parametrize("allow_missing", [False, True])
def test_missing_engine_logprobs_never_use_text_logprobs(
    payload: dict[str, Any], allow_missing: bool
) -> None:
    import math

    payload["nvext"]["engine_data"].pop("completion_logprobs")
    # Even equal-length text metadata is not evidence for the engine sequence.
    payload["choices"][0]["logprobs"]["content"] *= 3
    response = finalize_chat_completion(ChatCompletion.model_validate(payload))
    choice = response.choices[0]
    kwargs: dict[str, Any] = dict(
        tokenizer=cast(Any, None),
        histories=[LegacyHistory(messages_and_choices=[choice])],
        advantage=1.0,
        allow_training_without_logprobs=allow_missing,
        trajectory=Trajectory(messages_and_choices=[choice], reward=1),
    )
    if not allow_missing:
        with pytest.raises(RuntimeError, match="missing exact logprobs"):
            assemble_vllm_training_sequences(**kwargs)
    else:
        seq = assemble_vllm_training_sequences(**kwargs)[0]
        assert all(math.isnan(value) for value in seq.logprobs[3:])
    tokens = Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange(response)]), reward=1
    ).tokenize()
    assert all(math.isnan(value) for value in tokens.logprobs[3:])


@pytest.mark.asyncio
async def test_early_stream_stop_does_not_fabricate_training_tokens(
    payload: dict[str, Any],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse(payload), headers={"content-type": "text/event-stream"}
        )

    def stop(*args: Any) -> None:
        raise StopIteration

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        stream = await client.chat.completions.create(
            model="test", messages=[], stream=True
        )
        response = await consume_chat_completion_stream(stream, on_chunk=stop)
    assert not (response.choices[0].model_extra or {}).get("token_ids")
    assert choice_completion_logprobs(response.choices[0]) is None
