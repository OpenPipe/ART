from __future__ import annotations

from functools import cached_property
from typing import Any, Iterable, Mapping, cast

import httpx
from openai import AsyncOpenAI, BaseModel, _legacy_response
from openai._base_client import make_request_options
from openai._resource import AsyncAPIResource
from openai._response import async_to_streamed_response_wrapper
from openai._types import Body, Headers, NotGiven, Query, not_given
from openai.resources.models import AsyncModels
from openai.types import Model
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from art.types import Message, MessageOrChoice, MessagesAndChoices, Tools

ParsedMessageOrChoice = Choice | Message
ParsedMessagesAndChoices = list[ParsedMessageOrChoice]


def _message_or_choice_to_dict(message_or_choice: MessageOrChoice) -> dict[str, Any]:
    if isinstance(message_or_choice, dict):
        return cast(dict[str, Any], message_or_choice)
    if isinstance(message_or_choice, BaseModel):
        return cast(dict[str, Any], message_or_choice.to_dict())
    to_dict = getattr(message_or_choice, "to_dict", None)
    if to_dict is None:
        raise TypeError(
            "message_or_choice must be a dict or OpenAI model with to_dict()"
        )
    return cast(dict[str, Any], to_dict())


class MessagesAndChoicesWithLogprobs(BaseModel):
    messages_and_choices: ParsedMessagesAndChoices
    usages: list[CompletionUsage]


class TinkerAsyncModels(AsyncModels):
    @cached_property
    def with_raw_response(self) -> "TinkerAsyncModelsWithRawResponse":
        return TinkerAsyncModelsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> "TinkerAsyncModelsWithStreamingResponse":
        return TinkerAsyncModelsWithStreamingResponse(self)

    async def put(
        self,
        model: str,
        *,
        target: str,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Model:
        if not model:
            raise ValueError(
                f"Expected a non-empty value for `model` but received {model!r}"
            )

        return await self._put(
            f"/models/{model}",
            body={"target": target},
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=Model,
        )


class TinkerAsyncModelsWithRawResponse:
    def __init__(self, models: TinkerAsyncModels) -> None:
        self._models = models

        self.put = _legacy_response.async_to_raw_response_wrapper(models.put)
        self.retrieve = _legacy_response.async_to_raw_response_wrapper(models.retrieve)
        self.list = _legacy_response.async_to_raw_response_wrapper(models.list)
        self.delete = _legacy_response.async_to_raw_response_wrapper(models.delete)


class TinkerAsyncModelsWithStreamingResponse:
    def __init__(self, models: TinkerAsyncModels) -> None:
        self._models = models

        self.put = async_to_streamed_response_wrapper(models.put)
        self.retrieve = async_to_streamed_response_wrapper(models.retrieve)
        self.list = async_to_streamed_response_wrapper(models.list)
        self.delete = async_to_streamed_response_wrapper(models.delete)


class TinkerAsyncMessagesAndChoices(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> "TinkerAsyncMessagesAndChoicesWithRawResponse":
        return TinkerAsyncMessagesAndChoicesWithRawResponse(self)

    @cached_property
    def with_streaming_response(
        self,
    ) -> "TinkerAsyncMessagesAndChoicesWithStreamingResponse":
        return TinkerAsyncMessagesAndChoicesWithStreamingResponse(self)

    async def with_logprobs(
        self,
        messages_and_choices: MessagesAndChoices,
        *,
        models: Iterable[str],
        model_aliases: Mapping[str, str] | None = None,
        tools: Tools | None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> MessagesAndChoicesWithLogprobs:
        return await self._post(
            "/messages_and_choices/with_logprobs",
            body={
                "messages_and_choices": [
                    _message_or_choice_to_dict(item) for item in messages_and_choices
                ],
                "models": list(models),
                "model_aliases": dict(model_aliases or {}),
                "tools": tools,
            },
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=MessagesAndChoicesWithLogprobs,
        )


class TinkerAsyncMessagesAndChoicesWithRawResponse:
    def __init__(self, messages_and_choices: TinkerAsyncMessagesAndChoices) -> None:
        self._messages_and_choices = messages_and_choices

        self.with_logprobs = _legacy_response.async_to_raw_response_wrapper(
            messages_and_choices.with_logprobs
        )


class TinkerAsyncMessagesAndChoicesWithStreamingResponse:
    def __init__(self, messages_and_choices: TinkerAsyncMessagesAndChoices) -> None:
        self._messages_and_choices = messages_and_choices

        self.with_logprobs = async_to_streamed_response_wrapper(
            messages_and_choices.with_logprobs
        )


class TinkerAsyncOpenAI(AsyncOpenAI):
    @cached_property
    def models(self) -> TinkerAsyncModels:
        return TinkerAsyncModels(self)

    @cached_property
    def messages_and_choices(self) -> TinkerAsyncMessagesAndChoices:
        return TinkerAsyncMessagesAndChoices(self)
