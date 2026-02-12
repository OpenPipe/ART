import warnings

# Suppress pydantic warnings at module level
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

from aiohttp import web
import litellm
import litellm.litellm_core_utils.streaming_handler
import litellm.types.utils
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pytest
import pytest_asyncio

import art
from art.utils.litellm import convert_litellm_choice_to_openai

mock_response = {
    "id": "chatcmpl-293ce9f37dba40e5be39448acaf6fb49",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:9707",
                        "bytes": [72, 101, 108, 108, 111],
                        "logprob": -0.0017243054462596774,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:0",
                        "bytes": [33],
                        "logprob": -0.007611795328557491,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:2585",
                        "bytes": [32, 72, 111, 119],
                        "logprob": -0.03061593696475029,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:646",
                        "bytes": [32, 99, 97, 110],
                        "logprob": -1.1920858014491387e-05,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:358",
                        "bytes": [32, 73],
                        "logprob": -2.3841855067985307e-07,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:7789",
                        "bytes": [32, 97, 115, 115, 105, 115, 116],
                        "logprob": -0.020548323169350624,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:498",
                        "bytes": [32, 121, 111, 117],
                        "logprob": 0.0,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:3351",
                        "bytes": [32, 116, 111, 100, 97, 121],
                        "logprob": -4.410734163684538e-06,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:30",
                        "bytes": [63],
                        "logprob": -2.3841855067985307e-07,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:151645",
                        "bytes": [],
                        "logprob": -0.0083366259932518,
                        "top_logprobs": [],
                    },
                ],
                "refusal": None,
            },
            "message": {
                "content": "Hello! How can I assist you today?",
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": [],
                "reasoning_content": None,
            },
            "stop_reason": None,
        }
    ],
    "created": 1755801745,
    "model": "test",
    "object": "chat.completion",
    "service_tier": None,
    "system_fingerprint": None,
    "usage": {
        "completion_tokens": 10,
        "prompt_tokens": 31,
        "total_tokens": 41,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    },
    "prompt_logprobs": None,
    "kv_transfer_params": None,
}
mock_stream_response = b"""data: {"id":"chatcmpl-aa0d1e3261414f53acafc2f8e53bf9d6","object":"chat.completion.chunk","created":1755831263,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-aa0d1e3261414f53acafc2f8e53bf9d6","object":"chat.completion.chunk","created":1755831263,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-aa0d1e3261414f53acafc2f8e53bf9d6","object":"chat.completion.chunk","created":1755831263,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-29e663261e524fcfa2162f4f3d76a7f0","type":"function","index":0,"function":{"name":"get_current_weather","arguments":"{"}}]},"logprobs":{"content":[{"token":"token_id:314","logprob":-0.00015293381875380874,"bytes":[32,123],"top_logprobs":[]}]},"finish_reason":null}]}

data: {"id":"chatcmpl-aa0d1e3261414f53acafc2f8e53bf9d6","object":"chat.completion.chunk","created":1755831263,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-29e663261e524fcfa2162f4f3d76a7f0","type":"function","index":0,"function":{"name":"get_current_weather","arguments":"{"}}]},"logprobs":{"content":[{"token":"token_id:314","logprob":-0.00015293381875380874,"bytes":[32,123],"top_logprobs":[]}]},"finish_reason":null}]}

data: {"id":"chatcmpl-aa0d1e3261414f53acafc2f8e53bf9d6","object":"chat.completion.chunk","created":1755831263,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":null,"arguments":"}"}}]},"logprobs":{"content":[{"token":"token_id:3417","logprob":-3.576278118089249e-7,"bytes":[125,125],"top_logprobs":[]}]},"finish_reason":null}]}

data: {"id":"chatcmpl-aa0d1e3261414f53acafc2f8e53bf9d6","object":"chat.completion.chunk","created":1755831263,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":null,"arguments":"}"}}]},"logprobs":{"content":[{"token":"token_id:3417","logprob":-3.576278118089249e-7,"bytes":[125,125],"top_logprobs":[]}]},"finish_reason":null}]}

data: [DONE]

data: [DONE]

"""
mock_stream_choice = Choice(
    **{
        "finish_reason": "stop",
        "index": 0,
        "logprobs": {
            "content": [
                {
                    "token": "token_id:314",
                    "bytes": [32, 123],
                    "logprob": -0.00015293381875380874,
                    "top_logprobs": [],
                },
                {
                    "token": "token_id:314",
                    "bytes": [32, 123],
                    "logprob": -0.00015293381875380874,
                    "top_logprobs": [],
                },
                {
                    "token": "token_id:3417",
                    "bytes": [125, 125],
                    "logprob": -3.576278118089249e-07,
                    "top_logprobs": [],
                },
                {
                    "token": "token_id:3417",
                    "bytes": [125, 125],
                    "logprob": -3.576278118089249e-07,
                    "top_logprobs": [],
                },
            ],
            "refusal": None,
        },
        "message": {
            "content": None,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": [
                {
                    "id": "chatcmpl-tool-29e663261e524fcfa2162f4f3d76a7f0",
                    "function": {"arguments": "{{}}", "name": "get_current_weather"},
                    "type": "function",
                }
            ],
        },
    }  # ty:ignore[invalid-argument-type]
)


@pytest_asyncio.fixture
async def test_server():
    """Start a test server for the module."""

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        if body.get("stream", False):
            return web.Response(body=mock_stream_response)
        return web.json_response(mock_response)

    app = web.Application()
    app.router.add_route("POST", "/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8888)
    await site.start()
    print(f"Test server started on http://localhost:8888")

    yield  # Tests run here

    print("Cleaning up test server...")
    await runner.cleanup()


async def test_auto_trajectory(test_server: None) -> None:
    message: ChatCompletionMessageParam = {"role": "user", "content": "Hi!"}
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    async def say_hi() -> str | None:
        """A method that says hi to an assistant and returns the response."""
        client = AsyncOpenAI(base_url="http://localhost:8888/v1", api_key="default")
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[message],
            tools=tools,
        )
        # test a follow up message
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[
                message,
                {
                    "role": "assistant",
                    "content": chat_completion.choices[0].message.content,
                },
                message,
            ],
            tools=tools,
        )
        # and another call without tools (should create a new history)
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[
                message,
                {
                    "role": "assistant",
                    "content": chat_completion.choices[0].message.content,
                },
                message,
                {
                    "role": "assistant",
                    "content": chat_completion.choices[0].message.content,
                },
                message,
            ],
        )
        # and another call with tools, but limited messages (should create another history)
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[message],
            tools=tools,
        )
        # and another call with tool_choice="required" & stream=True
        async for _ in await client.chat.completions.create(
            model="test",
            messages=[message],
            tool_choice="required",
            tools=tools,
            stream=True,
        ):
            pass
        # Add ART support with a couple lines of optional code
        if trajectory := art.auto_trajectory():
            trajectory.reward = 1.0
        return chat_completion.choices[0].message.content

    # Use the capture_auto_trajectory utility to capture a trajectory automatically
    trajectory = await art.capture_auto_trajectory(say_hi())
    assert trajectory.messages_and_choices == [
        message,
        Choice(**mock_response["choices"][0]),  # ty:ignore[invalid-argument-type, not-subscriptable]
        message,
        Choice(**mock_response["choices"][0]),  # ty:ignore[invalid-argument-type, not-subscriptable]
    ]
    assert trajectory.tools == tools
    assert trajectory.additional_histories[0].messages_and_choices == [
        message,
        {
            "content": "Hello! How can I assist you today?",
            "role": "assistant",
        },
        message,
        {
            "content": "Hello! How can I assist you today?",
            "role": "assistant",
        },
        message,
        Choice(**mock_response["choices"][0]),  # ty:ignore[invalid-argument-type, not-subscriptable]
    ]
    assert trajectory.additional_histories[0].tools is None
    assert trajectory.additional_histories[1].messages_and_choices == [
        message,
        Choice(**mock_response["choices"][0]),  # ty:ignore[invalid-argument-type, not-subscriptable]
    ]
    assert trajectory.additional_histories[1].tools == tools
    assert trajectory.additional_histories[2].messages_and_choices == [
        message,
        mock_stream_choice,
    ]
    assert trajectory.additional_histories[2].tools == tools


@pytest.mark.filterwarnings("ignore::UserWarning:pydantic")
async def test_litellm_auto_trajectory(test_server: None) -> None:
    message: ChatCompletionMessageParam = {"role": "user", "content": "Hi!"}
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    async def say_hi() -> str | None:
        """A method that says hi to an assistant and returns the response."""
        response = await litellm.acompletion(
            model="openai/test",
            messages=[message],
            tools=tools,
            base_url="http://localhost:8888/v1",
            api_key="default",
        )
        assert isinstance(response, litellm.types.utils.ModelResponse)
        choice = convert_litellm_choice_to_openai(response.choices[0])
        # follow up message
        response = await litellm.acompletion(
            model="openai/test",
            messages=[
                message,
                {"role": "assistant", "content": choice.message.content},
                message,
            ],
            tools=tools,
            base_url="http://localhost:8888/v1",
            api_key="default",
        )
        assert isinstance(response, litellm.types.utils.ModelResponse)
        choice = convert_litellm_choice_to_openai(response.choices[0])
        # another call with tool_choice="required" & stream=True
        stream = await litellm.acompletion(
            model="openai/test",
            messages=[message],
            tool_choice="required",
            tools=tools,
            stream=True,
            base_url="http://localhost:8888/v1",
            api_key="default",
        )
        assert isinstance(
            stream, litellm.litellm_core_utils.streaming_handler.CustomStreamWrapper
        )
        async for _ in stream:
            pass
        # Add ART support with a couple lines of optional code
        if trajectory := art.auto_trajectory():
            trajectory.reward = 1.0
        return choice.message.content

    # Use the capture_auto_trajectory utility to capture a trajectory automatically
    trajectory = await art.capture_auto_trajectory(say_hi())
    assert trajectory.messages_and_choices == [
        message,
        Choice(**mock_response["choices"][0]),  # ty:ignore[invalid-argument-type, not-subscriptable]
        message,
        Choice(**mock_response["choices"][0]),  # ty:ignore[invalid-argument-type, not-subscriptable]
    ]
    assert trajectory.additional_histories[0].messages_and_choices == [
        message,
        mock_stream_choice,
    ]
    assert trajectory.additional_histories[0].tools == tools
