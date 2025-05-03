import pytest
from pydantic import BaseModel, ValidationError
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pytest_asyncio  # Import pytest_asyncio

from train_roflbot.tools import Tool

# Load environment variables from .env file
load_dotenv()

# Check if OPENAI_API_KEY is available
openai_api_key = os.getenv("OPENAI_API_KEY")


class EchoParams(BaseModel):
    message: str


class EchoTool(Tool[EchoParams]):
    name = "echo"
    description = "Echoes the input message."

    def _implementation(self, args) -> str:
        # Note: type checkers will now correctly infer that
        # args is of type EchoParams, so you get autocompletion
        # and type checking for args.message
        return args.message


def test_valid_tool_subclass():
    """Tests that a valid subclass like EchoTool works correctly."""
    assert EchoTool.name == "echo"
    assert EchoTool.description == "Echoes the input message."
    assert EchoTool.parameters is EchoParams
    # Instantiate the tool
    tool_instance = EchoTool()
    assert tool_instance.name == "echo"
    assert tool_instance.description == "Echoes the input message."


def test_subclass_without_params_type_raises_error():
    """Tests that subclassing Tool directly without specifying ParamsType raises TypeError."""
    with pytest.raises(TypeError, match="must inherit from Tool[YourParamsModel]"):

        class BadToolDirect(Tool):
            name = "bad"
            description = "bad"

            def _implementation(self, args):
                return "bad"


def test_subclass_with_non_basemodel_params_type_raises_error():
    """Tests that subclassing Tool with a non-BaseModel type parameter raises TypeError."""

    class NotABaseModel:
        pass

    with pytest.raises(TypeError, match="not a Pydantic BaseModel"):

        class BadToolNonModel(Tool[NotABaseModel]):
            name = "bad_non_model"
            description = "bad non model"

            def _implementation(self, args):
                return "bad"


# --- Test call() method ---


@pytest.fixture
def echo_tool_instance() -> EchoTool:
    """Fixture to provide an instance of EchoTool."""
    return EchoTool()


def test_call_with_correct_args(echo_tool_instance: EchoTool):
    """Tests calling the tool with valid arguments."""
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        function=Function(name="echo", arguments='{"message": "Hello, world!"}'),
        type="function",
    )
    result = echo_tool_instance.call(tool_call)
    assert result == "Hello, world!"


def test_call_with_wrong_function_name_raises_error(echo_tool_instance: EchoTool):
    """Tests calling the tool with a mismatched function name raises ValueError."""
    tool_call = ChatCompletionMessageToolCall(
        id="call_456",
        function=Function(
            name="not_echo",  # Wrong name
            arguments='{"message": "Irrelevant"}',
        ),
        type="function",
    )
    with pytest.raises(ValueError, match="Tool echo called with wrong name: not_echo"):
        echo_tool_instance.call(tool_call)


def test_call_with_invalid_json_args_raises_error(echo_tool_instance: EchoTool):
    """Tests calling the tool with invalid JSON arguments raises Pydantic ValidationError."""
    tool_call = ChatCompletionMessageToolCall(
        id="call_789",
        function=Function(
            name="echo",
            arguments='{"msg": "Hello"}',  # Wrong argument name 'msg' instead of 'message'
        ),
        type="function",
    )
    # pydantic.ValidationError is raised by model_validate_json
    with pytest.raises(ValidationError):
        echo_tool_instance.call(tool_call)


def test_call_with_incorrect_arg_type_raises_error(echo_tool_instance: EchoTool):
    """Tests calling the tool with correct arg name but wrong type."""
    tool_call = ChatCompletionMessageToolCall(
        id="call_abc",
        function=Function(
            name="echo",
            arguments='{"message": 123}',  # Incorrect type (int instead of str)
        ),
        type="function",
    )
    with pytest.raises(ValidationError):
        echo_tool_instance.call(tool_call)


# --- Test schema() method ---


def test_schema_generation(echo_tool_instance: EchoTool):
    """Tests the generated schema structure."""
    schema = echo_tool_instance.schema()
    assert schema["type"] == "function"
    function_def = schema["function"]
    assert function_def["name"] == "echo"
    assert function_def["description"] == "Echoes the input message."

    # Check parameters schema
    params_schema = function_def["parameters"]
    assert params_schema["type"] == "object"
    assert "message" in params_schema["properties"]
    assert params_schema["properties"]["message"]["type"] == "string"
    assert params_schema["required"] == ["message"]
    assert (
        params_schema["title"] == EchoParams.__name__
    )  # pydantic adds title by default


# --- OpenAI Integration Tests ---


@pytest_asyncio.fixture  # Use pytest_asyncio fixture
async def async_openai_client() -> AsyncOpenAI:
    """Fixture to provide an async OpenAI client instance."""
    return AsyncOpenAI()


@pytest.mark.asyncio  # Mark test as asyncio
async def test_openai_integration_echo_tool(
    echo_tool_instance: EchoTool, async_openai_client: AsyncOpenAI
):
    """Tests full loop: OpenAI call -> Tool Call -> Tool Execution."""
    tool_schema = echo_tool_instance.schema()
    message_content = "Please echo the message 'Testing 1 2 3'."

    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": message_content}],
            tools=[tool_schema],
            tool_choice="required",
        )
    except Exception as e:
        pytest.fail(f"OpenAI API call failed: {e}")

    # Check response
    message = response.choices[0].message
    assert message.tool_calls is not None, "OpenAI did not request a tool call."
    assert len(message.tool_calls) == 1, "Expected exactly one tool call."

    # Get the tool call object
    tool_call = message.tool_calls[0]

    result = echo_tool_instance.call(tool_call)

    assert result == "Testing 1 2 3"
