from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
from typing import Iterable, Literal

BaseModel = Literal[
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
ToolCall = ChatCompletionMessageToolCallParam
Tools = Iterable[ChatCompletionToolParam]


class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}
    tools: Tools | None = None


class TuneConfig(pydantic.BaseModel):
    learning_rate_multiplier: float = 1.0
    beta: float = 0.0

    # GRPO params
    clip_epsilon: float = 0.2
    kl_coef: float = pydantic.Field(
        default=0.0,
        deprecated="`kl_coef` is deprecated, use `beta` instead.",
    )

    # Optimizer params
    lr: float = pydantic.Field(
        default=5e-6,
        deprecated="`lr` is deprecated, use `learning_rate_multiplier` instead.",
    )
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1

    # Tensor packing params
    sequence_length: int | None = pydantic.Field(
        default=None,
        deprecated="Sequence length is now automatically determined from trajectory data.",
    )

    # Logging params
    plot_tensors: bool = False
    verbosity: "Verbosity" = 1


Verbosity = Literal[0, 1, 2]
