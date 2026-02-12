from dataclasses import dataclass, field
from typing import Annotated, Literal

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
from pydantic import SkipValidation

Message = Annotated[ChatCompletionMessageParam, SkipValidation]
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]


class TrainConfig(pydantic.BaseModel):
    learning_rate: float = 5e-6
    beta: float = 0.0


Verbosity = Literal[0, 1, 2]


# ---------------------------------------------------------------------------
# TrainResult classes
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Base result returned from backend.train().

    Attributes:
        step: The training step after this training call completed.
        metrics: Aggregated training metrics (loss, gradient norms, etc.).
    """

    step: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class LocalTrainResult(TrainResult):
    """Result from LocalBackend.train().

    Attributes:
        step: The training step after this training call completed.
        metrics: Aggregated training metrics (loss, gradient norms, etc.).
        checkpoint_path: Path to the saved checkpoint directory, or None if
            no checkpoint was saved.
    """

    checkpoint_path: str | None = None


@dataclass
class ServerlessTrainResult(TrainResult):
    """Result from ServerlessBackend.train().

    Attributes:
        step: The training step after this training call completed.
        metrics: Aggregated training metrics (loss, gradient norms, etc.).
        artifact_name: The W&B artifact name for the checkpoint
            (e.g., "entity/project/model:step5").
    """

    artifact_name: str | None = None
