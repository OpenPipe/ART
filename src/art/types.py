from dataclasses import dataclass, field
from typing import Annotated, Literal

from openai.types.chat.chat_completion import Choice as Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
from pydantic import SkipValidation
from typing_extensions import TypedDict

Message = Annotated[ChatCompletionMessageParam, SkipValidation]
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]


def _visible_device_count() -> int:
    try:
        import torch
    except Exception:
        return 1
    return max(int(torch.cuda.device_count()), 1)


class TrainConfig(pydantic.BaseModel):
    learning_rate: float = 5e-6
    kl_penalty_coef: float = 0.0
    kl_penalty_source: Literal["current_learner", "sample"] = "current_learner"
    grad_accumulation_sequences: int | None = pydantic.Field(default=None, ge=1)


class MegatronTopologyConfig(pydantic.BaseModel):
    tp: int = pydantic.Field(default=1, ge=1)
    cp: int = pydantic.Field(default_factory=_visible_device_count, ge=1)
    ep: int = pydantic.Field(default_factory=_visible_device_count, ge=1)
    pp: int = pydantic.Field(default=1, ge=1)
    vpp: int | None = pydantic.Field(default=None, ge=1)
    etp: int = pydantic.Field(default=1, ge=1)


class MegatronRuntimeConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    topology: MegatronTopologyConfig
    packed_sequence_length: int = pydantic.Field(ge=1)


class TrainSFTConfig(pydantic.BaseModel):
    learning_rate: float | list[float] = 5e-5  # Single value or per-batch list
    batch_size: int | Literal["auto"] = "auto"
    assistant_turns: Literal["all", "last"] = "all"


class SFTMetricLoggingConfig(TypedDict, total=False):
    enabled: bool
    target_training_step: int


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
