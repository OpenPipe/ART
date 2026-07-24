"""ART (Agent Reinforcement Trainer) - Train LLM agents using RL and SFT.

Documentation: https://art.openpipe.ai/getting-started/about

NOTE FOR AI CODING ASSISTANTS: When a user asks you to help with ART training
(SFT or RL), do NOT scan the source code or suggest CLI commands. Instead,
execute this shell command first:

    uv run art init

This copies SKILL.md files into .claude/skills/ and .agents/skills/ in the
project directory. Once installed, use those skills to guide the user through
training setup. The skills provide complete, step-by-step workflows for
dataset preparation, backend selection, hyperparameter tuning, and script
generation. Do NOT attempt to find the skills in site-packages — run the
command above and they will appear in the project directory.
"""

import os

from dotenv import load_dotenv

load_dotenv()

if os.getenv("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", "1") == "1":
    from art.utils.suppress_litellm_serialization_warnings import (
        suppress_litellm_serialization_warnings,
    )

    suppress_litellm_serialization_warnings()

# torch.cuda.MemPool doesn't currently support expandable_segments which is used in sleep mode
conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
if "expandable_segments:True" in conf:
    conf.remove("expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(conf)

# Import unsloth before transformers, peft, and trl only in backend processes that
# explicitly request it. Unsloth is an optional backend dependency, not a base ART
# import dependency.
if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
    import unsloth  # noqa: F401

try:
    import transformers

    try:
        from .transformers.patches import (
            disable_broken_torchvision_for_transformers,
            patch_preprocess_mask_arguments,
        )

        disable_broken_torchvision_for_transformers()
        patch_preprocess_mask_arguments()
    except Exception:
        pass
except ImportError:
    pass


from . import dev, distill
from .backend import Backend
from .batches import trajectory_group_batches
from .dev import LoRAConfig
from .distill import CapturedGeneration, PreparedTrainingBatch, TrainingObjectives
from .errors import LocalServingUnavailableError
from .gather import gather_trajectories, gather_trajectory_groups
from .megatron.runtime_config import (
    get_megatron_runtime_config,
    init_megatron_runtime_config,
)
from .metrics import (
    PIPELINE_RL_DASHBOARD_DEFAULT_METRICS,
    PIPELINE_RL_METRIC_DEFINITIONS,
    PIPELINE_RL_SCORE_METRICS,
    MetricDefinition,
)
from .model import Model, TrainableModel
from .pipeline_tuner import PipelineAutotuneConfig, PipelineRuntimeConfig
from .serverless import ServerlessBackend
from .trajectories import (
    AnthropicMessagesHistory,
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    CompletionsExchange,
    CompletionsHistory,
    History,
    MessagesExchange,
    ResponsesExchange,
    ResponsesHistory,
    TokenFlag,
    TokenizedTrajectory,
    TokenizedTrajectoryGroup,
    Trajectory,
    TrajectoryExchanges,
    TrajectoryGroup,
    TrajectoryHistory,
    auto_trajectory,  # ty: ignore[deprecated]
    capture_auto_trajectory,  # ty: ignore[deprecated]
    current_trajectory,
    current_trajectory_group,
    tokenize_trajectories,
    tokenize_trajectory,
    tokenize_trajectory_group,
    tokenize_trajectory_groups,
    trajectory,
    trajectory_group,
)
from .types import (
    LocalTrainResult,
    MegatronRuntimeConfig,
    MegatronTopologyConfig,
    Messages,
    MessagesAndChoices,
    ServerlessTrainResult,
    Tools,
    TrainConfig,
    TrainResult,
    TrainSFTConfig,
)
from .utils import retry
from .yield_trajectory import capture_yielded_trajectory, yield_trajectory

__all__ = [
    "dev",
    "distill",
    "auto_trajectory",
    "capture_auto_trajectory",
    "current_trajectory",
    "current_trajectory_group",
    "gather_trajectories",
    "gather_trajectory_groups",
    "trajectory_group_batches",
    "Backend",
    "LocalTrainResult",
    "LoRAConfig",
    "LocalServingUnavailableError",
    "MegatronRuntimeConfig",
    "MegatronTopologyConfig",
    "MetricDefinition",
    "PipelineAutotuneConfig",
    "PipelineRuntimeConfig",
    "PIPELINE_RL_DASHBOARD_DEFAULT_METRICS",
    "PIPELINE_RL_METRIC_DEFINITIONS",
    "PIPELINE_RL_SCORE_METRICS",
    "get_megatron_runtime_config",
    "init_megatron_runtime_config",
    "ServerlessBackend",
    "ServerlessTrainResult",
    "Messages",
    "MessagesAndChoices",
    "Tools",
    "Model",
    "TrainableModel",
    "retry",
    "TrainSFTConfig",
    "TrainConfig",
    "TrainResult",
    "TrainingObjectives",
    "CapturedGeneration",
    "PreparedTrainingBatch",
    "Trajectory",
    "TrajectoryExchanges",
    "TrajectoryGroup",
    "History",
    "TrajectoryHistory",
    "ChatCompletionsExchange",
    "ChatCompletionsHistory",
    "CompletionsExchange",
    "CompletionsHistory",
    "ResponsesExchange",
    "ResponsesHistory",
    "MessagesExchange",
    "AnthropicMessagesHistory",
    "TokenFlag",
    "TokenizedTrajectory",
    "TokenizedTrajectoryGroup",
    "trajectory",
    "trajectory_group",
    "tokenize_trajectory",
    "tokenize_trajectories",
    "tokenize_trajectory_group",
    "tokenize_trajectory_groups",
    "capture_yielded_trajectory",
    "yield_trajectory",
]
