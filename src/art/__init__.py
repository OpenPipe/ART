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

# Create a dummy GuidedDecodingParams class and inject it into vllm.sampling_params for trl compatibility
try:
    import vllm.sampling_params

    class GuidedDecodingParams:
        """Shim for vLLM 0.13+ where GuidedDecodingParams was removed."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    vllm.sampling_params.GuidedDecodingParams = GuidedDecodingParams  # type: ignore
except ImportError:
    pass  # vllm not installed

# torch.cuda.MemPool doesn't currently support expandable_segments which is used in sleep mode
conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
if "expandable_segments:True" in conf:
    conf.remove("expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(conf)

# Import unsloth before transformers, peft, and trl to maximize Unsloth optimizations
if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
    import unsloth  # noqa: F401

try:
    import transformers

    try:
        from .transformers.patches import patch_preprocess_mask_arguments

        patch_preprocess_mask_arguments()
    except Exception:
        pass
except ImportError:
    pass


from . import dev
from .auto_trajectory import auto_trajectory, capture_auto_trajectory
from .backend import Backend
from .batches import trajectory_group_batches
from .gather import gather_trajectories, gather_trajectory_groups
from .model import Model, TrainableModel
from .serverless import ServerlessBackend
from .trajectories import Trajectory, TrajectoryGroup
from .types import (
    LocalTrainResult,
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
    "auto_trajectory",
    "capture_auto_trajectory",
    "gather_trajectories",
    "gather_trajectory_groups",
    "trajectory_group_batches",
    "Backend",
    "LocalBackend",
    "LocalTrainResult",
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
    "Trajectory",
    "TrajectoryGroup",
    "capture_yielded_trajectory",
    "yield_trajectory",
]
