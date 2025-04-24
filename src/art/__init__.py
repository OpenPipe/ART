import os
import sys

# Import peft (and transformers by extension) before unsloth to enable sleep mode
if os.environ.get("IMPORT_PEFT", "0") == "1":
    import peft  # type: ignore

# Import unsloth before transformers, peft, and trl to maximize Unsloth optimizations
# NOTE: If we import peft before unsloth to enable sleep mode, a warning will be shown
if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
    import unsloth  # type: ignore

from . import dev
from .api import API
from .skypilot import SkypilotAPI
from .gather import gather_trajectories, gather_trajectory_groups
from .model import Model, TrainableModel
from .trajectories import Trajectory, TrajectoryGroup
from .types import Messages, MessagesAndChoices, Tools, TrainConfig
from .utils import retry

__all__ = [
    "dev",
    "gather_trajectories",
    "gather_trajectory_groups",
    "API",
    "SkypilotAPI",
    "Messages",
    "MessagesAndChoices",
    "Tools",
    "Model",
    "TrainableModel",
    "retry",
    "TrainConfig",
    "Trajectory",
    "TrajectoryGroup",
]

# Avoid importing LocalAPI on client machines
if sys.platform.startswith("linux"):
    from .local import LocalAPI

    __all__.append("LocalAPI")
    setattr(sys.modules[__name__], "LocalAPI", LocalAPI)
