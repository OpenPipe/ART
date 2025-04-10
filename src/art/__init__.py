import os

from .trajectories import Trajectory

# Import peft (and transformers by extension) before unsloth to enable sleep mode
if os.environ.get("IMPORT_PEFT", "0") == "1":
    import peft  # type: ignore

# Import unsloth before transformers, peft, and trl to maximize Unsloth optimizations
# NOTE: If we import peft before unsloth to enable sleep mode, a warning will be shown
if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
    import unsloth  # type: ignore

from .gather_trajectories import gather_trajectories
from .model import Model
from .trajectories import Trajectory, TrajectoryGroup
from .types import Messages, MessagesAndChoices, TrainConfig
from .local import LocalAPI
from .utils import retry

UnslothAPI = LocalAPI

__all__ = [
    "gather_trajectories",
    "LocalAPI",
    "Messages",
    "MessagesAndChoices",
    "Model",
    "retry",
    "TrainConfig",
    "Trajectory",
    "TrajectoryGroup",
    "UnslothAPI",
]
