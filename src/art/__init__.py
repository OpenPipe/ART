import os

# Import peft (and transformers by extension) before unsloth to enable sleep mode
if os.environ.get("IMPORT_PEFT", "0") == "1":
    import peft  # type: ignore

# Import unsloth before transformers, peft, and trl to maximize Unsloth optimizations
# NOTE: If we import peft before unsloth to enable sleep mode, a warning will be shown
if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
    import unsloth  # type: ignore

from .gather_trajectories import gather_trajectories
from .model import Model
from .types import Messages, MessagesAndChoices, ToolCall, Tools, Trajectory, TuneConfig
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
    "ToolCall",
    "Tools",
    "Trajectory",
    "TuneConfig",
    "UnslothAPI",
]
