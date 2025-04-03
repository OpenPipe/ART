import os

# Set UVICORN logging config path constant
UVICORN_LOGGING_CONFIG_PATH = __file__.replace(
    "__init__.py", "uvicorn_logging_config.json"
)

# Set VLLM logging config path environment variable
# os.environ["VLLM_LOGGING_CONFIG_PATH"] = __file__.replace(
#     "__init__.py", "vllm_logging_config.json"
# )

# Create logs directory if it doesn't exist
# os.makedirs("./logs", exist_ok=True)

# Empty vllm.log file if it exists
vllm_log_path = "./logs/vllm.log"
if os.path.exists(vllm_log_path):
    open(vllm_log_path, "w").close()

# Import peft (and transformers by extension) before unsloth to enable sleep mode
if os.environ.get("IMPORT_PEFT", "0") == "1":
    import peft  # type: ignore

# Import unsloth before transformers, peft, and trl to maximize Unsloth optimizations
# NOTE: If we import peft before unsloth to enable sleep mode, a warning will be shown
if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
    import unsloth  # type: ignore

from .api import API
from .gather_trajectories import gather_trajectories
from .local import LocalAPI
from .model import Model
from .types import Messages, MessagesAndChoices, ToolCall, Tools, Trajectory, TuneConfig
from .unsloth import UnslothAPI
from .utils import retry

__all__ = [
    "API",
    "gather_trajectories",
    "LocalAPI",
    "Messages",
    "MessagesAndChoices",
    "Model",
    "ToolCall",
    "Tools",
    "Trajectory",
    "TuneConfig",
    "UnslothAPI",
    "retry",
]
