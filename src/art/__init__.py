import os

# Set UVICORN logging config path constant
UVICORN_LOGGING_CONFIG_PATH = __file__.replace(
    "__init__.py", "uvicorn_logging_config.json"
)

# Set VLLM logging config path environment variable
os.environ["VLLM_LOGGING_CONFIG_PATH"] = __file__.replace(
    "__init__.py", "vllm_logging_config.json"
)

# Create logs directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)

# Empty vllm.log file if it exists
vllm_log_path = "./logs/vllm.log"
if os.path.exists(vllm_log_path):
    open(vllm_log_path, "w").close()

# Import unsloth before transformers
# try:
#     import unsloth  # type: ignore
# except ImportError:
#     # unsloth may not be available on all platforms
#     pass

from .api import API
from .gather_groups import gather_groups
from .gather_trajectories import gather_trajectories
from .local import LocalAPI
from .model import Model
from .types import Messages, MessagesAndChoices, ToolCall, Tools, Trajectory, TuneConfig
from .unsloth import UnslothAPI
from .utils import retry

__all__ = [
    "API",
    "gather_groups",
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
