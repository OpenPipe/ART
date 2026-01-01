import os

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
    import unsloth  # type: ignore # noqa: F401

try:
    import transformers  # type: ignore

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
from .types import Messages, MessagesAndChoices, Tools, TrainConfig
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
    "ServerlessBackend",
    "Messages",
    "MessagesAndChoices",
    "Tools",
    "Model",
    "TrainableModel",
    "retry",
    "TrainConfig",
    "Trajectory",
    "TrajectoryGroup",
    "capture_yielded_trajectory",
    "yield_trajectory",
]
