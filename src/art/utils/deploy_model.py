"""
DEPRECATED: This module is deprecated. Import from art.utils.deployment instead.

This file re-exports from the new location for backwards compatibility.
"""

# Re-export everything from the new deployment module
from art.utils.deployment import (
    # New API
    DeploymentConfig,
    DeploymentResult,
    # Legacy API
    LoRADeploymentJob,
    LoRADeploymentProvider,
    Provider,
    TogetherDeploymentConfig,
    WandbDeploymentConfig,
    deploy_model,
    deploy_wandb,
)

# Also export these for any code that imports them directly
from art.utils.deployment.together import (
    TOGETHER_SUPPORTED_BASE_MODELS,
    TogetherJobStatus,
)
from art.utils.deployment.wandb import (
    WANDB_SUPPORTED_BASE_MODELS,
)

# Keep these imports for any code that uses them
from art.utils.get_model_step import get_model_step
from art.utils.output_dirs import get_default_art_path
from art.utils.s3 import archive_and_presign_step_url, pull_model_from_s3

__all__ = [
    # New API
    "DeploymentConfig",
    "DeploymentResult",
    "Provider",
    "TogetherDeploymentConfig",
    "WandbDeploymentConfig",
    "deploy_model",
    "deploy_wandb",
    # Legacy API
    "LoRADeploymentJob",
    "LoRADeploymentProvider",
    # Constants
    "TOGETHER_SUPPORTED_BASE_MODELS",
    "WANDB_SUPPORTED_BASE_MODELS",
    "TogetherJobStatus",
    # Utilities (for backwards compat)
    "get_model_step",
    "get_default_art_path",
    "archive_and_presign_step_url",
    "pull_model_from_s3",
]
