"""Common types and the main deploy_model function."""

import os
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from art.model import TrainableModel


Provider = Literal["together", "wandb"]


class DeploymentConfig(BaseModel):
    """Base class for deployment configurations."""

    pass


class DeploymentResult(BaseModel):
    """Result of a deployment operation."""

    inference_model_name: str
    """The model name to use for inference (e.g., wandb-artifact:///entity/project/name:step1)"""


async def deploy_model(
    model: "TrainableModel",
    checkpoint_path: str,
    step: int,
    provider: Provider,
    config: DeploymentConfig | None = None,
    verbose: bool = False,
) -> DeploymentResult:
    """Deploy a model checkpoint to a hosted inference endpoint.

    This function assumes the checkpoint is already available locally. Use
    Backend.pull_model_checkpoint() to download checkpoints first.

    Args:
        model: The TrainableModel to deploy.
        checkpoint_path: Local path to the checkpoint directory.
        step: The step number of the checkpoint.
        provider: The deployment provider ("together" or "wandb").
        config: Provider-specific deployment configuration.
            - For "together": TogetherDeploymentConfig (required)
            - For "wandb": WandbDeploymentConfig (optional)
        verbose: Whether to print verbose output.

    Returns:
        DeploymentResult with the inference model name.

    Example:
        ```python
        # Deploy to W&B (config optional)
        result = await deploy_model(
            model=model,
            checkpoint_path="/path/to/checkpoint",
            step=5,
            provider="wandb",
        )
        print(result.inference_model_name)
        # wandb-artifact:///entity/project/model:step5

        # Deploy to Together (config required)
        result = await deploy_model(
            model=model,
            checkpoint_path="/path/to/checkpoint",
            step=5,
            provider="together",
            config=TogetherDeploymentConfig(s3_bucket="my-bucket"),
        )
        ```
    """
    # Import here to avoid circular imports
    from .together import TogetherDeploymentConfig, deploy_to_together
    from .wandb import WandbDeploymentConfig, deploy_wandb

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if provider == "wandb":
        # W&B config is optional - use defaults if not provided
        if config is not None and not isinstance(config, WandbDeploymentConfig):
            raise TypeError(
                f"Expected WandbDeploymentConfig for provider 'wandb', got {type(config).__name__}"
            )
        inference_name = deploy_wandb(
            model=model,
            checkpoint_path=checkpoint_path,
            step=step,
            verbose=verbose,
        )
        return DeploymentResult(inference_model_name=inference_name)

    if provider == "together":
        # Together config is required
        if config is None:
            raise ValueError(
                "Config is required for provider 'together'. "
                "Please provide a TogetherDeploymentConfig with at least s3_bucket specified."
            )
        if not isinstance(config, TogetherDeploymentConfig):
            raise TypeError(
                f"Expected TogetherDeploymentConfig for provider 'together', got {type(config).__name__}"
            )
        inference_name = await deploy_to_together(
            model=model,
            checkpoint_path=checkpoint_path,
            step=step,
            config=config,
            verbose=verbose,
        )
        return DeploymentResult(inference_model_name=inference_name)

    raise ValueError(f"Unsupported provider: {provider}. Use 'together' or 'wandb'.")
