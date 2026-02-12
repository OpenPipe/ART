"""W&B deployment functionality."""

import os
from typing import TYPE_CHECKING

from art.errors import UnsupportedBaseModelDeploymentError

from .common import DeploymentConfig

if TYPE_CHECKING:
    from art.model import TrainableModel


class WandbDeploymentConfig(DeploymentConfig):
    """Configuration for deploying to W&B.

    Supported base models:
    - meta-llama/Llama-3.1-8B-Instruct
    - meta-llama/Llama-3.1-70B-Instruct
    - OpenPipe/Qwen3-14B-Instruct
    - Qwen/Qwen2.5-14B-Instruct
    """

    pass


WANDB_SUPPORTED_BASE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "OpenPipe/Qwen3-14B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


def deploy_wandb(
    model: "TrainableModel",
    checkpoint_path: str,
    step: int,
    verbose: bool = False,
) -> str:
    """Deploy a model to W&B by uploading a LoRA artifact.

    Args:
        model: The TrainableModel to deploy.
        checkpoint_path: Local path to the checkpoint directory.
        step: The step number of the checkpoint.
        verbose: Whether to print verbose output.

    Returns:
        The model name for inference: wandb-artifact:///{entity}/{project}/{name}:step{step}
    """
    import wandb

    if model.base_model not in WANDB_SUPPORTED_BASE_MODELS:
        raise UnsupportedBaseModelDeploymentError(
            message=f"Base model {model.base_model} is not supported for serverless LoRA deployment by W&B. Supported models: {WANDB_SUPPORTED_BASE_MODELS}"
        )

    if "WANDB_API_KEY" not in os.environ:
        raise ValueError("WANDB_API_KEY is not set, cannot deploy LoRA to W&B")

    # Get the user's default entity from W&B if not set
    if model.entity is None:
        api = wandb.Api()
        model.entity = api.default_entity

    if verbose:
        print(f"Uploading checkpoint from {checkpoint_path} to W&B...")

    run = wandb.init(
        name=model.name + " (deployment)",
        entity=model.entity,
        project=model.project,
        settings=wandb.Settings(api_key=os.environ["WANDB_API_KEY"]),
    )
    try:
        artifact = wandb.Artifact(
            model.name,
            type="lora",
            metadata={"wandb.base_model": model.base_model},
            storage_region="coreweave-us",
        )
        artifact.add_dir(checkpoint_path)
        artifact = run.log_artifact(artifact, aliases=[f"step{step}", "latest"])
        try:
            artifact = artifact.wait()
        except ValueError as e:
            if "Unable to fetch artifact with id" in str(e):
                if verbose:
                    print(f"Warning: {e}")
            else:
                raise e
    finally:
        run.finish()

    inference_name = (
        f"wandb-artifact:///{model.entity}/{model.project}/{model.name}:step{step}"
    )
    if verbose:
        print(f"Successfully deployed to W&B. Inference model name: {inference_name}")

    return inference_name
