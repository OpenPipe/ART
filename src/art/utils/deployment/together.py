"""Together deployment functionality."""

import asyncio
from enum import Enum
import json
import os
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel

from art.errors import (
    LoRADeploymentTimedOutError,
    UnsupportedBaseModelDeploymentError,
)
from art.utils.s3 import archive_and_presign_step_url

from .common import DeploymentConfig

if TYPE_CHECKING:
    from art.model import TrainableModel


class TogetherDeploymentConfig(DeploymentConfig):
    """Configuration for deploying to Together.

    See supported base models: https://docs.together.ai/docs/lora-inference#supported-base-models

    Attributes:
        s3_bucket: S3 bucket to upload the checkpoint archive to (for presigned URL).
        prefix: S3 prefix for the upload.
        wait_for_completion: Whether to wait for deployment to complete (default: True).
    """

    s3_bucket: str | None = None
    prefix: str | None = None
    wait_for_completion: bool = True


class TogetherJobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETE = "Complete"
    FAILED = "Failed"


class TogetherJob(BaseModel):
    status: TogetherJobStatus
    job_id: str
    model_name: str
    failure_reason: str | None


TOGETHER_SUPPORTED_BASE_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]


def _init_session() -> aiohttp.ClientSession:
    """Initializes a session for interacting with Together."""
    if "TOGETHER_API_KEY" not in os.environ:
        raise ValueError("TOGETHER_API_KEY is not set, cannot deploy LoRA to Together")
    session = aiohttp.ClientSession()
    session.headers.update(
        {
            "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
            "Content-Type": "application/json",
        }
    )
    return session


def _model_checkpoint_id(model: "TrainableModel", step: int) -> str:
    """Generates a unique ID for a model checkpoint."""
    return f"{model.project}-{model.name}-{step}"


async def _upload_model(
    model: "TrainableModel",
    presigned_url: str,
    step: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Uploads a model to Together."""
    if model.base_model not in TOGETHER_SUPPORTED_BASE_MODELS:
        raise UnsupportedBaseModelDeploymentError(
            message=f"Base model {model.base_model} is not supported for serverless LoRA deployment by Together. Supported models: {TOGETHER_SUPPORTED_BASE_MODELS}"
        )

    async with _init_session() as session:
        async with session.post(
            url="https://api.together.xyz/v1/models",
            json={
                "model_name": _model_checkpoint_id(model=model, step=step),
                "model_source": presigned_url,
                "model_type": "adapter",
                "base_model": model.base_model,
                "description": f"Deployed from ART. Project: {model.project}. Model: {model.name}. Step: {step}",
            },
        ) as response:
            if response.status != 200:
                print("Error uploading to Together:", await response.text())
            response.raise_for_status()
            result = await response.json()
            if verbose:
                print(f"Successfully uploaded to Together: {result}")
            return result


def _convert_job_status(status: str, message: str | None = None) -> TogetherJobStatus:
    MODEL_ALREADY_EXISTS_ERROR_MESSAGE = "409 Client Error: Conflict for url: https://api.together.ai/api/admin/entity/Model"
    if (
        status == "Error"
        and message is not None
        and MODEL_ALREADY_EXISTS_ERROR_MESSAGE in message
    ):
        return TogetherJobStatus.COMPLETE
    if status == "Bad" or status == "Error":
        return TogetherJobStatus.FAILED
    if status == "Retry Queued":
        return TogetherJobStatus.QUEUED
    return TogetherJobStatus(status)


async def _find_existing_job_id(
    model: "TrainableModel",
    step: int,
) -> str | None:
    """Finds an existing model deployment job in Together."""
    checkpoint_id = _model_checkpoint_id(model, step)
    async with _init_session() as session:
        async with session.get(url="https://api.together.xyz/v1/jobs") as response:
            response.raise_for_status()
            result = await response.json()
            jobs = result["data"]
            jobs.sort(key=lambda x: x["updated_at"], reverse=True)
            for job in jobs:
                if checkpoint_id in job["args"]["modelName"]:
                    return job["job_id"]
            return None


async def _check_job_status(job_id: str, verbose: bool = False) -> TogetherJob:
    """Checks the status of a model deployment job in Together."""
    async with _init_session() as session:
        async with session.get(
            url=f"https://api.together.xyz/v1/jobs/{job_id}"
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if verbose:
                print(f"Job status: {json.dumps(result, indent=4)}")

            last_update = result["status_updates"][-1]
            status_body = TogetherJob(
                status=_convert_job_status(
                    result["status"], last_update.get("message")
                ),
                job_id=job_id,
                model_name=result["args"]["modelName"],
                failure_reason=result.get("failure_reason"),
            )

            if status_body.status == TogetherJobStatus.FAILED:
                status_body.failure_reason = last_update.get("message")
            return status_body


async def _wait_for_job(job_id: str, verbose: bool = False) -> TogetherJob:
    """Waits for a model deployment job to complete in Together."""
    print(f"checking status of job {job_id} every 15 seconds for 5 minutes")
    start_time = time.time()
    max_time = start_time + 300
    while time.time() < max_time:
        job_status = await _check_job_status(job_id, verbose)
        if job_status.status in (TogetherJobStatus.COMPLETE, TogetherJobStatus.FAILED):
            return job_status
        await asyncio.sleep(15)

    raise LoRADeploymentTimedOutError(
        message=f"LoRA deployment timed out after 5 minutes. Job ID: {job_id}"
    )


async def deploy_to_together(
    model: "TrainableModel",
    checkpoint_path: str,
    step: int,
    config: TogetherDeploymentConfig,
    verbose: bool = False,
) -> str:
    """Deploy a model checkpoint to Together.

    Args:
        model: The TrainableModel to deploy.
        checkpoint_path: Local path to the checkpoint directory.
        step: The step number of the checkpoint.
        config: Together deployment configuration.
        verbose: Whether to print verbose output.

    Returns:
        The inference model name.
    """
    # Archive and upload to S3 to get a presigned URL for Together
    presigned_url = await archive_and_presign_step_url(
        model_name=model.name,
        project=model.project,
        step=step,
        s3_bucket=config.s3_bucket,
        prefix=config.prefix,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
    )

    existing_job_id = await _find_existing_job_id(model, step)
    existing_job = None
    if existing_job_id is not None:
        existing_job = await _check_job_status(existing_job_id, verbose=verbose)

    if not existing_job or existing_job.status == TogetherJobStatus.FAILED:
        deployment_result = await _upload_model(
            model=model,
            presigned_url=presigned_url,
            step=step,
            verbose=verbose,
        )
        job_id = deployment_result["data"]["job_id"]
    else:
        job_id = existing_job_id
        assert job_id is not None
        print(
            f"Previous deployment for {model.name} at step {step} has status '{existing_job.status}', skipping redeployment"
        )

    if config.wait_for_completion:
        job = await _wait_for_job(job_id, verbose=verbose)
    else:
        job = await _check_job_status(job_id, verbose=verbose)

    if job.status == TogetherJobStatus.FAILED:
        raise RuntimeError(
            f"Together deployment failed for {model.name} step {step}. "
            f"Job ID: {job.job_id}. Reason: {job.failure_reason}"
        )

    return job.model_name
