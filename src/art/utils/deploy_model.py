import asyncio
import json
import os
import time
import aiohttp
from art.model import TrainableModel


def init_together_session() -> aiohttp.ClientSession:
    """
    Initializes a session for interacting with Together.
    """
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


def model_checkpoint_id(model: TrainableModel, step: int) -> str:
    """
    Generates a unique ID for a model checkpoint.
    """
    return f"{model.project}-{model.name}-{step}"


async def previously_deployed_model_id(model: TrainableModel) -> str | None:
    """
    Checks if a model with the same name has already been deployed to Together.
    If so, returns the model ID.
    """
    async with init_together_session() as session:
        async with session.get(url="https://api.together.xyz/v1/models") as response:
            response.raise_for_status()
            result = await response.json()

            # find a model with an "id" that contains the model.name
            for deployed_model in result:
                if model.name in deployed_model["id"]:
                    return deployed_model["id"]

            return None


async def deploy_together(
    model: TrainableModel,
    presigned_url: str,
    step: int,
    verbose: bool = False,
) -> None:
    """
    Deploys a model to Together.
    """
    async with init_together_session() as session:
        session.headers.update(
            {
                "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
                "Content-Type": "application/json",
            }
        )

        async with session.post(
            url="https://api.together.xyz/v1/models",
            json={
                "model_name": model_checkpoint_id(model=model, step=step),
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


async def check_together_job_status(job_id: str, verbose: bool = False) -> None:
    """
    Checks the status of a model deployment job in Together.
    """
    async with init_together_session() as session:
        async with session.get(
            url=f"https://api.together.xyz/v1/jobs/{job_id}"
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if verbose:
                print(f"Job status: {json.dumps(result, indent=4)}")
            return result


async def wait_for_together_job(job_id: str, verbose: bool = False) -> dict:
    """
    Waits for a model deployment job to complete in Together.

    Checks the status every 15 seconds for 5 minutes.
    """
    print(f"checking status of job {job_id} every 15 seconds for 5 minutes")
    start_time = time.time()
    max_time = start_time + 300
    while time.time() < max_time:
        job_status = await check_together_job_status(job_id, verbose)
        print(f"job status: {job_status['status']}")
        if job_status["status"] == "Complete":
            return job_status
        await asyncio.sleep(15)
