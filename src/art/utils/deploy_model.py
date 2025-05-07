import json
import os
import aiohttp
from art.model import TrainableModel


def init_together_session() -> aiohttp.ClientSession:
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


async def deploy_together(
    model: TrainableModel,
    presigned_url: str,
    step: int,
    verbose: bool = False,
) -> None:
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
                "model_name": f"{model.project}-{model.name}-{step}",
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
    async with init_together_session() as session:
        async with session.get(
            url=f"https://api.together.xyz/v1/jobs/{job_id}"
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if verbose:
                print(f"Job status: {json.dumps(result, indent=4)}")
            return result
