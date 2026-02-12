import argparse
import asyncio
import os

from rollout import ModelConfig, TicTacToeScenario, rollout
from train import BASE_MODEL, MODEL_NAME, PROJECT_NAME

import art
from art.utils.deployment import TogetherDeploymentConfig, deploy_model


async def deploy_step():
    parser = argparse.ArgumentParser(description="Train a model to play Tic-Tac-Toe")
    parser.add_argument(
        "--step",
        type=int,
        help="Step to deploy",
    )
    args = parser.parse_args()

    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
    )

    from art.local.backend import LocalBackend

    backend = LocalBackend()

    # Pull checkpoint from S3
    checkpoint_path = await backend._experimental_pull_model_checkpoint(
        model,
        step=args.step,
        s3_bucket=os.environ.get("BACKUP_BUCKET"),
        verbose=True,
    )

    # Deploy to Together
    deployment_result = await deploy_model(
        model=model,
        checkpoint_path=checkpoint_path,
        step=args.step,
        provider="together",
        config=TogetherDeploymentConfig(
            s3_bucket=os.environ.get("BACKUP_BUCKET"),
            wait_for_completion=True,
        ),
        verbose=True,
    )

    deployed_model_name = deployment_result.inference_model_name

    lora_model = art.Model(
        name=deployed_model_name,
        project="tic-tac-toe",
        inference_api_key=os.environ["TOGETHER_API_KEY"],
        inference_base_url="https://api.together.xyz/v1",
        inference_model_name=deployed_model_name,
        config=ModelConfig(),
    )

    print("Starting a rollout using the deployed model!")
    x_trajectory, y_trajectory = await rollout(
        x_model=lora_model,
        y_model=lora_model,
        scenario=TicTacToeScenario(step=0, split="val"),
    )

    print(x_trajectory)
    print(y_trajectory)


if __name__ == "__main__":
    asyncio.run(deploy_step())
