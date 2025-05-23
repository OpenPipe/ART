import asyncio
import art
import os
import argparse

from train import PROJECT_NAME, BASE_MODEL, MODEL_NAME
from train import CLUSTER_NAME
from rollout import ModelConfig, rollout, TicTacToeScenario


async def deploy_step():
    parser = argparse.ArgumentParser(description="Train a model to play Tic-Tac-Toe")
    parser.add_argument(
        "--backend",
        choices=["skypilot", "local"],
        default="local",
        help="Backend to use for training (default: local)",
    )
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

    # Avoid import unnecessary backend dependencies
    if args.backend == "skypilot":
        from art.skypilot.backend import SkyPilotBackend

        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=CLUSTER_NAME,
            art_version=".",
            env_path=".env",
            gpu="H100",
        )
    else:
        from art.local.backend import LocalBackend

        backend = LocalBackend()

    deployment_result = await backend._experimental_deploy(
        deploy_to="together",
        model=model,
        step=args.step,
        verbose=True,
        pull_s3=True,
        wait_for_completion=True,
    )
    if deployment_result.status == "Failed":
        raise Exception(f"Deployment failed: {deployment_result.failure_reason}")

    deployed_model_name = deployment_result.model_name

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
