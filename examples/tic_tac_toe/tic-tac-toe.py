import os
import random
import asyncio
import argparse
from dotenv import load_dotenv

import art
from rollout import rollout, TicTacToeScenario


load_dotenv()

random.seed(42)

DESTROY_AFTER_RUN = False
STEP = 36


async def main():
    parser = argparse.ArgumentParser(description="Train a model to play Tic-Tac-Toe")
    parser.add_argument(
        "--backend",
        choices=["skypilot", "local"],
        default="local",
        help="Backend to use for training (default: local)",
    )
    args = parser.parse_args()

    # Avoid import unnecessary backend dependencies
    if args.backend == "skypilot":
        from art.skypilot.backend import SkyPilotBackend

        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name="art3", art_version=".", env_path=".env", gpu="H100"
        )
    else:
        from art.local.backend import LocalBackend

        backend = LocalBackend()

    model = art.TrainableModel(
        name="llama-8b-001",
        project="tic-tac-toe",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    print("pulling from s3")
    await backend._experimental_pull_from_s3(model)

    print("registering")
    await model.register(backend)

    print("training")
    for i in range(await model.get_step(), STEP):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, TicTacToeScenario(step=i)) for _ in range(48)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
        await backend._experimental_push_to_s3(model)

    deployment_result = await backend._experimental_deploy(
        deploy_to="together",
        model=model,
        step=STEP,
        verbose=True,
        pull_s3=False,
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
    )

    print("Starting a rollout using the deployed model!")
    traj = await rollout(lora_model, TicTacToeScenario(step=0))

    print(traj)

    if DESTROY_AFTER_RUN:
        await backend.down()


if __name__ == "__main__":
    asyncio.run(main())
