import os
import random
import asyncio
import argparse
from dotenv import load_dotenv

import art
from art.trajectories import TrajectoryGroup
from gather_trajectory_groups_by_index import gather_trajectory_groups_by_index
from rollout import rollout, TicTacToeScenario


load_dotenv()

random.seed(42)

PULL_FROM_S3 = False
STEP = 100
DEPLOY_MODEL = False
GENERATE_BENCHMARKS = False
DESTROY_AFTER_RUN = False

CLUSTER_NAME = "art4"
MODEL_NAME = "llama-8b-self-play-010"

parser = argparse.ArgumentParser(description="Train a model to play Tic-Tac-Toe")
parser.add_argument(
    "--backend",
    choices=["skypilot", "local"],
    default="local",
    help="Backend to use for training (default: local)",
)
parser.add_argument(
    "--restart",
    action="store_true",
    help="Restart the ART server",
)
args = parser.parse_args()


async def main():
    # Avoid import unnecessary backend dependencies
    if args.backend == "skypilot":
        from art.skypilot.backend import SkyPilotBackend

        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=CLUSTER_NAME,
            art_version=".",
            env_path=".env",
            gpu="H100",
            force_restart=args.restart,
        )
    else:
        from art.local.backend import LocalBackend

        backend = LocalBackend()

    model = art.TrainableModel(
        name=MODEL_NAME,
        project="tic-tac-toe",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    o4_mini = art.Model(
        name="o4-mini",
        project="tic-tac-toe",
        inference_model_name="o4-mini",
        inference_api_key=os.environ["OPENAI_API_KEY"],
        inference_base_url="https://api.openai.com/v1",
    )

    if PULL_FROM_S3:
        print("pulling from s3")
        await backend._experimental_pull_from_s3(model)

    print("registering")
    await model.register(backend)
    await o4_mini.register(backend)

    print("commencing run")
    for i in range(await model.get_step(), STEP):
        (
            x_trajectory_group,
            y_trajectory_group,
        ) = await gather_trajectory_groups_by_index(
            [
                rollout(
                    x_model=model,
                    y_model=model,
                    scenario=TicTacToeScenario(step=i, split="train"),
                )
                for _ in range(96)
            ],
            pbar_desc="gather",
            trajectories_per_rollout=2,
        )

        if i % 10 == 0 or True:
            print("gathering val")
            x_val, y_val = await gather_trajectory_groups_by_index(
                [
                    rollout(
                        x_model=o4_mini if j % 2 == 0 else model,
                        y_model=model if j % 2 == 0 else o4_mini,
                        scenario=TicTacToeScenario(step=i, split="val"),
                    )
                    for j in range(4)
                ],
                pbar_desc="val",
                trajectories_per_rollout=2,
            )

            model_trajectories = list(
                filter(
                    lambda t: t.metadata["model_name"] == model.name,
                    x_val.trajectories + y_val.trajectories,
                )
            )

            await model.log(model_trajectories, split="val")

        await model.delete_checkpoints()
        await model.train(
            trajectory_groups=[x_trajectory_group, y_trajectory_group],
            config=art.TrainConfig(learning_rate=5e-5),
            verbose=True,
        )
        print("pushing to s3")
        await backend._experimental_push_to_s3(model)

    if DEPLOY_MODEL:
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

    if GENERATE_BENCHMARKS:
        gpt_4o_mini = art.Model(
            name="gpt-4o-mini",
            project="tic-tac-toe",
            inference_model_name="gpt-4o-mini",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
        )
        await gpt_4o_mini.register(backend)

        gpt_4o = art.Model(
            name="gpt-4o",
            project="tic-tac-toe",
            inference_model_name="gpt-4o",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
        )
        await gpt_4o.register(backend)

        async def benchmark_comparison_model(comparison_model: art.Model):
            trajectories = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(comparison_model, TicTacToeScenario(step=0))
                        for _ in range(12)
                    )
                    for _ in range(1)
                ),
                pbar_desc=f"gather {comparison_model.name}",
                max_exceptions=1,
            )
            await comparison_model.log(
                trajectories,
                split="val",
            )

        await benchmark_comparison_model(gpt_4o_mini)
        await benchmark_comparison_model(gpt_4o)


if __name__ == "__main__":
    asyncio.run(main())
