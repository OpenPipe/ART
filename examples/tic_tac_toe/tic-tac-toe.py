import asyncio
import os
import random

from dotenv import load_dotenv
from rollout import TicTacToeScenario, rollout
import weave

import art
from art.utils.deployment import TogetherDeploymentConfig, deploy_model
from art.utils.strip_logprobs import strip_logprobs

load_dotenv()

random.seed(42)

PULL_FROM_S3 = False
STEP = 50
DEPLOY_MODEL = False
GENERATE_BENCHMARKS = False

weave.init("tic-tac-toe", global_postprocess_output=strip_logprobs)


async def main():
    from art.local.backend import LocalBackend

    backend = LocalBackend()

    model = art.TrainableModel(
        name="llama-8b-007",
        project="tic-tac-toe",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    if PULL_FROM_S3:
        print("pulling from s3")
        await backend._experimental_pull_from_s3(model)

    print("registering")
    await model.register(backend)

    print("training")
    for i in range(await model.get_step(), STEP):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, TicTacToeScenario(step=i)) for _ in range(96)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        result = await backend.train(model, train_groups, learning_rate=5e-5)
        await model.log(
            train_groups, metrics=result.metrics, step=result.step, split="train"
        )
        await backend._experimental_push_to_s3(model)

    if DEPLOY_MODEL:
        print("deploying")
        # Pull checkpoint (already local since we just trained, but ensures correct path)
        checkpoint_path = await backend._experimental_pull_model_checkpoint(
            model,
            step=STEP,
            s3_bucket=os.environ.get("BACKUP_BUCKET"),
            verbose=True,
        )

        # Deploy to Together
        deployment_result = await deploy_model(
            model=model,
            checkpoint_path=checkpoint_path,
            step=STEP,
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
        )

        print("Starting a rollout using the deployed model!")
        traj = await rollout(lora_model, TicTacToeScenario(step=0))

        print(traj)

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
