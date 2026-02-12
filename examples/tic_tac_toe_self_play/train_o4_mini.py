import asyncio
import os
import random

from dotenv import load_dotenv
from game_utils import possible_moves
from gather_trajectory_groups_by_index import gather_trajectory_groups_by_index
from rollout import ModelConfig, TicTacToeScenario, rollout

import art
from art.trajectories import TrajectoryGroup

load_dotenv()

random.seed(42)

PULL_FROM_S3 = False
STEP = 200
PROJECT_NAME = "tic-tac-toe"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = "llama-8b-o4-mini-001"


async def main():
    from art.local.backend import LocalBackend

    backend = LocalBackend()

    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
        config=ModelConfig(),
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(
                num_scheduler_steps=1,
            ),
        ),
    )

    o4_mini = art.Model(
        name="o4-mini",
        project="tic-tac-toe",
        inference_model_name="o4-mini",
        inference_api_key=os.environ["OPENAI_API_KEY"],
        inference_base_url="https://api.openai.com/v1",
        config=ModelConfig(requires_reasoning=True),
    )

    if PULL_FROM_S3:
        print("pulling from s3")
        await backend._experimental_pull_from_s3(model)

    print("registering")
    await model.register(backend)
    await o4_mini.register(backend)

    def get_model_trajectories(
        trajectories: list[art.Trajectory],
    ) -> list[art.Trajectory]:
        return list(
            filter(
                lambda t: t.metadata["model_name"] == model.name,
                trajectories,
            )
        )

    print("commencing run")
    for i in range(await model.get_step(), STEP):
        (
            x_trajectory_group,
            o_trajectory_group,
        ) = await gather_trajectory_groups_by_index(
            [
                rollout(
                    x_model=o4_mini if j % 4 == 0 else model,
                    o_model=o4_mini if j % 4 == 1 else model,
                    scenario=TicTacToeScenario(
                        step=i,
                        split="train",
                        # ensure we learn how to play against all 9 possible opening moves
                        initial_move=possible_moves[j % 9] if j < 63 else None,
                    ),
                )
                for j in range(96)
            ],
            pbar_desc="gather",
            trajectories_per_rollout=2,
        )

        x_trajectory_group = TrajectoryGroup(
            get_model_trajectories(x_trajectory_group.trajectories)
        )
        o_trajectory_group = TrajectoryGroup(
            get_model_trajectories(o_trajectory_group.trajectories)
        )

        if i % 4 == 0:
            x_val, o_val = await gather_trajectory_groups_by_index(
                [
                    rollout(
                        x_model=o4_mini if j % 2 == 0 else model,
                        o_model=model if j % 2 == 0 else o4_mini,
                        scenario=TicTacToeScenario(
                            step=i,
                            split="val",
                        ),
                    )
                    for j in range(10)
                ],
                pbar_desc="val",
                trajectories_per_rollout=2,
            )

            model_trajectories = get_model_trajectories(
                x_val.trajectories + o_val.trajectories
            )

            await model.log(model_trajectories, split="val")

        # await model.delete_checkpoints()
        trajectory_groups = [x_trajectory_group, o_trajectory_group]
        result = await backend.train(
            model, trajectory_groups, learning_rate=2e-5, verbose=True
        )
        await model.log(
            trajectory_groups, metrics=result.metrics, step=result.step, split="train"
        )
        await backend._experimental_push_to_s3(model)

    if DESTROY_AFTER_RUN:
        await backend.down()


if __name__ == "__main__":
    asyncio.run(main())
