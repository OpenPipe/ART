import random
import asyncio
from dotenv import load_dotenv

import art
from rollout import rollout
from art.local.backend import LocalBackend


load_dotenv()

random.seed(42)

DESTROY_AFTER_RUN = False


async def main():
    # run from the root of the repo
    backend = LocalBackend()

    model = art.TrainableModel(
        name="agent-001",
        project="tic-tac-toe-agent",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    await backend._experimental_pull_from_s3(model)
    await model.register(backend)

    for i in range(await model.get_step(), 101):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(200)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
        await backend._experimental_push_to_s3(model)

    res = await backend._experimental_deploy(model=model, verbose=True)
    print(res)

    if DESTROY_AFTER_RUN:
        await backend.down()


if __name__ == "__main__":
    asyncio.run(main())
