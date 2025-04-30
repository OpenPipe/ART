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
    backend = LocalBackend(path="examples/tic_tac_toe/.art")

    model = art.TrainableModel(
        name="001-script",
        project="tic-tac-toe-local",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    await model.register(backend)

    for i in range(await model.get_step(), 100):
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

    if DESTROY_AFTER_RUN:
        await backend.down()


if __name__ == "__main__":
    asyncio.run(main())
