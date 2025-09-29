import asyncio
import os
import random
import sys
from dotenv import load_dotenv

# Add the 2048 example directory to the path so we can reuse its rollout
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2048"))
from rollout import rollout  # type: ignore

import art

load_dotenv()
random.seed(42)

# URL of the remote ART server (e.g. "http://123.45.67.89:7999")
REMOTE_URL = os.getenv("ART_SERVER_URL", "http://localhost:7999")
backend = art.Backend(base_url=REMOTE_URL)

# Declare a trainable model. Its weights and training will live on the server.
model = art.TrainableModel(
    name="remote-2048",
    project="2048-remote",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(max_seq_length=8192),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        num_scheduler_steps=1,
    ),
)
model.set_temperature_annealer(
    art.LinearTemperatureAnnealer(start=1.0, end=0.1, steps=5)
)


async def main() -> None:
    # Register the model with the remote backend. This also starts the
    # OpenAI-compatible inference server on the remote machine.
    await model.register(backend)

    for i in range(await model.get_step(), 5):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(18)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=18,
        )
        await model.delete_checkpoints()
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=3e-5),
            _config={"logprob_calculation_chunk_size": 8},
        )


if __name__ == "__main__":
    asyncio.run(main())
