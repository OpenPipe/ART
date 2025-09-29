"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel

import art
from art.skypilot.backend import SkyPilotBackend

load_dotenv()


class ComplexModelConfig(BaseModel):
    max_turns: int = 5
    max_tokens: int = 2048

    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    # Random seed to control which subset of the training data is sampled
    training_dataset_seed: int | None = None

    # Training configuration
    scale_rewards: bool = True


async def register_model():
    backend = await SkyPilotBackend().initialize_cluster(
        cluster_name="test-skypilot",
        gpu="H100-SXM",
        env_path=".env",
        # force_restart=True,
    )

    model = art.TrainableModel(
        name="complex-model",
        project="test-skypilot",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=ComplexModelConfig(
            num_epochs=160,
        ),
    )

    await backend.register(model)

    print("model registered")


if __name__ == "__main__":
    asyncio.run(register_model())
