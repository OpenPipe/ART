"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio

from dotenv import load_dotenv

from art.skypilot.backend import SkyPilotBackend

load_dotenv()


async def launch_tail():
    backend = await SkyPilotBackend().initialize_cluster(
        cluster_name="test-skypilot",
        gpu="H100-SXM",
        env_path=".env",
        force_restart=True,
        tail_logs=True,
    )
    print("successfully initialized skypilot server")

    # unforunately, we can't cancel the task programmatically, so we have to ctrl+c
    # to exit


if __name__ == "__main__":
    asyncio.run(launch_tail())
