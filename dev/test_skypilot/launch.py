"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio

from dotenv import load_dotenv

from art.skypilot.backend import SkyPilotBackend

load_dotenv()


async def launch():
    backend = await SkyPilotBackend().initialize_cluster(
        cluster_name="test-skypilot",
        gpu="H100-SXM",
        env_path=".env",
        force_restart=True,
    )

    print("successfully initialized skypilot server")


if __name__ == "__main__":
    asyncio.run(launch())
