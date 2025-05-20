import argparse
import asyncio

import sky
from art.skypilot.backend import SkyPilotBackend
from art.skypilot.utils import is_task_created, to_thread_typed

parser = argparse.ArgumentParser(
    description="Close the art server hosted on a skypilot cluster"
)
parser.add_argument(
    "--cluster",
    type=str,
    required=True,
    help="The name of the skypilot cluster to close the art server on",
)
args = parser.parse_args()


async def stop_server() -> None:
    cluster_status = await to_thread_typed(
        lambda: sky.status(cluster_names=[args.cluster])
    )
    if len(cluster_status) == 0 or cluster_status[0]["status"] != sky.ClusterStatus.UP:
        raise ValueError(f"Cluster {args.cluster} is not running")

    if not await is_task_created(cluster_name=args.cluster, task_name="art_server"):
        raise ValueError(f"Art server task for cluster {args.cluster} is not running")

    backend = await SkyPilotBackend.initialize_cluster(
        cluster_name=args.cluster, art_version=".", env_path=".env", gpu="H100"
    )
    await backend.close()

    # cancel the art server task
    await to_thread_typed(lambda: sky.cancel(cluster_name=args.cluster, all=True))


def main() -> None:
    asyncio.run(stop_server())
