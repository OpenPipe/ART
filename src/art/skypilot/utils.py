import sky
import asyncio
from typing import Callable, TypeVar
from sky.core import endpoints
import httpx

T = TypeVar("T")


async def to_thread_typed(func: Callable[[], T]) -> T:
    return await asyncio.to_thread(func)


async def get_task_status(cluster_name: str, task_name: str) -> sky.JobStatus:
    job_queue = await to_thread_typed(lambda: sky.queue(cluster_name))

    for job in job_queue:
        if job["job_name"] == task_name:
            return job["status"]
    return None


async def is_task_created(cluster_name: str, task_name: str) -> bool:
    task_status = await get_task_status(cluster_name, task_name)
    if task_status is None:
        return False
    return task_status in (
        sky.JobStatus.INIT,
        sky.JobStatus.PENDING,
        sky.JobStatus.SETTING_UP,
        sky.JobStatus.RUNNING,
    )


# wait for task to start running
# checks every 5 seconds for 1 minute
async def wait_for_task_to_start(cluster_name: str, task_name: str) -> None:
    task_status = await get_task_status(cluster_name, task_name)

    num_checks = 0
    while num_checks < 12:
        task_status = await get_task_status(cluster_name, task_name)
        if task_status is None:
            raise ValueError(f"Task {task_name} not found in cluster {cluster_name}")
        if task_status == sky.JobStatus.RUNNING:
            return
        await asyncio.sleep(5)
        num_checks += 1

    raise ValueError(
        f"Task {task_name} in cluster {cluster_name} failed to start within 60s"
    )


async def wait_for_art_server_to_start(cluster_name: str) -> None:
    print(f"Waiting for art server task to start on cluster {cluster_name}...")
    await wait_for_task_to_start(cluster_name, "art_server")
    print(
        f"Art server started on cluster {cluster_name}. Waiting for it to be ready..."
    )

    base_url = await get_art_server_base_url(cluster_name)

    num_checks = 0
    client = httpx.AsyncClient(
        base_url=base_url,
        timeout=10,
    )
    while num_checks < 12:
        try:
            response = await client.get("/healthcheck")
            if response.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(5)
        num_checks += 1

    return base_url


async def get_art_server_base_url(cluster_name: str) -> str:
    art_endpoint = await to_thread_typed(
        lambda: endpoints(cluster=cluster_name, port=7999)[7999]
    )
    return f"http://{art_endpoint}"


async def get_vllm_base_url(cluster_name: str) -> str:
    vllm_endpoint = await to_thread_typed(
        lambda: endpoints(cluster=cluster_name, port=8000)[8000]
    )
    return f"http://{vllm_endpoint}/v1"
