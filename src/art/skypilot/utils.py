import sky
import asyncio
from typing import Callable, TypeVar

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
            # Waiting for our server to start up
            await asyncio.sleep(30)
            return
        await asyncio.sleep(5)
        num_checks += 1

    raise ValueError(
        f"Task {task_name} in cluster {cluster_name} failed to start within 60s"
    )
