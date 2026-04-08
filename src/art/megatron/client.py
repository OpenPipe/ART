import asyncio
import datetime
import json
import os
from typing import Any, AsyncIterator

from .jobs import DEFAULT_JOBS_DIR, MegatronJob
from .merge import merge_lora_adapter

DEFAULT_TRAINING_LOG_DIR = "/tmp/megatron_training_logs"


def create_megatron_job_paths() -> tuple[str, str]:
    timestamp = datetime.datetime.now().isoformat()
    os.makedirs(DEFAULT_JOBS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_TRAINING_LOG_DIR, exist_ok=True)
    return (
        os.path.join(DEFAULT_JOBS_DIR, f"{timestamp}.json"),
        os.path.join(DEFAULT_TRAINING_LOG_DIR, f"{timestamp}.jsonl"),
    )


def write_megatron_job(job: MegatronJob, *, job_path: str) -> None:
    os.makedirs(os.path.dirname(job_path), exist_ok=True)
    with open(job_path, "w", encoding="utf-8") as handle:
        handle.write(job.model_dump_json())


async def stream_megatron_job(
    job: MegatronJob,
    *,
    job_path: str,
    poll_interval: float = 0.1,
) -> AsyncIterator[dict[str, Any]]:
    num_lines = 0
    try:
        while True:
            await asyncio.sleep(poll_interval)
            try:
                with open(job.log_path, "a+", encoding="utf-8") as log_file:
                    log_file.seek(0)
                    lines = log_file.readlines()[num_lines:]
            except FileNotFoundError:
                continue

            for line in lines:
                if not (line := line.strip()):
                    continue
                if line == "all done":
                    merge_lora_adapter(job.lora_path)
                    return
                num_lines += 1
                yield json.loads(line)
    finally:
        if os.path.exists(job_path):
            os.remove(job_path)
        if os.path.exists(job.log_path):
            os.remove(job.log_path)
