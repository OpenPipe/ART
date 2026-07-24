import asyncio
import datetime
import json
import os
import tempfile
from typing import Any, AsyncIterator

from .jobs import DEFAULT_JOBS_DIR, MegatronJob, dump_megatron_job

DEFAULT_TRAINING_LOG_DIR = "/tmp/megatron_training_logs"


def create_megatron_job_paths(
    *,
    jobs_dir: str = DEFAULT_JOBS_DIR,
    training_log_dir: str = DEFAULT_TRAINING_LOG_DIR,
) -> tuple[str, str]:
    timestamp = datetime.datetime.now().isoformat()
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(training_log_dir, exist_ok=True)
    return (
        os.path.join(jobs_dir, f"{timestamp}.json"),
        os.path.join(training_log_dir, f"{timestamp}.jsonl"),
    )


def write_megatron_job(job: MegatronJob, *, job_path: str) -> None:
    jobs_dir = os.path.dirname(job_path)
    os.makedirs(jobs_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=jobs_dir,
        prefix=f".{os.path.basename(job_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(dump_megatron_job(job).encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, job_path)
        directory_descriptor = os.open(jobs_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


async def stream_megatron_job(
    job: MegatronJob,
    *,
    job_path: str,
    process: Any | None = None,
    process_log_path: str | None = None,
    poll_interval: float = 0.05,
) -> AsyncIterator[dict[str, Any]]:
    num_lines = 0
    try:
        while True:
            await asyncio.sleep(poll_interval)
            process_returncode = None
            if process is not None:
                process_returncode = process.returncode
                poll = getattr(process, "poll", None)
                if process_returncode is None and callable(poll):
                    process_returncode = poll()
            if process_returncode is not None:
                raise RuntimeError(
                    f"Megatron worker exited with code {process_returncode}. "
                    f"Check logs at {process_log_path or job.log_path}"
                )
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
                    return
                num_lines += 1
                yield json.loads(line)
    finally:
        if os.path.exists(job_path):
            os.remove(job_path)
        if os.path.exists(job.log_path):
            os.remove(job.log_path)
