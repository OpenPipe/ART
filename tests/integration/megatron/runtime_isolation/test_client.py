import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from art.megatron.runtime import client
from art.megatron.runtime.client import stream_megatron_job, write_megatron_job
from art.megatron.runtime.jobs import (
    MegatronSyncJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
    dump_megatron_job,
)


def _job(*, log_path: Path) -> MegatronSyncJob:
    return MegatronSyncJob(
        lora_path="/tmp/lora",
        merged_weight_transfer=MergedWeightTransferSpec(
            init_info=MergedWeightTransferInitInfo(
                master_address="127.0.0.1",
                master_port=12345,
                rank_offset=1,
                world_size=2,
            ),
            vllm_base_url="http://127.0.0.1:8000",
            served_model_name="test@0",
        ),
        log_path=str(log_path),
    )


def test_write_megatron_job_publishes_only_complete_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path = tmp_path / "job.json"
    job = _job(log_path=tmp_path / "job.log")
    replace = os.replace
    observed_temporary_path: Path | None = None

    def assert_atomic_replace(source: Any, destination: Any) -> None:
        nonlocal observed_temporary_path
        observed_temporary_path = Path(source)
        assert Path(destination) == job_path
        assert not job_path.exists()
        assert observed_temporary_path.exists()
        assert observed_temporary_path.suffix != ".json"
        assert observed_temporary_path.read_text() == dump_megatron_job(job)
        replace(source, destination)

    monkeypatch.setattr(client.os, "replace", assert_atomic_replace)

    write_megatron_job(job, job_path=str(job_path))

    assert observed_temporary_path is not None
    assert job_path.read_text() == dump_megatron_job(job)
    assert list(tmp_path.iterdir()) == [job_path]


def test_write_megatron_job_cleans_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path = tmp_path / "job.json"
    job = _job(log_path=tmp_path / "job.log")

    def fail_replace(_source: Any, _destination: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(client.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_megatron_job(job, job_path=str(job_path))

    assert not job_path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_stream_megatron_job_raises_when_worker_exits(
    tmp_path: Path,
) -> None:
    job_path = tmp_path / "job.json"
    log_path = tmp_path / "job.log"
    job = _job(log_path=log_path)
    write_megatron_job(job, job_path=str(job_path))

    with pytest.raises(RuntimeError, match="Megatron worker exited with code 17"):
        async for _ in stream_megatron_job(
            job,
            job_path=str(job_path),
            process=SimpleNamespace(returncode=17),
            process_log_path="/tmp/megatron-runtime.log",
            poll_interval=0.0,
        ):
            pass
