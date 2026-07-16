from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.utils.lifecycle import complete_task


class _NcclRuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str = Field(min_length=1)
    runtime_kind: Literal["trainer", "vllm"]
    master_addr: str = Field(min_length=1)
    timeout_s: float = Field(gt=0)
    probe_command: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_probe_command(self) -> "_NcclRuntimeRequest":
        if self.probe_command is not None and (
            not self.probe_command or any(not value for value in self.probe_command)
        ):
            raise ValueError("NCCL probe command must contain non-empty arguments")
        if self.runtime_kind == "trainer" and self.probe_command is not None:
            raise ValueError("trainer NCCL probes use the ART runtime")
        return self


class NcclRendezvousRequest(_NcclRuntimeRequest):
    pass


class NcclRendezvousResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host_id: str
    port: int = Field(ge=1, le=65535)


class NcclProbeRequest(_NcclRuntimeRequest):
    rank: int = Field(ge=0)
    world_size: int = Field(ge=2)
    master_port: int = Field(ge=1, le=65535)
    gpu_id: int = Field(ge=0)
    net_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_rank(self) -> "NcclProbeRequest":
        if self.rank >= self.world_size:
            raise ValueError("NCCL probe rank must be smaller than world_size")
        return self


class NcclProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host_id: str
    rank: int = Field(ge=0)
    net_name: str
    duration_s: float = Field(ge=0)


_PARENT_DEATH = r"""
import ctypes
import os
import signal

parent = os.getppid()
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGTERM) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
if os.getppid() != parent:
    os.kill(os.getpid(), signal.SIGTERM)
"""

_RENDEZVOUS_SCRIPT = (
    _PARENT_DEATH
    + r"""
from datetime import timedelta
import sys

import torch.distributed as dist

store = dist.TCPStore(
    os.environ["MASTER_ADDR"],
    0,
    None,
    True,
    timedelta(seconds=float(os.environ["ART_NCCL_TIMEOUT_S"])),
    wait_for_workers=False,
)
print(f"ART_NCCL_RENDEZVOUS_PORT={store.port}", flush=True)
sys.stdin.buffer.read()
"""
)

_CHILD_SCRIPT = (
    _PARENT_DEATH
    + r"""
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
timeout = timedelta(seconds=float(os.environ["ART_NCCL_TIMEOUT_S"]))
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
store = dist.TCPStore(
    os.environ["MASTER_ADDR"],
    int(os.environ["MASTER_PORT"]),
    None,
    False,
    timeout,
)
options = dist.ProcessGroupNCCL.Options()
options.config.net_name = os.environ["ART_NCCL_EXPECTED_NET"]
try:
    dist.init_process_group(
        "nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        pg_options=options,
        device_id=device,
    )
    value = torch.tensor(rank + 1, device=device, dtype=torch.int64)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    expected = world_size * (world_size + 1) // 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL preflight reduced {value.item()}, expected {expected}")
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
"""
)

_VLLM_EXEC_SCRIPT = r"""
import os
import sys

from art.vllm_runtime import (
    RUNTIME_SERVER,
    _runtime_python_for_nccl_discovery,
    _vllm_runtime_subprocess_env,
)

try:
    python = _runtime_python_for_nccl_discovery()
except RuntimeError as error:
    raise RuntimeError(
        "Cannot derive the Python environment behind ART_VLLM_RUNTIME_BIN; "
        "set ClusterSpec.nccl_transport.vllm_probe_command"
    ) from error
environment = _vllm_runtime_subprocess_env([str(python.parent / RUNTIME_SERVER)])
os.execve(str(python), [str(python), "-c", sys.argv[1]], environment)
"""

_RENDEZVOUS_PREFIX = b"ART_NCCL_RENDEZVOUS_PORT="
_SELECTED_NETWORK = re.compile(r"NCCL INFO Using network ([^\r\n]+)$", re.MULTILINE)


class NcclRendezvous:
    def __init__(self, process: asyncio.subprocess.Process, port: int) -> None:
        self.process = process
        self.port = port

    async def close(self) -> None:
        await complete_task(asyncio.create_task(_stop_process(self.process)))


def parse_selected_network(log: str, expected: str) -> str:
    selected = tuple(value.strip() for value in _SELECTED_NETWORK.findall(log))
    if selected != (expected,):
        raise RuntimeError(
            f"NCCL selected-network proof mismatch: expected={expected!r}, "
            f"reported={selected!r}"
        )
    return selected[0]


async def start_nccl_rendezvous(request: NcclRendezvousRequest) -> NcclRendezvous:
    command, environment = _runtime_launch(request, _RENDEZVOUS_SCRIPT)
    environment.update(
        {
            "ART_NCCL_TIMEOUT_S": str(request.timeout_s),
            "CUDA_VISIBLE_DEVICES": "",
            "MASTER_ADDR": request.master_addr,
        }
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        env=environment,
        start_new_session=True,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async with asyncio.timeout(request.timeout_s):
            port = await _read_rendezvous_port(process)
        return NcclRendezvous(process, port)
    except BaseException:
        await complete_task(asyncio.create_task(_stop_process(process)))
        raise


async def run_nccl_probe(host_id: str, request: NcclProbeRequest) -> NcclProbeResult:
    command, environment = _runtime_launch(request, _CHILD_SCRIPT)
    log_path = Path(tempfile.gettempdir()) / (
        f"art-nccl-{request.probe_id}-{request.rank}-{uuid.uuid4().hex}.log"
    )
    environment.update(
        {
            "ART_NCCL_EXPECTED_NET": request.net_name,
            "ART_NCCL_TIMEOUT_S": str(request.timeout_s),
            "CUDA_VISIBLE_DEVICES": str(request.gpu_id),
            "MASTER_ADDR": request.master_addr,
            "MASTER_PORT": str(request.master_port),
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_FILE": str(log_path),
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
            "NCCL_NET": request.net_name,
            "RANK": str(request.rank),
            "WORLD_SIZE": str(request.world_size),
        }
    )
    started = time.monotonic()
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            env=environment,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async with asyncio.timeout(request.timeout_s):
            output, _ = await process.communicate()
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        detail = output.decode(errors="replace")[-4000:]
        if process.returncode:
            raise RuntimeError(
                f"NCCL {request.runtime_kind} preflight rank {request.rank} exited "
                f"{process.returncode}:\n{detail}\n{log[-4000:]}"
            )
        selected = parse_selected_network(log, request.net_name)
        return NcclProbeResult(
            host_id=host_id,
            rank=request.rank,
            net_name=selected,
            duration_s=time.monotonic() - started,
        )
    except BaseException:
        if process is not None:
            await complete_task(asyncio.create_task(_stop_process(process)))
        raise
    finally:
        log_path.unlink(missing_ok=True)


def _runtime_launch(
    request: _NcclRuntimeRequest, script: str
) -> tuple[tuple[str, ...], dict[str, str]]:
    if request.runtime_kind == "trainer":
        return (_art_python(), "-c", script), os.environ.copy()
    if request.probe_command is not None:
        from art.vllm_runtime import _vllm_runtime_subprocess_env

        command = (*request.probe_command, "-c", script)
        return command, _vllm_runtime_subprocess_env(list(command))
    return (
        _art_python(),
        "-c",
        _VLLM_EXEC_SCRIPT,
        script,
    ), os.environ.copy()


def _art_python() -> str:
    candidate = Path(os.environ.get("ART_VIRTUAL_ENV", sys.prefix)) / "bin/python"
    return str(candidate if candidate.exists() else Path(sys.executable))


async def _read_rendezvous_port(process: asyncio.subprocess.Process) -> int:
    assert process.stdout is not None
    output = bytearray()
    while line := await process.stdout.readline():
        if line.startswith(_RENDEZVOUS_PREFIX):
            return int(line.removeprefix(_RENDEZVOUS_PREFIX))
        output.extend(line)
        del output[:-4000]
    await process.wait()
    raise RuntimeError(
        f"NCCL rendezvous exited {process.returncode} before binding a port:\n"
        f"{output.decode(errors='replace')}"
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        async with asyncio.timeout(2):
            await process.wait()
            return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()
