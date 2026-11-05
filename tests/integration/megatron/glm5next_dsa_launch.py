"""Bounded workstation launch/rehearsal for M3. No cluster resources are created.

Rehearsal exercises the --execute controller with CPU R2 workers and meta
projections at the full F4/F8 token volumes. A real launch additionally requires
the exact approved seal and a matching BOARD GRANT. CP8 uses the parity driver
directly under M4's node grant, rather than this workstation controller.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

LANE = "glm-m3-dsa"
MODULE = "tests.integration.megatron.glm5next_dsa_parity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def board_line(board: Path, message: str) -> None:
    stamp = datetime.now(timezone(timedelta(hours=-7))).strftime("%Y-%m-%d %I:%M %p")
    with board.open("a") as stream:
        stream.write(f"{stamp} MST {message}\n")


def gpu_idle(uuids: list[str]) -> bool:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    memory = {
        line.split(",")[0].strip(): int(line.split(",")[1].strip())
        for line in output.splitlines()
    }
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    return all(memory.get(uuid) == 0 and uuid not in processes for uuid in uuids)


def tree_memory(pid: int, *, proc: Path = Path("/proc")) -> tuple[int, int]:
    """Sum PSS and RSS across descendants, including separate process sessions."""
    pss, rss, pending, seen = 0, 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            rollup = (proc / str(current) / "smaps_rollup").read_text()
            values = {
                line.split(":", 1)[0]: int(line.split()[1]) * 1024
                for line in rollup.splitlines()
                if line.startswith(("Pss:", "Rss:"))
            }
            pss += values["Pss"]
            rss += values["Rss"]
            pending.extend(
                map(
                    int,
                    (proc / str(current) / "task" / str(current) / "children")
                    .read_text()
                    .split(),
                )
            )
        except (FileNotFoundError, ProcessLookupError):
            pass
    return pss, rss


def run_step(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    log: Path,
    timeout: int,
    pss_limit: int,
) -> dict[str, Any]:
    start, peak_pss, peak_rss = time.monotonic(), 0, 0
    with log.open("w") as stream:
        process = subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                pss, rss = tree_memory(process.pid)
                peak_pss, peak_rss = max(peak_pss, pss), max(peak_rss, rss)
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Step exceeded {timeout}s: {argv}")
                if pss > pss_limit:
                    raise RuntimeError("Process tree exceeded sealed PSS limit.")
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f"Step failed ({process.returncode}); see {log}")
        finally:
            # Also reap descendants left behind by an exited launcher.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            deadline = time.monotonic() + 15
            while True:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGKILL)
                    raise RuntimeError(
                        "Process group survived cleanup; inspect before another launch."
                    )
                time.sleep(0.1)
    return {
        "argv": argv,
        "log": str(log),
        "elapsed_seconds": time.monotonic() - start,
        "peak_tree_pss_bytes": peak_pss,
        "peak_tree_rss_bytes": peak_rss,
        "exit_code": process.returncode,
    }


def command(
    python: str,
    *,
    fixture: str,
    cp: int,
    device: str,
    lengths: str,
    output: Path,
    meta: bool = False,
) -> list[str]:
    argv = [python]
    if cp > 1:
        argv += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={cp}",
            "--module",
            MODULE,
        ]
    else:
        argv += ["-m", MODULE]
    argv += [
        "--fixture",
        fixture,
        "--device",
        device,
        "--lengths",
        lengths,
        "--output",
        str(output),
    ]
    if meta:
        argv += ["--meta-plan"]
    return argv


def execute(seal_path: Path, approved: str, output: Path, recording: bool) -> None:
    digest = sha256(seal_path)
    if approved != digest:
        raise ValueError("Approved seal hash differs from the requested seal.")
    seal = json.loads(seal_path.read_text())
    if seal["holds"] or seal["launch_ready"] is not True:
        raise ValueError("Seal is not launch-ready.")
    repo, board = Path(seal["repo"]), Path(seal["board"])
    for name, expected in seal["scripts"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Sealed source changed: {name}")
    # Fixture records are a stable snapshot, independent of M1's evolving manifest.
    for record in seal["fixture_files"]:
        path = Path(record["path"])
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Fixture integrity failure: {path}")
    if not recording and not any(
        "GRANT glm-m3-dsa" in line and digest in line
        for line in board.read_text().splitlines()
    ):
        raise PermissionError("No matching GRANT glm-m3-dsa seal on BOARD.")
    if not recording:
        predecessor = seal["admission"]["after_lane"]
        after_event = seal["admission"]["after_event"]
        lifecycle = [
            line
            for line in board.read_text().splitlines()
            if any(
                f"{event} {predecessor} " in line
                for event in ("GRANT", "GPU-START", "RELEASED", "CLEANUP-DONE")
            )
        ]
        if not lifecycle or f"{after_event} {predecessor} " not in lifecycle[-1]:
            raise PermissionError(f"Wait for {predecessor}'s BOARD {after_event} line.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "STARTED").open("x").close()
    receipt: dict[str, Any] = {
        "seal_sha256": digest,
        "recording": recording,
        "memory_guard": "process-tree PSS from /proc/<pid>/smaps_rollup",
        "pss_limit_bytes": seal["pss_limit_bytes"],
        "gpu_uuids": seal["gpu_uuids"],
        "admission": seal["admission"],
        "steps": [],
        "cleanup_done": False,
        "passed": False,
    }
    locks = []
    admitted = False
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join((str(repo / "src"), str(repo))),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        MAX_JOBS="2",
    )
    started = time.monotonic()
    try:
        if not recording:
            for uuid in seal["gpu_uuids"]:
                lock = open(f"/tmp/{LANE}-{uuid}.lock", "a+")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(lock)
            if not gpu_idle(seal["gpu_uuids"]):
                raise RuntimeError(
                    "Requested GPUs require 0 MiB and no compute processes at admission."
                )
            admitted = True
            board_line(
                board,
                f"GPU-START {LANE} seal_sha256={digest} GPUs={','.join(seal['gpu_uuids'])} deadline_seconds={seal['window_seconds']} output={output}",
            )
        runtime = run_step(
            [
                seal["python"],
                "-c",
                "import json,torch; assert torch.version.cuda=='12.8'; assert not torch.cuda.is_initialized(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_initialized':False}))",
            ],
            env=env,
            cwd=repo,
            log=output / "runtime.log",
            timeout=120,
            pss_limit=seal["pss_limit_bytes"],
        )
        receipt["steps"].append(runtime)
        for fixture in ("F4", "F8"):
            argv = command(
                seal["python"],
                fixture=fixture,
                cp=1,
                device="cpu",
                lengths=seal["lengths"],
                output=output / f"meta-{fixture}",
                meta=True,
            )
            receipt["steps"].append(
                run_step(
                    argv,
                    env=env,
                    cwd=repo,
                    log=output / f"meta-{fixture}.log",
                    timeout=180,
                    pss_limit=seal["pss_limit_bytes"],
                )
            )
        for stage in seal["stages"]:
            remaining = seal["window_seconds"] - 120 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Protected cleanup window reached.")
            path = output / stage["name"]
            planned = command(
                seal["python"],
                fixture=stage["fixture"],
                cp=stage["cp"],
                device="cuda",
                lengths=seal["lengths"],
                output=path,
            )
            argv = (
                command(
                    seal["python"],
                    fixture="R2",
                    cp=stage["cp"],
                    device="cpu",
                    lengths=seal["recording_lengths"],
                    output=path,
                )
                if recording
                else planned
            )
            stage_env = dict(
                env,
                CUDA_VISIBLE_DEVICES=""
                if recording
                else ",".join(seal["gpu_uuids"][: stage["cp"]]),
            )
            result = run_step(
                argv,
                env=stage_env,
                cwd=repo,
                log=output / f"{stage['name']}.log",
                timeout=min(stage["timeout_seconds"], int(remaining)),
                pss_limit=seal["pss_limit_bytes"],
            )
            result["planned_argv"] = planned
            receipt["steps"].append(result)
            for rank in range(stage["cp"]):
                worker = json.loads(
                    (path / f"cp{stage['cp']}-rank{rank}.json").read_text()
                )
                if not worker["cases"] or not all(
                    case["passed"] for case in worker["cases"]
                ):
                    raise RuntimeError("Worker parity receipt failed.")
        for name, expected in seal["scripts"].items():
            if sha256(repo / name) != expected:
                raise ValueError(f"Sealed source changed during execution: {name}")
        receipt["passed"] = True
    finally:
        if recording:
            receipt["cleanup_done"] = True
        elif admitted:
            deadline = time.monotonic() + 90
            while not gpu_idle(seal["gpu_uuids"]) and time.monotonic() < deadline:
                time.sleep(1)
            receipt["cleanup_done"] = gpu_idle(seal["gpu_uuids"])
            if receipt["cleanup_done"]:
                board_line(
                    board,
                    f"RELEASED {LANE} seal_sha256={digest} verified 0 MiB and no compute processes; result={output}",
                )
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for lock in locks:
            lock.close()
    if not receipt["cleanup_done"]:
        raise RuntimeError("GPU release verification failed; notify the lead.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--approved-seal", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.seal, args.approved_seal, args.output, args.recording)


if __name__ == "__main__":
    main()
