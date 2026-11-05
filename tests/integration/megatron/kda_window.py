"""One sealed workstation KDA window, with a CPU rehearsal of the same driver.

--execute --rehearsal substitutes R2/CPU for F4/CUDA in the worker. The F4
schema step still reads and validates every real KDA weight on both paths.
No GPU allocation is performed here: Flash/Jarvis grant the named UUIDs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def group_rss(group: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            if int(fields[2]) == group:
                total += int((entry / "statm").read_text().split()[1]) * os.sysconf(
                    "SC_PAGE_SIZE"
                )
        except (OSError, ValueError, IndexError):
            continue
    return total


def stop_group(group: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        limit = time.monotonic() + 5
        while time.monotonic() < limit:
            if group_rss(group) == 0:
                return
            time.sleep(0.1)
    if group_rss(group):
        raise RuntimeError(f"Owned process group {group} survived cleanup")


def run_step(
    argv: list[str],
    *,
    env: dict[str, str],
    output: Path,
    timeout: int,
    owned_groups: list[int],
) -> dict[str, Any]:
    started = time.monotonic()
    peak = 0
    with output.open("w") as log:
        child = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        owned_groups.append(child.pid)
        try:
            while child.poll() is None:
                peak = max(peak, group_rss(child.pid))
                if peak > 120 * (1 << 30):
                    raise RuntimeError("KDA process tree exceeded 120 GiB RSS")
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"KDA step exceeded {timeout}s")
                time.sleep(0.25)
            if child.returncode:
                raise RuntimeError(f"KDA worker exited {child.returncode}: {output}")
        finally:
            stop_group(child.pid)
            child.wait(timeout=15)
    return {
        "argv": argv,
        "log": str(output),
        "elapsed_seconds": time.monotonic() - started,
        "peak_tree_rss_bytes": peak,
        "exit_code": child.returncode,
    }


def verify(seal: dict[str, Any]) -> None:
    if seal.get("holds") != [] or seal.get("launch_ready") is not True:
        raise ValueError("KDA seal is not launch ready")
    for item in seal["files"]:
        path = Path(item["path"])
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError(f"KDA sealed file changed: {path}")


def idle(uuids: list[str]) -> list[str]:
    records = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    memory = {
        row.split(",")[0].strip(): int(row.split(",")[1].strip())
        for row in records.splitlines()
    }
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    for uuid in uuids:
        if memory.get(uuid) != 0 or uuid in processes:
            raise RuntimeError(f"GPU {uuid} is not idle and at 0 MiB")
    return records.splitlines()


def execute(args: Any) -> dict[str, Any]:
    seal_sha = digest(args.seal)
    if args.approved_seal != seal_sha:
        raise ValueError("--approved-seal must equal the exact sealed request SHA256")
    seal = json.loads(args.seal.read_text())
    verify(seal)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "seal_sha256": seal_sha,
        "rehearsal": args.rehearsal,
        "steps": [],
        "pass": False,
        "cleanup_done": False,
    }
    env = dict(os.environ)
    env.update(seal["environment"])
    env["CUDA_VISIBLE_DEVICES"] = "" if args.rehearsal else ",".join(seal["gpu_uuids"])
    if not args.rehearsal:
        board = Path(seal["grant_board"]).read_text()
        if not any(
            line.startswith("GRANT glm-m2-kda ") and seal_sha in line
            for line in board.splitlines()
        ):
            raise ValueError("No BOARD GRANT names glm-m2-kda and this seal")
    python = seal["python"]
    worker = seal["worker"]
    started = time.monotonic()
    owned_groups: list[int] = []
    with ExitStack() as locks:
        # Rehearsal exercises real lock acquisition/release as well.
        for uuid in seal["gpu_uuids"]:
            lock = locks.enter_context(open(f"/tmp/stark-gpu/kda-{uuid}.lock", "a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not args.rehearsal:
                receipt["before"] = idle(seal["gpu_uuids"])
            schema = [
                python,
                worker,
                "--device",
                "cpu",
                "--fixture",
                "F4",
                "--schema-only",
                "--tokens",
                str(seal["tokens"]),
                "--output",
                str(output / "schema.json"),
            ]
            schema_env = dict(env, CUDA_VISIBLE_DEVICES="")
            receipt["steps"].append(
                run_step(
                    schema,
                    env=schema_env,
                    output=output / "schema.log",
                    timeout=120,
                    owned_groups=owned_groups,
                )
            )
            for step in seal["steps"]:
                world = step["world_size"]
                argv = [python]
                if world > 1:
                    argv += [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc-per-node={world}",
                    ]
                argv += [
                    worker,
                    "--device",
                    "cpu" if args.rehearsal else "cuda",
                    "--fixture",
                    "R2" if args.rehearsal else "F4",
                    "--tokens",
                    str(seal["tokens"]),
                    "--output",
                    str(output / f"cp{world}.json"),
                ]
                remaining = (
                    seal["window_seconds"] - 120 - int(time.monotonic() - started)
                )
                receipt["steps"].append(
                    run_step(
                        argv,
                        env=env,
                        output=output / f"cp{world}.log",
                        timeout=min(step["timeout_seconds"], remaining),
                        owned_groups=owned_groups,
                    )
                )
                for rank in range(world):
                    result = json.loads(
                        (output / f"cp{world}-rank{rank}.json").read_text()
                    )
                    if not result["pass"]:
                        raise RuntimeError(f"CP{world} rank {rank} failed")
            verify(seal)
            receipt["pass"] = True
        except BaseException as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["cleanup_done"] = True
            for group in owned_groups:
                if group_rss(group):
                    receipt["cleanup_done"] = False
                    receipt["cleanup_error"] = f"Surviving owned process group {group}"
            if not args.rehearsal:
                try:
                    receipt["after"] = idle(seal["gpu_uuids"])
                except RuntimeError as exc:
                    receipt["cleanup_done"] = False
                    receipt["cleanup_error"] = str(exc)
            receipt["elapsed_seconds"] = time.monotonic() - started
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--approved-seal", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = execute(args)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "seal_sha256",
                    "rehearsal",
                    "pass",
                    "cleanup_done",
                    "elapsed_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
