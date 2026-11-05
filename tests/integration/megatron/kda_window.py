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


def tree_pids(root: int, owned: dict[int, int]) -> set[int]:
    """Track descendants across setsid(), retaining PID start times on exit."""
    processes: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            if fields[0] != "Z":
                processes[int(entry.name)] = (
                    int(fields[1]),
                    int(fields[19]),
                )
        except (OSError, ValueError, IndexError):
            continue
    if root not in owned and root in processes:
        owned[root] = processes[root][1]
    live = {
        pid
        for pid, start in owned.items()
        if pid in processes and processes[pid][1] == start
    }
    while True:
        children = {
            pid
            for pid, (parent, _) in processes.items()
            if parent in live and pid not in live
        }
        if not children:
            break
        live.update(children)
        for pid in children:
            owned[pid] = processes[pid][1]
    return live


def tree_memory(root: int, owned: dict[int, int]) -> tuple[int, int]:
    """Sum PSS for admission and RSS for reference over the same live tree."""
    pss = rss = 0
    for pid in tree_pids(root, owned):
        try:
            fields = {}
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                key, separator, value = line.partition(":")
                if separator and key in ("Pss", "Rss"):
                    fields[key] = int(value.split()[0]) * 1024
            if fields.keys() != {"Pss", "Rss"}:
                raise RuntimeError(f"Incomplete smaps_rollup memory counters for {pid}")
            pss += fields["Pss"]
            rss += fields["Rss"]
        except (FileNotFoundError, ProcessLookupError):
            # A worker may exit between discovery and opening its rollup.
            continue
    return pss, rss


def stop_tree(root: int, owned: dict[int, int]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        tree_pids(root, owned)
        for pid, start in list(owned.items()):
            try:
                fields = (
                    Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
                )
                if int(fields[19]) == start:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass
        limit = time.monotonic() + 5
        while time.monotonic() < limit:
            if not tree_pids(root, owned):
                return
            time.sleep(0.1)
    if tree_pids(root, owned):
        raise RuntimeError(f"Owned process tree {root} survived cleanup")


def run_step(
    argv: list[str],
    *,
    env: dict[str, str],
    output: Path,
    timeout: int,
    max_pss_bytes: int,
    owned_groups: list[tuple[int, dict[int, int]]],
    step_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "argv": argv,
        "log": str(output),
        "peak_tree_pss_bytes": 0,
        "peak_tree_rss_bytes": 0,
        "exit_code": None,
    }
    step_receipts.append(record)
    with output.open("w") as log:
        child = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        owned: dict[int, int] = {}
        owned_groups.append((child.pid, owned))
        tree_pids(child.pid, owned)
        try:
            while child.poll() is None:
                pss, rss = tree_memory(child.pid, owned)
                record["peak_tree_pss_bytes"] = max(record["peak_tree_pss_bytes"], pss)
                record["peak_tree_rss_bytes"] = max(record["peak_tree_rss_bytes"], rss)
                if pss > max_pss_bytes:
                    raise RuntimeError(
                        f"KDA process tree exceeded {max_pss_bytes} bytes PSS"
                    )
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"KDA step exceeded {timeout}s")
                time.sleep(0.25)
            if child.returncode:
                raise RuntimeError(f"KDA worker exited {child.returncode}: {output}")
        finally:
            stop_tree(child.pid, owned)
            child.wait(timeout=15)
            record["elapsed_seconds"] = time.monotonic() - started
            record["exit_code"] = child.returncode
    return record


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
        "gpu_uuids": seal["gpu_uuids"],
        "memory_guard": {
            "metric": "sum of descendant Pss from /proc/<pid>/smaps_rollup",
            "max_tree_pss_bytes": seal["max_tree_pss_bytes"],
            "rss": "reference only",
            "sampling_interval_seconds": 0.25,
        },
        "steps": [],
        "rank_results": [],
        "pass": False,
        "blocking_pass": False,
        "report_only_pass": True,
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
    owned_groups: list[tuple[int, dict[int, int]]] = []
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
            run_step(
                schema,
                env=schema_env,
                output=output / "schema.log",
                timeout=120,
                max_pss_bytes=seal["max_tree_pss_bytes"],
                owned_groups=owned_groups,
                step_receipts=receipt["steps"],
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
                    seal["window_seconds"]
                    - seal["cleanup_reserve_seconds"]
                    - int(time.monotonic() - started)
                )
                if remaining <= 0:
                    raise TimeoutError("KDA window has no remaining compute time")
                run_step(
                    argv,
                    env=env,
                    output=output / f"cp{world}.log",
                    timeout=min(step["timeout_seconds"], remaining),
                    max_pss_bytes=seal["max_tree_pss_bytes"],
                    owned_groups=owned_groups,
                    step_receipts=receipt["steps"],
                )
                for rank in range(world):
                    result = json.loads(
                        (output / f"cp{world}-rank{rank}.json").read_text()
                    )
                    receipt["rank_results"].append(
                        {
                            "world_size": world,
                            "rank": rank,
                            **{
                                key: result[key]
                                for key in ("pass", "blocking_pass", "report_only_pass")
                            },
                        }
                    )
                    receipt["report_only_pass"] &= result["report_only_pass"]
                    if not result["blocking_pass"]:
                        raise RuntimeError(
                            f"CP{world} rank {rank} failed blocking correctness"
                        )
            verify(seal)
            receipt["blocking_pass"] = True
            receipt["pass"] = receipt["report_only_pass"]
        except BaseException as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["cleanup_done"] = True
            for root, owned in owned_groups:
                if tree_pids(root, owned):
                    receipt["cleanup_done"] = False
                    receipt["cleanup_error"] = f"Surviving owned process tree {root}"
            if not args.rehearsal:
                try:
                    receipt["after"] = idle(seal["gpu_uuids"])
                except RuntimeError as exc:
                    receipt["cleanup_done"] = False
                    receipt["cleanup_error"] = str(exc)
            receipt["elapsed_seconds"] = time.monotonic() - started
            for metric in ("pss", "rss"):
                key = f"peak_tree_{metric}_bytes"
                receipt[key] = max((step[key] for step in receipt["steps"]), default=0)
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
                    "blocking_pass",
                    "report_only_pass",
                    "cleanup_done",
                    "elapsed_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
