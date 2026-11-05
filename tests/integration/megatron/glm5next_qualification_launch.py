"""Sealed M4 controller. RecordingClient exercises execute, bootstrap and cleanup."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
WORKER = "tests/integration/megatron/glm5next_qualification.py"
IMAGE = "ghcr.io/openpipe/caladan-stark-trainer@sha256:7bebcc01d83c60b1d019cef07637339690730649684d5a49a88f30ccdaa1f6ae"
BOARD = Path("/home/brad/.local/share/stark-v0/2026-10-20/workstreams/_shared/BOARD.md")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def board(message: str) -> None:
    now = datetime.now(timezone(timedelta(hours=-7))).strftime("%Y-%m-%d %I:%M %p MST")
    with BOARD.open("a") as f:
        f.write(f"\n{now} {message}\n")


def process_tree_pss(root: int, owned: dict[int, int]) -> int:
    """Track birth times across sessions/reparenting; never add RSS."""
    processes = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = path.read_text().rsplit(")", 1)[1].split()
            processes[int(path.parent.name)] = (int(values[1]), int(values[19]))
        except (OSError, ValueError, IndexError):
            continue
    if root in processes and root not in owned:
        owned[root] = processes[root][1]
    changed = True
    while changed:
        changed = False
        for pid, (parent, born) in processes.items():
            if (
                parent in owned
                and parent in processes
                and owned[parent] == processes[parent][1]
                and pid not in owned
            ):
                owned[pid] = born
                changed = True
    total = 0
    for pid, born in tuple(owned.items()):
        if pid not in processes or processes[pid][1] != born:
            continue
        try:
            lines = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
            total += (
                int(next(line for line in lines if line.startswith("Pss:")).split()[1])
                * 1024
            )
        except (OSError, StopIteration):
            # Exited/zombie processes have no mappings. A live unreadable worker is unsafe.
            try:
                state = (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                )
            except OSError:
                continue
            if state != "Z" and Path(f"/proc/{pid}/smaps_rollup").exists():
                raise RuntimeError(f"cannot read PSS for owned PID {pid}") from None
    return total


def terminate(owned: dict[int, int]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, born in tuple(owned.items()):
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[19]) == born:
                    os.kill(pid, sig)
            except (OSError, ValueError, IndexError):
                pass
        if sig == signal.SIGTERM:
            time.sleep(0.3)


def run(
    argv: list[str],
    seconds: int,
    log: Path,
    *,
    env: dict | None = None,
    cwd: Path = ROOT,
    pss_limit: int = 120 << 30,
) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started, peak, owned = time.monotonic(), 0, {}
    with (
        log.open("w") as f,
        subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc,
    ):
        try:
            while proc.poll() is None:
                peak = max(peak, process_tree_pss(proc.pid, owned))
                if peak > pss_limit:
                    raise MemoryError(f"process tree PSS {peak} exceeds {pss_limit}")
                if time.monotonic() - started > seconds:
                    raise TimeoutError(f"step timeout {seconds}s: {argv[0]}")
                time.sleep(0.2)
            if proc.returncode:
                raise RuntimeError(f"exit {proc.returncode}: {log}")
        finally:
            terminate(owned)
            proc.wait(timeout=30)
    return {
        "seconds": time.monotonic() - started,
        "peak_tree_pss_bytes": peak,
        "argv": argv,
        "log": str(log),
    }


def preflight(seal_path: Path, approved: str) -> dict:
    if sha(seal_path) != approved:
        raise ValueError("approved seal mismatch")
    seal = json.loads(seal_path.read_text())
    if seal["holds"] or seal["launch_ready"] is not True:
        raise ValueError("seal is not launch ready")
    for record in seal["files"]:
        path = Path(record["path"])
        if path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
            raise ValueError(f"sealed input changed: {path}")
    return seal


def task_argv(
    python: str,
    root: Path,
    task: str,
    inputs: Path,
    model: Path,
    output: Path,
    tiny: bool,
) -> list[str]:
    args = [
        python,
        str(root / WORKER),
        "--task",
        task,
        "--inputs",
        str(inputs),
        "--model",
        str(model),
        "--output",
        str(output),
    ]
    return args + (["--tiny"] if tiny else [])


def steps(seal: dict, root: Path, work: Path, *, tiny: bool) -> list[dict]:
    python = (
        seal["native_python"]
        if seal["kind"] == "workstation"
        else "/opt/art/megatron_runtime/.venv/bin/python"
    )
    if tiny:
        python = seal["native_python"]
    inputs = root / "qualification-inputs"
    model = work / "model"
    commands = []

    def add(
        name: str,
        task: str,
        seconds: int,
        cp: int = 1,
        ep: int = 1,
        count: int = 1,
        source: Path | None = None,
    ) -> None:
        argv = task_argv(
            python, root, task, inputs, source or model, work / "results" / name, tiny
        )
        if count > 1 and not tiny:
            argv = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={count}",
                *argv[1:],
            ]
        commands.append(
            {
                "name": name,
                "argv": argv,
                "seconds": seconds,
                "env": {
                    "ART_MEGATRON_CONTEXT_PARALLEL_SIZE": str(cp),
                    "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE": str(ep),
                },
            }
        )

    add("stage0", "stage0", 600)
    add("volumes", "volumes", 180)
    if seal["kind"] == "workstation":
        for fixture in ("F4", "F8"):
            path = Path(seal["fixtures"]) / fixture
            add(f"identity-{fixture}", "identity", 300, source=path)
            add(fixture, "parity", 2100, count=2, source=path)
    else:
        add("acquire", "acquire", 4200)
        add("identity", "identity", 600)
        add("EP8-CP1", "forward", 1200, ep=8, count=8)
        # Pinned, lane-owned operator workers, using the separate HF overlay only for M3.
        for lane in ("kda", "dsa"):
            oracle = str(work / "oracle-site")
            env = (
                {
                    "PYTHONPATH": oracle
                    + os.pathsep
                    + str(root / "src")
                    + os.pathsep
                    + str(root)
                }
                if lane == "dsa"
                else {}
            )
            script = "kda_parity.py" if lane == "kda" else "glm5next_dsa_parity.py"
            argv = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=8",
                str(root / "tests/integration/megatron" / script),
                "--fixture",
                "R2" if tiny else "F4" if lane == "kda" else "F8",
                "--device",
                "cpu" if tiny else "cuda",
                "--fixtures",
                str(Path(seal["fixtures"]) if tiny else work / "fixtures"),
                "--output",
                str(work / "results" / f"{lane}-cp8"),
            ]
            argv += (
                ["--tokens", "128" if tiny else "1024"]
                if lane == "kda"
                else [
                    "--lengths",
                    "128" if tiny else "512,2048,2052,8192,16384",
                    "--lora-rank",
                    "2" if tiny else "8",
                ]
            )
            commands.append(
                {"name": f"{lane}-cp8", "argv": argv, "seconds": 900, "env": env}
            )
        add("CP8-EP8", "forward", 1200, cp=8, ep=8, count=8)
        add("rank32", "learn", 1200, cp=8, ep=8, count=8)
        for length in (32768, 65536, 131072, 262144):
            add(f"context-{length}", f"context-{length}", 600, cp=8, ep=8, count=8)
    return commands


def prepare_worker_fixtures(root: Path, work: Path, *, tiny: bool) -> None:
    import shutil
    import urllib.request

    if not tiny:
        # M2/M3 read one layer from the full snapshot; no second weight download.
        config = json.loads((work / "model/config.json").read_text())
        config = config.get("text_config", config)
        for name, layers in (("F4", 4), ("F8", 8)):
            target = work / "fixtures" / name
            target.mkdir(parents=True, exist_ok=True)
            value = {**config, "num_hidden_layers": layers}
            for key in ("layer_types", "mlp_layer_types"):
                if key in value:
                    value[key] = value[key][:layers]
            write(target / "config.json", value)
            for path in (work / "model").glob("*.safetensors*"):
                (target / path.name).symlink_to(path)
    wheel = json.loads((root / "qualification-inputs/oracle-wheel.json").read_text())
    path = work / wheel["filename"]
    with (
        urllib.request.urlopen(wheel["url"], timeout=120) as src,
        path.open("wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    if sha(path) != wheel["sha256"]:
        raise ValueError("oracle wheel hash mismatch")
    import zipfile

    with zipfile.ZipFile(path) as archive:
        archive.extractall(work / "oracle-site")


def pipeline(seal: dict, root: Path, work: Path, *, tiny: bool) -> dict:
    from tests.integration.megatron.glm5next_qualification import compare

    records = []
    deadline = time.monotonic() + seal["window_seconds"] - 600
    for step in steps(seal, root, work, tiny=tiny):
        env = dict(
            os.environ,
            PYTHONPATH=str(root / "src") + os.pathsep + str(root),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            HF_HOME=str(work / "hf"),
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            PYTHON_DOTENV_DISABLED="1",
            FLA_TILELANG="0",
            **{},
        )
        env.update(step["env"])
        if tiny:
            env["CUDA_VISIBLE_DEVICES"] = ""
        if step["name"] == "dsa-cp8" and tiny:
            env["PYTHONPATH"] = (
                str(work / "oracle-site")
                + os.pathsep
                + str(root / "src")
                + os.pathsep
                + str(root)
            )
        try:
            receipt = run(
                step["argv"],
                min(step["seconds"], max(1, int(deadline - time.monotonic()))),
                work / "logs" / (step["name"] + ".log"),
                env=env,
                cwd=root,
                pss_limit=min(seal["pss_limit_bytes"], 120 << 30)
                if tiny
                else seal["pss_limit_bytes"],
            )
            records.append({"name": step["name"], "status": "pass", **receipt})
            if step["name"] == "acquire":
                prepare_worker_fixtures(root, work, tiny=tiny)
            if step["name"] in ("F4", "F8"):
                compare(
                    work / "results" / step["name"],
                    Path(seal["fixtures"]),
                    step["name"],
                    work / "results" / (step["name"] + "-comparison.json"),
                    tiny,
                )
        except Exception as exc:
            records.append(
                {"name": step["name"], "status": "failed", "error": str(exc)}
            )
            write(work / "results/pipeline.json", records)
            # Each context is isolated in its own torchrun; OOM/refusal is a measured result.
            if not step["name"].startswith("context-"):
                raise
        write(work / "results/pipeline.json", records)
    return {"steps": records, "execution": "tiny_cpu" if tiny else "native_gpu"}


def manifest(seal: dict, name: str, gpu: bool) -> dict:
    resources = {
        "cpu": "96" if gpu else "4",
        "memory": "1400Gi" if gpu else "16Gi",
        "ephemeral-storage": "1200Gi" if gpu else "8Gi",
    }
    if gpu:
        resources["nvidia.com/gpu"] = "8"
    volumes = [
        {"name": "work", "emptyDir": {"sizeLimit": "1200Gi"}},
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
    ]
    mounts = [
        {"name": "work", "mountPath": "/work"},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    for i, _ in enumerate(seal["chunks"]):
        volumes.append({"name": f"input-{i}", "configMap": {"name": f"m4-source-{i}"}})
        mounts.append(
            {"name": f"input-{i}", "mountPath": f"/sealed/{i}", "readOnly": True}
        )
    # This bootstrap uses only the real image's stdlib before the sealed ART import.
    bootstrap = (
        "import pathlib,hashlib,tarfile,os,json; "
        "p=pathlib.Path('/work/source.tar.gz'); "
        f"p.write_bytes(b''.join(pathlib.Path('/sealed/'+str(i)+'/chunk').read_bytes() for i in range({len(seal['chunks'])}))); "
        f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={seal['source_sha256']!r}; "
        "tarfile.open(p).extractall('/work/source',filter='data'); "
        "os.environ['PYTHONPATH']='/work/source/src:/work/source'; "
        f"os.execv('/opt/art/megatron_runtime/.venv/bin/python',['python','/work/source/tests/integration/megatron/glm5next_qualification_launch.py','--pod-mode',{'pipeline' if gpu else 'stage0'!r},'--seal','/work/source/qualification-inputs/pod-plan.json','--output','/work'])"
    )
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": seal["namespace"],
            "labels": {"stark-lane": "glm-m4-model"},
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": 13200 if gpu else 600,
            "priorityClassName": "caladan-serving",
            "nodeSelector": {"caladan.openpipe.ai/pool": "caladan-b300-spot"},
            "tolerations": [
                {
                    "key": "caladan.openpipe.ai/spot",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
            "imagePullSecrets": [{"name": "caladan-ghcr-pull"}],
            "securityContext": {"fsGroup": 1000},
            "containers": [
                {
                    "name": "qualification",
                    "image": IMAGE,
                    "command": [
                        "/opt/art/megatron_runtime/.venv/bin/python",
                        "-u",
                        "-c",
                        bootstrap,
                    ],
                    "resources": {"requests": resources, "limits": resources},
                    "volumeMounts": mounts,
                }
            ],
            "volumes": volumes,
        },
    }


class KubeClient:
    def __init__(self, seal: dict, output: Path):
        self.seal, self.output = seal, output

    def call(
        self, args: list[str], data: dict | None = None, seconds: int = 120
    ) -> str:
        value = subprocess.run(
            [
                "kubectl",
                "--context",
                self.seal["context"],
                "--namespace",
                self.seal["namespace"],
                *args,
            ],
            input=json.dumps(data) if data else None,
            text=True,
            capture_output=True,
            timeout=seconds,
            check=True,
        )
        return value.stdout

    def create(self, value: dict) -> None:
        self.call(["create", "-f", "-"], value)

    def pod(self, value: dict, root: Path, work: Path, seal: dict) -> None:
        self.create(value)
        name = value["metadata"]["name"]
        deadline = time.monotonic() + value["spec"]["activeDeadlineSeconds"] + 60
        while time.monotonic() < deadline:
            status = json.loads(self.call(["get", "pod", name, "-o", "json"]))
            phase = status.get("status", {}).get("phase")
            if phase == "Running" and name == "m4-qualification":
                try:
                    done = json.loads(
                        self.call(["exec", name, "--", "cat", "/work/done.json"])
                    )
                except (subprocess.CalledProcessError, json.JSONDecodeError):
                    done = None
                if done is not None:
                    target = self.output / "pod-results"
                    target.mkdir(exist_ok=True)
                    self.call(
                        ["cp", f"{name}:/work/results", str(target / "results")],
                        seconds=300,
                    )
                    self.call(
                        ["cp", f"{name}:/work/logs", str(target / "logs")], seconds=300
                    )
                    write(target / "done.json", done)
                    self.call(["exec", name, "--", "touch", "/work/collected"])
                    if done["status"] != "pass":
                        raise RuntimeError(f"qualification failed: {done}")
            if phase in ("Succeeded", "Failed"):
                logs = self.call(["logs", name], seconds=120)
                (self.output / (name + ".log")).write_text(logs)
                if phase != "Succeeded":
                    raise RuntimeError(f"Pod failed: {name}")
                if name.endswith("stage0") and "M4_STAGE0_PASS" not in logs:
                    raise RuntimeError("stage0 missing success marker")
                return
            time.sleep(5)
        raise TimeoutError(name)

    def cleanup(self) -> None:
        self.call(
            [
                "delete",
                "namespace",
                self.seal["namespace"],
                "--wait=true",
                "--timeout=180s",
            ],
            seconds=210,
        )
        result = self.call(
            [
                "get",
                "namespace",
                self.seal["namespace"],
                "--ignore-not-found",
                "-o",
                "name",
            ]
        )
        if result.strip():
            raise RuntimeError("namespace cleanup incomplete")


class RecordingClient:
    def __init__(self):
        self.events: list[dict] = []
        self.ran = False

    def create(self, value: dict) -> None:
        self.events.append({"create": value})

    def pod(self, value: dict, root: Path, work: Path, seal: dict) -> None:
        self.create(value)
        assert "preemptionPolicy" not in value["spec"]
        assert value["spec"]["nodeSelector"] == {
            "caladan.openpipe.ai/pool": "caladan-b300-spot"
        }
        if value["metadata"]["name"].endswith("stage0"):
            run(
                task_argv(
                    seal["native_python"],
                    root,
                    "stage0",
                    root / "qualification-inputs",
                    work / "model",
                    work / "stage0",
                    True,
                ),
                600,
                work / "stage0.log",
                env=dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES="",
                    PYTHONPATH=str(root / "src") + os.pathsep + str(root),
                ),
                cwd=root,
            )
        else:
            self.result = pipeline(seal, root, work, tiny=True)
            self.ran = True

    def cleanup(self) -> None:
        self.events.append({"delete_namespace": True, "verified_absent": True})


def idle(uuids: list[str]) -> None:
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    memory = {
        line.split(",")[0].strip(): int(line.split(",")[1])
        for line in value.splitlines()
    }
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(memory[u] != 0 or u in processes for u in uuids):
        raise RuntimeError("granted workstation GPUs are not idle")


def execute(
    seal_path: Path, approved: str, output: Path, client: Any | None = None
) -> None:
    seal = preflight(seal_path, approved)
    recording = isinstance(client, RecordingClient)
    if not recording and not any(
        "GRANT" in line and "glm-m4-model" in line and approved in line
        for line in BOARD.read_text().splitlines()
    ):
        raise ValueError("no BOARD grant for this exact seal")
    output.mkdir(parents=True, exist_ok=False)
    # Exercise the actual sealed-source acquisition and bootstrap extraction on CPU too.
    source = output / "source"
    archive_path = output / "sealed-source.tar.gz"
    with archive_path.open("wb") as dst:
        for chunk in seal["chunks"]:
            dst.write(Path(chunk).read_bytes())
    if sha(archive_path) != seal["source_sha256"]:
        raise ValueError("source chunk binding mismatch")
    with tarfile.open(archive_path) as archive:
        archive.extractall(source, filter="data")
    for path in source.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    work = output / "work"
    work.mkdir()
    receipt: dict = {
        "seal_sha256": approved,
        "kind": seal["kind"],
        "recording": recording,
        "cleanup_done": False,
    }
    try:
        if seal["kind"] == "workstation":
            if not recording:
                idle(seal["uuids"])
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(seal["uuids"])
                board(
                    f"GPU-START glm-m4-model workstation seal={approved} uuids={seal['uuids']}"
                )
            receipt["science"] = pipeline(seal, source, work, tiny=recording)
        else:
            client = client or KubeClient(seal, output)
            for i, chunk in enumerate(seal["chunks"]):
                client.create(
                    {
                        "apiVersion": "v1",
                        "kind": "ConfigMap",
                        "metadata": {
                            "name": f"m4-source-{i}",
                            "namespace": seal["namespace"],
                        },
                        "immutable": True,
                        "binaryData": {
                            "chunk": base64.b64encode(Path(chunk).read_bytes()).decode()
                        },
                    }
                )
            client.pod(manifest(seal, "m4-stage0", False), source, work, seal)
            if not recording:
                board(
                    f"GPU-START glm-m4-model b300 namespace={seal['namespace']} seal={approved}"
                )
            client.pod(manifest(seal, "m4-qualification", True), source, work, seal)
            if recording:
                receipt["science"] = cast(RecordingClient, client).result
        receipt["status"] = "pass"
    finally:
        if seal["kind"] == "cluster":
            if client is not None:
                client.cleanup()
                if recording:
                    receipt["events"] = [
                        {"operation": list(event)[0]}
                        for event in cast(RecordingClient, client).events
                    ]
                else:
                    board(
                        f"CLEANUP-DONE glm-m4-model namespace={seal['namespace']} verified absent"
                    )
        elif not recording:
            idle(seal["uuids"])
            board(
                "RELEASED glm-m4-model workstation verified 0 MiB and no compute processes"
            )
        receipt["cleanup_done"] = True
        if sha(archive_path) != seal["source_sha256"]:
            raise ValueError("sealed source changed during run")
        write(output / "receipt.json", receipt)
        if not recording:
            board(f"RESULT-LINK glm-m4-model {output / 'receipt.json'} seal={approved}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--approved-seal")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pod-mode", choices=("stage0", "pipeline"))
    args = parser.parse_args()
    if args.pod_mode:
        seal = json.loads(args.seal.read_text())
        if args.pod_mode == "stage0":
            from tests.integration.megatron.glm5next_qualification import stage0

            stage0(args.output / "stage0.json")
        else:
            status = {"status": "failed"}
            try:
                result = pipeline(seal, ROOT, args.output, tiny=False)
                print(json.dumps(result), flush=True)
                status = {"status": "pass"}
            finally:
                write(args.output / "done.json", status)
                deadline = time.monotonic() + 360
                while (
                    not (args.output / "collected").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(1)
    elif args.execute and args.approved_seal:
        execute(
            args.seal,
            args.approved_seal,
            args.output,
            RecordingClient() if args.recording else None,
        )
    else:
        parser.error("--execute and --approved-seal required")


if __name__ == "__main__":
    main()
