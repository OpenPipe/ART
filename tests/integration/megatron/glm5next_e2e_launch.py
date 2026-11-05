"""Sealed, one-node GLM positive check; --recording creates no real resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from unittest.mock import patch

from tests.integration.megatron import glm5next_qualification_launch as base
from tests.integration.megatron.glm5next_qualification import sha, write

GRANT_REGEX = r"^(?:[0-9-]+ [0-9:]+ [AP]M MST )?GRANT glm-m4-model\b"
NATIVE_MANIFEST = base.manifest
NATIVE_KUBE = base.KubeClient
BOARD_WRITE = base.board


def literals(seal: dict, approved: str) -> list[str]:
    return [
        "GLM end-to-end CP8 vs CP1",
        f"seal_sha256={approved}",
        seal["context"],
        seal["namespace"],
        seal["pool"],
        "caladan-serving",
        f"node={seal['target_node']}",
        f"window_seconds={seal['window_seconds']}",
    ]


def has_grant(seal: dict, approved: str, contents: str) -> bool:
    return any(
        re.match(GRANT_REGEX, line)
        and all(term in line for term in literals(seal, approved))
        for line in contents.splitlines()
    )


def pending_stop(age: float, *, gpu: bool, own_scaleup: bool) -> bool:
    return age >= (2700 if own_scaleup and not gpu else 300)


def pending_state(pod: dict, events: dict, *, now: datetime | None = None) -> dict:
    """Use API timestamps and this Pod's UID; assigned startup gets 300 seconds."""
    uid = pod["metadata"]["uid"]
    gpu = pod["metadata"]["name"].endswith("qualification")
    own = any(
        e.get("involvedObject", {}).get("uid") == uid
        and e.get("reason") == "TriggeredScaleUp"
        for e in events["items"]
    )
    since = pod["metadata"]["creationTimestamp"]
    if pod.get("spec", {}).get("nodeName") and not gpu:
        scheduled = next(
            c
            for c in pod["status"]["conditions"]
            if c["type"] == "PodScheduled" and c["status"] == "True"
        )
        since = scheduled["lastTransitionTime"]
        own = False
    age = (
        (now or datetime.now(timezone.utc))
        - datetime.fromisoformat(since.replace("Z", "+00:00"))
    ).total_seconds()
    return {
        "age_s": age,
        "own_TriggeredScaleUp": own,
        "gpu": gpu,
        "stopped": pending_stop(age, gpu=gpu, own_scaleup=own),
    }


class Recording(base.RecordingClient):
    def __init__(self, health: str, pending: str):
        super().__init__(health)
        self.pending_case = pending

    def pod(self, value: dict, root: Path, work: Path, seal: dict) -> None:
        gpu = value["metadata"]["name"].endswith("qualification")
        case = self.pending_case
        if case != "pass" and (not gpu or case == "gpu-pending"):
            scaleup = case in ("scale-success", "scale-timeout", "gpu-pending")
            age = (
                1804
                if case == "scale-success"
                else (2700 if case == "scale-timeout" else 300)
            )
            stopped = pending_stop(age, gpu=gpu, own_scaleup=scaleup)
            self.events.append(
                {
                    "pending": {
                        "case": case,
                        "age_s": age,
                        "own_TriggeredScaleUp": scaleup,
                        "gpu": gpu,
                        "stopped": stopped,
                        "simulated": True,
                    }
                }
            )
            if stopped:
                self.create(value)
                raise RuntimeError("NEEDS-LEAD: synthetic Pending deadline reached")
        super().pod(value, root, work, seal)


class Kube(NATIVE_KUBE):
    """Reuse the qualified cluster lifecycle, add UID-scoped Pending deadlines."""

    def __init__(self, seal: dict, output: Path):
        super().__init__(seal, output)
        self.pending: dict[str, float] = {}
        self.identities: dict[tuple[str, str], str] = {}
        namespace_uid = self.call(
            ["get", "namespace", seal["namespace"], "-o", "jsonpath={.metadata.uid}"]
        ).strip()
        if namespace_uid != seal["namespace_uid"]:
            raise RuntimeError("standing namespace UID changed")
        self.started_notice = False

    def create(self, value: dict) -> None:
        super().create(value)
        key = (value["kind"].lower(), value["metadata"]["name"])
        obj = json.loads(super().call(["get", *key, "-o", "json"]))
        self.identities[key] = obj["metadata"]["uid"]

    def call(
        self, args: list[str], data: dict | None = None, seconds: int = 120
    ) -> str:
        out = super().call(args, data, seconds)
        if (
            len(args) >= 5
            and args[:2] == ["get", "pod"]
            and args[-2:] == ["-o", "json"]
        ):
            pod = json.loads(out)
            if (
                pod.get("status", {}).get("phase") == "Running"
                and pod["metadata"]["name"].endswith("qualification")
                and not self.started_notice
            ):
                BOARD_WRITE(
                    f"GPU-START glm-m4-model GLM end-to-end CP8 vs CP1 node={self.seal['target_node']} GPUs=8"
                )
                self.started_notice = True
            if pod.get("status", {}).get("phase") == "Pending":
                uid, name = pod["metadata"]["uid"], pod["metadata"]["name"]
                events = json.loads(
                    super().call(
                        [
                            "get",
                            "events",
                            "--field-selector",
                            f"involvedObject.uid={uid}",
                            "-o",
                            "json",
                        ],
                        seconds=30,
                    )
                )
                state = pending_state(pod, events)
                if state["stopped"]:
                    write(
                        self.output / "pending-stop.json",
                        {
                            "status": "NEEDS-LEAD",
                            "pod": name,
                            "uid": uid,
                            **state,
                            "events": events,
                        },
                    )
                    raise RuntimeError("NEEDS-LEAD: sealed Pending deadline reached")
        return out

    def cleanup(self) -> None:
        # Refuse to delete a replacement with a different UID.
        for key, uid in self.identities.items():
            found = super().call(["get", *key, "--ignore-not-found", "-o", "json"])
            if found.strip() and json.loads(found)["metadata"]["uid"] != uid:
                raise RuntimeError(f"owned UID changed: {key}")
        super().cleanup()
        if (
            self.call(
                [
                    "get",
                    "namespace",
                    self.seal["namespace"],
                    "-o",
                    "jsonpath={.metadata.uid}",
                ]
            ).strip()
            != self.seal["namespace_uid"]
        ):
            raise RuntimeError("standing namespace UID changed at cleanup")


def steps(seal: dict, root: Path, work: Path, *, tiny: bool) -> list[dict]:
    return [
        {"name": name, "trainer_plan": None}
        for name in ("volumes", "hybridep-setup", "acquire", "identity")
    ] + [
        {
            "name": variant,
            "trainer_plan": {
                "task": "forward",
                "world_size": world,
                "cp": world,
                "ep": world,
            },
        }
        for variant, world in (("B0", 1), ("B1", 8))
    ]


def manifest(seal: dict, name: str, gpu: bool) -> dict:
    pod = NATIVE_MANIFEST(seal, name, gpu)
    pod["spec"]["activeDeadlineSeconds"] = seal["gpu_ads" if gpu else "stage0_ads"]
    container = pod["spec"]["containers"][0]
    container["command"][-1] = container["command"][-1].replace(
        "glm5next_qualification_launch.py", "glm5next_e2e_launch.py"
    )
    # Native source extracted once by the parent, before any torchrun processes.
    container["env"] = [
        {"name": key, "value": value}
        for key, value in {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MAX_JOBS": "8",
            "TORCH_CUDA_ARCH_LIST": "10.0",
            "HYBRID_EP_MULTINODE": "0",
            "USE_NIXL": "0",
            "ART_MEGATRON_RANDOM_STATE": "20261105",
        }.items()
    ]
    return pod


def pipeline(
    seal: dict, root: Path, work: Path, *, tiny: bool, deadline: float | None = None
) -> dict:
    from tests.integration.megatron.glm5next_e2e import compare

    deadline = min(deadline or float("inf"), time.monotonic() + seal["gpu_ads"] - 300)
    env = {
        **os.environ,
        "PYTHONPATH": f"{root / 'src'}:{root}",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ART_MEGATRON_RANDOM_STATE": "20261105",
    }
    python = (
        seal["native_python"] if tiny else "/opt/art/megatron_runtime/.venv/bin/python"
    )
    worker = root / "tests/integration/megatron/glm5next_e2e.py"
    inputs = root / "qualification-inputs"
    records = []

    def run(name: str, argv: list[str], visible: str = "") -> None:
        native_cap = seal["caps_s"][name]
        # CPU import/reference arithmetic is not a native timing calibration.
        # Native execution retains the doubled receipt cap exactly.
        execution_cap = max(600, native_cap) if tiny else native_cap
        records.append(
            {
                "name": name,
                "native_cap_s": native_cap,
                "execution_cap_s": execution_cap,
                **base.run(
                    argv,
                    max(1, min(execution_cap, int(deadline - time.monotonic()))),
                    work / "logs" / f"{name}.log",
                    env={**env, "CUDA_VISIBLE_DEVICES": visible},
                    cwd=root,
                    pss_limit=(120 if tiny else 1200) << 30,
                ),
            }
        )
        write(work / "results/stages.json", records)

    for task in ("volumes", "hybridep-setup", "acquire", "identity"):
        run(
            task,
            base.task_argv(
                python,
                root,
                task,
                inputs,
                work / "model",
                work / f"results/{task}",
                tiny and task != "identity",
            ),
        )
    for variant, world in (("B0", 1), ("B1", 8)):
        run(
            variant,
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={world}",
                str(worker),
                "--variant",
                variant,
                "--model",
                str(work / "model"),
                "--identity",
                str(work / "initial"),
                "--documents",
                str(inputs / "documents.json"),
                "--output",
                str(work / "results" / variant),
                *(["--cpu"] if tiny else []),
            ],
            "" if tiny else ("0" if world == 1 else "0,1,2,3,4,5,6,7"),
        )
    result = compare(work / "results", work / "results/comparison.json")
    result.update(
        steps=records,
        execution="real_TrainerRank_tiny_CPU_Gloo" if tiny else "native_GPU",
        gpu_only_unmeasured=tiny,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recording", action="store_true")
    parser.add_argument(
        "--recording-pending",
        choices=("pass", "blocked", "scale-success", "scale-timeout", "gpu-pending"),
        default="pass",
    )
    parser.add_argument(
        "--recording-node-health",
        default="pass",
        choices=("pass", "match-name", "match-label", "permission-denied"),
    )
    parser.add_argument("--approved-seal")
    parser.add_argument("--controller-owner")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pod-mode", choices=("stage0", "pipeline"))
    args = parser.parse_args()
    if not args.recording and (
        args.recording_node_health != "pass" or args.recording_pending != "pass"
    ):
        parser.error("recording scenarios require --recording")

    def board_notice(message: str) -> None:
        if not message.startswith("GPU-START"):
            BOARD_WRITE(message)

    with patch.multiple(
        base,
        steps=steps,
        pipeline=pipeline,
        manifest=manifest,
        KubeClient=Kube,
        has_grant=has_grant,
        board=board_notice,
    ):
        if args.pod_mode:
            seal = json.loads(args.seal.read_text())
            root = Path(__file__).parents[3]
            if args.pod_mode == "stage0":
                from tests.integration.megatron.glm5next_qualification import stage0

                stage0(
                    args.output / "stage0.json",
                    expected_cuda="13.0",
                    require_toolkit=True,
                    inputs=root / "qualification-inputs",
                )
            else:
                status = {"status": "failed"}
                try:
                    status = pipeline(seal, root, args.output, tiny=False)
                finally:
                    write(args.output / "done.json", {"status": status["status"]})
                    # Keep results available to the controller on numeric or
                    # worker failure. Deadline and finally cleanup still bound it.
                    until = time.monotonic() + 300
                    while (
                        not (args.output / "collected").exists()
                        and time.monotonic() < until
                    ):
                        time.sleep(1)
                if status["status"] != "pass":
                    raise SystemExit(1)
            return
        if not args.execute or not args.approved_seal:
            parser.error("--execute and --approved-seal are required")
        seal = base.preflight(args.seal, args.approved_seal)
        if args.recording:
            grant = "GRANT glm-m4-model " + " ".join(literals(seal, args.approved_seal))
            assert has_grant(seal, args.approved_seal, grant)
        else:
            if not has_grant(seal, args.approved_seal, base.BOARD.read_text()):
                raise ValueError("no exact GRANT with every required literal")
        client = (
            Recording(args.recording_node_health, args.recording_pending)
            if args.recording
            else None
        )
        try:
            base.execute(
                args.seal,
                args.approved_seal,
                args.output,
                client,
                controller_owner=args.controller_owner,
            )
        finally:
            path = args.output / "receipt.json"
            if path.exists():
                receipt = json.loads(path.read_text())
                receipt.update(
                    requested=sha(args.seal),
                    approved=args.approved_seal,
                    final_source_head=seal["source_head"],
                    grant_regex=GRANT_REGEX,
                    required_literals=literals(seal, args.approved_seal),
                    real_resources_created=0 if args.recording else None,
                    gpu_pod_created=bool(
                        client
                        and any(
                            event.get("create", {}).get("kind") == "Pod"
                            and event["create"]["metadata"]["name"].endswith(
                                "qualification"
                            )
                            for event in client.events
                        )
                    ),
                )
                write(path, receipt)
                if client:
                    write(args.output / "kubernetes-recording.json", client.events)


if __name__ == "__main__":
    main()
