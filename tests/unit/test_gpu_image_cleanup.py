import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location(
    "image_cleanup", ROOT / "scripts/ci/gpu-image-cleanup.py"
)
assert spec is not None and spec.loader is not None
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
LABEL = "art.openpipe/image-run"


def resource(kind="Pod", uid="owned-uid", **metadata):
    return {
        "kind": kind,
        "metadata": {
            "namespace": "default",
            "name": "owned",
            "uid": uid,
            "resourceVersion": "1",
            "labels": {LABEL: "123-1"},
            **metadata,
        },
    }


@pytest.fixture
def api(monkeypatch):
    state: dict[str, Any] = {
        "pod": [resource()],
        "service": [resource("Service")],
        "events": [],
    }
    elapsed = 0
    clock = cleanup.time.monotonic

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds

    monkeypatch.setattr(cleanup.time, "monotonic", lambda: clock() + elapsed)
    monkeypatch.setattr(cleanup.time, "sleep", sleep)

    def command(argv, seconds, output, *, stdout):
        assert argv[:3] == ["kubectl", "--context", "offline"]
        assert 0 < seconds <= 2
        args = argv[4:]
        state["events"].append(args)
        if args[0] == "config":
            Path(stdout.name).write_text(
                json.dumps(
                    {
                        "contexts": [
                            {
                                "name": state.get("context_name", "offline"),
                                "context": {
                                    "namespace": state.get("namespace", "default")
                                },
                            }
                        ]
                    }
                )
            )
            output.write_text("")
        elif args[0] == "get":
            assert args[2:] == [
                "-n",
                state.get("namespace", "default"),
                "-l",
                f"{LABEL}=123-1",
                "-o",
                "json",
            ]
            if state.get("get_error"):
                output.write_text("offline API unavailable")
                raise subprocess.CalledProcessError(1, argv)
            Path(stdout.name).write_text(json.dumps({"items": state[args[1]]}))
            output.write_text("kubectl warning on stderr\n")
        else:
            assert args[:2] == ["delete", "--raw"]
            body = json.loads(Path(args[4]).read_text())
            assert body["preconditions"]["uid"] == "owned-uid"
            namespace = state.get("namespace", "default")
            assert args[2] in (
                f"/api/v1/namespaces/{namespace}/pods/owned",
                f"/api/v1/namespaces/{namespace}/services/owned",
            )
            output.write_text("")
            Path(stdout.name).write_text("")
            if state.get("delete_error"):
                output.write_text("offline delete forbidden")
                raise subprocess.CalledProcessError(1, argv)
            if state.get("replacement"):
                # The UID precondition rejects a peer replacement. The next
                # ownership census excludes it; it must never be deleted.
                state["replacement_preserved"] = True
                output.write_text("Error from server (Conflict)")
                for kind in ("pod", "service"):
                    state[kind] = []
                raise subprocess.CalledProcessError(1, argv)
            kind = "pod" if "/pods/" in args[2] else "service"
            metadata = state[kind][0]["metadata"]
            race = state.get("race")
            if race and (not state.get("races") or race == "continuous_update"):
                metadata["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)
                state["races"] = state.get("races", 0) + 1
                if race == "relabel":
                    metadata["labels"] = {LABEL: "peer"}
                    state["peer"] = state[kind][0]
            if (
                "resourceVersion" in body["preconditions"]
                and body["preconditions"]["resourceVersion"]
                != metadata["resourceVersion"]
            ):
                state["conflicts"] = state.get("conflicts", 0) + 1
                if race == "relabel":
                    state[kind] = []  # The next selector census excludes the peer.
                output.write_text("Error from server (Conflict)")
                raise subprocess.CalledProcessError(1, argv)
            if race == "relabel":
                state["peer_deleted"] = True
            if not state.get("finalizer"):
                state[kind] = []

    monkeypatch.setattr(cleanup.gpu_ci, "run_child", command)
    return state


def exercise(tmp_path):
    return cleanup.cleanup(
        "offline",
        "default",
        f"{LABEL}=123-1",
        ["pod", "service"],
        tmp_path / "receipt.json",
        seconds=2,
    )


def test_uid_cleanup_observes_absence_and_retains_only_metadata(api, tmp_path):
    receipt = exercise(tmp_path)
    assert receipt["outcome"] == "ABSENT" and receipt["remaining"] == []
    assert receipt["creator_quiescence"] == "UNKNOWN"
    assert {item["kind"] for item in receipt["observed"]} == {"pod", "service"}
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt
    assert all(
        set(item) == {"kind", "name", "uid", "resourceVersion"}
        for item in receipt["observed"]
    )


def test_uid_replacement_conflict_preserves_peer(api, tmp_path):
    api["replacement"] = True
    assert exercise(tmp_path)["outcome"] == "ABSENT"
    assert api["replacement_preserved"]


def test_same_uid_relabel_after_census_preserves_peer(api, tmp_path):
    api["race"] = "relabel"
    api["service"] = []
    assert exercise(tmp_path)["outcome"] == "ABSENT"
    assert not api.get("peer_deleted"), "UID-only deletion removed the relabelled peer"
    assert api["peer"]["metadata"]["uid"] == "owned-uid"
    assert api["peer"]["metadata"]["labels"] == {LABEL: "peer"}
    assert api["conflicts"] == 1


def test_owned_version_update_recensuses_before_delete(api, tmp_path):
    api["race"] = "owned_update"
    api["service"] = []
    result = exercise(tmp_path)
    assert result["outcome"] == "ABSENT" and api["conflicts"] == 1
    assert [item["resourceVersion"] for item in result["observed"]] == ["1", "2"]


def test_continuous_version_conflicts_exhaust_budget_as_unknown(api, tmp_path):
    api["race"] = "continuous_update"
    api["service"] = []
    with pytest.raises(TimeoutError):
        exercise(tmp_path)
    assert api["conflicts"] > 1
    assert json.loads((tmp_path / "receipt.json").read_text())["outcome"] == "UNKNOWN"


def test_smoke_census_uses_the_effective_context_namespace(api, tmp_path):
    api["namespace"] = "smoke-owned"
    api["pod"] = [resource(namespace="smoke-owned")]
    api["service"] = [resource("Service", namespace="smoke-owned")]
    result = cleanup.cleanup(
        "offline",
        None,
        f"{LABEL}=123-1",
        ["pod", "service"],
        tmp_path / "receipt.json",
        seconds=2,
    )
    assert result["namespace"] == "smoke-owned" and result["outcome"] == "ABSENT"


def test_missing_smoke_context_cannot_claim_absence(api, tmp_path):
    api["context_name"] = "peer"
    with pytest.raises(ValueError, match="Exact smoke"):
        cleanup.cleanup(
            "offline",
            None,
            f"{LABEL}=123-1",
            ["pod"],
            tmp_path / "receipt.json",
            seconds=2,
        )
    result = json.loads((tmp_path / "receipt.json").read_text())
    assert result["outcome"] == "UNKNOWN" and result["namespace"] is None
    assert all(event[0] == "config" for event in api["events"])


@pytest.mark.parametrize("fault", ["get_error", "delete_error", "finalizer"])
def test_unproved_cleanup_is_unknown_and_fails(api, tmp_path, fault):
    api[fault] = True
    with pytest.raises((subprocess.CalledProcessError, TimeoutError)):
        exercise(tmp_path)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["outcome"] == "UNKNOWN"
    assert receipt["error_type"]


@pytest.mark.parametrize(
    "item",
    [
        resource(namespace="peer"),
        resource(labels={LABEL: "peer"}),
        resource(uid=""),
        resource(resourceVersion=""),
        resource(resourceVersion=None),
        resource(resourceVersion=1),
        resource(name="../peer"),
        resource("Secret"),
    ],
)
def test_foreign_identity_never_reaches_delete(api, tmp_path, item):
    api["pod"] = [item]
    with pytest.raises(ValueError, match="unowned"):
        exercise(tmp_path)
    assert all(event[0] != "delete" for event in api["events"])
    assert json.loads((tmp_path / "receipt.json").read_text())["outcome"] == "UNKNOWN"


def test_resource_version_is_opaque(api, tmp_path):
    api["pod"] = [resource(resourceVersion="opaque:version")]
    api["service"] = []
    result = exercise(tmp_path)
    assert result["outcome"] == "ABSENT"
    assert result["observed"][0]["resourceVersion"] == "opaque:version"


def test_workflow_cleanup_is_always_bounded_and_uploaded():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/build-gpu-image.yml").read_text()
    )
    steps = workflow["jobs"]["build-gpu-image"]["steps"]
    fallback = next(
        step
        for step in steps
        if step.get("name") == "Remove temporary image builder and prewarm pods"
    )
    assert fallback["if"] == "${{ always() }}" and fallback["timeout-minutes"] == 6
    assert "--kinds pod service" in fallback["run"]
    assert "--namespace-from-context" in fallback["run"]
    assert "|| true" not in fallback["run"]
    upload = next(
        step for step in steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert (
        upload["if"] == "${{ always() }}"
        and upload["with"]["if-no-files-found"] == "error"
    )
    source = (ROOT / "scripts/build-gpu-image.sh").read_text()
    assert 'art.openpipe/build-run: "${prewarm_run_uid}"' in source
    assert '"${kubectl_cmd[@]}" create -n "${buildkit_namespace}"' in source
    smoke = next(step for step in steps if step.get("id") == "smoke")["run"]
    assert "kubernetes.custom_metadata.labels=" in smoke
    assert "skypilot-cluster=${cluster}" not in smoke
    assert "timeout --signal=TERM --kill-after=5s 60s" in smoke


def test_owned_runner_keeps_json_separate_from_stderr(tmp_path):
    stdout, stderr = tmp_path / "stdout", tmp_path / "stderr"
    with stdout.open("w") as stream:
        cleanup.gpu_ci.run_child(
            [
                sys.executable,
                "-c",
                "import sys; print('{}'); print('warning', file=sys.stderr)",
            ],
            2,
            stderr,
            stdout=stream,
        )
    assert json.loads(stdout.read_text()) == {}
    assert stderr.read_text() == "warning\n"


def test_smoke_metadata_reaches_pods_and_services_offline(monkeypatch):
    pytest.importorskip("sky")
    from sky.provision.kubernetes import utils

    monkeypatch.setattr(
        utils.skypilot_config, "get_effective_region_config", lambda **_: {}
    )
    template = {
        "provider": {
            "autoscaler_service_account": {"metadata": {}},
            "autoscaler_role": {"metadata": {}},
            "autoscaler_role_binding": {"metadata": {}},
            "services": [{"metadata": {"labels": {"existing": "kept"}}}],
        },
        "available_node_types": {"ray_head_default": {"node_config": {"metadata": {}}}},
    }
    config = {"kubernetes": {"custom_metadata": {"labels": {LABEL: "123-1"}}}}
    merged = utils.combine_metadata_fields(template, config, context="offline")
    assert merged["provider"]["services"][0]["metadata"]["labels"] == {
        "existing": "kept",
        LABEL: "123-1",
    }
    assert merged["available_node_types"]["ray_head_default"]["node_config"][
        "metadata"
    ]["labels"] == {LABEL: "123-1"}


@pytest.mark.parametrize(
    "work,down,fault,expected",
    [
        (0, 0, None, 0),
        (0, 7, None, 7),
        (23, 0, None, 23),
        (23, 7, None, 23),
        (0, 0, "receipt", 1),
        (23, 0, "receipt", 23),
        (0, 0, "mkdir", 37),
        (23, 0, "mkdir", 23),
    ],
)
def test_actual_smoke_trap_records_cleanup_and_preserves_work_failure(
    tmp_path, work, down, fault, expected
):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/build-gpu-image.yml").read_text()
    )
    steps = workflow["jobs"]["build-gpu-image"]["steps"]
    source = next(step for step in steps if step.get("id") == "smoke")["run"]
    trap = source[
        source.index("cleanup() {") : source.index('"${sky_cmd[@]}" check kubernetes')
    ]
    sky = tmp_path / "offline-sky"
    sky.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\nassert sys.argv[1:3] == ['down', '-y']\n"
        f"pathlib.Path({str(tmp_path / 'down-attempted')!r}).touch()\nsys.exit({down})\n"
    )
    sky.chmod(0o700)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    if fault == "receipt":
        (receipts / "smoke-down.json").mkdir()
    script = tmp_path / "smoke.sh"
    script.write_text(
        f'set -euo pipefail\nsky_cmd=("{sky}")\ncluster=art-gpu-smoke-123-1\n'
        "dump_diagnostics() { :; }\n"
        + ("mkdir() { return 37; }\n" if fault == "mkdir" else "")
        + trap
        + f"\nexit {work}\n"
    )
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "GPU_IMAGE_CLEANUP_ROOT": str(receipts)},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stderr
    assert (tmp_path / "down-attempted").exists()
    if fault is None:
        assert (
            json.loads((receipts / "smoke-down.json").read_text())["returncode"] == down
        )
