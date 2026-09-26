import base64
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).parents[2]


def test_gpu_image_build_context_copies_every_local_docker_source() -> None:
    dockerfile = (ROOT / "docker/art-gpu.Dockerfile").read_text()
    build_script = (ROOT / "scripts/build-gpu-image.sh").read_text()

    for line in dockerfile.splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        for source in shlex.split(line)[1:-1]:
            assert f"${{repo_root}}/{source}" in build_script


def test_gpu_image_build_cleans_only_its_oneshot_prewarm_pods() -> None:
    build_script = (ROOT / "scripts/build-gpu-image.sh").read_text()
    workflow = (ROOT / ".github/workflows/build-gpu-image.yml").read_text()

    run_label = 'art.openpipe/prewarm-run: "${prewarm_run_uid}"'
    run_selector = (
        "art.openpipe/prewarm-name=${prewarm_name},"
        "art.openpipe/prewarm-run=${prewarm_run_uid}"
    )
    assert build_script.count(run_label) == 2
    assert build_script.count(run_selector) == 1
    assert "PREWARM_RUN_UID must be a valid Kubernetes label value" in build_script
    assert "PREWARM_RUN_UID: ${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert (
        "art.openpipe/prewarm-name=art-gpu-image-prewarm,"
        "art.openpipe/prewarm-run=${PREWARM_RUN_UID}"
    ) in workflow


def test_gpu_image_node_prewarm_reuses_an_existing_digest(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$KUBECTL_LOG"\n')
    kubectl.chmod(0o755)
    digest = "sha256:" + "a" * 64
    env = {
        **os.environ,
        "DOCKER_CONFIG_PATH": str(tmp_path / "missing-docker-config.json"),
        "KUBECTL_LOG": str(kubectl_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REGISTRY_AUTH_JSON_B64": base64.b64encode(b"{}").decode(),
    }

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build-gpu-image.sh"),
            "--image-digest",
            digest,
            "--image-repo",
            "registry/art-gpu",
            "--prewarm-infra",
            "k8s/test",
            "--prewarm-nodes-only",
            "--pull-image-repo",
            "registry/art-gpu",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Prewarming Kubernetes context test" in result.stdout
    assert "Skipping GPU node prewarm" in result.stdout
    assert "Launching temporary BuildKit" not in result.stdout
    kubectl_calls = kubectl_log.read_text()
    assert "get nodes" in kubectl_calls
    assert " apply " not in kubectl_calls


def test_gpu_image_node_prewarm_requires_an_immutable_digest() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build-gpu-image.sh"),
            "--image-digest",
            "latest",
            "--prewarm-nodes-only",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "requires --image-digest sha256:<64 lowercase hex>" in result.stderr


def test_gpu_image_workflow_qualifies_digest_before_fleet_prewarm() -> None:
    workflow = (ROOT / ".github/workflows/build-gpu-image.yml").read_text()

    steps = [
        workflow.index(f"- name: {name}")
        for name in (
            "Build, push, and prewarm Modal image",
            "Smoke launch immutable image",
            "Dispatch Caladan GPU image build",
            "Prewarm GPU image on nodes",
            "Remove temporary image builder and prewarm pods",
        )
    ]
    assert steps == sorted(steps)
    assert workflow.count("GH_TOKEN: ${{ github.token }}") == 2
    assert "IMAGE_DIGEST: ${{ steps.build.outputs.image_digest }}" in workflow
    assert '"art_image": f"{image_repo}@{image_digest}"' in workflow
    assert "!cancelled() &&" in workflow
    assert "steps.build.outcome == 'success' &&" in workflow
    assert (
        "steps.smoke.outcome == 'success' || steps.smoke.outcome == 'skipped'"
        in workflow
    )
