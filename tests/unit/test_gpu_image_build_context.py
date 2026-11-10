import base64
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

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
    kubectl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$KUBECTL_LOG"\n'
        'case " $* " in *" get pod "*" -o json "*) printf \'{"items":[]}\' ;; esac\n'
    )
    kubectl.chmod(0o755)
    digest = "sha256:" + "a" * 64
    env = {
        **os.environ,
        "DOCKER_CONFIG_PATH": str(tmp_path / "missing-docker-config.json"),
        "KUBECTL_LOG": str(kubectl_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REGISTRY_AUTH_JSON_B64": base64.b64encode(b"{}").decode(),
        "GPU_IMAGE_CLEANUP_ROOT": str(tmp_path / "cleanup"),
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


def _digest_resolver() -> str:
    script = (ROOT / "scripts/build-gpu-image.sh").read_text()
    start = script.index('"${image_repo}" "${image_tag}" <<\'PY\'\n') + len(
        '"${image_repo}" "${image_tag}" <<\'PY\'\n'
    )
    return script[start : script.index('\nPY\n)"', start)]


@pytest.mark.parametrize(
    "repo,logged",
    [
        ("docker.io/bradhiltonnw/art-gpu", "docker.io/bradhiltonnw/art-gpu"),
        ("bradhiltonnw/art-gpu", "docker.io/bradhiltonnw/art-gpu"),
        ("art-gpu", "docker.io/library/art-gpu"),
        ("art.gpu", "docker.io/library/art.gpu"),
        ("localhost", "docker.io/library/localhost"),
        ("index.docker.io/bradhiltonnw/art-gpu", "docker.io/bradhiltonnw/art-gpu"),
        ("ghcr.io/openpipe/art-gpu", "ghcr.io/openpipe/art-gpu"),
        ("localhost:5000/art-gpu", "localhost:5000/art-gpu"),
    ],
)
def test_gpu_image_digest_resolution_normalizes_repository_names(
    tmp_path: Path, repo: str, logged: str
) -> None:
    digest = "sha256:" + "b" * 64
    log = tmp_path / "build.log"
    log.write_text(
        "#12 pushing manifest for docker.io/other/image:latest@sha256:"
        + "c" * 64
        + "\n"
        + f"#13 pushing manifest for {logged}:latest@{digest} 0.4s done\n"
    )
    result = subprocess.run(
        [sys.executable, "-", str(log), repo, "latest"],
        input=_digest_resolver(),
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == digest
    unrelated = subprocess.run(
        [sys.executable, "-", str(log), repo, "other-tag"],
        input=_digest_resolver(),
        check=True,
        text=True,
        capture_output=True,
    )
    assert unrelated.stdout.strip() == ""


def test_gpu_image_node_prewarm_pulls_the_digest_on_present_nodes(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    applied = tmp_path / "applied"
    applied.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    digest = "sha256:" + "a" * 64
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$KUBECTL_LOG"\n'
        'case " $* " in\n'
        '  *" get pod "*" -o json "*) printf \'{"items":[]}\' ;;\n'
        '  *" get nodes "*"hypervisor=true"*) exit 0 ;;\n'
        '  *" get nodes "*) printf "gpu-node-a\\n" ;;\n'
        '  *" create secret "*) printf "kind: Secret\\n" ;;\n'
        '  *" apply "*) cat > "$APPLIED_DIR/$(date +%s%N).yaml" ;;\n'
        '  *" get pod "*"initContainerStatuses"*) printf "%s" "$IMAGE_ID" ;;\n'
        '  *" get pods "*) printf "art-gpu-image-prewarm-steady\\n" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    env = {
        **os.environ,
        "APPLIED_DIR": str(applied),
        "DOCKER_CONFIG_PATH": str(tmp_path / "missing-docker-config.json"),
        "IMAGE_ID": f"registry/art-gpu@{digest}",
        "KUBECTL_LOG": str(kubectl_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REGISTRY_AUTH_JSON_B64": base64.b64encode(b"{}").decode(),
        "GPU_IMAGE_CLEANUP_ROOT": str(tmp_path / "cleanup"),
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

    assert "Launching temporary BuildKit" not in result.stdout
    assert (
        f"Prewarming registry/art-gpu@{digest} and refreshing registry/art-gpu:latest "
        "on 1 GPU node(s)"
    ) in result.stdout
    assert f"Mutable tags converged to {digest}" in result.stdout
    assert "Installing steady-state art-gpu-image-prewarm DaemonSet" in result.stdout
    manifests = [path.read_text() for path in sorted(applied.iterdir())]
    # Tag-check pods only refresh the mutable tag; prewarm pods and the steady
    # DaemonSet carry the prepull init container that must pin the digest.
    pods = [m for m in manifests if "kind: Pod" in m and "-tag-check-" not in m]
    tag_checks = [m for m in manifests if "kind: Pod" in m and "-tag-check-" in m]
    daemonsets = [m for m in manifests if "kind: DaemonSet" in m]
    assert pods and tag_checks and len(daemonsets) == 1
    for manifest in [*pods, *daemonsets]:
        prepull = manifest.index("- name: prepull")
        assert manifest[prepull:].split("\n")[1].strip() == (
            f"image: registry/art-gpu@{digest}"
        )
        assert "image: registry/art-gpu:latest" in manifest  # The refresh-tag pull.
    assert all("nodeName: gpu-node-a" in m for m in [*pods, *tag_checks])
    kubectl_calls = kubectl_log.read_text()
    assert "rollout status" in kubectl_calls
    assert "--for=condition=Ready pod/art-gpu-image-prewarm-gpu-node-a" in kubectl_calls


def _run_node_prewarm_with_tag_image(
    tmp_path: Path,
    image_id: str,
    *extra_args: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
    """Prewarm one fake GPU node whose mutable-tag pulls resolve to image_id."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    applied = tmp_path / "applied"
    applied.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$KUBECTL_LOG"\n'
        'case " $* " in\n'
        '  *" get pod "*" -o json "*) printf \'{"items":[]}\' ;;\n'
        '  *" get nodes "*"hypervisor=true"*) exit 0 ;;\n'
        '  *" get nodes "*) printf "gpu-node-a\\n" ;;\n'
        '  *" create secret "*) printf "kind: Secret\\n" ;;\n'
        '  *" apply "*) cat > "$APPLIED_DIR/$(date +%s%N).yaml" ;;\n'
        '  *" get pod "*"initContainerStatuses"*) printf "%s" "$IMAGE_ID" ;;\n'
        '  *" get pods "*) printf "art-gpu-image-prewarm-steady\\n" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    env: dict[str, str] = {
        **os.environ,
        "APPLIED_DIR": str(applied),
        "DOCKER_CONFIG_PATH": str(tmp_path / "missing-docker-config.json"),
        "IMAGE_ID": image_id,
        "KUBECTL_LOG": str(kubectl_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PREWARM_TAG_CONVERGENCE_INTERVAL": "1",
        "REGISTRY_AUTH_JSON_B64": base64.b64encode(b"{}").decode(),
        "GPU_IMAGE_CLEANUP_ROOT": str(tmp_path / "cleanup"),
    }
    env.pop("GITHUB_ACTIONS", None)
    env.pop("GITHUB_STEP_SUMMARY", None)
    env.update(extra_env or {})
    command: list[str] = [
        "bash",
        str(ROOT / "scripts/build-gpu-image.sh"),
        "--image-digest",
        "sha256:" + "a" * 64,
        "--image-repo",
        "registry/art-gpu",
        "--prewarm-infra",
        "k8s/test",
        "--prewarm-nodes-only",
        "--pull-image-repo",
        "registry/art-gpu",
        *extra_args,
    ]
    result = subprocess.run(
        command, check=False, env=env, text=True, capture_output=True
    )
    manifests = [path.read_text() for path in sorted(applied.iterdir())]
    return result, manifests, kubectl_log.read_text()


def test_gpu_image_node_prewarm_warns_on_a_stale_mutable_tag(tmp_path: Path) -> None:
    # A registry mirror can keep serving an older digest for the mutable tag far
    # longer than the CI budget. The digest prewarm must still complete.
    digest = "sha256:" + "a" * 64
    stale = "sha256:" + "b" * 64
    result, manifests, kubectl_calls = _run_node_prewarm_with_tag_image(
        tmp_path,
        f"registry/art-gpu@{stale}",
        extra_env={"PREWARM_TAG_CONVERGENCE_BEST_EFFORT_ATTEMPTS": "3"},
    )

    assert result.returncode == 0, result.stderr
    tag_checks = [m for m in manifests if "kind: Pod" in m and "-tag-check-" in m]
    assert len(tag_checks) == 3  # Bounded by the best-effort budget, not 120.
    assert "Mutable tag check 3/3 has not converged" in result.stdout
    assert f"registry/art-gpu:latest has not converged to {digest}" in result.stderr
    assert f"last seen: registry/art-gpu@{stale}" in result.stderr
    assert f"Mutable tags converged to {digest}" not in result.stdout
    pods = [m for m in manifests if "kind: Pod" in m and "-tag-check-" not in m]
    daemonsets = [m for m in manifests if "kind: DaemonSet" in m]
    assert pods and len(daemonsets) == 1
    for manifest in [*pods, *daemonsets]:
        prepull = manifest.index("- name: prepull")
        assert manifest[prepull:].split("\n")[1].strip() == (
            f"image: registry/art-gpu@{digest}"
        )
    assert "rollout status" in kubectl_calls
    # A stale tag never triggers per-node or steady-state retry loops.
    assert "retrying after tag convergence delay" not in result.stdout
    assert "recreating them" not in result.stdout


def test_gpu_image_node_prewarm_can_require_mutable_tag_convergence(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    stale = "sha256:" + "b" * 64
    result, manifests, _ = _run_node_prewarm_with_tag_image(
        tmp_path,
        f"registry/art-gpu@{stale}",
        "--require-tag-convergence",
        extra_env={
            "PREWARM_TAG_CONVERGENCE_ATTEMPTS": "2",
            "PREWARM_TAG_CONVERGENCE_BEST_EFFORT_ATTEMPTS": "1",
        },
    )

    assert result.returncode != 0
    assert (
        f"Timed out waiting for registry/art-gpu:latest to resolve to {digest}"
        in result.stderr
    )
    tag_checks = [m for m in manifests if "kind: Pod" in m and "-tag-check-" in m]
    assert len(tag_checks) == 2  # Strict mode keeps the full configured budget.
    assert not [m for m in manifests if "kind: DaemonSet" in m]


def test_gpu_image_node_prewarm_reports_stale_tags_in_github_actions(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.md"
    result, _, _ = _run_node_prewarm_with_tag_image(
        tmp_path,
        "registry/art-gpu@sha256:" + "b" * 64,
        extra_env={"PREWARM_TAG_CONVERGENCE_BEST_EFFORT_ATTEMPTS": "1"},
    )
    assert result.returncode == 0, result.stderr

    gha_env = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_STEP_SUMMARY": str(summary),
        "PREWARM_TAG_CONVERGENCE_BEST_EFFORT_ATTEMPTS": "1",
    }
    second = tmp_path / "gha"
    second.mkdir()
    result, _, _ = _run_node_prewarm_with_tag_image(
        second, "registry/art-gpu@sha256:" + "b" * 64, extra_env=gha_env
    )
    assert result.returncode == 0, result.stderr
    assert "::warning title=Mutable tag not yet converged::" in result.stdout
    assert "Mutable tag not yet converged" in summary.read_text()


def test_gpu_image_workflow_passes_tag_convergence_policy() -> None:
    workflow = (ROOT / ".github/workflows/build-gpu-image.yml").read_text()
    assert "require_tag_convergence:" in workflow
    prewarm = workflow[workflow.index("- name: Prewarm GPU image on nodes") :]
    prewarm = prewarm[: prewarm.index("- name: Remove temporary image builder")]
    assert "PREWARM_REQUIRE_TAG_CONVERGENCE:" in prewarm
