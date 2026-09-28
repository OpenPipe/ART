from pathlib import Path
import shlex

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
