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


def test_gpu_image_retains_only_managed_megatron_project_metadata() -> None:
    dockerfile = (ROOT / "docker/art-gpu.Dockerfile").read_text()
    final_stage = dockerfile.rsplit("\nFROM ", 1)[1]
    copies = [
        shlex.split(line)
        for line in final_stage.splitlines()
        if line.startswith("COPY ") and "/opt/src/art" in line
    ]
    assert copies == [
        [
            "COPY",
            "--from=builder",
            "--chown=sky:sky",
            "/opt/src/art/megatron_runtime/pyproject.toml",
            "/opt/src/art/megatron_runtime/uv.lock",
            "/opt/src/art/megatron_runtime/",
        ]
    ]
