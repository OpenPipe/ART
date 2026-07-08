import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from art import vllm_runtime as runtime

ROOT = Path(__file__).resolve().parents[4]


def test_get_vllm_runtime_project_root_defaults_to_repo_subdir(monkeypatch) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_PROJECT_ROOT", raising=False)
    runtime_root = runtime.get_vllm_runtime_project_root()
    assert runtime_root == ROOT / "vllm_runtime"


def test_get_vllm_runtime_project_root_honors_override(monkeypatch) -> None:
    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", "/tmp/custom-runtime")
    assert runtime.get_vllm_runtime_project_root() == Path("/tmp/custom-runtime")


def test_build_runtime_server_cmd_uses_runtime_project(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_BIN", raising=False)
    runtime_root = tmp_path / "custom-runtime"
    runtime_bin = runtime_root / ".venv" / "bin" / "art-vllm-runtime-server"
    runtime_bin.parent.mkdir(parents=True, exist_ok=True)
    runtime_bin.write_text("#!/bin/sh\n", encoding="ascii")
    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", str(runtime_root))
    command = runtime.build_vllm_runtime_server_cmd(
        runtime.VllmRuntimeLaunchConfig(
            base_model="Qwen/Qwen3-14B",
            port=8000,
            host="127.0.0.1",
            cuda_visible_devices="1",
            lora_path="/tmp/lora",
            served_model_name="test@0",
            rollout_weights_mode="merged",
            engine_args={"weight_transfer_config": {"backend": "nccl"}},
            server_args={"tool_call_parser": "hermes"},
        )
    )
    assert command[0] == str(runtime_bin)
    assert "--model=Qwen/Qwen3-14B" in command
    assert (
        '--engine-args-json={"weight_transfer_config": {"backend": "nccl"}}' in command
    )
    assert '--server-args-json={"tool_call_parser": "hermes"}' in command


def test_build_runtime_server_cmd_honors_runtime_bin_override(monkeypatch) -> None:
    monkeypatch.setenv("ART_VLLM_RUNTIME_BIN", "/opt/art/bin/runtime --wrapped")
    command = runtime.build_vllm_runtime_server_cmd(
        runtime.VllmRuntimeLaunchConfig(
            base_model="Qwen/Qwen3-14B",
            port=8000,
            host="127.0.0.1",
            cuda_visible_devices="1",
            lora_path="/tmp/lora",
            served_model_name="test@0",
            rollout_weights_mode="merged",
        )
    )
    assert command[:2] == ["/opt/art/bin/runtime", "--wrapped"]


def test_get_vllm_runtime_nccl_so_path_queries_runtime_python(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_BIN", raising=False)
    runtime_root = tmp_path / "custom-runtime"
    runtime_bin = runtime_root / ".venv" / "bin" / "art-vllm-runtime-server"
    runtime_python = runtime_root / ".venv" / "bin" / "python"
    runtime_bin.parent.mkdir(parents=True, exist_ok=True)
    runtime_bin.write_text("#!/bin/sh\n", encoding="ascii")
    runtime_python.write_text("#!/bin/sh\n", encoding="ascii")
    nccl_so_path = tmp_path / "libnccl.so.2"
    nccl_so_path.write_text("nccl\n", encoding="ascii")
    seen: dict[str, object] = {}

    def fake_run(command, *, capture_output: bool, text: bool):
        seen["command"] = command
        seen["capture_output"] = capture_output
        seen["text"] = text
        return SimpleNamespace(returncode=0, stdout=f"{nccl_so_path}\n", stderr="")

    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", str(runtime_root))
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime.get_vllm_runtime_nccl_so_path() == nccl_so_path.resolve()
    command = seen["command"]
    assert isinstance(command, list)
    assert command[0] == str(runtime_python)
    assert seen["capture_output"] is True
    assert seen["text"] is True


def test_cleanup_old_managed_runtimes_only_deletes_marked_venvs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_KEEP_OLD", raising=False)
    cache_root = tmp_path.resolve()
    keep_hash = "a" * 64
    old_hash = "b" * 64
    invalid_hash = "c" * 64

    def write_runtime(path: Path, manifest_hash: str) -> None:
        (path / ".venv").mkdir(parents=True)
        (path / ".venv" / "pyvenv.cfg").write_text("venv\n")
        marker = runtime.VllmRuntimeInstallMarker(
            runtime_version="0.1.0",
            protocol_version=runtime.RUNTIME_PROTOCOL_VERSION,
            manifest_hash=manifest_hash,
            runtime_wheel_sha256="wheel",
            cache_root=str(cache_root),
        )
        runtime._install_marker_path(path).write_text(marker.model_dump_json())

    keep_dir = cache_root / keep_hash
    old_dir = cache_root / old_hash
    invalid_dir = cache_root / invalid_hash
    arbitrary_dir = cache_root / "not-art"
    write_runtime(keep_dir, keep_hash)
    write_runtime(old_dir, old_hash)
    invalid_dir.mkdir()
    arbitrary_dir.mkdir()
    (arbitrary_dir / "important.txt").write_text("do not delete\n")

    runtime._cleanup_old_managed_runtimes(cache_root, keep_hash=keep_hash)

    assert keep_dir.exists()
    assert not old_dir.exists()
    assert invalid_dir.exists()
    assert arbitrary_dir.exists()
    assert (arbitrary_dir / "important.txt").exists()


def test_cleanup_old_managed_runtimes_respects_keep_old(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ART_VLLM_RUNTIME_KEEP_OLD", "1")
    old_hash = "d" * 64
    old_dir = tmp_path / old_hash
    (old_dir / ".venv").mkdir(parents=True)
    (old_dir / ".venv" / "pyvenv.cfg").write_text("venv\n")
    marker = runtime.VllmRuntimeInstallMarker(
        runtime_version="0.1.0",
        protocol_version=runtime.RUNTIME_PROTOCOL_VERSION,
        manifest_hash=old_hash,
        runtime_wheel_sha256="wheel",
        cache_root=str(tmp_path.resolve()),
    )
    runtime._install_marker_path(old_dir).write_text(marker.model_dump_json())

    runtime._cleanup_old_managed_runtimes(tmp_path.resolve(), keep_hash="e" * 64)

    assert old_dir.exists()


def test_install_managed_runtime_installs_entrypoint_after_promote(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    runtime_wheel = bundle_dir / "art_vllm_runtime-0.1.0-py3-none-any.whl"
    pyproject = bundle_dir / "pyproject.toml"
    lockfile = bundle_dir / "uv.lock"
    runtime_wheel.write_text("wheel\n")
    pyproject.write_text("[project]\nname = 'art-vllm-runtime'\n")
    lockfile.write_text("version = 1\n")
    manifest = runtime.VllmRuntimeManifest(
        art_version="0.5.17",
        runtime_version="0.1.0",
        python=">=3.11",
        runtime_wheel=runtime_wheel.name,
        runtime_wheel_sha256=runtime._sha256_file(runtime_wheel),
        pyproject_sha256=runtime._sha256_file(pyproject),
        lockfile_sha256=runtime._sha256_file(lockfile),
    )
    manifest_hash = runtime._manifest_hash(manifest)
    cache_root = (tmp_path / "cache").resolve()

    def fake_run_install_command(command: list[str], *, cwd=None) -> None:
        del cwd
        if command[:2] == ["uv", "sync"]:
            stage = Path(command[command.index("--project") + 1])
            bin_dir = stage / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            (stage / ".venv" / "pyvenv.cfg").write_text("venv\n")
            (bin_dir / "python").write_text("#!/bin/sh\n")
            return
        assert command[:3] == ["uv", "pip", "install"]
        runtime_python = Path(command[command.index("--python") + 1])
        assert runtime_python == cache_root / manifest_hash / ".venv" / "bin" / "python"
        runtime_bin = runtime_python.parent / runtime.RUNTIME_SERVER
        runtime_bin.write_text(f"#!{runtime_python}\n")
        runtime_bin.chmod(runtime_bin.stat().st_mode | 0o111)

    monkeypatch.setattr(runtime, "_run_install_command", fake_run_install_command)

    runtime_bin = runtime._install_managed_runtime(
        bundle_dir=bundle_dir,
        cache_root=cache_root,
        manifest=manifest,
        manifest_hash=manifest_hash,
    )

    assert (
        runtime_bin
        == cache_root / manifest_hash / ".venv" / "bin" / runtime.RUNTIME_SERVER
    )
    assert runtime_bin.read_text().startswith(
        f"#!{runtime._runtime_python(cache_root / manifest_hash)}"
    )
    assert runtime._read_install_marker(cache_root / manifest_hash) is not None


def test_validate_managed_runtime_rejects_non_executable_entrypoint(
    tmp_path: Path,
) -> None:
    manifest = runtime.VllmRuntimeManifest(
        art_version="0.5.17",
        runtime_version="0.1.0",
        python=">=3.11",
        runtime_wheel="art_vllm_runtime-0.1.0-py3-none-any.whl",
        runtime_wheel_sha256="wheel",
        pyproject_sha256="pyproject",
        lockfile_sha256="lockfile",
    )
    manifest_hash = runtime._manifest_hash(manifest)
    runtime_dir = tmp_path / manifest_hash
    runtime_bin = runtime._runtime_bin(runtime_dir)
    runtime_bin.parent.mkdir(parents=True)
    (runtime_dir / ".venv" / "pyvenv.cfg").write_text("venv\n")
    runtime_bin.write_text("#!/bin/sh\n")
    runtime_bin.chmod(runtime_bin.stat().st_mode & ~0o111)
    marker = runtime.VllmRuntimeInstallMarker(
        runtime_version=manifest.runtime_version,
        protocol_version=manifest.protocol_version,
        manifest_hash=manifest_hash,
        runtime_wheel_sha256=manifest.runtime_wheel_sha256,
        cache_root=str(tmp_path.resolve()),
    )
    runtime._install_marker_path(runtime_dir).write_text(marker.model_dump_json())

    assert not os.access(runtime_bin, os.X_OK)
    assert (
        runtime._validate_managed_runtime(
            runtime_dir,
            cache_root=tmp_path.resolve(),
            manifest=manifest,
            manifest_hash=manifest_hash,
        )
        is None
    )


@pytest.mark.asyncio
async def test_wait_for_vllm_runtime_polls_http_health(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

    class FakeResponse:
        status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, timeout: float):
            seen["url"] = url
            seen["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda: FakeClient())
    await runtime.wait_for_vllm_runtime(
        process=cast(Any, FakeProcess()),
        host="127.0.0.1",
        port=8123,
        timeout=12.0,
    )
    assert seen == {
        "url": "http://127.0.0.1:8123/health",
        "timeout": 5.0,
    }
