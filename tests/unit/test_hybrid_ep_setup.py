from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

from art.megatron import hybrid_ep_setup
from art.megatron.runtime import managed


def test_setup_serializes_shared_environment_install(
    monkeypatch, tmp_path: Path
) -> None:
    environment = tmp_path / "shared-environment"
    wheel = tmp_path / "art_deep_ep.whl"
    wheel.touch()
    installed_version: str | None = None
    install_count = 0

    monkeypatch.setattr(hybrid_ep_setup.sys, "prefix", str(environment))
    monkeypatch.setattr(
        hybrid_ep_setup, "_build_identity", lambda: ("1.0+art.test", "10.0")
    )
    monkeypatch.setattr(
        hybrid_ep_setup, "_installed_version", lambda: installed_version
    )
    monkeypatch.setattr(hybrid_ep_setup, "_build_wheel", lambda *_: wheel)
    monkeypatch.setattr(hybrid_ep_setup, "_uv", lambda: "uv")

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal install_count, installed_version
        if command[:3] == ["uv", "pip", "install"]:
            install_count += 1
            time.sleep(0.05)
            installed_version = "1.0+art.test"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hybrid_ep_setup.subprocess, "run", run)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: hybrid_ep_setup.setup_hybrid_ep(), range(2)))

    assert results == ("1.0+art.test", "1.0+art.test")
    assert install_count == 1


def test_prepare_overlay_does_not_mutate_shared_environment(
    monkeypatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "art_deep_ep.whl"
    wheel.touch()
    installs: list[list[str]] = []

    monkeypatch.setenv("ART_MEGATRON_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        hybrid_ep_setup, "_build_identity", lambda: ("1.0+art.test", "10.0")
    )
    monkeypatch.setattr(hybrid_ep_setup, "_build_wheel", lambda *_: wheel)
    monkeypatch.setattr(hybrid_ep_setup, "_uv", lambda: "uv")
    monkeypatch.setattr(
        hybrid_ep_setup, "_validate_overlay", lambda *_args, **_kwargs: None
    )

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        installs.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hybrid_ep_setup.subprocess, "run", run)

    first = hybrid_ep_setup.prepare_hybrid_ep_overlay()
    second = hybrid_ep_setup.prepare_hybrid_ep_overlay()

    assert first == second
    assert first[1].is_dir()
    assert len(installs) == 1
    assert installs[0][1:4] == ["pip", "install", "--target"]
    assert installs[0][5:7] == ["--python", hybrid_ep_setup.sys.executable]


def test_hybrid_ep_launcher_pins_overlay(monkeypatch, tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nprintf '%s\\n' \"$PYTHONPATH\"\n")
    python.chmod(0o755)
    overlay = tmp_path / "overlays" / "1.0+art.test"
    overlay.mkdir(parents=True)
    monkeypatch.setenv("ART_MONARCH_PROGRAM_PYTHONPATH", "/art/source")

    launcher = managed._hybrid_ep_launcher(python, overlay)

    assert subprocess.check_output([launcher], text=True).strip() == (
        f"{overlay.resolve()}:/art/source"
    )
