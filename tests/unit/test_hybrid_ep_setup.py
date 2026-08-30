from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

from art.megatron import hybrid_ep_setup


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
