from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from tests.integration.megatron import glm5next_dsa_launch as launch


def test_tree_memory_sums_pss_and_records_rss(tmp_path: Path) -> None:
    for pid, children, pss, rss in [(10, "11 12", 100, 400), (11, "", 30, 90)]:
        root = tmp_path / str(pid)
        (root / "task" / str(pid)).mkdir(parents=True)
        (root / "task" / str(pid) / "children").write_text(children)
        (root / "smaps_rollup").write_text(
            f"Rss: {rss} kB\nPss: {pss} kB\nPss_Anon: 9 kB\n"
        )
    # PID 12 exited between reading the child list and its rollup.
    assert launch.tree_memory(10, proc=tmp_path) == (130 * 1024, 490 * 1024)


@pytest.mark.parametrize("pss,exceeded", [(100, False), (101, True)])
def test_step_guards_pss_independently_of_rss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pss: int, exceeded: bool
) -> None:
    monkeypatch.setattr(launch, "tree_memory", lambda pid: (pss, 1000))

    def run() -> dict:
        return launch.run_step(
            [sys.executable, "-c", "import time; time.sleep(0.7)"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            cwd=tmp_path,
            log=tmp_path / "worker.log",
            timeout=10,
            pss_limit=100,
        )

    if exceeded:
        with pytest.raises(RuntimeError, match="PSS limit"):
            run()
    else:
        receipt = run()
        assert receipt["peak_tree_pss_bytes"] == pss
        assert receipt["peak_tree_rss_bytes"] == 1000


@pytest.mark.parametrize(
    "memory,processes,idle",
    [(0, "", True), (1, "", False), (0, "GPU-test, 123", False)],
)
def test_admission_requires_zero_memory_and_no_processes(
    monkeypatch: pytest.MonkeyPatch, memory: int, processes: str, idle: bool
) -> None:
    def query(argv: list[str], **kwargs: object) -> str:
        return f"GPU-test, {memory}\n" if "--query-gpu" in argv[1] else processes

    monkeypatch.setattr(launch.subprocess, "check_output", query)
    assert launch.gpu_idle(["GPU-test"]) is idle


def test_admission_rejects_cleanup_from_an_earlier_window(tmp_path: Path) -> None:
    board, seal_path = tmp_path / "BOARD.md", tmp_path / "seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "holds": [],
                "launch_ready": True,
                "repo": str(tmp_path),
                "board": str(board),
                "scripts": {},
                "fixture_files": [],
                "admission": {
                    "after_lane": "glm-m2-kda",
                    "after_event": "CLEANUP-DONE",
                    "after_cleanup_contains": "CLEANUP-DONE glm-m2-kda window2",
                },
            }
        )
    )
    digest = launch.sha256(seal_path)
    board.write_text(
        f"GRANT glm-m3-dsa seal_sha256={digest}\nCLEANUP-DONE glm-m2-kda window1 idle\n"
    )
    with pytest.raises(PermissionError, match="sealed predecessor window"):
        launch.execute(seal_path, digest, tmp_path / "output", recording=False)
    assert not (tmp_path / "output").exists()
