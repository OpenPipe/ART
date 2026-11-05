from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import sys

import pytest

from tests.integration.megatron import kda_window as window


def test_tree_pss_counts_descendants_and_retains_reparented_workers(
    tmp_path, monkeypatch
):
    def process(pid, parent, start, pss, rss):
        directory = tmp_path / "proc" / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        fields = ["S", str(parent)] + ["0"] * 17 + [str(start)]
        (directory / "stat").write_text(f"{pid} (worker) {' '.join(fields)}")
        if pss is not None:
            (directory / "smaps_rollup").write_text(
                f"Rss: {rss} kB\nPss: {pss} kB\nPss_Anon: 999999 kB\n"
            )

    process(100, 1, 10, 20, 80)
    process(101, 100, 11, 30, 90)
    process(102, 101, 12, 40, 100)
    # Neither unrelated processes nor a reused PID are part of the tree.
    process(200, 1, 99, None, None)
    monkeypatch.setattr(window, "Path", lambda path: tmp_path / str(path).lstrip("/"))
    owned = {200: 1}
    assert window.tree_memory(100, owned) == (90 * 1024, 270 * 1024)
    assert owned == {100: 10, 101: 11, 102: 12, 200: 1}
    (tmp_path / "proc" / "100" / "stat").unlink()
    process(101, 1, 11, 30, 90)
    assert window.tree_memory(100, owned) == (70 * 1024, 190 * 1024)


@pytest.mark.parametrize("over_limit", [False, True])
def test_guard_enforces_pss_and_records_rss_on_success_or_failure(
    tmp_path, monkeypatch, over_limit
):
    pss = 101 if over_limit else 99
    monkeypatch.setattr(window, "tree_memory", lambda root, owned: (pss, 1000))
    groups = []
    records = []
    argv = [sys.executable, "-c", "import time; time.sleep(0.5)"]
    expectation = (
        pytest.raises(RuntimeError, match="bytes PSS") if over_limit else nullcontext()
    )
    with expectation:
        window.run_step(
            argv,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            output=tmp_path / "worker.log",
            timeout=10,
            max_pss_bytes=100,
            owned_groups=groups,
            step_receipts=records,
        )
    assert records[0]["peak_tree_pss_bytes"] == pss
    assert records[0]["peak_tree_rss_bytes"] == 1000
    assert not window.tree_pids(*groups[0])


def test_timeout_reaps_descendant_in_separate_session(tmp_path):
    groups = []
    records = []
    child_pid = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        window.run_step(
            [sys.executable, "-c", code, str(child_pid)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            output=tmp_path / "timeout.log",
            timeout=1,
            max_pss_bytes=120 * (1 << 30),
            owned_groups=groups,
            step_receipts=records,
        )
    root, owned = groups[0]
    assert int(child_pid.read_text()) in owned
    assert len(owned) == 2
    assert not window.tree_pids(root, owned)
    assert records[0]["peak_tree_pss_bytes"] > 0
    assert records[0]["peak_tree_rss_bytes"] >= records[0]["peak_tree_pss_bytes"]


def test_unreadable_rollup_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "tree_pids", lambda root, owned: {123})
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/123/smaps_rollup":
            raise PermissionError("PSS unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(PermissionError, match="PSS unavailable"):
        window.tree_memory(123, {})
