"""Actual current physical capture/spill into the recovered selective reader."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from safetensors.torch import load_file
from test_checkpoint_moment_capture import captured_state  # noqa: F401
import torch

from art.trainer_rank import _checkpoint as cp


@pytest.mark.parametrize("present", [(), ("exp_avg",), ("exp_avg", "exp_avg_sq")])
def test_current_physical_spill_selective_parity(captured_state, tmp_path, present):
    trainer, params, masters, optimizer, *_ = captured_state
    if trainer._checkpoint_slots["a"].custom:
        pytest.skip("custom tensors use the unchanged direct-copy finalization path")
    for master in masters:
        optimizer.state[master] = {
            "step": torch.tensor(7.0),
            **{
                name: torch.full_like(master, index + 0.25)
                for index, name in enumerate(present)
            },
        }
    payloads = {}
    captured, _, _ = cp._local_state(trainer, "a", payloads)
    # Mutating the physical owners after capture must not leak through the spill.
    with torch.no_grad():
        for value in (*params, *masters):
            value.add_(100)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    writer = cp._SnapshotSpill()
    shards = writer.submit(snapshot, payloads, captured).result(timeout=5)
    assert shards
    prepared = cast(Any, SimpleNamespace(snapshot=snapshot))
    retained = []
    with cp._snapshot_block(None) as block:
        for filename in sorted({record.file for record in shards}):
            expected = load_file(snapshot / filename)
            for component in ("lora", "master", "exp_avg", "exp_avg_sq", "step"):
                wanted = {
                    key.removeprefix(component + "/"): value
                    for key, value in expected.items()
                    if key.startswith(component + "/")
                }
                actual = cp._read_snapshot(
                    prepared, filename, component, snapshot=block
                )
                assert actual.keys() == wanted.keys()
                for key, value in actual.items():
                    torch.testing.assert_close(value, wanted[key], rtol=0, atol=0)
                    if component == "step":
                        assert value.item() == 7.0
                retained.extend(actual.values())
    for path in snapshot.iterdir():
        path.unlink()
    assert retained and all(torch.isfinite(value).all() for value in retained)
