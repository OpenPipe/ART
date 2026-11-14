"""CPU Gloo probes of local work surrounding the split payload gather."""

from datetime import timedelta
import json
from pathlib import Path
import sys
import time
from typing import Any

import pytest
from test_trainer_rank_calibration_seed import _rank
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _checkpoint


def _worker(index, directory, site):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    checkpoint_group = (
        dist.new_group(backend="gloo")
        if site.startswith("checkpoint")
        else dist.group.WORLD
    )
    rank, gathers = _rank(), []
    row: dict[str, Any] = {"injected": False}
    value = {"rank": index, "data": "x" * (13 + index * 8192)}
    values = [None, None]
    try:
        with pytest.MonkeyPatch.context() as patch:
            real_resize, real_item, real_empty = (
                torch.Tensor.resize_,
                torch.Tensor.item,
                torch.empty,
            )
            real_getitem, real_decode = torch.Tensor.__getitem__, c10d._tensor_to_object
            real_gather = dist.all_gather

            def inject():
                row["injected"] = True
                current = protocol.active()
                assert current is not None
                row["open_at_failure"] = current.open
                if site == "interrupt":
                    raise KeyboardInterrupt("injected padding interruption")
                raise MemoryError("injected local payload " + site)

            def resize(tensor, *args, **kwargs):
                if (
                    index == 1
                    and not row["injected"]
                    and site in ("resize", "interrupt", "checkpoint_phase")
                    and tensor.dtype == torch.uint8
                ):
                    inject()
                return real_resize(tensor, *args, **kwargs)

            def item(tensor, *args, **kwargs):
                if (
                    index == 1
                    and site == "maximum"
                    and sys._getframe(1).f_code.co_filename == protocol.__file__
                ):
                    inject()
                return real_item(tensor, *args, **kwargs)

            def empty(*args, **kwargs):
                if (
                    index == 1
                    and site == "checkpoint_storage"
                    and not row["injected"]
                    and kwargs.get("dtype") == torch.uint8
                    and sys._getframe(1).f_code.co_filename == protocol.__file__
                ):
                    inject()
                return real_empty(*args, **kwargs)

            def getitem(tensor, key):
                if (
                    index == 1
                    and site == "views"
                    and tensor.dtype == torch.uint8
                    and sys._getframe(1).f_code.co_filename == protocol.__file__
                ):
                    inject()
                return real_getitem(tensor, key)

            def decode(*args, **kwargs):
                if index == 1 and site == "decode":
                    inject()
                return real_decode(*args, **kwargs)

            def gather(outputs, tensor, **kwargs):
                gathers.append(str(tensor.dtype))
                return real_gather(outputs, tensor, **kwargs)

            patch.setattr(torch.Tensor, "resize_", resize)
            patch.setattr(torch, "empty", empty)
            patch.setattr(torch.Tensor, "item", item)
            patch.setattr(torch.Tensor, "__getitem__", getitem)
            patch.setattr(c10d, "_tensor_to_object", decode)
            patch.setattr(dist, "all_gather", gather)
            patch.setattr(c10d, "all_gather", gather)
            try:
                if site == "outside":
                    protocol.all_gather_object(values, value)
                else:
                    with protocol.Driver(rank):
                        if site.startswith("checkpoint"):
                            patch.setattr(
                                rank, "_local_lora_adapter_templates", lambda: {}
                            )
                            patch.setattr(
                                rank, "_checkpoint_group", lambda: checkpoint_group
                            )
                            prepared = _checkpoint._phase(
                                lambda: rank._prepare_adapter_model(
                                    "adapter", {}, canonicalized=True
                                ),
                                "localize checkpoint adapter",
                                checkpoint_group,
                            )
                            row["matched"] = prepared == {}
                        else:
                            protocol.all_gather_object(values, value)
            except BaseException as exc:
                row.update(error=type(exc).__name__, message=str(exc))
            row.update(active=protocol.active() is not None, gathers=gathers)
            try:
                row["outer"] = rank._recovery_reduce(
                    [float("error" in row)],
                    op="MAX",
                    sync_across_dp=True,
                )
            except BaseException as exc:
                row["outer_error"] = str(exc)
            if site in ("success", "outside"):
                row["matched"] = values == [
                    {"rank": i, "data": "x" * (13 + i * 8192)} for i in range(2)
                ]
    finally:
        Path(directory, f"rank{index}.json").write_text(json.dumps(row))
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "site",
    [
        "maximum",
        "resize",
        "interrupt",
        "views",
        "decode",
        "success",
        "outside",
        "checkpoint_phase",
        "checkpoint_storage",
        "checkpoint_success",
    ],
)
def test_payload_seams(tmp_path, site):
    _rank()
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "test_trainer_rank_calibration_seed"]
    )
    context = mp.start_processes(
        _worker,
        args=(str(tmp_path), site),
        nprocs=2,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=0.1):
            assert time.monotonic() < deadline, "payload probe timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    failure = site not in ("success", "outside", "checkpoint_success")
    assert all(not row["active"] for row in rows), rows
    assert all(row.get("outer") == [float(failure)] for row in rows), rows
    if failure:
        assert rows[1]["injected"] and rows[1]["open_at_failure"], rows
        assert all(
            row.get("message") == "Calibration failed on one or more ranks"
            for row in rows
        ), rows
        expected = (
            ["torch.int64", "torch.uint8"] if site == "decode" else ["torch.int64"]
        )
        assert all(row["gathers"] == expected for row in rows), rows
    else:
        assert all(row["matched"] and "error" not in row for row in rows), rows
