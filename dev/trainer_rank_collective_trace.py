"""Run one cost-calibration cell with every collective traced per rank.

Diagnostic for hangs and desynchronization in distributed TrainerRank runs
(used to root-cause issue #840). Runs
``dev/trainer_rank_landing_acceptance.py --phase cost-calibrate`` unchanged
while logging every ``torch.distributed`` collective to a per-rank JSONL:
operation, calling frames, split sizes, element counts, the forced-layout
label in effect and a running forward index. A stall watchdog dumps all thread
stacks and exits when no collective completes for ``--stall-seconds``, so a
deadlock surfaces in minutes instead of the NCCL timeout.

Usage (from the repo root, e.g. 4 GPUs at CP4):
  torchrun --standalone --nproc-per-node=4 dev/trainer_rank_collective_trace.py \
    --cell cal-ellavox --group 4 --repeat 1 --log-dir <dir>
  python dev/trainer_rank_collective_diff.py <dir>   # first divergent collective
"""

from __future__ import annotations

import argparse
import faulthandler
import functools
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dev"))

COLLECTIVES = (
    "all_to_all_single",
    "all_to_all",
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_object",
    "reduce_scatter_tensor",
    "reduce_scatter",
    "broadcast",
    "broadcast_object_list",
    "barrier",
    "send",
    "recv",
)


class _Logger:
    def __init__(self, log_path: Path, rank: int) -> None:
        self.rank = rank
        self.handle = open(log_path, "a", encoding="utf-8")
        self.seq = 0
        self.forward_index = 0
        self.last_event = time.time()
        self.lock = threading.Lock()

    def write(self, record: dict[str, object]) -> None:
        with self.lock:
            self.seq += 1
            record = {"seq": self.seq, "rank": self.rank, "t": time.time(), **record}
            self.handle.write(json.dumps(record, default=str) + "\n")
            self.handle.flush()
            self.last_event = record["t"]

    def anchor(self) -> str:
        return os.environ.get("ART_TRAINER_RANK_TEST_ANCHOR", "automatic")


def _numel(value: object) -> int | None:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_numel(v) or 0 for v in value)
    return None


def _shape(value: object) -> object:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape(v) for v in value[:4]]
    return None


def _install(logger: _Logger) -> None:
    import torch.distributed as dist

    import art.megatron.gdn.layout as gdn_layout
    from art.trainer_rank import TrainerRank

    def wrap(name: str):
        original = getattr(dist, name)
        signature = inspect.signature(original)

        def logged(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            params = bound.arguments
            group = params.get("group")
            frames = inspect.stack()[1:7]
            record: dict[str, object] = {
                "op": name,
                "forward_index": logger.forward_index,
                "anchor": logger.anchor(),
                "caller": "/".join(f.function for f in frames),
                "group_size": dist.get_world_size(group)
                if dist.is_initialized()
                else None,
                "async_op": bool(params.get("async_op", False)),
            }
            if name == "all_to_all_single":
                record["in_splits"] = (
                    list(map(int, params["input_split_sizes"]))
                    if params.get("input_split_sizes") is not None
                    else None
                )
                record["out_splits"] = (
                    list(map(int, params["output_split_sizes"]))
                    if params.get("output_split_sizes") is not None
                    else None
                )
                record["numel_in"] = _numel(params.get("input"))
                record["numel_out"] = _numel(params.get("output"))
                record["shape_in"] = _shape(params.get("input"))
                record["shape_out"] = _shape(params.get("output"))
            else:
                first = next(iter(params.values()), None)
                record["numel"] = _numel(first)
                record["shape"] = _shape(first)
            logger.write(record)
            return original(*args, **kwargs)

        logged.__name__ = name
        return logged

    for name in COLLECTIVES:
        if hasattr(dist, name):
            setattr(dist, name, wrap(name))
    # gdn.layout imported the function by name; rebind it to the logged one.
    gdn_layout.all_to_all_single = dist.all_to_all_single

    original_forward = TrainerRank.dp_rank_forward

    @functools.wraps(original_forward)
    def logged_forward(self, *args, **kwargs):
        logger.forward_index += 1
        logger.write(
            {
                "op": "forward_start",
                "forward_index": logger.forward_index,
                "anchor": logger.anchor(),
            }
        )
        outputs = original_forward(self, *args, **kwargs)
        telemetry = self.last_forward_telemetry()
        logger.write(
            {
                "op": "forward_end",
                "forward_index": logger.forward_index,
                "anchor": logger.anchor(),
                "selected_max_depth": telemetry.get("selected_max_depth"),
                "subforward_count": telemetry.get("subforward_count"),
            }
        )
        return outputs

    TrainerRank.dp_rank_forward = logged_forward  # type: ignore[method-assign]


def _start_watchdog(logger: _Logger, log_dir: Path, stall_seconds: float) -> None:
    def watch() -> None:
        while True:
            time.sleep(5)
            idle = time.time() - logger.last_event
            if logger.seq > 0 and idle > stall_seconds:
                path = log_dir / f"stall.rank{logger.rank}.txt"
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(
                        f"rank {logger.rank}: no collective for {idle:.0f}s after seq "
                        f"{logger.seq} (forward_index {logger.forward_index}, anchor "
                        f"{logger.anchor()}); dumping all thread stacks\n"
                    )
                    handle.flush()
                    faulthandler.dump_traceback(handle, all_threads=True)
                print(
                    f"repro: rank {logger.rank} stalled after collective #{logger.seq} "
                    f"(forward {logger.forward_index}, anchor {logger.anchor()}); see {path}",
                    flush=True,
                )
                os._exit(3)

    threading.Thread(target=watch, name="stall-watchdog", daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="cal-ellavox")
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--stall-seconds", type=float, default=300.0)
    arguments = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    log_dir = Path(arguments.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _Logger(log_dir / f"collectives.rank{rank}.jsonl", rank)
    _install(logger)
    _start_watchdog(logger, log_dir, arguments.stall_seconds)

    import trainer_rank_landing_acceptance as driver

    sys.argv = [
        "trainer_rank_landing_acceptance.py",
        "--phase",
        "cost-calibrate",
        "--cell",
        arguments.cell,
        "--model",
        arguments.model,
        "--layers",
        str(arguments.layers),
        "--group",
        str(arguments.group),
        "--repeat",
        str(arguments.repeat),
        "--evidence",
        str(log_dir / "evidence.jsonl"),
    ]
    driver.main()
    if rank == 0:
        print("repro: harness completed without hanging", flush=True)


if __name__ == "__main__":
    main()
