from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import time
from typing import Any, cast

from ..model_support import oracle_worker
from ..model_support.oracle_harness import (
    LIVE_TRAINING_LOG_PATH,
    WorkerRunRequest,
    _format_elapsed,
    _read_json,
    _write_json,
)


@contextmanager
def _apply_attention_only_mlp_noop():
    """Disables decoder-layer MLP for the attention-only oracle worker."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    transformer_layer = cast(Any, TransformerLayer)
    original_forward_mlp = transformer_layer._forward_mlp

    def _noop_forward_mlp(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states

    transformer_layer._forward_mlp = _noop_forward_mlp
    try:
        yield
    finally:
        transformer_layer._forward_mlp = original_forward_mlp


def run_worker_subprocess(
    request: WorkerRunRequest,
    topology_dir: Path,
    *,
    repo_root: Path,
) -> None:
    """Runs the attention-only distributed worker subprocess and stores combined logs."""
    request_path = topology_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    worker_module = "integration.megatron.cp_attn.megatron_attention_oracle_worker"
    worker_cwd = repo_root / "tests"
    pythonpath_entries = [str(repo_root / "src"), str(repo_root / "tests")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(request.topology.world_size()),
        "-m",
        worker_module,
        "--worker-run",
        "--run-request",
        str(request_path),
    ]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        }
    )
    env.pop("ART_FLEX_BACKEND", None)
    for cache_env in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        cache_root = env.get(cache_env)
        if not cache_root:
            continue
        env[cache_env] = str(Path(cache_root) / topology_dir.name)
    worker_log_path = topology_dir / "worker.log"
    launch_start = time.perf_counter()
    LIVE_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (
        worker_log_path.open("w", encoding="utf-8") as log_file,
        LIVE_TRAINING_LOG_PATH.open("a", encoding="utf-8") as live_log_file,
    ):
        header = (
            "[attention-oracle-harness] launching_worker_subprocess "
            f"topology={request.topology.slug()} world_size={request.topology.world_size()} "
            f"cwd={worker_cwd} command={shlex.join(command)}\n"
        )
        log_file.write(header)
        log_file.flush()
        live_log_file.write(header)
        live_log_file.flush()
        run: subprocess.Popen[bytes] = subprocess.Popen(
            command,
            cwd=str(worker_cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert run.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(run.stdout, selectors.EVENT_READ)
        while True:
            events = selector.select(timeout=0.1)
            if not events and run.poll() is not None:
                break
            for key, _ in events:
                fileobj = cast(Any, key.fileobj)
                chunk = os.read(fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                text = chunk.decode("utf-8", errors="replace")
                log_file.write(text)
                log_file.flush()
                live_log_file.write(text)
                live_log_file.flush()
        run.wait()
        footer = (
            "\n[attention-oracle-harness] worker_subprocess_exit "
            f"topology={request.topology.slug()} returncode={run.returncode} "
            f"elapsed={_format_elapsed(time.perf_counter() - launch_start)}\n"
        )
        log_file.write(footer)
        log_file.flush()
        live_log_file.write(footer)
        live_log_file.flush()
    if run.returncode != 0:
        tail = "\n".join(worker_log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(
            f"Topology run failed for {request.topology.slug()} with exit code "
            f"{run.returncode}.\n{tail}"
        )


def run_worker_cli(run_request_path: Path) -> None:
    request = WorkerRunRequest.model_validate(_read_json(run_request_path))
    with _apply_attention_only_mlp_noop():
        oracle_worker._worker_run(request)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-run",
        action="store_true",
        help="Run one distributed attention-only worker invocation from a JSON request.",
    )
    parser.add_argument(
        "--run-request",
        type=Path,
        help="Path to the worker run request JSON file.",
    )
    args = parser.parse_args(argv)
    if args.worker_run:
        if args.run_request is None:
            parser.error("--run-request is required with --worker-run")
        run_worker_cli(args.run_request)
        return 0
    parser.error("No action specified")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
