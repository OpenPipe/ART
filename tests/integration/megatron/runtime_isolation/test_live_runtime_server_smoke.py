import json
import os
from pathlib import Path
import socket
import subprocess
import uuid

import httpx
import pytest

import art.vllm_runtime as runtime

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.12
DEFAULT_MAX_MODEL_LEN = 512
LIVE_RUNTIME_SMOKE_ENV = "ART_RUN_LIVE_VLLM_RUNTIME_SMOKE"


def _require_live_runtime_smoke_opt_in() -> None:
    if os.environ.get(LIVE_RUNTIME_SMOKE_ENV) != "1":
        pytest.skip(f"set {LIVE_RUNTIME_SMOKE_ENV}=1 to run the live runtime smoke")


def _safe_gpu_memory_utilization() -> float:
    min_free_gib = float(os.environ.get("ART_TEST_MIN_FREE_GPU_GIB", "8"))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < min_free_gib:
        pytest.skip(
            f"Insufficient free GPU memory for live runtime smoke: "
            f"{free_gib:.1f} GiB free < {min_free_gib:.1f} GiB required."
        )
    requested = float(
        os.environ.get(
            "ART_TEST_GPU_MEMORY_UTILIZATION",
            str(DEFAULT_GPU_MEMORY_UTILIZATION),
        )
    )
    return max(0.02, min(requested, (free_bytes / total_bytes) * 0.8))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
@pytest.mark.asyncio
async def test_external_runtime_server_live_smoke(
    artifact_dir: Path,
) -> None:
    _require_live_runtime_smoke_opt_in()

    port = _find_free_port()
    served_model_name = f"vllm-runtime-live-{uuid.uuid4().hex[:8]}"
    log_path = artifact_dir / "runtime.log"
    launch_config = runtime.VllmRuntimeLaunchConfig(
        base_model=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
        port=port,
        host="127.0.0.1",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        served_model_name=served_model_name,
        engine_args={
            "gpu_memory_utilization": _safe_gpu_memory_utilization(),
            "max_model_len": int(
                os.environ.get("ART_TEST_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
            ),
            "max_num_seqs": 4,
            "enforce_eager": True,
        },
    )
    command = runtime.build_vllm_runtime_server_cmd(launch_config)
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        await runtime.wait_for_vllm_runtime(
            process=process,
            host=launch_config.host,
            port=launch_config.port,
            timeout=600.0,
        )
        async with httpx.AsyncClient(
            base_url=f"http://{launch_config.host}:{launch_config.port}",
            timeout=120.0,
        ) as client:
            models_response = await client.get("/v1/models")
            models_response.raise_for_status()
            original_model_ids = [
                model_info["id"] for model_info in models_response.json()["data"]
            ]

            sleep_response = await client.post(
                "/sleep",
                params={"level": 1, "mode": "wait"},
            )
            sleep_response.raise_for_status()
            sleeping_response = await client.get("/is_sleeping")
            sleeping_response.raise_for_status()
            sleeping_before_wake = bool(sleeping_response.json()["is_sleeping"])

            wake_response = await client.post("/wake_up")
            wake_response.raise_for_status()
            awake_response = await client.get("/is_sleeping")
            awake_response.raise_for_status()
            sleeping_after_wake = bool(awake_response.json()["is_sleeping"])

            completion_response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8,
                    "logprobs": True,
                    "top_logprobs": 0,
                },
            )
            completion_response.raise_for_status()
            completion = completion_response.json()

        (artifact_dir / "runtime_smoke_result.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "base_model": launch_config.base_model,
                    "original_model_ids": original_model_ids,
                    "sleeping_before_wake": sleeping_before_wake,
                    "sleeping_after_wake": sleeping_after_wake,
                    "text": completion["choices"][0]["message"]["content"],
                    "has_logprobs": completion["choices"][0]["logprobs"] is not None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        assert served_model_name in original_model_ids
        assert sleeping_before_wake is True
        assert sleeping_after_wake is False
        assert completion["choices"][0]["logprobs"] is not None
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
