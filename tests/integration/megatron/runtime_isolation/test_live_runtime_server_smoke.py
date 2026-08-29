import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import uuid

import httpx
import pytest

from art.serving_capabilities import PairedInferenceEndpoint, ServingProfileIdentity
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
    runtime_target_id = "a" * 64
    private_dispatch_token = "private-dispatch-token-" + uuid.uuid4().hex
    log_path = artifact_dir / "runtime.log"
    launch_config = runtime.VllmRuntimeLaunchConfig(
        base_model=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
        port=port,
        host="127.0.0.1",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        served_model_name=served_model_name,
        runtime_target_id=runtime_target_id,
        runtime_source_id="live-smoke-runtime",
        runtime_source_epoch=0,
        private_dispatch_token=private_dispatch_token,
        serving_profile_identity=ServingProfileIdentity(
            base_model=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
            model_identifier=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
            model_revision="default",
            model_support_key="live-smoke",
            handler_name="live-smoke",
            lora_rank=1,
            lora_alpha=1,
            lora_target_modules=("q_proj",),
            trainer_dtype="bfloat16",
            route_replay=False,
            lora_transport="local",
            retained_route_transport="none",
            retained_route_max_bytes=0,
            retained_route_max_bundles=0,
        ),
        engine_args={
            "gpu_memory_utilization": _safe_gpu_memory_utilization(),
            "max_model_len": int(
                os.environ.get("ART_TEST_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
            ),
            "max_num_seqs": 4,
            "tensor_parallel_size": int(
                os.environ.get("ART_TEST_TENSOR_PARALLEL_SIZE", "1")
            ),
            "enforce_eager": True,
        },
    )
    command = runtime.build_vllm_runtime_server_cmd(launch_config)
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["ART_PRIVATE_DISPATCH_TOKEN"] = private_dispatch_token

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
            capabilities_response = await client.get("/art/capabilities")
            capabilities_response.raise_for_status()
            profile = capabilities_response.json()["profile"]
            endpoint = PairedInferenceEndpoint(
                url=f"{client.base_url}/art/internal/v1/chat/completions",
                target_id=runtime_target_id,
                runtime_generation=0,
                runtime_source_id="live-smoke-runtime",
                runtime_source_epoch=0,
                authorization_token=private_dispatch_token,
                profile=profile,
            )
            unauthorized = await client.post(
                endpoint.url.path,
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 1,
                },
            )
            stale_headers = endpoint.request_headers(
                request_identity="b" * 64,
                cache_identity="c" * 64,
                tenant_id="tenant-live",
                run_id="run-live",
                service_tier="standard",
            )
            stale_headers["x-art-runtime-target"] = "d" * 64
            stale = await client.post(
                endpoint.url.path,
                headers=stale_headers,
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 1,
                },
            )

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
            assert completion_response.status_code == 200, completion_response.text
            completion = completion_response.json()
            private_completion_response = await client.post(
                endpoint.url.path,
                headers=endpoint.request_headers(
                    request_identity="e" * 64,
                    cache_identity="f" * 64,
                    tenant_id="tenant-live",
                    run_id="run-live",
                    service_tier="standard",
                ),
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8,
                    "logprobs": True,
                    "top_logprobs": 0,
                },
            )
            private_completion_response.raise_for_status()
            private_completion = private_completion_response.json()
            private_receipt = await client.get(
                f"/art/internal/v1/requests/{'e' * 64}",
                headers=endpoint.runtime_headers(),
            )
            request_usage_page = {}
            for _ in range(100):
                usage_response = await client.get(
                    "/art/internal/v1/usage",
                    headers=endpoint.runtime_headers(),
                )
                usage_response.raise_for_status()
                request_usage_page = usage_response.json()
                if any(
                    receipt["producer"] == "inference_request"
                    for receipt in request_usage_page["receipts"]
                ):
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(
                    f"terminal usage was not published: {request_usage_page}"
                )
            duplicate_private_completion = await client.post(
                endpoint.url.path,
                headers=endpoint.request_headers(
                    request_identity="e" * 64,
                    cache_identity="f" * 64,
                    tenant_id="tenant-live",
                    run_id="run-live",
                    service_tier="standard",
                ),
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8,
                    "logprobs": True,
                    "top_logprobs": 0,
                },
            )
            reset_cache = await client.post("/art/reset_prefix_cache", json={})
            reset_cache.raise_for_status()
            assert reset_cache.json()["success"] is True
            usage_page = {}
            for _ in range(100):
                usage_response = await client.get(
                    "/art/internal/v1/usage",
                    headers=endpoint.runtime_headers(),
                )
                usage_response.raise_for_status()
                usage_page = usage_response.json()
                if {receipt["producer"] for receipt in usage_page["receipts"]} == {
                    "inference_request",
                    "residency",
                }:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(f"KV usage was not published: {usage_page}")
            usage_high_watermark = usage_page["high_watermark_sequence"]
            usage_ack = await client.post(
                "/art/internal/v1/usage:ack",
                headers=endpoint.runtime_headers(),
                json={
                    "source_id": "live-smoke-runtime",
                    "source_epoch": 0,
                    "through_sequence": usage_high_watermark,
                },
            )
            usage_ack.raise_for_status()
            usage_state = await client.get(
                "/art/internal/v1/usage/state",
                headers=endpoint.runtime_headers(),
            )
            usage_state.raise_for_status()

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
                    "kv_block_bytes_per_rank": profile["kv_block_bytes_per_rank"],
                    "kv_capacity_bytes_per_rank": profile["kv_capacity_bytes_per_rank"],
                    "private_text": private_completion["choices"][0]["message"][
                        "content"
                    ],
                    "private_receipt": private_receipt.json(),
                    "duplicate_private_execution": (
                        duplicate_private_completion.json()["execution"]
                    ),
                    "usage_page": usage_page,
                    "usage_state": usage_state.json(),
                    "stale_status": stale.status_code,
                    "unauthorized_status": unauthorized.status_code,
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
        assert profile["kv_block_bytes_per_rank"] > 0
        assert profile["kv_capacity_bytes_per_rank"] > 0
        assert unauthorized.status_code == 401
        assert stale.status_code == 409
        assert stale.json()["execution"] == "not_started"
        assert private_completion["choices"][0]["logprobs"] is not None
        assert private_receipt.status_code == 200
        assert private_receipt.json()["execution"] == "completed"
        assert duplicate_private_completion.status_code == 409
        assert duplicate_private_completion.json()["execution"] == "completed"
        assert usage_page["source_id"] == "live-smoke-runtime"
        assert usage_page["high_watermark_sequence"] == 2
        request_usage = next(
            receipt
            for receipt in usage_page["receipts"]
            if receipt["producer"] == "inference_request"
        )
        kv_usage = next(
            receipt
            for receipt in usage_page["receipts"]
            if receipt["producer"] == "residency"
        )
        assert request_usage["tenant_id"] == "tenant-live"
        request_measurements = {
            measurement["metric"] for measurement in request_usage["measurements"]
        }
        assert request_measurements == {
            "inference_gpu_ms",
            "live_request_ms",
            "cached_prefill_tokens",
            "uncached_prefill_tokens",
            "decode_tokens",
        }
        assert (
            float(
                next(
                    measurement["quantity"]
                    for measurement in request_usage["measurements"]
                    if measurement["metric"] == "inference_gpu_ms"
                )
            )
            > 0
        )
        assert kv_usage["measurements"][0]["metric"] == "kv_byte_ms"
        assert float(kv_usage["measurements"][0]["quantity"]) > 0
        assert usage_state.json()["acknowledged_through_sequence"] == 2
        assert usage_state.json()["active_kv_residencies"] == 0
        assert usage_state.json()["retained_receipts"] == 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
