import json
import os
from pathlib import Path
import uuid

import pytest

torch = pytest.importorskip("torch")

import art
from art.local import LocalBackend

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.12
DEFAULT_MAX_MODEL_LEN = 512
DEFAULT_MAX_SEQ_LENGTH = 512
LIVE_SMOKE_ENV = "ART_RUN_LIVE_VLLM_SEPARATION"


def _require_live_smoke_opt_in() -> None:
    if os.environ.get(LIVE_SMOKE_ENV) != "1":
        pytest.skip(f"set {LIVE_SMOKE_ENV}=1 to run the live runtime smoke")


def _safe_gpu_memory_utilization() -> float:
    min_free_gib = float(os.environ.get("ART_TEST_MIN_FREE_GPU_GIB", "8"))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < min_free_gib:
        pytest.skip(
            f"Insufficient free GPU memory for live vLLM separation smoke: "
            f"{free_gib:.1f} GiB free < {min_free_gib:.1f} GiB required."
        )
    requested = float(
        os.environ.get(
            "ART_TEST_GPU_MEMORY_UTILIZATION",
            str(DEFAULT_GPU_MEMORY_UTILIZATION),
        )
    )
    return max(0.02, min(requested, (free_bytes / total_bytes) * 0.8))


def _live_test_config() -> art.dev.InternalModelConfig:
    return {
        "engine_args": {
            "gpu_memory_utilization": _safe_gpu_memory_utilization(),
            "max_model_len": int(
                os.environ.get("ART_TEST_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
            ),
            "max_num_seqs": 4,
            "enforce_eager": True,
        },
        "init_args": {
            "max_seq_length": int(
                os.environ.get("ART_TEST_MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
            ),
        },
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
@pytest.mark.asyncio
async def test_local_backend_external_runtime_live_smoke(
    tmp_path: Path,
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_live_smoke_opt_in()
    monkeypatch.setenv("WANDB_MODE", "offline")

    model_name = f"vllm-separation-live-{uuid.uuid4().hex[:8]}"
    backend = LocalBackend(path=str(tmp_path))
    model = art.TrainableModel(
        run_name=model_name,
        name=model_name,
        project="integration-tests",
        base_model=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
        _internal_config=_live_test_config(),
    )

    try:
        await model.register(backend)
        client = model.openai_client()
        try:
            step0_name = model.get_inference_name(step=0)
            model_ids = [model_info.id async for model_info in client.models.list()]
            completion = await client.chat.completions.create(
                model=step0_name,
                messages=[{"role": "user", "content": "Say hello."}],
                max_tokens=8,
                timeout=120,
                logprobs=True,
                top_logprobs=0,
            )
            payload = {
                "step0_name": step0_name,
                "model_ids": model_ids,
                "text": completion.choices[0].message.content,
                "has_logprobs": completion.choices[0].logprobs is not None,
            }
            (artifact_dir / "live_smoke_result.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True)
            )
            assert step0_name in model_ids
            assert completion.choices[0].logprobs is not None
        finally:
            await client.close()
    finally:
        await backend.close()
