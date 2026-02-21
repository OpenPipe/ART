"""End-to-end vLLM contract tests for ART LocalBackend."""

import os
import tempfile
import uuid

import openai
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import art
from art.local import LocalBackend
from art.types import LocalTrainResult

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.2
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_SEQ_LENGTH = 2048


def get_base_model() -> str:
    return os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)


def get_safe_gpu_memory_utilization() -> float:
    requested = float(
        os.environ.get(
            "ART_TEST_GPU_MEMORY_UTILIZATION",
            str(DEFAULT_GPU_MEMORY_UTILIZATION),
        )
    )
    min_free_gib = float(os.environ.get("ART_TEST_MIN_FREE_GPU_GIB", "8"))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < min_free_gib:
        pytest.skip(
            f"Insufficient free GPU memory for vLLM contract test: {free_gib:.1f} GiB free < {min_free_gib:.1f} GiB required."
        )
    # Keep requested utilization below currently free memory with headroom.
    return max(0.02, min(requested, (free_bytes / total_bytes) * 0.8))


def get_vllm_test_config() -> art.dev.InternalModelConfig:
    return {
        "engine_args": {
            "gpu_memory_utilization": get_safe_gpu_memory_utilization(),
            "max_model_len": int(
                os.environ.get("ART_TEST_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
            ),
            "max_num_seqs": 8,
            "enforce_eager": True,
        },
        "init_args": {
            "max_seq_length": int(
                os.environ.get("ART_TEST_MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
            ),
        },
    }


async def simple_rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": prompt}]
    completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=10,
        timeout=60,
        temperature=1,
        logprobs=True,
        top_logprobs=0,
    )
    choice = completion.choices[0]
    content = (choice.message.content or "").lower()
    if "yes" in content:
        reward = 1.0
    elif "no" in content:
        reward = 0.5
    elif "maybe" in content:
        reward = 0.25
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


async def assert_chat_logprobs(
    client: openai.AsyncOpenAI,
    model_name: str,
) -> None:
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello."}],
        model=model_name,
        max_tokens=8,
        timeout=60,
        logprobs=True,
        top_logprobs=0,
    )
    assert completion.choices[0].logprobs is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in this environment",
)
async def test_local_backend_vllm_contract() -> None:
    model_name = f"test-vllm-contract-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(path=tmpdir)
        model = art.TrainableModel(
            name=model_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        object.__setattr__(model, "_internal_config", get_vllm_test_config())
        try:
            await model.register(backend)
            client = model.openai_client()

            step0_name = model.get_inference_name(step=0)
            await assert_chat_logprobs(client, step0_name)

            model_ids = [m.id async for m in client.models.list()]
            assert f"{model.name}@0" in model_ids

            train_groups = await art.gather_trajectory_groups(
                [
                    art.TrajectoryGroup(
                        [simple_rollout(client, step0_name, prompt) for _ in range(2)]
                    )
                    for prompt in ("Say yes", "Say no")
                ]  # ty:ignore[invalid-argument-type]
            )
            result = await backend.train(model, train_groups, learning_rate=1e-5)
            assert isinstance(result, LocalTrainResult)
            assert result.step > 0

            latest_name = model.get_inference_name(step=result.step)
            await assert_chat_logprobs(client, latest_name)
            await assert_chat_logprobs(client, step0_name)

            model_ids_after = [m.id async for m in client.models.list()]
            assert f"{model.name}@0" in model_ids_after
            assert f"{model.name}@{result.step}" in model_ids_after
        finally:
            await backend.close()
