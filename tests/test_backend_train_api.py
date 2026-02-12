"""Test the new backend.train() API with real GPU training.

This test runs a simple yes-no-maybe training loop using the new backend-first API.

Usage:
    cd /workspace/ART && source .venv/bin/activate
    python tests/test_backend_train_api.py
"""

import asyncio
import os
import tempfile

import art
from art.local import LocalBackend
from art.types import LocalTrainResult

DEFAULT_GPU_MEMORY_UTILIZATION = 0.2
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_SEQ_LENGTH = 2048


def get_vllm_test_config() -> tuple[art.dev.InternalModelConfig, str | None]:
    requested = float(
        os.environ.get(
            "ART_TEST_GPU_MEMORY_UTILIZATION",
            str(DEFAULT_GPU_MEMORY_UTILIZATION),
        )
    )
    min_free_gib = float(os.environ.get("ART_TEST_MIN_FREE_GPU_GIB", "8"))
    safe_utilization = requested
    skip_reason: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / (1024**3)
            if free_gib < min_free_gib:
                skip_reason = (
                    f"Skipping backend.train API test: free GPU memory is too low "
                    f"({free_gib:.2f} GiB < {min_free_gib:.2f} GiB)."
                )
            safe_utilization = min(requested, (free_bytes / total_bytes) * 0.8)
    except Exception:
        pass

    return {
        "engine_args": {
            "gpu_memory_utilization": safe_utilization,
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
    }, skip_reason


async def simple_rollout(client, model_name: str, prompt: str) -> art.Trajectory:
    """A simple rollout function for testing."""
    messages: art.Messages = [{"role": "user", "content": prompt}]
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=10,
        timeout=60,
        temperature=1,
    )
    choice = chat_completion.choices[0]
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


async def main():
    print("=" * 60)
    print("Testing new backend.train() API")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temp directory: {tmpdir}")

        # Create backend and model
        backend = LocalBackend(path=tmpdir)
        model = art.TrainableModel(
            name="test-backend-train-api",
            project="api-test",
            base_model="Qwen/Qwen3-0.6B",
        )
        test_config, skip_reason = get_vllm_test_config()
        if skip_reason is not None:
            print(f"\n{skip_reason}")
            return
        object.__setattr__(model, "_internal_config", test_config)

        try:
            print("\n1. Registering model with backend...")
            await model.register(backend)
            print("   ✓ Model registered")

            # Get OpenAI client
            openai_client = model.openai_client()

            # Use get_inference_name() for the correct model name
            # After registration, this returns the proper name (e.g., model.name@0)
            inference_name = model.get_inference_name()
            print(f"   Using model for inference: {inference_name}")

            print("\n2. Gathering trajectories...")
            prompts = ["Say yes", "Say no", "Say maybe", "Say hello"]
            train_groups = await art.gather_trajectory_groups(
                [
                    art.TrajectoryGroup(
                        [
                            simple_rollout(openai_client, inference_name, prompt)
                            for _ in range(4)  # 4 rollouts per prompt
                        ]
                    )
                    for prompt in prompts
                ]  # ty:ignore[invalid-argument-type]
            )
            print(f"   ✓ Gathered {len(train_groups)} trajectory groups")

            # Print some sample rewards
            for i, group in enumerate(train_groups):
                rewards = [t.reward for t in group]
                print(f"   Group {i} ({prompts[i]}): rewards = {rewards}")

            print("\n3. Training with backend.train()...")
            result = await backend.train(
                model,
                train_groups,
                learning_rate=1e-5,
                verbose=True,
            )

            print("\n4. Logging trajectories and training metrics...")
            await model.log(
                train_groups, metrics=result.metrics, step=result.step, split="train"
            )
            print("   ✓ Trajectories and metrics logged")

            print("\n5. Checking TrainResult...")
            print(f"   Result type: {type(result).__name__}")
            print(f"   Step: {result.step}")
            print(f"   Metrics: {result.metrics}")

            assert isinstance(result, LocalTrainResult), (
                f"Expected LocalTrainResult, got {type(result)}"
            )
            print(f"   Checkpoint path: {result.checkpoint_path}")

            assert result.step > 0, f"Expected step > 0, got {result.step}"
            assert isinstance(result.metrics, dict), (
                f"Expected dict metrics, got {type(result.metrics)}"
            )

            print("\n" + "=" * 60)
            print("✓ All checks passed! New backend.train() API works correctly.")
            print("=" * 60)

        finally:
            print("\nCleaning up...")
            await backend.close()
            print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
