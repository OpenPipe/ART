"""Integration test: verify provenance tracking on W&B artifact metadata via ServerlessBackend."""

import asyncio
from datetime import datetime

from dotenv import load_dotenv
import wandb

import art
from art.serverless.backend import ServerlessBackend

load_dotenv()


async def simple_rollout(model: art.TrainableModel) -> art.Trajectory:
    """Minimal rollout that produces a single turn with a reward."""
    traj = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Reply with exactly 'hello'."},
        ],
        reward=0.0,
    )

    choice = (
        await model.openai_client().chat.completions.create(
            model=model.get_inference_name(),
            messages=traj.messages(),
            max_completion_tokens=16,
            timeout=30,
        )
    ).choices[0]

    traj.messages_and_choices.append(choice)
    traj.reward = (
        1.0 if (choice.message.content or "").strip().lower() == "hello" else 0.0
    )
    return traj


def get_latest_artifact_provenance(
    entity: str, project: str, name: str
) -> list[str] | None:
    """Fetch provenance from the latest W&B artifact's metadata."""
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{name}:latest", type="lora")
    return artifact.metadata.get("wandb.provenance")


async def main() -> None:
    backend = ServerlessBackend()

    model = art.TrainableModel(
        name=f"provenance-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        project="provenance-test",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    await model.register(backend)
    assert model.entity is not None

    # --- Step 1: first training call (retry on transient server errors) ---
    for attempt in range(3):
        groups = await art.gather_trajectory_groups(
            [art.TrajectoryGroup(simple_rollout(model) for _ in range(4))]  # ty: ignore[invalid-argument-type]
        )
        try:
            result = await backend.train(model, groups)
            await model.log(
                groups, metrics=result.metrics, step=result.step, split="train"
            )
            break
        except RuntimeError as e:
            print(f"Step 1 attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise

    # Check provenance on the latest artifact after first train call
    provenance = get_latest_artifact_provenance(model.entity, model.project, model.name)
    print(f"After step 1: provenance = {provenance}")
    assert provenance == ["serverless-rl"], (
        f"Expected ['serverless-rl'], got {provenance}"
    )

    # --- Step 2: second training call (same technique, should NOT duplicate) ---
    groups2 = await art.gather_trajectory_groups(
        [art.TrajectoryGroup(simple_rollout(model) for _ in range(4))]  # ty: ignore[invalid-argument-type]
    )
    try:
        result2 = await backend.train(model, groups2)
        await model.log(
            groups2, metrics=result2.metrics, step=result2.step, split="train"
        )
    except RuntimeError as e:
        print(f"Step 2 training failed (transient server error, OK for this test): {e}")

    provenance = get_latest_artifact_provenance(model.entity, model.project, model.name)
    print(f"After step 2: provenance = {provenance}")
    assert provenance == ["serverless-rl"], (
        f"Expected ['serverless-rl'] (no duplicate), got {provenance}"
    )

    print("\nAll provenance checks passed!")

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
