"""Integration test: verify provenance tracking in W&B run config via ServerlessBackend."""

import asyncio
from datetime import datetime

from dotenv import load_dotenv

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


async def make_group(model: art.TrainableModel) -> art.TrajectoryGroup:
    return art.TrajectoryGroup(simple_rollout(model) for _ in range(4))


async def main() -> None:
    backend = ServerlessBackend()

    model = art.TrainableModel(
        name=f"provenance-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        project="provenance-test",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    await model.register(backend)

    # --- Step 1: first training call ---
    groups = await art.gather_trajectory_groups(make_group(model) for _ in range(1))
    result = await backend.train(model, groups)
    await model.log(groups, metrics=result.metrics, step=result.step, split="train")

    # Check provenance after first train call
    run = model._get_wandb_run()
    assert run is not None, "W&B run should exist"
    provenance = run.config.get("provenance")
    print(f"After step 1: provenance = {provenance}")
    assert provenance == ["serverless-rl"], (
        f"Expected ['serverless-rl'], got {provenance}"
    )

    # --- Step 2: second training call (same technique, should NOT duplicate) ---
    # Provenance is recorded at the start of train(), before the remote call,
    # so we can verify deduplication even if the server-side training fails.
    groups2 = await art.gather_trajectory_groups(make_group(model) for _ in range(1))
    try:
        result2 = await backend.train(model, groups2)
        await model.log(
            groups2, metrics=result2.metrics, step=result2.step, split="train"
        )
    except RuntimeError as e:
        print(f"Step 2 training failed (transient server error, OK for this test): {e}")

    provenance = run.config.get("provenance")
    print(f"After step 2: provenance = {provenance}")
    assert provenance == ["serverless-rl"], (
        f"Expected ['serverless-rl'] (no duplicate), got {provenance}"
    )

    print("\nAll provenance checks passed!")

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
