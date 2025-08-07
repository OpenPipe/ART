import asyncio
from dotenv import load_dotenv
import art
from art_e.project_types import ProjectPolicyConfig
from art_e.data.query_iterators import load_synthetic_queries
from art_e.rollout import rollout
from tqdm.asyncio import tqdm
from art.rewards import ruler_score_group

load_dotenv()


async def main():
    """Run a quick smoke-test: generate one rollout per model and judge them."""

    MODEL_CONFIGS = [
        "openai/gpt-4o",
        "openai/gpt-4.1",
        "openai/o4-mini",
        "openai/o3",
    ]

    # Create the four models we want to benchmark.
    models: list[art.Model] = []
    for litellm_name in MODEL_CONFIGS:
        models.append(
            art.Model(
                name=litellm_name,
                inference_model_name=litellm_name,
                project="email_agent",
                config=ProjectPolicyConfig(),
            )
        )

    # Grab the first N test-set scenarios and evaluate them one by one so we
    # can inspect the results serially.
    scenarios = load_synthetic_queries(split="test", limit=1)

    for scenario_idx, scenario in enumerate(scenarios, start=1):
        print(f"\n\n=== Scenario {scenario_idx} / {len(scenarios)} ===")

        # Generate one rollout per model for the current scenario.
        rollouts = await tqdm.gather(
            *[rollout(model, scenario) for model in models],
            desc=f"Rollouts for scenario {scenario_idx}",
        )

        print("Independent rewards (before judging):")
        for m, t in zip(models, rollouts):
            print(f"  {m.name:10s}: {t.reward:.3f}")

        # Create a TrajectoryGroup from the rollouts
        group = art.TrajectoryGroup(rollouts)

        judged_group = await ruler_score_group(
            group,
            "openai/o3",
            debug=True,
        )

        if judged_group:
            print("\nRULER rewards:")
            for m, t in zip(models, judged_group.trajectories):
                print(f"  {m.name:10s}: {t.reward:.3f}")


asyncio.run(main())
