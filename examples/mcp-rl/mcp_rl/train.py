"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import argparse
import asyncio
import fnmatch
import json
import os

from dotenv import load_dotenv
import weave

import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset

from .benchmarks.generate_benchmarks import calculate_beat_comp, generate_val_groups
from .rollout import McpScenario, rollout

load_dotenv()

# Import models from all_experiments.py
try:
    from all_experiments import models
except ImportError:
    models = {}


if os.getenv("WANDB_API_KEY"):
    print("Initializing Weave")
    weave.init("mcp-agent-training")


async def train_mcp_agent(model: art.TrainableModel):
    """Example training function that creates AlphaMcpServer and passes it in scenarios."""
    load_dotenv()

    gpt_4o = art.Model(
        name="gpt-4o",
        project=model.project,
        inference_model_name="gpt-4o",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1",
    )

    # Get configuration from model config or use defaults
    config = getattr(model, "config", None)

    if config is None:
        raise ValueError("Model config is required")

    max_turns = config.max_turns
    trajectories_per_group = config.trajectories_per_group
    groups_per_step = config.groups_per_step
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    eval_steps = config.eval_steps
    ruler_judge_model = config.ruler_judge_model
    training_dataset_size = config.training_dataset_size
    mcp_server_name = config.mcp_server_name

    # Load server params dynamically based on config
    try:
        server_params_module = __import__(
            f"servers.python.{mcp_server_name}.server_params",
            fromlist=["server_params"],
        )
        server_params = server_params_module.server_params
    except ImportError:
        raise ValueError(
            f"Could not import server_params for MCP server: {mcp_server_name}"
        )

    # Load pre-split scenarios from scenarios directory
    scenarios_dir = f"servers/python/{mcp_server_name}/scenarios"

    with open(f"{scenarios_dir}/train.jsonl") as f:
        raw_train_scenarios = [json.loads(line.strip()) for line in f if line.strip()]

    with open(f"{scenarios_dir}/val.jsonl") as f:
        raw_val_scenarios = [json.loads(line.strip()) for line in f if line.strip()]

    # Limit training dataset size if specified in config
    if training_dataset_size and training_dataset_size < len(raw_train_scenarios):
        raw_train_scenarios = raw_train_scenarios[:training_dataset_size]

    print(f"Loaded {len(raw_train_scenarios)} training scenarios")
    print(f"Loaded {len(raw_val_scenarios)} validation scenarios")
    print(
        f"Using config: max_turns={max_turns}, trajectories_per_group={trajectories_per_group}, groups_per_step={groups_per_step}, num_epochs={num_epochs}, learning_rate={learning_rate}"
    )

    from art.local.backend import LocalBackend

    backend = LocalBackend()

    await backend._experimental_pull_from_s3(
        model,
    )

    await model.register(backend)

    train_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=max_turns,
        )
        for scenario in raw_train_scenarios
    ]

    # Create validation scenarios from pre-split data (used for periodic evaluation)
    val_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=max_turns,
        )
        for scenario in raw_val_scenarios
    ]

    # Create dataset iterator using raw scenarios (not McpScenario objects)
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        initial_step=await model.get_step(),  # Resume from checkpoint
    )

    control_groups = await generate_val_groups(gpt_4o, val_scenarios)

    # Main training loop using iterate_dataset
    for batch in train_iterator:
        print("Gathering trajectory groups with RULER scoring...")

        # Use gather_trajectory_groups with ruler_score_group
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, False)
                    for _ in range(trajectories_per_group)
                )
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
            after_each=lambda group: ruler_score_group(
                group,
                judge_model=ruler_judge_model,
                debug=True,  # Show judge reasoning
                swallow_exceptions=True,
            ),
        )

        print("train groups finished")

        if batch.step % eval_steps == 0:
            print("starting comparison val gather")
            val_groups = await generate_val_groups(model, val_scenarios)
            await calculate_beat_comp(val_groups, control_groups, control_first=True)
            await calculate_beat_comp(val_groups, control_groups, control_first=False)

            await model.log(val_groups, split="val")

        print("starting train")
        result = await backend.train(model, groups, learning_rate=learning_rate)
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")

        await backend._experimental_push_to_s3(
            model,
        )


def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(description="Train MCP agent models.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Model key to train from all_experiments.py (e.g. 001, 002, etc.). If not provided, uses default model.",
    )
    args = parser.parse_args()

    if args.models not in models:
        # Check for wildcard patterns
        matching_keys = [
            key
            for key in models.keys()
            if fnmatch.fnmatch(key, args.models.replace("%", "*"))
        ]
        if matching_keys:
            if len(matching_keys) > 1:
                print(
                    f"Multiple models matched pattern '{args.models}': {', '.join(sorted(matching_keys))}"
                )
                print("Please specify a single model key.")
                return
            model_key = matching_keys[0]
        else:
            print(
                f"Unknown model key: {args.models}. Valid keys: {', '.join(sorted(models.keys()))}"
            )
            return
    else:
        model_key = args.models

    model = models[model_key].model_copy(deep=True)
    print(f"Using model configuration: {model_key} ({model.name})")

    print("Starting MCP agent training...")
    try:
        asyncio.run(train_mcp_agent(model))
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise the exception to ensure it reaches the user


if __name__ == "__main__":
    main()
