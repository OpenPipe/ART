# Copyright OpenPipe

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import litellm
from dotenv import load_dotenv
from litellm import provider_list

# Import evaluate_model and rollout functions from run_rl
from run_rl import rollout_tau_bench_task
from tau_bench.envs import get_env
from tau_bench.envs.user import UserStrategy
from tau_bench.run import display_metrics
from tau_bench.types import EnvRunResult, RunConfig, TauBenchPolicyConfig

import art

# Load environment variables
load_dotenv(override=True)


def parse_args() -> tuple[RunConfig, argparse.Namespace]:
    """Parse command line arguments for benchmarking"""
    parser = argparse.ArgumentParser(
        description="Benchmark off-the-shelf models on tau-bench using RL evaluation"
    )

    # Model configuration
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gpt-4o"],
        help="List of models to benchmark (default: gpt-4o)",
    )
    parser.add_argument(
        "--model-providers",
        type=str,
        nargs="+",
        default=["openai"],
        choices=provider_list,
        help="List of model providers corresponding to each model",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model provider",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the model provider",
    )

    # Environment configuration
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        default="openai",
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--user-strategy",
        type=str,
        default="llm",
        choices=[item.value for item in UserStrategy],
    )

    # Task configuration
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to benchmark on",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index", type=int, default=100, help="End index for benchmark tasks"
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs",
    )

    # Evaluation configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The sampling temperature for the models",
    )
    parser.add_argument(
        "--max-num-steps",
        type=int,
        default=30,
        help="Maximum number of steps per rollout",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run for each task",
    )
    parser.add_argument(
        "--litellm-drop-params",
        action="store_true",
        default=False,
        help="Drop litellm params that are not supported by the model",
    )

    # Output configuration
    parser.add_argument("--log-dir", type=str, default="benchmark_results")
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()
    if args.litellm_drop_params:
        litellm.drop_params = True

    # Ensure model providers match models
    if len(args.model_providers) == 1 and len(args.models) > 1:
        args.model_providers = args.model_providers * len(args.models)
    elif len(args.model_providers) != len(args.models):
        raise ValueError(
            "Number of model providers must match number of models or be 1"
        )

    # Create RunConfig
    run_config = RunConfig(
        model_provider=args.model_providers[0],  # Will be updated per model
        user_model_provider=args.user_model_provider,
        model=args.models[0],  # Will be updated per model
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy="tool-calling-rl",
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=50,
        seed=args.seed,
        shuffle=0,
        user_strategy=args.user_strategy,
        max_num_steps=args.max_num_steps,
        reward_type="real",
        messages_only=True,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    return run_config, args


async def benchmark_model(
    model_name: str,
    model_provider: str,
    config: RunConfig,
    task_indices: List[int],
    num_trials: int,
) -> Dict[str, Any]:
    """Benchmark a single model on the given tasks"""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking model: {model_name} (provider: {model_provider})")
    print(f"{'=' * 60}")

    # Update config for this model
    config.model = model_name
    config.model_provider = model_provider

    # Create a mock trainable model for evaluation
    model = art.Model(
        project="tau_bench_rl",
        name=model_name,
        config=TauBenchPolicyConfig(
            run_config=config,
            training_config=None,  # No training config needed for evaluation
        ),
    )

    # Store results for each trial
    all_results = []
    trial_rewards = {}

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")

        # Run evaluation for this trial
        trial_results = []
        total_reward = 0.0

        # Collect trajectories for all tasks in this trial
        trajectories = await art.gather_trajectories(
            [
                rollout_tau_bench_task(
                    model=model,
                    task_index=task_idx,
                    step=0,
                    phase="eval",
                    is_shadow=False,
                )
                for task_idx in task_indices
            ]
        )

        for task_idx in task_indices:
            traj = trajectories[task_idx]
            # Track results
            reward = traj.metrics["outcome_correct"]
            total_reward += reward

            result = EnvRunResult(
                task_id=task_idx,
                reward=reward,
                info=traj.metadata,
                traj=traj.messages_and_choices,
                trial=trial,
            )
            trial_results.append(result)
            all_results.append(result)

            # print(
            #     "" if reward == 1 else "L",
            #     f"task_id={task_idx}",
            #     f"reward={reward}",
            # )

        avg_reward = total_reward / len(task_indices)
        trial_rewards[trial] = avg_reward
        print(f"\nTrial {trial + 1} average reward: {avg_reward:.3f}")

    # Calculate overall metrics
    print(f"\n{'-' * 40}")
    print(f"Overall Results for {model_name}:")
    display_metrics(all_results)

    # Return summary
    return {
        "model": model_name,
        "provider": model_provider,
        "num_tasks": len(task_indices),
        "num_trials": num_trials,
        "trial_rewards": trial_rewards,
        "all_results": [r.model_dump() for r in all_results],
        "average_reward": sum(trial_rewards.values()) / len(trial_rewards),
    }


async def main():
    """Main benchmarking function"""
    config, args = parse_args()

    # Create output directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Get task indices
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )

    if args.task_ids:
        task_indices = args.task_ids
    else:
        end_index = (
            min(args.end_index, len(env.tasks))
            if args.end_index != -1
            else len(env.tasks)
        )
        task_indices = list(range(args.start_index, end_index))

    print(
        f"Benchmarking on {len(task_indices)} tasks from {config.env} {config.task_split} split"
    )
    print(f"Models to benchmark: {args.models}")

    # Benchmark each model
    all_benchmark_results = {}
    for model_name, model_provider in zip(args.models, args.model_providers):
        results = await benchmark_model(
            model_name=model_name,
            model_provider=model_provider,
            config=config,
            task_indices=task_indices,
            num_trials=args.num_trials,
        )
        all_benchmark_results[model_name] = results

    # Save results
    time_str = datetime.now().strftime("%m%d%H%M%S")
    output_path = os.path.join(
        args.log_dir, f"benchmark_{config.env}_{config.task_split}_{time_str}.json"
    )

    with open(output_path, "w") as f:
        json.dump(all_benchmark_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Display comparison
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<30} {'Provider':<10} {'Avg Reward':<12} {'Pass@1':<10}")
    print(f"{'-' * 60}")

    for model_name, results in all_benchmark_results.items():
        # Calculate Pass@1
        pass_1 = sum(
            1
            for r in results["all_results"]
            if r["reward"] >= 0.999 and r["trial"] == 0
        ) / len(task_indices)

        print(
            f"{model_name:<30} {results['provider']:<10} "
            f"{results['average_reward']:<12.3f} {pass_1:<10.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
    # https://mqjymgipdsp3xw-8000.proxy.runpod.net/v1
