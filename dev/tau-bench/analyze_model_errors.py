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
from tau_bench.rl_utils import (
    ErrorAnalysis,
    analyze_failed_task,
    log_full_analysis_to_openpipe,
    log_rollout_analysis_to_openpipe,
)
from tau_bench.run import display_metrics
from tau_bench.types import EnvRunResult, RunConfig, TauBenchPolicyConfig
from tqdm.asyncio import tqdm_asyncio

import art

# Load environment variables
load_dotenv(override=True)


def parse_args() -> tuple[RunConfig, argparse.Namespace]:
    """Parse command line arguments for error analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze model errors on tau-bench tasks using multiple trials"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to analyze (default: gpt-4o)",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        default="openai",
        choices=provider_list,
        help="Model provider",
    )
    parser.add_argument(
        "--analyzer-model",
        type=str,
        default="o3",
        help="Model to use for error analysis (default: o3)",
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
        help="The split of tasks to analyze",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index", type=int, default=100, help="End index for analysis tasks"
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="(Optional) analyze only the tasks with the given IDs",
    )

    # Analysis configuration
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
        default=3,
        help="Number of trials to run for each task (default: 3)",
    )
    parser.add_argument(
        "--litellm-drop-params",
        action="store_true",
        default=False,
        help="Drop litellm params that are not supported by the model",
    )

    # Output configuration
    parser.add_argument("--log-dir", type=str, default="error_analysis_results")
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()
    if args.litellm_drop_params:
        litellm.drop_params = True

    # Create RunConfig
    run_config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
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


async def analyze_model_errors(
    model_name: str,
    model_provider: str,
    config: RunConfig,
    task_indices: List[int],
    num_trials: int,
    analyzer_model: str,
) -> Dict[str, Any]:
    """Analyze model errors across multiple trials."""

    print(f"\n{'=' * 60}")
    print(f"Analyzing errors for model: {model_name} (provider: {model_provider})")
    print(f"Running {num_trials} trials per task")
    print(f"{'=' * 60}")

    # Update config for this model
    config.model = model_name
    config.model_provider = model_provider

    # Create model for evaluation
    model = art.Model(
        project="tau_bench_error_analysis",
        name=model_name,
        config=TauBenchPolicyConfig(
            run_config=config,
            training_config=None,
        ),
    )

    # Get environment to access task information
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )

    # Store results
    all_results = []
    failed_tasks = {}  # task_id -> list of failed trajectories
    analysis_results = {}
    analysis_prompts = {}  # task_id -> prompt used for analysis

    # Run all trials for all tasks in parallel
    print(f"\nRunning {num_trials} trials for {len(task_indices)} tasks...")

    # Create all rollout tasks
    all_rollout_tasks = []
    task_trial_mapping = []

    for task_idx in task_indices:
        for trial_idx in range(num_trials):
            rollout_task = rollout_tau_bench_task(
                model=model,
                task_index=task_idx,
                step=0,
                phase="eval",
                is_shadow=False,
            )
            all_rollout_tasks.append(rollout_task)
            task_trial_mapping.append((task_idx, trial_idx))

    # Run all rollouts in parallel with progress bar
    trajectories = await tqdm_asyncio.gather(
        *all_rollout_tasks, desc="Running rollouts", total=len(all_rollout_tasks)
    )

    # Group trajectories by task
    task_trajectories = {}
    for i, traj in enumerate(trajectories):
        task_idx, trial_idx = task_trial_mapping[i]
        if task_idx not in task_trajectories:
            task_trajectories[task_idx] = []
        task_trajectories[task_idx].append(traj)

    # Process results for each task
    for task_idx in task_indices:
        trajectories_for_task = task_trajectories.get(task_idx, [])

        # Classify trials as passed/failed
        failed_trajectories = []
        passed_count = 0

        for trial_idx, traj in enumerate(trajectories_for_task):
            reward = traj.metrics.get("outcome_correct", 0)
            result = EnvRunResult(
                task_id=task_idx,
                reward=reward,
                info=traj.metadata,
                traj=traj.messages_and_choices,
                trial=trial_idx,
            )
            all_results.append(result)

            if reward < 0.999:  # Consider as failed
                failed_trajectories.append(traj)
            else:
                passed_count += 1

        print(f"Task {task_idx}: {passed_count}/{len(trajectories_for_task)} passed")

        # If all trials failed, collect for analysis
        if (
            len(failed_trajectories) == len(trajectories_for_task)
            and len(failed_trajectories) > 0
        ):
            print(f"Task {task_idx} failed all trials, queuing for analysis...")
            failed_tasks[task_idx] = failed_trajectories

    # Parallelize analysis of failed tasks
    if failed_tasks:
        print(f"\nAnalyzing {len(failed_tasks)} failed tasks in parallel...")

        # Create analysis tasks for parallel execution
        analysis_tasks = []
        task_ids_for_analysis = []

        for task_idx, failed_trajectories in failed_tasks.items():
            # Get task information from environment
            env_task = env.tasks[task_idx] if task_idx < len(env.tasks) else None
            if env_task:
                analysis_task = analyze_failed_task(
                    task_idx, failed_trajectories, env_task.model_dump(), analyzer_model
                )
                analysis_tasks.append(analysis_task)
                task_ids_for_analysis.append(task_idx)
            else:
                print(f"Could not find task {task_idx} in environment")

        # Run all analyses in parallel with progress bar
        if analysis_tasks:
            analysis_results_list = await tqdm_asyncio.gather(
                *analysis_tasks, desc="Running analyses", total=len(analysis_tasks)
            )

            # Process results
            for i, result in enumerate(analysis_results_list):
                task_idx = task_ids_for_analysis[i]
                if result:
                    analysis, prompt = result
                    analysis_results[task_idx] = analysis
                    analysis_prompts[task_idx] = prompt
                    print(f"Analysis complete for task {task_idx}")
                else:
                    print(f"Failed to analyze task {task_idx}")

    # Display overall metrics
    print(f"\n{'-' * 40}")
    print(f"Overall Results for {model_name}:")
    display_metrics(all_results)

    print(f"\nTasks that failed all trials: {len(failed_tasks)}")
    print(f"Tasks successfully analyzed: {len(analysis_results)}")

    return {
        "model": model_name,
        "provider": model_provider,
        "num_tasks": len(task_indices),
        "num_trials": num_trials,
        "total_failed_tasks": len(failed_tasks),
        "analyzed_tasks": len(analysis_results),
        "all_results": [r.model_dump() for r in all_results],
        "failed_tasks": {k: len(v) for k, v in failed_tasks.items()},
        "analysis_results": {k: v.model_dump() for k, v in analysis_results.items()},
        "analysis_prompts": analysis_prompts,
    }


async def main():
    """Main error analysis function"""
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
        f"Analyzing {len(task_indices)} tasks from {config.env} {config.task_split} split"
    )
    print(f"Model: {args.model}")
    print(f"Analyzer model: {args.analyzer_model}")

    # Run error analysis
    results = await analyze_model_errors(
        model_name=args.model,
        model_provider=args.model_provider,
        config=config,
        task_indices=task_indices,
        num_trials=args.num_trials,
        analyzer_model=args.analyzer_model,
    )

    # Log analysis results to OpenPipe
    if results["analysis_results"]:
        print("\nLogging analysis results to OpenPipe...")

        for task_id, analysis_dict in results["analysis_results"].items():
            analysis = ErrorAnalysis(**analysis_dict)

            # Find failed trajectories for this task from the stored results
            failed_trajectories = [
                r
                for r in results["all_results"]
                if str(r["task_id"]) == str(task_id) and r["reward"] < 0.999
            ]

            # 1. Log each failing trajectory separately with its rollout analysis
            for i, rollout_analysis in enumerate(analysis.error_analysis_rollouts):
                if i < len(failed_trajectories):
                    trajectory_data = failed_trajectories[i]

                    # Create prompt for this specific rollout
                    prompt = f"Error analysis for Task {task_id}, Rollout {i + 1}"

                    # Log individual rollout with its trajectory data
                    await log_rollout_analysis_to_openpipe(
                        task_id=int(task_id),
                        rollout_analysis=rollout_analysis,
                        trajectory_data=trajectory_data,
                        rollout_index=i,
                        prompt=prompt,
                    )

            # 2. Log the full analysis summary for this task
            full_prompt = results["analysis_prompts"][task_id]
            full_response = f"Analysis of {len(analysis.error_analysis_rollouts)} failed rollouts for task {task_id}:\n\n"

            for i, rollout_analysis in enumerate(analysis.error_analysis_rollouts):
                full_response += f"ROLLOUT {i + 1}:\n"
                full_response += f"Summary: {rollout_analysis.summary}\n"
                full_response += f"Reasoning: {rollout_analysis.reasoning}\n"
                full_response += f"Blame: {rollout_analysis.blame_assignment}\n"
                full_response += f"Category: {rollout_analysis.category}\n\n"

            await log_full_analysis_to_openpipe(
                task_id=int(task_id),
                analysis=analysis,
                prompt=full_prompt,
                response=full_response,
            )

            print(
                f"Logged {len(analysis.error_analysis_rollouts)} rollouts + full analysis for task {task_id}"
            )

    # Save results
    time_str = datetime.now().strftime("%m%d%H%M%S")
    output_path = os.path.join(
        args.log_dir, f"error_analysis_{config.env}_{config.task_split}_{time_str}.json"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Display summary
    print(f"\n{'=' * 60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tasks analyzed: {results['num_tasks']}")
    print(f"Tasks that failed all trials: {results['total_failed_tasks']}")
    print(f"Tasks successfully analyzed: {results['analyzed_tasks']}")

    if results["analysis_results"]:
        print("\nBlame assignment distribution:")
        blame_counts = {}
        category_counts = {}

        for analysis_dict in results["analysis_results"].values():
            analysis = ErrorAnalysis(**analysis_dict)
            for rollout_analysis in analysis.error_analysis_rollouts:
                blame = rollout_analysis.blame_assignment
                category = rollout_analysis.category
                blame_counts[blame] = blame_counts.get(blame, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1

        for blame, count in blame_counts.items():
            print(f"  {blame}: {count}")

        print("\nFailure category distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
