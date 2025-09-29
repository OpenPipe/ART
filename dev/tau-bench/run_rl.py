# Copyright OpenPipe

import argparse
import asyncio
import concurrent.futures
import copy
import json
import random
from typing import Any, Dict, List

from dotenv import load_dotenv
from tau_bench.agents.tool_calling_agent import ToolCallingRLAgent
from tau_bench.envs import get_env
from tau_bench.general_rm import calculate_reward, create_general_rm_trajectory_groups
from tau_bench.rl_utils import (
    log_trajectory_to_openpipe,
    update_steps_for_openpipe_logs,
)
from tau_bench.run import agent_factory
from tau_bench.types import RunConfig, SolveResult, TauBenchPolicyConfig
from tqdm.asyncio import tqdm_asyncio

import art
from art.local import LocalBackend
from art.utils import iterate_dataset, limit_concurrency

# Load environment variables
load_dotenv(override=True)


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_messages = []
    for msg in messages:
        cleaned_msg = {k: v for k, v in msg.items() if v is not None}
        cleaned_messages.append(cleaned_msg)
    return cleaned_messages


@limit_concurrency(256)
async def rollout_tau_bench_task(
    model: art.Model[TauBenchPolicyConfig],
    task_index: int,
    step: int = 0,
    phase: str = "train",
    reward_type: str = "real",
    is_shadow: bool = False,
) -> art.Trajectory:
    """
    Generate a trajectory for a single tau-bench task using the given model.
    This adapts the tau-bench evaluation loop for RL trajectory generation.
    Now truly async.
    """
    # print(f"Rolling out task {task_index} (step {step}, phase {phase})")
    config = copy.deepcopy(model.config.run_config)
    if is_shadow:
        config.model = "gpt-4.1"
        config.model_provider = "openai"
        config.api_key = None
        config.base_url = None

    # Get isolated environment for this task
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        task_index=task_index,
    )
    if config.add_no_think:
        env.wiki += "\n/no_think"

    # Create agent with the trainable model
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )

    if not isinstance(agent, ToolCallingRLAgent):
        raise ValueError("Agent must be a ToolCallingRLAgent")

    # Create trajectory object
    traj = art.Trajectory(
        messages_and_choices=[],
        tools=env.tools_info,
        reward=0,
        metadata={
            "task_index": str(task_index),
            "env": config.env,
            "training_step": str(step),
            "phase": phase,
            "model": model.name,
            "reward_type": config.reward_type,
            "is_shadow": str(is_shadow),
        },
    )

    try:
        # Run the agent on the task (now async call)
        result = await agent.solve(
            env=env,
            task_index=task_index,
            max_num_steps=config.max_num_steps,
        )
        outcome_correct = 1 if result.reward == 1 else 0

        # Convert result to trajectory format
        traj.reward, explanation = await calculate_reward(result, config)
        traj.metrics = {
            "total_steps": result.info["total_steps"],
            "final_prompt_tokens": result.info["final_prompt_tokens"],
            "avg_completion_tokens": result.info["avg_completion_tokens"],
            "max_completion_tokens": result.info["max_completion_tokens"],
            "outcome_correct": outcome_correct,
            "forced_stop": result.info["forced_stop"],
        }
        traj.metadata.update(result.info)
        traj.metadata["reward"] = (
            "pending_general_rm" if config.reward_type == "general_rm" else traj.reward
        )
        traj.metadata["outcome_correct"] = traj.metrics["outcome_correct"]
        traj.metadata["judge_explanation"] = explanation

        if config.messages_only:
            traj.messages_and_choices = clean_messages(result.messages)  # type: ignore
        else:
            traj.messages_and_choices = agent.create_messages_and_choices()  # type: ignore
    except Exception as e:
        print(f"Error in rollout for task {task_index}: {e}")
        traj.reward = -1.0
        traj.metadata["error"] = str(e)
        traj.messages_and_choices = agent.create_messages_and_choices()  # type: ignore
        result = SolveResult(
            reward=-1.0,
            info={"error": str(e)},
            messages=agent.messages,
            total_cost=0.0,
        )

    traj.finish()

    # Log to langfuse/openpipe
    try:
        await log_trajectory_to_openpipe(traj, result.messages)
    except Exception as e:
        print(f"Error logging trajectory to openpipe: {e}")

    # print(f"Finished rolling out task {task_index} (reward: {traj.reward})")
    return traj


async def evaluate_model(
    model: art.Model[TauBenchPolicyConfig],
    config: RunConfig,
    step: int,
    val_task_indices: List[int],
) -> float:
    """Evaluate the model on a subset of tasks"""
    print(f"Evaluating model on {len(val_task_indices)} tasks...")

    total_reward = 0.0

    trajectories = await art.gather_trajectories(
        (
            rollout_tau_bench_task(
                model, val_task_index, step, "val", reward_type=config.reward_type
            )
            for val_task_index in val_task_indices
        )
    )
    await model.log(trajectories=trajectories, split="val")

    for traj in trajectories:
        total_reward += traj.reward
        print(f"Eval task {traj.metadata['task_index']}: reward={traj.reward}")

    avg_reward = total_reward / len(val_task_indices)
    print(f"Average evaluation reward: {avg_reward}")
    return avg_reward


async def train(model: art.TrainableModel[TauBenchPolicyConfig]):
    """Main training loop adapted from art-e example"""
    loop = asyncio.get_event_loop()
    big_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(big_pool)

    config = model.config.run_config
    training_config = model.config.training_config

    if training_config is None:
        raise ValueError("Training config is not set")

    with LocalBackend() as backend:
        # Setup model with backend
        await model.register(backend)
        config.api_key = model.inference_api_key
        config.base_url = model.inference_base_url
        config.base_model = model.base_model

        print("Loading training tasks...")
        # Get environment to access tasks
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
        )

        # Create list of task indices for training
        end_index = (
            min(config.end_index, len(env.tasks))
            if config.end_index != -1
            else len(env.tasks)
        )
        if config.task_ids:
            train_task_indices = config.task_ids
        else:
            train_task_indices = list(
                range(
                    config.start_index,
                    min(end_index, training_config.training_dataset_size),
                )
            )

        # Validation task indices
        val_task_indices = list(
            range(
                len(train_task_indices),
                len(train_task_indices) + training_config.val_set_size,
            )
        )

        print(f"Training on {len(train_task_indices)} tasks")
        print(f"Validation on {len(val_task_indices)} tasks")

        if training_config.train_mode == "async_rl":
            global_step = 0
            train_task_indices_async_rl = []
            for _ in range(training_config.num_epochs):
                train_task_indices_async_rl.extend(
                    random.sample(train_task_indices, len(train_task_indices))
                )

            async for trajectory_groups in art.trajectory_group_batches(
                (
                    art.TrajectoryGroup(
                        (
                            rollout_tau_bench_task(model, task_index, -1, "train")
                            for _ in range(training_config.trajectories_per_group)
                        )
                    )
                    for task_index in train_task_indices_async_rl
                ),
                batch_size=training_config.groups_per_step,
                max_concurrent_batches=3,
                skip_batches=await model.get_step(),
            ):
                # NOT UPDATED FOR TRAINING WITH SHADOW TRAJECTORIES
                if (
                    global_step % training_config.eval_steps == 0
                    and not config.skip_eval
                ):
                    print(f"\n--- Evaluating at Step {global_step} ---")
                    await evaluate_model(model, config, global_step, val_task_indices)
                    # await model.delete_checkpoints()

                if config.reward_type == "general_rm":
                    print("Creating general RM trajectory groups...")
                    updated_groups = await tqdm_asyncio.gather(
                        *[
                            create_general_rm_trajectory_groups(group, config)
                            for group in trajectory_groups
                        ],
                        desc="Creating general RM trajectory groups",
                        total=len(trajectory_groups),
                    )
                    trajectory_groups = updated_groups

                try:
                    await update_steps_for_openpipe_logs(trajectory_groups, global_step)
                except Exception as e:
                    print(f"Error updating steps for openpipe logs: {e}")

                # Training step
                print(f"Training on {len(trajectory_groups)} trajectory groups...")
                await model.train(
                    trajectory_groups,
                    config=art.TrainConfig(learning_rate=training_config.learning_rate),
                    _config=art.dev.TrainConfig(plot_tensors=config.plot_tensors),
                )
                if config.is_multi_gpu:
                    await model.delete_checkpoints()
                global_step += 1
        else:
            # Training iterator
            train_iterator = iterate_dataset(
                train_task_indices,
                groups_per_step=training_config.groups_per_step,
                num_epochs=training_config.num_epochs,
                initial_step=await model.get_step(),
            )

            for batch in train_iterator:
                print(
                    f"\n--- Training Step {batch.step} (Epoch {batch.epoch}, Step {batch.epoch_step}) ---"
                )

                # Evaluation
                if (
                    batch.step % training_config.eval_steps == 0
                    and not config.skip_eval
                ):
                    print(f"\n--- Evaluating at Step {batch.step} ---")
                    await evaluate_model(model, config, batch.step, val_task_indices)
                    await model.delete_checkpoints()

                # Generate trajectory groups
                print(f"Generating trajectories for {len(batch.items)} tasks...")
                groups = await art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            (
                                rollout_tau_bench_task(
                                    model,
                                    task_index,
                                    batch.step,
                                    "train",
                                    reward_type=config.reward_type,
                                    is_shadow=config.add_shadow_trajectory
                                    and rollout_idx
                                    % training_config.trajectories_per_group
                                    == 0,
                                )
                                for rollout_idx in range(
                                    training_config.trajectories_per_group
                                )
                            )
                        )
                        for task_index in batch.items
                    )
                )
                if config.reward_type == "general_rm":
                    print("Creating general RM trajectory groups...")
                    updated_groups = await tqdm_asyncio.gather(
                        *[
                            create_general_rm_trajectory_groups(group, config)
                            for group in groups
                        ],
                        desc="Creating general RM trajectory groups",
                        total=len(groups),
                    )
                    groups = updated_groups

                # Training step
                print(f"Training on {len(groups)} trajectory groups...")
                await model.train(
                    groups,
                    config=art.TrainConfig(learning_rate=training_config.learning_rate),
                    _config=art.dev.TrainConfig(
                        importance_sampling_level=training_config.importance_sampling_level,
                        allow_training_without_logprobs=True
                        if config.messages_only
                        else False,
                    ),
                )
                if config.is_multi_gpu:
                    await model.delete_checkpoints()

                # Log progress
                total_reward = sum(
                    sum(traj.reward for traj in group.trajectories) for group in groups
                )
                num_trajectories = sum(len(group.trajectories) for group in groups)
                avg_reward = (
                    total_reward / num_trajectories if num_trajectories > 0 else 0
                )
                print(f"Step {batch.step}: Average training reward = {avg_reward}")

        # Final evaluation
        print("\n--- Final Evaluation ---")
        final_step = await model.get_step()
        final_reward = await evaluate_model(model, config, final_step, val_task_indices)
        print(f"Final average reward: {final_reward}")

        print("Training completed!")


def main():
    """Entry point: expects a JSON-serialized TrainableModel (model_json) just like art-e/train.py"""

    parser = argparse.ArgumentParser(
        description="Run RL training for a serialized TrainableModel"
    )
    parser.add_argument(
        "model_json",
        help="JSON string serialization of the TrainableModel to train",
    )
    args = parser.parse_args()

    print("Model JSON:", args.model_json)

    # Recreate the TrainableModel from the serialized JSON.
    model_dict = json.loads(args.model_json)

    # The nested `config` needs to be converted back into the proper pydantic model.
    model_dict["config"] = TauBenchPolicyConfig(**model_dict["config"])

    is_multi_gpu = False

    # the nested "_internal_config" needs to be converted back into the proper pydantic model.
    if "_internal_config" in model_dict and model_dict["_internal_config"] is not None:
        model_dict["_internal_config"] = art.dev.InternalModelConfig(
            **model_dict["_internal_config"]
        )

    model: art.TrainableModel[TauBenchPolicyConfig] = art.TrainableModel(**model_dict)
    if model._internal_config is not None:
        is_multi_gpu = (
            model._internal_config.get("engine_args", {}).get("tensor_parallel_size", 1)
            > 1
        )
    model.config.run_config.model = (
        model.name
    )  # set run_config model name to model name
    model.config.run_config.is_multi_gpu = is_multi_gpu

    print(model)

    run_config = model.config.run_config

    print(f"Starting RL training for model: {model.name}")
    print(f"Base model: {model.base_model}")
    print(f"Environment: {run_config.env}")
    print(f"Task split: {run_config.task_split}")
    print(f"Reward type: {run_config.reward_type}")

    # Run training
    asyncio.run(train(model))


if __name__ == "__main__":
    main()
