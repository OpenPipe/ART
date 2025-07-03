import argparse
import json
import textwrap
import concurrent.futures
import traceback
import sky
from dotenv import dotenv_values
from sky import ClusterStatus

# New imports for model serialization
import art
from tau_bench.types import RunConfig, TauBenchTrainingConfig, TauBenchPolicyConfig

# Usage:
# uv run run_training.py 001 --fast
# uv run run_training.py 001,002,003 --fast

# moved older models and how they were created to old_models.py
trainable_models = {
    "001": art.TrainableModel(
        name="tau-bench-rl-001-tm",
        project="tau_bench_rl",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=TauBenchPolicyConfig(
            training_config=TauBenchTrainingConfig(
                trajectories_per_group=6,
                groups_per_step=10,
                learning_rate=1.2e-5,
                eval_steps=10,
                val_set_size=85,
                training_dataset_size=30,
                num_epochs=50,
                train_mode="sync_rl",
            ),
            run_config=RunConfig(
                model_provider="hosted_vllm",
                user_model_provider="openai",
                user_model="gpt-4o",
                agent_strategy="tool-calling-rl",
                temperature=1.0,
                task_split="test",
                log_dir="rl_results",
                skip_eval=True,
            ),
        ),
    )
}

trainable_models["002"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["002"].config.training_config is not None
trainable_models["002"].name = "tau-bench-rl-002-tm"
trainable_models["002"].config.training_config.trajectories_per_group = 15
trainable_models["002"].config.training_config.groups_per_step = 4
trainable_models["002"].config.training_config.training_dataset_size = 4
trainable_models["002"].config.training_config.learning_rate = 5e-6

# v high lr, v low gn, because twitter said so
trainable_models["003"] = trainable_models["002"].model_copy(deep=True)
assert trainable_models["003"].config.training_config is not None
trainable_models["003"].name = "tau-bench-rl-003-tm"
trainable_models["003"].config.training_config.learning_rate = 1e-2
trainable_models["003"]._internal_config = art.dev.InternalModelConfig(
    trainer_args=art.dev.TrainerArgs(
        max_grad_norm=1e-7,
    )
)

parser = argparse.ArgumentParser(
    description="Train one or more tau-bench RL models (comma separated)."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. 001,002,003).",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Whether to use fast launch (skip setup).",
)
args = parser.parse_args()

# Parse and validate the requested model keys
requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
unknown = [m for m in requested_models if m not in trainable_models]
if unknown:
    raise ValueError(
        f"Unknown model keys requested: {', '.join(unknown)}. Valid keys: {', '.join(trainable_models.keys())}"
    )


def launch_model(model_key: str):
    trainable_model = trainable_models[model_key]
    print(f"Launching {model_key} on SkyPilot…")

    model_json = json.dumps(trainable_model.model_dump())

    # ------------------------------------------------------------------
    # 2. Prepare the setup and run scripts
    # ------------------------------------------------------------------

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt update && apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            # Install project in editable mode
            uv remove openpipe-art
            uv add --editable ~/ART --extra backend

            # Sync dependencies
            uv sync
        """
    )

    run_script = textwrap.dedent(
        f"""
        # Run the RL training
        uv remove openpipe-art
        uv add --editable ~/ART --extra backend

        uv run run_rl.py '{model_json}'
    """
    )

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"tau-bench-rl-{model_key}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),
    )
    task.set_resources(sky.Resources(accelerators="H100-SXM:1"))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = f"tau-bench-rl-{model_key}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs…")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch the task; stream_and_get blocks until the task starts running, but
    # running this in its own thread means all models run in parallel.
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
            fast=args.fast,
        )
    )

    print(f"Job submitted for {model_key} (ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} for {model_key} finished with exit code {exit_code}.")


# Launch all requested models in parallel threads
with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(requested_models)
) as executor:
    futures = [executor.submit(launch_model, key) for key in requested_models]
    for future in concurrent.futures.as_completed(futures):
        # Propagate any exceptions raised inside threads
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            print(f"Traceback: {traceback.format_exc()}")
