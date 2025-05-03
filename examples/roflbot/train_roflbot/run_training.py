# Usage: uv run train_roflbot/run_training.py --models=003
import art
from project_types import PolicyConfig, TrainingConfig
import argparse
import sky
import textwrap
from dotenv import load_dotenv, dotenv_values
from sky import ClusterStatus
import concurrent.futures
import json
import traceback

load_dotenv()

models = {
    "003": art.TrainableModel(
        name="roflbot-003",
        project="roflbot",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=PolicyConfig(
            max_turns=1,
            max_tokens=8192,
            thinking_char_budget=5000,
            training_config=TrainingConfig(
                trajectories_per_group=6,
                groups_per_step=8,
                learning_rate=1.2e-5,
                eval_steps=30,
                val_set_size=100,
                training_dataset_size=4000,
                num_epochs=1,
            ),
        ),
    ),
    "004": art.TrainableModel(
        name="roflbot-004",
        project="roflbot",
        base_model="Qwen/Qwen3-32B",
        config=PolicyConfig(
            max_turns=1,
            max_tokens=8192,
            thinking_char_budget=5000,
            training_config=TrainingConfig(
                trajectories_per_group=6,
                groups_per_step=8,
                learning_rate=1.2e-5,
                eval_steps=30,
                val_set_size=100,
                training_dataset_size=4000,
                num_epochs=1,
            ),
        ),
    ),
}


parser = argparse.ArgumentParser(
    description="Train one or more roflbot models (comma separated)."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. 003).",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Whether to use fast launch (skip setup).",
)
args = parser.parse_args()

# Parse and validate the requested model keys
requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
unknown = [m for m in requested_models if m not in models]
if unknown:
    raise ValueError(
        f"Unknown model keys requested: {', '.join(unknown)}. Valid keys: {', '.join(models.keys())}"
    )


def launch_model(model_key: str):
    model = models[model_key]
    print(f"Launching {model_key} ({model.name}) on SkyPilot…")

    if (
        not model.config
        or not isinstance(model.config, PolicyConfig)
        or not model.config.training_config
    ):
        raise ValueError(
            f"Training config not found or is invalid for model {model_key}"
        )

    # Assert type for linter
    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    training_config = model.config.training_config

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            uv remove openpipe-art
            uv add --editable ~/ART

            uv sync
        """
    )

    model_dict = model.model_dump()
    model_dict["config"] = model.config.model_dump()
    # Pass the model key as run_name and the serialized model
    run_script = f"uv run train_roflbot/train.py '{json.dumps(model_dict)}'"

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"roflbot-train-{model_key}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )

    task.set_resources(sky.Resources(accelerators=training_config.gpus))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = f"kyle-roflbot-{model_key}"
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
