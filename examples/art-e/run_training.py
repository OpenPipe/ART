import argparse
import sky
from art_e.project_types import ProjectPolicyConfig
import json
import textwrap
import concurrent.futures
import traceback
from dotenv import dotenv_values
from sky import ClusterStatus

from all_experiments import models

# Usage:
# uv run run_training.py --models=207 --fast

parser = argparse.ArgumentParser(
    description="Train one or more models (comma separated)."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. art-1).",
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

    if not model.config or not isinstance(model.config, ProjectPolicyConfig):
        raise ValueError(
            f"Training config not found or is invalid for model {model_key}"
        )

    # Assert type for linter
    assert isinstance(model.config, ProjectPolicyConfig)

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env
        """
    )

    model_dict = model.model_dump()
    model_dict["config"] = model.config.model_dump()
    # Pass the model key as run_name and the serialized model
    # Remove --no-managed-python and revert to python 3.12 once https://github.com/astral-sh/python-build-standalone/pull/667#issuecomment-3059073433 is addressed.
    run_script = textwrap.dedent(f"""
        uv remove openpipe-art
        uv add --editable ~/ART --extra backend

        uv run --no-managed-python art_e/train.py '{json.dumps(model_dict)}'
    """)

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"art-e-{model_key}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )
    task.set_resources(sky.Resources(accelerators="H100-SXM:1"))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = f"art-e-{model_key}"
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
