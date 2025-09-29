# Usage:
# uv run run_training.py --models=207 --fast
# uv run run_training.py --models=210-% --fast  # Run all 210-* models
# uv run run_training.py --models=210-16,210-% --fast  # Run 210-16 and all 210-* models

import argparse
import concurrent.futures
import fnmatch
import json
import os
import random
import textwrap
import traceback

import sky
from all_experiments import models
from art_e.project_types import ProjectPolicyConfig
from dotenv import dotenv_values, load_dotenv
from sky import ClusterStatus

load_dotenv()

parser = argparse.ArgumentParser(
    description="Train one or more models (comma separated)."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. 207 or 210-% for wildcards).",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Whether to use fast launch (skip setup).",
)
parser.add_argument(
    "--add-suffix",
    type=str,
    help="Add suffix to model name. Use 'random' to generate a random suffix.",
)
args = parser.parse_args()

# Parse and validate the requested model keys
requested_patterns = [m.strip() for m in args.models.split(",") if m.strip()]
requested_models = []

for pattern in requested_patterns:
    # Replace % with * for fnmatch
    glob_pattern = pattern.replace("%", "*")

    # Find all matching model keys
    matching_keys = [key for key in models.keys() if fnmatch.fnmatch(key, glob_pattern)]

    if matching_keys:
        requested_models.extend(matching_keys)
        if len(matching_keys) > 1:
            print(f"Pattern '{pattern}' matched: {', '.join(sorted(matching_keys))}")
    else:
        # If no wildcard and exact match doesn't exist, it's an error
        if "%" not in pattern and pattern not in models:
            raise ValueError(
                f"Unknown model key: {pattern}. Valid keys: {', '.join(sorted(models.keys()))}"
            )
        elif "%" in pattern:
            print(f"Warning: Pattern '{pattern}' matched no models")

# Remove duplicates while preserving order
requested_models = list(dict.fromkeys(requested_models))

if not requested_models:
    raise ValueError("No models matched the provided patterns")


def launch_model(model_key: str):
    model = models[model_key].model_copy(deep=True)

    # Add suffix if requested
    if args.add_suffix:
        if args.add_suffix.lower() == "random":
            # Generate a random suffix
            suffix = f"-{random.randint(1000, 9999)}"
            print(f"Generated random suffix: {suffix}")
        else:
            # Use the provided suffix
            suffix = f"-{args.add_suffix}"

        model.name = f"{model.name}{suffix}"
        print(f"Launching {model_key} as {model.name} on SkyPilot…")
    else:
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
    task.set_resources(sky.Resources(accelerators=model.config.accelerator))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = f"art-e-{model_key}"
    # Add cluster prefix if defined in environment
    cluster_prefix = os.environ.get("CLUSTER_PREFIX")
    if cluster_prefix:
        cluster_name = f"{cluster_prefix}-{cluster_name}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
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
