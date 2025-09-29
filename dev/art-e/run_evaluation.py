#!/usr/bin/env python3

import argparse
import os
import textwrap

import sky
from dotenv import dotenv_values, load_dotenv
from sky import ClusterStatus

load_dotenv()

parser = argparse.ArgumentParser(
    description="Run validation noise evaluation for a model on a remote GPU."
)
parser.add_argument(
    "--model",
    type=str,
    default="231",
    help="Model key to evaluate (default: 231)",
)
parser.add_argument(
    "--runs",
    type=int,
    default=20,
    help="Number of evaluation runs (default: 20)",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Whether to use fast launch (skip setup).",
)

args = parser.parse_args()


def launch_evaluation():
    print(
        f"Launching evaluation for model {args.model} with {args.runs} runs on SkyPilot…"
    )

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env
        """
    )

    # Create the remote evaluation script
    run_script = textwrap.dedent(f"""
        uv remove openpipe-art
        uv add --editable ~/ART --extra backend

        uv run --no-managed-python evaluate_noise.py --model {args.model} --runs {args.runs}
    """)

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"art-e-eval-{args.model}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )
    task.set_resources(sky.Resources(accelerators="H100-SXM:1"))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = f"art-e-eval-{args.model}"
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

    # Launch the task
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

    print(f"Job submitted for evaluation (ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Evaluation job {job_id} finished with exit code {exit_code}.")


if __name__ == "__main__":
    launch_evaluation()
