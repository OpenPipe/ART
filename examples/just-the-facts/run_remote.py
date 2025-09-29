# Usage:
# uv run run_training.py --models=207 --fast

import argparse
import textwrap
from datetime import datetime

import sky
from dotenv import dotenv_values, load_dotenv
from sky import ClusterStatus

load_dotenv()

parser = argparse.ArgumentParser(
    description="Run a Python module remotely on a SkyPilot cluster. All arguments after the flags will be treated as the module path (and any additional arguments) to execute remotely."
)

# Required/optional flags that we want to parse explicitly
parser.add_argument(
    "--cluster-name",
    type=str,
    required=True,
    help="Name of the SkyPilot cluster to run the command on.",
)
parser.add_argument(
    "--cancel-all",
    action="store_true",
    help="Cancel all jobs on the cluster before submitting the new one.",
)

# Everything that comes after the above flags will be captured here and treated as the command to run
parser.add_argument(
    "command",
    nargs=argparse.REMAINDER,
    help="Python module path (and optional arguments) to execute remotely, e.g. 'mcp_rl.benchmarks.generate_benchmarks'.",
)

args = parser.parse_args()

cluster_name = args.cluster_name
cancel_all = args.cancel_all

# Join the remaining arguments into a single string to pass to `python -m ...`
if len(args.command) == 0:
    parser.error("No command specified. Provide a module path to run after the flags.")

remote_command = " ".join(args.command).strip()


def run_remote(command: str):
    print(f"Running command: '{command}'")
    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env
        """
    )

    run_script = textwrap.dedent(f"""
        cd ~/ART && uv add "transformers==4.53.2" --no-sync
        cd ~/sky_workdir
        uv remove openpipe-art
        uv add --editable ~/ART --extra backend
        uv sync --locked

        uv run python {command}
    """)

    seconds_str = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"art-remote-{seconds_str}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )
    task.set_resources(sky.Resources(accelerators="H100-SXM:1"))
    task.set_file_mounts({"~/ART": "../.."})

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
    if (
        len(cluster_status) > 0
        and cluster_status[0]["status"] == ClusterStatus.UP
        and cancel_all
    ):
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs…")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch the task; stream_and_get blocks until the task starts running, but
    # running this in its own thread means all models run in parallel.
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=120,
            down=True,
        )
    )

    print(f"Job submitted for {cluster_name} (ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} for {cluster_name} finished with exit code {exit_code}.")


if __name__ == "__main__":
    run_remote(remote_command)
