import argparse
import sky
from train import RunConfig
import textwrap
from dotenv import load_dotenv, dotenv_values
from sky import exceptions as sky_exceptions
from sky import ClusterStatus

load_dotenv()

models = {
    "002": RunConfig(
        run_name="002",
        base_model="unsloth/Llama-3.2-1B",
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=1024,
    ),
    "test_gh200": RunConfig(
        run_name="test_gh200",
        base_model="unsloth/Llama-3.2-1B",
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=1024,
        accelerator="GH200",
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specific model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=models.keys(),
        help="Name of the model to train.",
    )
    args = parser.parse_args()

    config = models[args.model]

    print(f"Launching {args.model} on SkyPilot...")

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            curl -LsSf https://astral.sh/uv/install.sh | sh

            source $HOME/.local/bin/env

            uv sync
        """
    )

    config_json = config.model_dump_json()
    run_script = f"uv run python train_rm/train.py '{config_json}'"

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"train-{config.run_name}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )
    task.set_resources(sky.Resources(accelerators=config.accelerator))

    # Generate cluster name
    cluster_name = f"kyle-rofl-{config.run_name}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs...")
    cluster_status = sky.get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs...")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch the task and stream logs
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
            fast=True,
        )
    )

    print(f"Job submitted (ID: {job_id}). Streaming job logs...")
    # Stream the job logs until completion
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} finished with exit code {exit_code}.")
