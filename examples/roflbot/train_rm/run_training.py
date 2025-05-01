import argparse
import sky
from training_helpers import RunConfig
import textwrap
from dotenv import load_dotenv, dotenv_values
from sky import ClusterStatus
import concurrent.futures

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
}

models["003"] = models["002"].model_copy(deep=True)
models["003"].run_name = "003"
models["003"].base_model = "Qwen/Qwen3-4B"

models["004"] = models["003"].model_copy(deep=True)
models["004"].run_name = "004"
models["004"].base_model = "Qwen/Qwen3-8B"

models["005"] = models["003"].model_copy(deep=True)
models["005"].run_name = "005"
models["005"].base_model = "Qwen/Qwen3-14B"

models["006"] = models["003"].model_copy(deep=True)
models["006"].run_name = "006"
models["006"].base_model = "Qwen/Qwen3-32B"

models["007"] = models["002"].model_copy(deep=True)
models["007"].run_name = "007"
models["007"].num_epochs = 4

models["008"] = models["006"].model_copy(deep=True)
models["008"].run_name = "008"
models["008"].num_epochs = 4
models["008"].gradient_accumulation_steps = 8
models["008"].learning_rate = 1e-4

models["009"] = models["002"].model_copy(deep=True)
models["009"].run_name = "009"
models["009"].num_epochs = 4
models["009"].batch_size = 16

models["010"] = models["009"].model_copy(deep=True)
models["010"].run_name = "010"
models["010"].base_model = "Qwen/Qwen3-1.7B"

models["011"] = models["009"].model_copy(deep=True)
models["011"].run_name = "011"
models["011"].gradient_accumulation_steps = 1
models["011"].batch_size = 64

models["012"] = models["011"].model_copy(deep=True)
models["012"].run_name = "012"
models["012"].num_epochs = 8
models["012"].batch_size = 128

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one or more models (comma separated)."
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model keys to train (e.g. 002,003,004).",
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
        config = models[model_key]
        print(f"Launching {model_key} on SkyPilot…")

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
            future.result()
