import argparse
import concurrent.futures
import json
import textwrap
import traceback

import sky
from dotenv import dotenv_values
from sky import ClusterStatus
from tau_bench.types import RunConfig, TauBenchPolicyConfig, TauBenchTrainingConfig

# New imports for model serialization
import art

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
trainable_models["002"].name = "tau-bench-rl-002-tm-main-32"
trainable_models["002"].config.training_config.trajectories_per_group = 32
trainable_models["002"].config.training_config.groups_per_step = 4
trainable_models["002"].config.training_config.training_dataset_size = 4
trainable_models["002"].config.training_config.learning_rate = 1e-6

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

trainable_models["008"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["008"].config.training_config is not None
trainable_models["008"].name = "tau-bench-rl-008-tm-small-2"
trainable_models["008"].config.training_config.trajectories_per_group = 64
trainable_models["008"].config.training_config.groups_per_step = 4
trainable_models["008"].config.training_config.training_dataset_size = 4
trainable_models["008"].config.training_config.learning_rate = 1e-6
trainable_models["008"].config.run_config.skip_eval = False
trainable_models["008"].config.training_config.val_set_size = 60
trainable_models["008"].config.training_config.eval_steps = 8
trainable_models["008"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["009"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["009"].config.training_config is not None
trainable_models["009"].name = "tau-bench-rl-009-tm-too-big"
trainable_models["009"].config.training_config.trajectories_per_group = 64
trainable_models["009"].config.training_config.groups_per_step = 16
trainable_models["009"].config.training_config.training_dataset_size = 32
trainable_models["009"].config.training_config.learning_rate = 1e-6
trainable_models["009"].config.run_config.skip_eval = False
trainable_models["009"].config.training_config.val_set_size = 60
trainable_models["009"].config.training_config.eval_steps = 8
trainable_models["009"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["010"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["010"].config.training_config is not None
trainable_models["010"].name = "tau-bench-rl-010-tm-too-big"
trainable_models["010"].config.training_config.trajectories_per_group = 64
trainable_models["010"].config.training_config.groups_per_step = 16
trainable_models["010"].config.training_config.training_dataset_size = 32
trainable_models["010"].config.training_config.learning_rate = 1e-6
trainable_models["010"].config.run_config.skip_eval = False
trainable_models["010"].config.run_config.reward_type = "real+llm"
trainable_models["010"].config.run_config.judge_model = "o4-mini"
trainable_models["010"].config.training_config.val_set_size = 60
trainable_models["010"].config.training_config.eval_steps = 8
trainable_models["010"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["011"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["011"].config.training_config is not None
trainable_models["011"].name = "tau-bench-rl-011-tm-32b-4"
trainable_models["011"].base_model = "Qwen/Qwen2.5-32B-Instruct"
trainable_models["011"].config.training_config.trajectories_per_group = 64
trainable_models["011"].config.training_config.groups_per_step = 32
trainable_models["011"].config.training_config.training_dataset_size = 32
trainable_models["011"].config.training_config.learning_rate = 5e-7
trainable_models["011"].config.run_config.skip_eval = False
trainable_models["011"].config.run_config.reward_type = "real"
trainable_models["011"].config.run_config.user_model = "gpt-4.1"
trainable_models["011"].config.training_config.val_set_size = 60
trainable_models["011"].config.training_config.eval_steps = 8
trainable_models["011"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=8, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_32b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)
trainable_models["011"].config.run_config.plot_tensors = True

trainable_models["012"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["012"].config.training_config is not None
trainable_models["012"].name = "tau-bench-rl-012-airline-multi-2"
trainable_models["012"].config.training_config.trajectories_per_group = 32
trainable_models["012"].config.training_config.groups_per_step = 16
trainable_models["012"].config.training_config.training_dataset_size = 16
trainable_models["012"].config.training_config.learning_rate = 1e-6
trainable_models["012"].config.run_config.skip_eval = False
trainable_models["012"].config.training_config.val_set_size = 32
trainable_models["012"].config.training_config.eval_steps = 8
trainable_models["012"].config.run_config.reward_type = "real"
trainable_models["012"].config.run_config.user_model = "gpt-4.1"
trainable_models["012"].config.run_config.env = "airline"
trainable_models["012"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

trainable_models["015"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["015"].config.training_config is not None
trainable_models["015"].name = "tau-bench-rl-015-tm-2"
trainable_models["015"].base_model = "willcb/Qwen3-14B"
trainable_models["015"].config.training_config.trajectories_per_group = 64
trainable_models["015"].config.training_config.groups_per_step = 32
trainable_models["015"].config.training_config.training_dataset_size = 32
trainable_models["015"].config.training_config.learning_rate = 5e-6
trainable_models["015"].config.run_config.skip_eval = False
trainable_models["015"].config.run_config.reward_type = "real"
trainable_models["015"].config.run_config.user_model = "gpt-4.1"
trainable_models["015"].config.run_config.add_no_think = True
trainable_models["015"].config.training_config.val_set_size = 60
trainable_models["015"].config.training_config.eval_steps = 8
trainable_models["015"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen3_14b_instruct", model_type="QWEN3", async_weight_syncing=True
    ),
)

trainable_models["016"] = trainable_models["001"].model_copy(deep=True)
assert trainable_models["016"].config.training_config is not None
trainable_models["016"].name = "tau-bench-rl-016-tm-14b-gspo-2"
trainable_models["016"].base_model = "Qwen/Qwen2.5-14B-Instruct"
trainable_models["016"].config.training_config.trajectories_per_group = 32
trainable_models["016"].config.training_config.groups_per_step = 32
trainable_models["016"].config.training_config.training_dataset_size = 32
trainable_models["016"].config.training_config.learning_rate = 5e-7
trainable_models["016"].config.training_config.importance_sampling_level = "sequence"
trainable_models["016"].config.run_config.skip_eval = False
trainable_models["016"].config.run_config.reward_type = "real"
trainable_models["016"].config.run_config.user_model = "gpt-4.1"
trainable_models["016"].config.training_config.val_set_size = 60
trainable_models["016"].config.training_config.eval_steps = 8
trainable_models["016"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(tensor_parallel_size=8, gpu_memory_utilization=0.75),
    torchtune_args=art.dev.TorchtuneArgs(
        model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
    ),
)

# trainable_models["002"] = trainable_models["001"].model_copy(deep=True)
# assert trainable_models["002"].config.training_config is not None
# trainable_models["002"].name = "tau-bench-rl-002-tm-llm-reward-test"
# trainable_models["002"].config.training_config.trajectories_per_group = 3
# trainable_models["002"].config.training_config.groups_per_step = 3
# trainable_models["002"].config.training_config.training_dataset_size = 32
# trainable_models["002"].config.training_config.learning_rate = 1e-6
# trainable_models["002"].config.run_config.skip_eval = True
# trainable_models["002"].config.run_config.reward_type = "real+llm"
# trainable_models["002"].config.run_config.judge_model = "o4-mini"
# trainable_models["002"].config.training_config.val_set_size = 60
# trainable_models["002"].config.training_config.eval_steps = 8
# trainable_models["002"]._internal_config = art.dev.InternalModelConfig(
#     engine_args=art.dev.EngineArgs(tensor_parallel_size=4, gpu_memory_utilization=0.75),
#     torchtune_args=art.dev.TorchtuneArgs(
#         model="qwen2_5_14b_instruct", model_type="QWEN2", async_weight_syncing=True
#     ),
# )


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
    "--use-cluster-name",
    type=str,
    required=False,
    help="Use a specific cluster name for the task.",
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
            uv add --editable ~/ART --extra backend --extra plotting

            # Sync dependencies
            uv sync
        """
    )

    run_script = textwrap.dedent(
        f"""
        # Run the RL training
        uv remove openpipe-art
        uv add --editable ~/ART --extra backend --extra plotting

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
    num_gpus = 1
    if trainable_model._internal_config is not None:
        num_gpus = trainable_model._internal_config.get("engine_args", {}).get(
            "tensor_parallel_size", 1
        )

    task.set_resources(
        sky.Resources(
            accelerators=f"H200-SXM:{num_gpus}",
            cloud=sky.clouds.RunPod(),
            region="US",
        )
    )
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = args.use_cluster_name or f"tau-bench-rl-{model_key}"
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
