"""Launch a multi-step Qwen3.5 LocalBackend yes-no-maybe run on SkyPilot."""

import argparse
import os
import textwrap

from dotenv import load_dotenv
import sky
from sky import ClusterStatus

load_dotenv()

DEFAULT_IMAGE_ID = "docker:nvidia/cuda:12.8.1-devel-ubuntu22.04"


def _format_env_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_int_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


parser = argparse.ArgumentParser(
    description="Launch a Qwen3.5 LocalBackend yes-no-maybe convergence run."
)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-4B")
parser.add_argument("--accelerator", type=str, default="H200:1")
parser.add_argument(
    "--cluster-name", type=str, default="art-qwen35-localbackend-yes-no-maybe"
)
parser.add_argument("--image-id", type=str, default=DEFAULT_IMAGE_ID)
parser.add_argument("--project", type=str, default="qwen35-localbackend-yes-no-maybe")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
parser.add_argument("--max-model-len", type=int, default=1024)
parser.add_argument("--max-seq-length", type=int, default=1024)
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--num-steps", type=int, default=10)
parser.add_argument("--rollouts-per-prompt", type=int, default=8)
parser.add_argument("--eval-prompts", type=int, default=24)
parser.add_argument("--eval-every-n-steps", type=int, default=1)
parser.add_argument("--max-tokens", type=int, default=5)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument(
    "--load-in-4bit", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--load-in-16bit", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--enable-thinking", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument("--trainer-gpu-ids", type=int, nargs="+")
parser.add_argument("--inference-gpu-ids", type=int, nargs="+")
args = parser.parse_args()

assert (args.trainer_gpu_ids is None) == (args.inference_gpu_ids is None), (
    "--trainer-gpu-ids and --inference-gpu-ids must both be set or both unset"
)

cluster_name = args.cluster_name
cluster_prefix = os.environ.get("CLUSTER_PREFIX")
if cluster_prefix:
    cluster_name = f"{cluster_prefix}-{cluster_name}"

setup_script = textwrap.dedent("""\
    echo 'Setting up environment...'
    apt-get update
    apt-get install -y python3 python3-pip python-is-python3 git curl ninja-build
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
""")

env = [
    f"PROJECT={args.project}",
    "MODEL_NAME=qwen35-localbackend-ynm-$(date +%Y%m%d-%H%M%S)",
    f"BASE_MODEL={args.base_model}",
    f"GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization}",
    f"MAX_MODEL_LEN={args.max_model_len}",
    f"MAX_SEQ_LENGTH={args.max_seq_length}",
    f"MAX_NUM_SEQS={args.max_num_seqs}",
    "ENFORCE_EAGER=true",
    f"LOAD_IN_4BIT={_format_env_bool(args.load_in_4bit)}",
    f"LOAD_IN_16BIT={_format_env_bool(args.load_in_16bit)}",
    f"ENABLE_THINKING={_format_env_bool(args.enable_thinking)}",
    f"NUM_STEPS={args.num_steps}",
    f"ROLLOUTS_PER_PROMPT={args.rollouts_per_prompt}",
    f"EVAL_PROMPTS={args.eval_prompts}",
    f"EVAL_EVERY_N_STEPS={args.eval_every_n_steps}",
    f"MAX_TOKENS={args.max_tokens}",
    f"LEARNING_RATE={args.learning_rate}",
]
if args.trainer_gpu_ids is not None:
    env.extend(
        [
            f"TRAINER_GPU_IDS={_format_int_list(args.trainer_gpu_ids)}",
            f"INFERENCE_GPU_IDS={_format_int_list(args.inference_gpu_ids)}",
        ]
    )
env_block = " \\\n    ".join(env)

run_script = textwrap.dedent(
    f"""\
    source $HOME/.local/bin/env
    cd ~/sky_workdir
    ~/.local/bin/uv sync --extra backend
    {env_block} \
    ~/.local/bin/uv run dev/yes-no-maybe-metrics.py
"""
)

task = sky.Task(
    name="qwen3.5-localbackend-yes-no-maybe",
    setup=setup_script,
    run=run_script,
    workdir=".",
)
task.set_resources(
    sky.Resources(
        accelerators=args.accelerator,
        cloud=sky.clouds.Kubernetes(),
        image_id=args.image_id,
    )
)
task.set_file_mounts({"~/sky_workdir/.env": ".env"})

print(f"Launching on cluster: {cluster_name}")
print(f"  base_model: {args.base_model}")
print(f"  project: {args.project}")
print(f"  accelerator: {args.accelerator}")
print(f"  image_id: {args.image_id}")
print(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
print(f"  max_model_len: {args.max_model_len}")
print(f"  max_seq_length: {args.max_seq_length}")
print(f"  max_num_seqs: {args.max_num_seqs}")
print(f"  num_steps: {args.num_steps}")
print(f"  rollouts_per_prompt: {args.rollouts_per_prompt}")
print(f"  eval_prompts: {args.eval_prompts}")
print(f"  eval_every_n_steps: {args.eval_every_n_steps}")
print(f"  max_tokens: {args.max_tokens}")
print(f"  learning_rate: {args.learning_rate}")
print(f"  load_in_4bit: {args.load_in_4bit}")
print(f"  load_in_16bit: {args.load_in_16bit}")
print(f"  enable_thinking: {args.enable_thinking}")
print(f"  trainer_gpu_ids: {args.trainer_gpu_ids}")
print(f"  inference_gpu_ids: {args.inference_gpu_ids}")

cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
if cluster_status and cluster_status[0]["status"] == ClusterStatus.UP:
    print(f"Cluster {cluster_name} is UP. Canceling any active jobs...")
    sky.stream_and_get(sky.cancel(cluster_name, all=True))

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

print(f"Job submitted (ID: {job_id}). Streaming logs...")
exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
print(f"Job {job_id} finished with exit code {exit_code}.")
