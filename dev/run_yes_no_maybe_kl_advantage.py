"""Launch yes-no-maybe-kl-advantage training on SkyPilot (Kubernetes).

Usage:
    uv run dev/run_yes_no_maybe_kl_advantage.py
    uv run dev/run_yes_no_maybe_kl_advantage.py --fast
    uv run dev/run_yes_no_maybe_kl_advantage.py --base-model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import textwrap

from dotenv import load_dotenv
import sky
from sky import ClusterStatus

load_dotenv()

parser = argparse.ArgumentParser(
    description="Launch yes-no-maybe KL advantage training on SkyPilot."
)
parser.add_argument(
    "--fast", action="store_true", help="Skip setup (for re-runs on existing cluster)."
)
parser.add_argument(
    "--base-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
)
parser.add_argument("--num-steps", type=int, default=20)
parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
parser.add_argument("--accelerator", type=str, default="H200:1")
parser.add_argument("--cluster-name", type=str, default=None)
parser.add_argument(
    "--kl-ref-step",
    type=int,
    default=None,
    help="Checkpoint step of training model to use as KL reference",
)
parser.add_argument(
    "--kl-ref-adapter-path",
    type=str,
    default=None,
    help="Path to LoRA adapter checkpoint to use as KL reference",
)
args = parser.parse_args()

cluster_name = args.cluster_name or f"ynm-kl-{args.kl_penalty_coef}"
cluster_prefix = os.environ.get("CLUSTER_PREFIX")
if cluster_prefix:
    cluster_name = f"{cluster_prefix}-{cluster_name}"

setup_script = textwrap.dedent("""\
    echo 'Setting up environment...'
    apt install -y nvtop
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
""")

kl_ref_env = ""
if args.kl_ref_step is not None:
    kl_ref_env = f"KL_REF_STEP={args.kl_ref_step} "
elif args.kl_ref_adapter_path is not None:
    kl_ref_env = f"KL_REF_ADAPTER_PATH={args.kl_ref_adapter_path} "

run_script = textwrap.dedent(f"""\
    source $HOME/.local/bin/env
    cd ~/sky_workdir
    {kl_ref_env}BASE_MODEL={args.base_model} NUM_STEPS={args.num_steps} KL_PENALTY_COEF={args.kl_penalty_coef} uv run --python 3.11 --extra backend dev/yes-no-maybe-kl-advantage.py
""")

task = sky.Task(
    name="yes-no-maybe-kl-advantage",
    setup=setup_script,
    run=run_script,
    workdir=".",
)
task.set_resources(
    sky.Resources(accelerators=args.accelerator, cloud=sky.clouds.Kubernetes())
)
task.set_file_mounts(
    {
        "~/sky_workdir/.env": ".env",
    }
)

print(f"Launching on cluster: {cluster_name}")
print(f"  base_model: {args.base_model}")
print(f"  accelerator: {args.accelerator}")
print(f"  num_steps: {args.num_steps}")
print(f"  kl_penalty_coef: {args.kl_penalty_coef}")
if args.kl_ref_step is not None:
    print(f"  kl_ref_step: {args.kl_ref_step}")
if args.kl_ref_adapter_path is not None:
    print(f"  kl_ref_adapter_path: {args.kl_ref_adapter_path}")

# Cancel any existing jobs on this cluster
cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
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
