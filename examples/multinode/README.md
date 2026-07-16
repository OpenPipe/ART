# ART multi-node smoke

`program.py` is a bounded CPU example using only public ART APIs. Its top-level
controller admits the attached hosts, then its top-level rollout returns one
synthetic Yes/No/Maybe `Trajectory` per host for each answer. It never loads a
model or starts Megatron or vLLM.

Run the same controller on one local Monarch worker from the project root:

```fish
env PYTHONPATH=(pwd)/examples/multinode .venv/bin/art-monarch local \
  --program program:main \
  --port 0 \
  --startup-timeout 90
```

Or let SkyPilot run it on every node in one allocation:

```fish
sky launch -c art-multinode examples/multinode/skypilot.yaml
```

SkyPilot synchronizes `workdir` and runs `setup` on every node before starting
the same `run` command on every node. ART starts one Monarch worker per node and
calls `program:main` only on rank 0. User controllers and rollouts must remain
importable at the same paths on every node; ART sends import references rather
than pickled closures.

Edit the accelerator and setup commands for your infrastructure. The example
assumes `uv` is installed in the image and installs ART's `distributed` extra
from the synchronized source checkout. GPU training also needs the `megatron`
extra and the locked `vllm_runtime` project; release wheels instead carry the
managed vLLM runtime bundle. Use `sky launch` after changing setup, and reuse an
unchanged cluster without rerunning setup with:

```fish
sky exec art-multinode examples/multinode/skypilot.yaml
```

GPU workloads spanning hosts must also set `NCCL_NET` on every node and provide
the same exact registered name through `ClusterSpec.nccl_transport`. ART proves
that selected module before trainer or vLLM model allocation and never falls
back to Socket. `ART_VLLM_RUNTIME_BIN`, when set, must point directly to a
standard `.venv/bin/art-vllm-runtime-server`; arbitrary wrappers fail closed.

Each invocation terminates every worker loop before the task exits. Reusing the
cluster starts fresh loops; Monarch 0.6 worker addresses are generation-owned and
completed loops are not reattached.

Setting `num_nodes: 1` exercises the same API on one node. Do not expose the
default private ports `22222` and `22223`; pinned Monarch 0.6 does not
authenticate its transport.
