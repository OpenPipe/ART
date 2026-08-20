# ART multi-node smoke

`program.py` is a bounded CPU example using only public ART APIs. Its top-level
controller receives a typed launch context and admits the attached hosts, then
its top-level rollout returns one synthetic Yes/No/Maybe `Trajectory` per host
for each answer. It never loads a model or starts Megatron or vLLM.

Run the same controller on one local Monarch worker from the project root:

```fish
.venv/bin/art-monarch local \
  --program examples/multinode/program.py:main \
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

The CPU smoke installs ART's `distributed` extra from the synchronized source
checkout. `skypilot_training.yaml` is the corresponding real two-host Megatron
qualification. It synchronizes only this example directory and installs a
published ART wheel, so no ART checkout or setup script exists on the cluster.
Set `ART_SHARED_ROOT` to storage mounted at the same path on every host before
launching it:

```fish
sky launch -c art-multinode-training \
  --env ART_SHARED_ROOT=/mnt/shared/art-multinode-release \
  examples/multinode/skypilot_training.yaml
```

The default training run uses one GPU per host and DP2. Set
`ART_TRAINER_RANKS_PER_HOST`, `ART_EXAMPLE_MODEL`, and the
`ART_EXAMPLE_{TP,CP,EP,PP}` variables to qualify larger topologies. Set
`ART_EXAMPLE_USE_NIXL=1` when an EP group crosses hosts; ART then provisions its
metadata store and builds the multi-node HybridEP runtime automatically.

Use `sky launch` after changing setup. Reuse unchanged CPU-smoke and training
clusters without rerunning setup with, respectively:

```fish
sky exec art-multinode examples/multinode/skypilot.yaml
sky exec art-multinode-training examples/multinode/skypilot_training.yaml
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
