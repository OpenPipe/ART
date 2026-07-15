# ART multi-node bootstrap

This example consumes one SkyPilot allocation directly. From the project root:

```fish
sky launch -c art-multinode examples/multinode/skypilot.yaml
```

SkyPilot synchronizes `workdir` and runs `setup` on every node before starting
the same `run` command on every node. ART starts one Monarch worker per node and
calls the installed deployment smoke only on rank 0. The smoke admits every host
and runs one CPU rollout on each; replace its import path with an installed
top-level async function that owns your ART run.

Edit the accelerator and setup commands for your infrastructure. The example
assumes `uv` is installed in the image and builds both ART and the pinned vLLM
runtime because it runs from a source checkout. Wheel installs provision the
bundled managed runtime on first use. Use `sky launch` after changing setup;
reuse an unchanged cluster without rerunning setup with:

```fish
sky exec art-multinode examples/multinode/skypilot.yaml
```

Each invocation terminates every worker loop before the task exits. Reusing the
cluster starts fresh loops; completed Monarch 0.5 loops are not reattached.

Setting `num_nodes: 1` exercises the same API on one node. Do not expose the
default private ports `22222` and `22223`; pinned Monarch 0.5 does not
authenticate its transport.

Without SkyPilot, the equivalent owned one-host smoke is:

```fish
.venv/bin/art-monarch local \
  --program art.distributed.monarch_bootstrap:deployment_smoke
```
