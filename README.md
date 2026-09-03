<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer</h1>
</p>

<p>
Train language models to become reliable agents through experience.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/EceeVdhpxD)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## ART Overview

**ART** is an open-source framework for teaching agents to improve through
reinforcement learning and supervised fine-tuning. You write the agent, its
environment, and a reward signal. ART gathers experience, trains a LoRA, makes
the new checkpoint available for inference, and records the lineage needed to
understand the run.

ART is designed for the parts of language-model training that are awkward in a
normal application:

- **Multi-turn and tool-using agents.** Train on complete trajectories rather
  than reducing an agent to a single prompt and response.
- **Online reinforcement learning.** Generate experience, score it, train, and
  continue from the new policy in one loop.
- **SFT and RL in the same project.** Warm-start behavior with demonstrations,
  then improve it from task rewards.
- **LoRA-first checkpoints.** Move small adapters between training and vLLM
  without repeatedly copying the full model.
- **Flexible infrastructure.** Use managed W&B Training, your own GPUs, or a
  Tinker-compatible backend without rewriting the agent.
- **Production-scale training.** The Megatron path supports packed sequences,
  expert and context parallelism, asynchronous rollout/training pipelines, and
  multi-node execution.

## 🚀 Quick Start

Install the ART client in any Python 3.12+ project:

```bash
pip install openpipe-art
```

Initialize the bundled ART skills if you use Claude Code or OpenAI Codex:

```bash
art init
```

The fastest hands-on introduction is the free
[2048 notebook](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb).
For an existing application, register a trainable model with a backend:

```python
import asyncio

import art
from art.serverless.backend import ServerlessBackend


async def main() -> None:
    backend = ServerlessBackend()  # reads WANDB_API_KEY
    model = art.TrainableModel(
        project="my-agent",
        name="agent-001",
        run_name="agent-001",
        base_model="Qwen/Qwen3.6-27B",
    )
    await model.register(backend)
    # Generate trajectory groups, assign rewards, then call backend.train(...).


asyncio.run(main())
```

See the [Quick Start](https://art.openpipe.ai/getting-started/quick-start) for a
complete rollout and training loop.

## Choose Where Training Runs

The agent-facing API stays the same across ART's backends:

| Backend | Best for | Infrastructure |
| --- | --- | --- |
| `ServerlessBackend` | Getting started quickly and scaling online RL | W&B Training manages inference, training, checkpoints, and deployment |
| `LocalBackend` | Development or training on GPUs you control | ART manages colocated or dedicated vLLM and training processes |
| Tinker backends | Training through an existing Tinker service | Tinker owns the remote training runtime |

### W&B Training

[W&B Training](https://docs.wandb.ai/guides/training) runs ART on managed GPU
infrastructure. It scales rollout inference and training independently, records
metrics and traces in W&B, stores LoRA checkpoints as artifacts, and makes new
checkpoints available through W&B Inference. Your agent code can continue to run
on a laptop, CI worker, or application server.

### Your Own GPUs

`LocalBackend` can run vLLM and training on one machine or use dedicated GPU
pools with `PipelineTrainer`. ART's packaged Megatron runtime extends that path
to large dense, MoE, and hybrid models across multiple hosts. The release wheel
contains the exact runtime contracts; users do not need an ART checkout or a
manual Megatron setup script.

## Installation Profiles

The base package is sufficient for clients and W&B Training:

```bash
pip install openpipe-art
```

For a local CUDA 12 backend, including H100 and H200 hosts:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "openpipe-art[backend]"
```

CUDA 13 hosts such as B300 use the matching profile and PyTorch index:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "openpipe-art[backend-cu130]"
```

For ART's locked Megatron runtime, replace `backend` with `megatron`:

```bash
# H100 / H200
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "openpipe-art[megatron]"

# B300
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "openpipe-art[megatron-cu130]"
```

The host image remains responsible for the NVIDIA driver and CUDA toolkit. A
multi-node image must also expose its NCCL network transport and RDMA devices.

The first Megatron launch creates content-addressed vLLM and trainer environments
from the locks bundled in the ART wheel. This can download several gigabytes and
compile GPU-specific extensions. Later launches reuse the immutable environments
while their cache persists. Prepare them before a job, or inspect an existing
installation, with:

```bash
art runtime prepare
art runtime status
```

Use `art runtime prepare --hybrid-ep` for an MoE topology that needs HybridEP,
or add `--multinode` for its cross-host variant. Runtime environments and
compiler caches default to `/tmp/art-cache`. Keep them on fast node-local
storage; use `ART_MEGATRON_CACHE_ROOT`, `ART_MEGATRON_RUNTIME_CACHE_DIR`, or
`ART_VLLM_RUNTIME_CACHE_DIR` when the default is not suitable. Store durable
checkpoints on shared storage separately.

The full installation and deployment contracts are in
[Installation + Setup](https://art.openpipe.ai/getting-started/installation-setup)
and the [multi-node guide](https://art.openpipe.ai/getting-started/multi-node).

## 🔁 How the Training Loop Works

1. **Generate experience.** Your application runs several copies of an agent in
   parallel. ART's OpenAI-compatible model client records messages, tool calls,
   responses, token provenance, and the policy checkpoint used by each rollout.
2. **Score outcomes.** Your environment assigns rewards directly, or a judge such
   as [RULER](https://art.openpipe.ai/fundamentals/ruler) compares the results.
3. **Build a training batch.** Related trajectories are grouped so ART can
   calculate relative advantages, reject unusable groups, and pack useful tokens
   efficiently.
4. **Train.** The selected backend applies an RL or SFT update and commits the
   optimizer and checkpoint state.
5. **Publish the policy.** ART loads the new LoRA into inference. New requests use
   the new policy while requests already in flight retain their original policy.
6. **Repeat and evaluate.** Training continues from the durable checkpoint while
   reward, throughput, freshness, and cost metrics describe the run.

## 📒 Examples

| Agent or task | Start here | What it demonstrates |
| --- | --- | --- |
| **ART·E email research** | [Train in Colab](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb) | Multi-step search, tool use, and RULER evaluation |
| **2048** | [Train in Colab](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb) | A complete serverless RL loop with an interactive environment |
| **MCP·RL** | [Open notebook](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb) | Teaching a model to use an MCP server |
| **Tic Tac Toe** | [View example](examples/tic_tac_toe/tic-tac-toe.py) | Local rollouts, grouped rewards, and checkpoint iteration |
| **OpenEnv** | [View example](examples/openenv_echo.py) | Connecting an external agent environment to ART |
| **SFT from a dataset** | [Train in Colab](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/sft/train_from_file.ipynb) | Supervised fine-tuning from a local data file |

More tutorials cover LangGraph, MCP servers, deep research, checkpoint forking,
custom rewards, and SFT-to-RL workflows in the
[ART documentation](https://art.openpipe.ai).

## 🧩 Models

Model availability depends on the backend. W&B Training exposes a curated set
of qualified models. The standard local backend works with many causal language
models supported by Hugging Face, vLLM, and ART's LoRA path.

The packaged Megatron backend uses an explicit registry with per-model readiness
rather than assuming every architecture is interchangeable. It currently
includes models from these families:

- Llama 3, 3.1, 3.2, and 3.3
- Qwen 3 dense and MoE
- Qwen 3.5 and 3.6 dense and MoE, plus Qwen 3.8 27B
- Gemma 4 dense and MoE
- DeepSeek V4
- GLM 5.2 and GLM 5.3 BF16
- GPT-OSS
- NVIDIA Nemotron 3 Nano and Nemotron 3.5 Lightning

Exact model IDs and feature readiness live in ART's model-support registry. If a
model is not listed, ask in [Discord](https://discord.gg/zbBHRUpwf4) or open a
[GitHub issue](https://github.com/OpenPipe/ART/issues) rather than assuming a
nearby architecture is automatically safe.

## Why ART?

- **Bring the application you already have.** ART wraps inference and training
  behind a model and backend instead of requiring the agent to understand a
  trainer service.
- **Train from anywhere.** Keep the environment on your laptop or application
  server while the GPUs run locally, in your cluster, or through W&B Training.
- **Use rewards that match the task.** Unit tests, simulators, human feedback,
  business metrics, and LLM judges can all become learning signals.
- **Debug the whole loop.** Trajectories, traces, rewards, optimizer checkpoints,
  and policy versions remain connected.
- **Start simple and keep control.** Defaults provide a working path, while
  batching, loss, inference, topology, and deployment remain configurable.

## 🤝 Contributing

ART is in active development, and contributions are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) to set up the repository, run checks, and send
a change.

## 📖 Citation

```bibtex
@misc{hilton2025art,
  author = {Brad Hilton and Kyle Corbitt and David Corbitt and Saumya Gandhi and Angky William and Bohdan Kovalevskyi and Andie Jones},
  title = {ART: Agent Reinforcement Trainer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openpipe/art}}
}
```

## ⚖️ License

ART is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART stands on the shoulders of the open-source training and inference community.
We are especially grateful to the authors and maintainers of
[vLLM](https://github.com/vllm-project/vllm),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[Hugging Face Transformers](https://github.com/huggingface/transformers),
[Unsloth](https://github.com/unslothai/unsloth),
[TRL](https://github.com/huggingface/trl), and
[torchtune](https://github.com/pytorch/torchtune).

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
