<div align="center">


  <a href="https://openpipe.ai"><picture>
    <img alt="ART header" src="https://github.com/user-attachments/assets/d5441604-59fe-415d-a90a-9e9e2cbd5c2c" width="100%">
  </picture></a>

<a href="https://colab.research.google.com/github/OpenPipe/ART/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb"><img src="https://github.com/user-attachments/assets/8d655cbd-6498-4ef0-a4bb-c6c353c63c0e" height="48"></a>
<a href="https://discord.gg/F6MxpujP"><img src="https://github.com/user-attachments/assets/9d257702-a4a5-4824-901a-5155aa032a27" height="48"></a>
<a href="https://docs.openpipe.ai"><img src="https://github.com/user-attachments/assets/33acfb02-6920-4636-b66f-38dacdbe59ca" height="48"></a>

### Train free-range RL agents with minimal code changes and maximal performance!

![](https://github.com/user-attachments/assets/8296d076-508e-4689-ab13-f55599baced3)

</div>

# The OpenPipe Agent Reinforcement Trainer (ART)

An open-source reinforcement training library for LLMs and agentic workflows

## Getting Started

Clone the repository:

```bash
git clone https://github.com/OpenPipe/agent-reinforcement-training.git
cd agent-reinforcement-training
```

Install the dependencies:

```bash
uv sync
```

Then follow the SkyPilot or Local Training instructions below.

> **Warning:** There is currently a bug with tool use functionality. The issue appears to be that vLLM does not return all the token log probabilities for tool use. Further investigation is needed to determine the exact cause. For now, teaching use case-specific tool use with non-tool use models is the recommended workaround.

### SkyPilot

Copy the `.env.example` file to `.env` and set the environment variables:

```bash
cp .env.example .env
```

Ensure you have a valid SkyPilot cloud available:

```bash
uv run sky check
```

Launch a cluster:

```bash
./launch-cluster.sh # you can pass any sky launch arguments here
```

SSH into the `art` cluster with VSCode or from the command line:

```bash
ssh art
```

When you're done, you can tear down the cluster with:

```bash
uv run sky down art
```

### Local Training

Make sure you are on a machine with at least one H100 or A100-80GB GPU.

Reinstall torchtune due to a CLI naming conflict:

```bash
uv remove torchtune
uv add torchtune
```

### "Temporal Clue" example

Now you can run the "Temporal Clue" example in `/examples/temporal-clue.ipynb`.

It has been tested with the `NousResearch/Hermes-2-Theta-Llama-3-8B` model on a 1xH100 instance.

You can monitor training progress with Weights & Biases at https://wandb.ai/your-wandb-organization/agent-reinforcement-training.

You should see immediate improvement in `val/reward` after one step.

If you run into any issues, the training output is set to maximum verbosity. Copying the outputs such as the vLLM or torchtune logs, or copying/screenshotting the plotted packed tensors, may help me debug the issue.
