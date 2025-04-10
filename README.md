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

ART is an open-source reinforcement training library for improving LLM performance in agentic workflows. Unlike existing RL libraries, ART allows you to execute agent runs **in your existing codebase** while offloading all the complexity of the RL training loop to the ART backend. Read the [architecture overview](#brief-architecture-overview) to learn more. Then try out one of the notebooks below!

## Notebooks

TODO: Add notebooks

## Brief Architecture Overview

ART's functionality is divided into two parts, a client and a server. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server can run separately on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1. Your code uses the ART client to execute agentic workflows (usually several in parallel).
   1. As the agent executes and the LLM generates completions, each `system`, `user`, and `assistant` message is stored in a Trajectory.
   2. Completion requests are routed to the ART server, which runs the most recently trained LoRA on vLLM.
   3. After a run finishes, your code assigns a `reward` to the Trajectory, which indicates the overall performance of the LLM during that run.

2. After a batch of runs completes, Trajectories are grouped together and sent to server. Inference is blocked while training executes.
   1. The server uses GRPO to train your model, starting from the most recently trained checkpoint (or an empty LoRA on the first iteration).
   2. The server saves the LoRA in a local directory to be read from on the next training run.
   3. Inference is unblocked and the loop resumes at step 1.
  
This training loop runs until a specified number of inference and training steps have been completed.

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
