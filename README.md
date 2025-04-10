<div align="center">

  <a href="https://openpipe.ai"><picture>
    <img alt="ART header" src="https://github.com/user-attachments/assets/d5441604-59fe-415d-a90a-9e9e2cbd5c2c" width="100%">
  </picture></a>

<a href="https://colab.research.google.com/github/OpenPipe/ART/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb"><img src="https://github.com/user-attachments/assets/8d655cbd-6498-4ef0-a4bb-c6c353c63c0e" height="48"></a>
<a href="https://discord.gg/F6MxpujP"><img src="https://github.com/user-attachments/assets/9d257702-a4a5-4824-901a-5155aa032a27" height="48"></a>
<a href="https://docs.openpipe.ai"><img src="https://github.com/user-attachments/assets/33acfb02-6920-4636-b66f-38dacdbe59ca" height="48"></a>

### Train free-range RL agents with minimal code changes and maximal performance!

![](https://github.com/user-attachments/assets/13eae0af-a7bf-4fd3-9e13-20847438cd66)

</div>

# The OpenPipe Agent Reinforcement Trainer (ART)

ART is an open-source reinforcement training library for improving LLM performance in agentic workflows. Unlike existing RL libraries, ART allows you to execute agent runs **in your existing codebase** while offloading all the complexity of the RL training loop to the ART backend. Read about the [ training loop](#training-loop-overview). Then try out one of the notebooks below!

## Notebooks

TODO: Add notebooks

## Training Loop Overview

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1. **Inference**
   1. Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
   2. Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
   3. As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
   4. When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.

3. **Training**
   1. When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
   2. The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
   3. The server saves the newly trained LoRA to a local directory and loads it into vLLM.
   4. Inference is unblocked and the loop resumes at step 1.
  
This training loop runs until a specified number of inference and training iterations have completed.

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
