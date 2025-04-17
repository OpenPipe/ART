<div align="center">

<a href="https://openpipe.ai"><picture>
<img alt="ART header" src="https://github.com/openpipe/art/raw/main/assets/ART_header.png" width="100%">
</picture></a>

<a href="https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb"><img src="https://github.com/openpipe/art/raw/main/assets/Train_pill.png" height="48"></a>
<a href="https://discord.com/invite/EceeVdhpxD"><img src="https://github.com/openpipe/art/raw/main/assets/Discord_pill.png" height="48"></a>
<a href="https://openpipe.ai/blog/art-trainer-a-new-rl-trainer-for-agents"><img src="https://github.com/openpipe/art/raw/main/assets/Launch_pill.png" height="48"></a>

### Train free-range RL agents with minimal code changes and maximal performance!

![](https://github.com/openpipe/art/raw/main/assets/Header_separator.png)

</div>

# Agent Reinforcement Trainer (ART)

ART is an open-source reinforcement training library for improving LLM performance in agentic workflows. Unlike most RL libraries, ART allows you to execute agent runs **in your existing codebase** while offloading all the complexity of the RL training loop to the ART backend. Read about the [ training loop](#training-loop-overview). Then try out one of the notebooks below!

## 📒 Notebooks

| Agent Task | Example Notebook                                                                                                | Description                     | Comparative Performance |
| ---------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------- | ----------------------- |
| **2048**   | [🏋️ Train your agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb) | Qwen 2.5 3B learns to play 2048 | [Link coming soon]      |

## 🔁 Training Loop Overview

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1. **Inference**

   1. Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
   2. Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
   3. As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
   4. When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.

2. **Training**
   1. When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
   2. The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
   3. The server saves the newly trained LoRA to a local directory and loads it into vLLM.
   4. Inference is unblocked and the loop resumes at step 1.

This training loop runs until a specified number of inference and training iterations have completed.

## Supported Models

ART should work with any vLLM/HuggingFace-transformers compatible causal language model, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If a model isn't working for you, please let us know on [Discord](https://discord.com/invite/EceeVdhpxD) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## ⚠️ Disclaimer

ART is currently in alpha and has only been tested on a few projects in the wild! We're working hard to make it work for everyone, but if you run into any issues, please let us know on [Discord](https://discord.com/invite/EceeVdhpxD) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is in very active development, and contributions are most welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## 🙏 Credits

ART stands on the shoulders of giants. While we owe many of the ideas and early experiments that led to ART's development to the open source RL community at large, we're especially grateful to the authors of the following projects:

- [Unsloth](https://github.com/unslothai/unsloth)
- [vLLM](https://github.com/vllm-project/vllm)
- [trl](https://github.com/huggingface/trl)
- [SkyPilot](https://github.com/skypilot-org/skypilot)

Finally, thank you to our partners who've helped us test ART in the wild! We're excited to see what you all build with it.
