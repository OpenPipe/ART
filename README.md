<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer</h1>
</p>

<p>
Train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/EceeVdhpxD)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## 🚀 W&B Training: Serverless RL

**W&B Training (Serverless RL)** is the first publicly available service for flexibly training models with reinforcement learning. It manages your training and inference infrastructure automatically, letting you focus on defining your data, environment and reward function—leading to faster feedback cycles, lower costs, and far less DevOps.

✨ **Key Benefits:**

- **40% lower cost** - Multiplexing on shared production-grade inference cluster
- **28% faster training** - Scale to 2000+ concurrent requests across many GPUs
- **Zero infra headaches** - Fully managed infrastructure that stays healthy
- **Instant deployment** - Every checkpoint instantly available via W&B Inference

```python
# Before: Hours of GPU setup and infra management
# RuntimeError: CUDA error: out of memory 😢

# After: Serverless RL with instant feedback
from art.serverless.backend import ServerlessBackend

model = art.TrainableModel(
  project="voice-agent",
  name="agent-001",
  base_model="OpenPipe/Qwen3-14B-Instruct"
)

backend = ServerlessBackend(
    api_key="your_wandb_api_key"
)
model.register(backend)
# Edit and iterate in minutes, not hours!
```

[📖 Learn more about W&B Training →](https://docs.wandb.ai/guides/training)

## ART Overview

ART is an open-source RL framework that improves agent reliability by allowing LLMs to **learn from experience**. ART provides an ergonomic harness for integrating GRPO into any python application. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## 📒 Notebooks

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [Serverless]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen3 14B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048 [Serverless]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen3 14B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

## 📰 ART News

Explore our latest research and updates on building SOTA agents.

- 🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
- 🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
- 🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
- 🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
- 🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
- 🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why ART?

- ART provides convenient wrappers for introducing RL training into **existing applications**. We abstract the training server into a modular service that your code doesn't need to interface with.
- **Train from anywhere.** Run the ART client on your laptop and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU.
- Integrations with hosted platforms like W&B, Langfuse, and OpenPipe provide flexible observability and **simplify debugging**.
- ART is customizable with **intelligent defaults**. You can configure training parameters and inference engine configurations to meet specific needs, or take advantage of the defaults, which have been optimized for training efficiency and stability.

## Installation

ART agents can be trained from any client machine that runs python. To add to an existing project, run this command:

```
pip install openpipe-art
```

## 🤖 ART•E Agent

Curious about how to use ART for a real-world task? Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail how we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

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

## 🧩 Supported Models

ART should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being. If any other model isn't working for you, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is in active development, and contributions are most welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## 📖 Citation

```bibtex
@misc{hilton2025art,
  author = {Brad Hilton and Kyle Corbitt and David Corbitt and Saumya Gandhi and Angky William and Bohdan Kovalenskyi and Andie Jones},
  title = {ART: Agent Reinforcement Trainer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openpipe/art}}
}
```

## ⚖️ License

This repository's source code is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART stands on the shoulders of giants. While we owe many of the ideas and early experiments that led to ART's development to the open source RL community at large, we're especially grateful to the authors of the following projects:

- [Unsloth](https://github.com/unslothai/unsloth)
- [vLLM](https://github.com/vllm-project/vllm)
- [trl](https://github.com/huggingface/trl)
- [torchtune](https://github.com/pytorch/torchtune)

Finally, thank you to our partners who've helped us test ART in the wild! We're excited to see what you all build with it.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
