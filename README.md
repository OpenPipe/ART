<div align="center">

<a href="https://openpipe.ai"><picture>
<img alt="ART header" src="https://github.com/openpipe/art/raw/main/assets/ART_header.png" width="100%">
</picture></a>

<a href="https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb"><img src="https://github.com/openpipe/art/raw/main/assets/Train_pill.png" height="48"></a>
<a href="https://discord.gg/zbBHRUpwf4"><img src="https://github.com/openpipe/art/raw/main/assets/Discord_pill.png" height="48"></a>
<a href="https://openpipe.ai/blog/art-e-mail-agent"><img src="https://github.com/openpipe/art/raw/main/assets/ART_E_pill.png" height="48"></a>

### Train GRPO-powered RL agents with minimal code changes and maximal performance!

![](https://github.com/openpipe/art/raw/main/assets/Header_separator.png)

</div>

# Agent Reinforcement Trainer (ART)

ART is an open-source reinforcement training library for improving LLM performance in agentic workflows. ART utilizes the powerful GRPO reinforcement learning algorithm to train models from their own experiences. Unlike most RL libraries, ART allows you to execute agent runs **in your existing codebase** while offloading all the complexity of the RL training loop to the ART backend. Read about the [ training loop](#training-loop-overview). Then try out one of the notebooks below!

## 📒 Notebooks

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                                                                                                           |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)     | Qwen 2.5 3B learns to play Codenames    | <img src="/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb) |

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
- [SkyPilot](https://github.com/skypilot-org/skypilot)

Finally, thank you to our partners who've helped us test ART in the wild! We're excited to see what you all build with it.
