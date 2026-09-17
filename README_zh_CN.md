<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>智能体强化训练器（Agent Reinforcement Trainer）</h1>
</p>

<p>
使用 GRPO 训练多步智能体以完成真实世界任务。
</p>

[![欢迎 PR][contribute-image]][contribute-url]
[![下载量][downloads-image]][pypi-url]
[![训练智能体](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![加入 Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![文档](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## 📏 RULER：零样本智能体奖励

**RULER**（相对通用 LLM 引导奖励）通过使用 LLM 作为评判者自动为智能体轨迹打分，消除了手工设计奖励函数的需求。只需在系统提示中定义你的任务，其余交给 RULER——**无需标注数据、专家反馈或奖励工程**。

✨ **核心优势：**

- **开发速度提升 2-3 倍** -完全跳过奖励函数工程
- **通用性强** -适用于任何任务，无需修改
- **性能优异** - 在 3/4 基准测试中与手工设计奖励持平或更优
- **易于集成** -可直接替换现有手工设计奖励函数

```python
# 以前：需要数小时的奖励工程
def complex_reward_function(trajectory):
    # 50 多行精心编写的评分逻辑...
    pass

# 现在：用 RULER只需一行代码
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 了解关于RULER的更多→](https://art.openpipe.ai/fundamentals/ruler)

## ART 概览

ART 是一个开源的强化学习框架，通过让 LLM **从经验中学习**，提升智能体的可靠性。ART 提供了便捷的工具，可将 GRPO 集成到任何 Python 应用中。想快速上手？可以运行下方的示例笔记本。想深入了解，请查阅[官方文档](https://art.openpipe.ai)。

## 📒 示例笔记本

| 智能体任务                   | 示例笔记本                                                                                                                     | 描述                                   | 对比性能                                                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**       | [🏋️ 开始训练智能体](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B 使用 RULER 学习邮件搜索          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [基准](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**                | [🏋️ 开始训练智能体](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B 学习玩 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [基准](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue（时空谜题）** | [🏋️ 开始训练智能体](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B 学习解决 Temporal Clue（时空谜题） | [链接即将上线]                                                                                                                                                                                    |
| **井字棋**                 | [🏋️ 开始训练智能体](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B 学习玩井字棋                   | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [基准](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames（行动代号）**     | [🏋️ 开始训练智能体](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B 学习玩 Codenames（行动代号）      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [基准](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]**      | [🏋️ 开始训练智能体](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | 训练 Qwen 2.5 7B 掌握任意任务                | [链接即将上线]                                                                                                                                                                                    |

## 📰 ART 新闻

探索我们关于构建 SOTA 智能体的最新研究和更新。

- 🗞️ **[AutoRL：零数据训练任何任务](https://x.com/mattshumer_/status/1950572449025650733)** —— 利用自动输入生成和 RULER 评估，无需标注数据即可训练自定义 AI 模型。
- 🗞️ **[RULER：强化学习奖励的简单模式](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** —— 现已推出，用于强化学习中的自动奖励生成。
- 🗞️ **[ART·E：我们是如何构建击败 o3 的邮件研究智能体的](https://openpipe.ai/blog/art-e-mail-agent)** —— 展示 Qwen 2.5 14B 邮件智能体超越 OpenAI o3 的过程。
- 🗞️ **[ART Trainer：全新强化学习智能体训练器](https://openpipe.ai/blog/art-trainer)** —— 轻松用 GRPO 训练基于 LLM 的智能体。

[📖 查看所有博客文章 →](https://openpipe.ai/blog)

## 为什么选择 ART？

- ART 为将强化学习训练引入**现有应用**提供了便捷的封装。我们将训练服务器抽象为一个模块化服务，您的代码无需与其直接交互。
- **随时随地训练。** 在笔记本电脑上运行 ART 客户端，让 ART 服务器启动临时 GPU 环境，或者直接在本地 GPU 上运行。
- 与 W&B、Langfuse 和 OpenPipe 等托管平台的集成提供了灵活的可观测性，并**简化了调试流程**。
- ART 提供了**智能默认设置**，您可以根据具体需求配置训练参数和推理引擎，或者直接使用经过优化的默认设置，这些默认设置旨在提高训练效率和稳定性。

## 安装

ART 智能体可在任何运行 Python 的客户端机器上训练。要集成到现有项目，请运行：

```
pip install openpipe-art
```

## 🤖 ART•E 智能体

想了解如何用 ART 解决现实世界任务吗？来看看这篇关于 [ART·E 智能体](https://openpipe.ai/blog/art-e-mail-agent) 的博客文章吧，我们详细介绍了如何训练 Qwen 2.5 14B 在邮件检索任务上击败 o3！

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 训练循环概览

ART 的功能分为**客户端**和**服务器端**。兼容 OpenAI 的客户端负责在 ART 和你的代码库之间进行交互。使用客户端，你可以传递消息并从正在改进的 LLM 中获取补全结果。服务器端独立运行在任何带有 GPU 的机器上。它抽象了强化学习循环中推理和训练部分的复杂性，同时允许一些自定义配置。以下是训练循环的概述：

1. **推理**

   
   1. 你的代码使用 ART 客户端执行一个智能体工作流（通常并行执行多个 rollout 以更快地收集数据）。
   2. 补全请求被路由到 ART 服务器，服务器在 vLLM 中运行模型的LoRA。
   3. 智能体执行过程中，每个 `system`、`user` 和 `assistant` 消息都被存储在一个轨迹中。
   4. 当一个 rollout 结束后，你的代码为其轨迹分配一个 `reward`，表示 LLM 的性能。


2. **训练**


   1. 当每个 rollout 结束后，轨迹被分组并发送到服务器。训练执行期间会阻塞推理。
   2. 服务器使用 GRPO 训练你的模型，从最新的检查点初始化（或在第一次迭代时从空的 LoRA 开始）。
   3. 服务器将新训练的 LoRA 保存到本地目录并加载到 vLLM 中。
   4. 解除推理阻塞，循环返回步骤 1 继续执行。

这个训练循环会一直运行，直到完成指定数量的推理和训练迭代。

## 🧩 支持的模型

ART 应适用于大多数 vLLM/HuggingFace-transformers 兼容的因果语言模型，或至少是 [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) 支持的模型。目前 Gemma 3 似乎暂不支持。如果你遇到其他模型无法使用，请在 [Discord](https://discord.gg/zbBHRUpwf4) 上告诉我们，或者在 [GitHub](https://github.com/openpipe/art/issues) 上写一份 issue 。

## 🤝 贡献

ART 正在积极开发中，非常欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 获取更多信息。

## 📖 引用

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

## ⚖️ 许可证

本仓库源代码采用 [Apache-2.0 License](LICENSE) 许可。

## 🙏 鸣谢

ART 站在巨人的肩膀上。ART 的许多理念和早期实验都得益于整个开源强化学习社区，在此我们特别感谢以下项目的作者：

- [Unsloth](https://github.com/unslothai/unsloth)
- [vLLM](https://github.com/vllm-project/vllm)
- [trl](https://github.com/huggingface/trl)
- [torchtune](https://github.com/pytorch/torchtune)
- [SkyPilot](https://github.com/skypilot-org/skypilot)

最后，感谢所有帮助我们在实际环境中测试 ART 的合作伙伴！我们很期待看到大家用 ART 构建的精彩项目。

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
