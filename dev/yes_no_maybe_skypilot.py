# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openpipe-art[skypilot]==0.4.3",
#     "skypilot[runpod]",
# ]
# ///
import art
from art.skypilot import SkyPilotBackend
from dotenv import load_dotenv
import openai
import asyncio

load_dotenv()


async def main():
    backend = await SkyPilotBackend.initialize_cluster(
        cluster_name="kyle-yes-no-maybe",
        gpu="H100",
    )

    model = art.TrainableModel(
        name="001",
        project="yes-no-maybe",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        _internal_config=art.dev.InternalModelConfig(
            _decouple_vllm_and_unsloth=True,
            engine_args=art.dev.EngineArgs(gpu_memory_utilization=0.7),
        ),
    )
    await model.register(backend)

    async def rollout(client: openai.AsyncOpenAI, prompt: str) -> art.Trajectory:
        messages: art.Messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        chat_completion = await client.chat.completions.create(
            messages=messages, model=model.name, max_tokens=100, timeout=100
        )
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        if content == "yes":
            reward = 0.5
        elif content == "no":
            reward = 0.75
        elif content == "maybe":
            reward = 1.0
        else:
            reward = 0.0
        return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)

    def with_quotes(w):
        return f"'{w}'"

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in [
            ["yes", "no", "maybe"],
            ["maybe", "yes", "no"],
            ["no", "yes", "maybe"],
            ["yes", "maybe", "no"],
            ["yes", "no"],
            ["maybe", "no"],
            ["no", "maybe"],
            ["no", "yes"],
            ["yes", "no"],
        ]
    ]

    openai_client = model.openai_client()
    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(openai_client, prompt) for _ in range(32))
                for prompt in prompts
            ),
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
        )


if __name__ == "__main__":
    asyncio.run(main())
