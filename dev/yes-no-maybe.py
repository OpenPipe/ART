import asyncio
from itertools import permutations

import openai
from dotenv import load_dotenv

import art
from art.local import LocalBackend


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


def with_quotes(w: str) -> str:
    return f"'{w}'"


async def main():
    load_dotenv()

    backend = LocalBackend()
    global model
    model = art.TrainableModel(
        name="011",
        project="yes-no-maybe",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        _internal_config=art.dev.InternalModelConfig(
            _decouple_vllm_and_unsloth=True,
            # engine_args=art.dev.EngineArgs(gpu_memory_utilization=0.7),
        ),
    )
    await model.register(backend)

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in (
            list(p) for n in [3, 2] for p in permutations(["yes", "no", "maybe"], n)
        )
    ]

    openai_client = model.openai_client()
    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(openai_client, prompt) for _ in range(32))
                for prompt in prompts
            )
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
            # _config=art.dev.TrainConfig(
            #     precalculate_logprobs=True,
            # ),
        )


if __name__ == "__main__":
    asyncio.run(main())

