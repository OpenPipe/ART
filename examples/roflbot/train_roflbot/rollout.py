import art
from typing import List, Any
from art import Trajectory
from litellm import acompletion
import litellm
from litellm.caching.caching import LiteLLMCacheType, Cache
from litellm.types.utils import Choices, ModelResponse
from dataclasses import asdict
from art.utils.litellm import convert_litellm_choice_to_openai
from dataclasses import dataclass
from art.utils import limit_concurrency
from datetime import datetime
import textwrap
from tenacity import retry, stop_after_attempt
from project_types import PolicyConfig
from dataset import RedditJoke
import re
from score_joke import score_joke
import math

litellm.cache = Cache(type=LiteLLMCacheType.DISK)


@dataclass
class Rubric:
    rm_score: float | None = None
    thinking_length: int | None = None
    joke_length: int | None = None
    matches_prefix: bool | None = None
    no_content: bool | None = None
    finish_reason_length: bool | None = None
    successfully_parsed_thinking: bool | None = None
    successfully_parsed_joke: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_metrics(self) -> dict[str, float | int]:
        metrics = {}
        for k, v in asdict(self).items():
            if v is not None:
                metrics[k] = v if isinstance(v, float) else int(v)

        return metrics


def calculate_reward(
    policy_config: PolicyConfig, rubric: Rubric, traj: Trajectory
) -> float:
    # Early exits for failure conditions
    if rubric.no_content:
        return 0.0
    if not rubric.successfully_parsed_joke:
        return 0.0

    # Handle thinking length penalty
    if (rubric.thinking_length or 0) > policy_config.thinking_char_budget:
        # If budget is 0 or negative, any thinking results in 0 reward
        if policy_config.thinking_char_budget <= 0:
            return 0.0
        else:
            # Calculate ratio, ensure it's > 1
            # (since rubric.thinking_length > policy_config.thinking_char_budget)
            ratio = (rubric.thinking_length or 0) / policy_config.thinking_char_budget
            # Apply log penalty: 1 / (1 + log(ratio))
            # This approaches 1 as ratio -> 1+, and decreases towards 0 as ratio increases.
            reward = 1.0 / (1.0 + math.log(ratio))
            return reward  # Return the penalized score directly

    # If thinking length is within budget:
    # Rule 3: Joke generated, thinking OK, but doesn't match prefix
    if not rubric.matches_prefix:
        return 1.0
    # Rule 4: Joke generated, thinking OK, matches prefix
    else:
        # Add rm_score if available, otherwise base reward is 1.0
        base_reward = 1.0
        bonus = rubric.rm_score if rubric.rm_score is not None else 0.0
        return base_reward + bonus


@retry(stop=stop_after_attempt(3))
async def determine_if_completion_matches_prefix(joke: str, prefix: str) -> bool:
    system_prompt = textwrap.dedent("""\
        You will be given a joke prefix and then the rest of the joke. Your job is to determine whether the joke body is a natural continuation of the joke prefix. It's ok if it takes things in an unexpected direction, but there should be a clear connection between the prefix and the joke body.
                                    
        Simply return True if the joke body is a natural continuation of the joke prefix, and False otherwise. Do not return anything except True or False. /no_think
        """)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Joke prefix: {prefix}\nJoke body: {joke}",
        },
    ]

    response = await acompletion(
        model="openrouter/qwen/qwen3-235b-a22b",
        messages=messages,
        temperature=0,
        caching=True,
    )
    print(response.choices[0])

    return response.choices[0].message.content.strip().lower().startswith("t")  # type: ignore


async def finish_trajectory(
    traj: Trajectory, rubric: Rubric, rollout_start_time: datetime
) -> Trajectory:
    traj.finish()
    traj.metrics = {**traj.metrics, **rubric.to_metrics()}
    return traj


@retry(stop=stop_after_attempt(3))
@limit_concurrency(10, derive_key=lambda model, scenario, **kwargs: model.name)
async def rollout(
    model: art.Model,
    scenario: RedditJoke,
) -> Trajectory:
    rollout_start_time = datetime.now()
    rubric = Rubric()
    trajectory = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"joke_id": scenario.id},
    )
    assert isinstance(model.config, PolicyConfig)
    rollout_start_time = datetime.now()

    system_prompt = textwrap.dedent(f"""
        You are a joke generator, and your goal is to generate hilarious jokes. As a starting point, you will be given a joke prefix. Your job is to generate a continuation of the joke, maximizing humor.
                                    
        You should take some time to think about the best way to complete the joke. You can think for up to {model.config.thinking_char_budget} characters. If you take longer than that to think, your response will be penalized. Do not repeat the joke prefix in your response, only return the continuation.
                                    
        You should respond in the following format:
        <think>some thoughts</think>
        <joke_continuation>the joke continuation</joke_continuation>
                                    
        Here are some examples of jokes:
        <joke_prefix>What is a pirate's favorite letter?</joke_prefix>
        <think>I should respond with a letter that pirates are known for.</think>
        <joke_continuation>ARRR!</joke_continuation>

        <joke_prefix>What do you call a belt made of watches?</joke_prefix>
        <think>What is a phrase that would make sense here? Maybe something about time? Should be related to belts too, hmm. Oh, I know! A waist of time.</think>
        <joke_continuation>A waist of time.</joke_continuation>
    """)

    trajectory.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<joke_prefix>{scenario.title}</joke_prefix>"},
    ]

    model_name = model.get_inference_name()
    if isinstance(model, art.TrainableModel):
        model_name = f"hosted_vllm/{model_name}"

    async with trajectory.track_duration("policy_completion"):
        llm_response = await acompletion(
            model=model_name,
            base_url=model.inference_base_url,
            messages=trajectory.messages(),
            caching=not isinstance(model, art.TrainableModel),
            api_key=model.inference_api_key,
            max_completion_tokens=model.config.max_tokens,
        )
    assert isinstance(llm_response, ModelResponse)
    rubric.prompt_tokens = llm_response.usage.prompt_tokens  # type: ignore
    rubric.completion_tokens = llm_response.usage.completion_tokens  # type: ignore
    choice = llm_response.choices[0]  # type: ignore
    assert isinstance(choice, Choices)

    trajectory.messages_and_choices.append(convert_litellm_choice_to_openai(choice))

    content = choice.message.content

    if content is None:
        rubric.no_content = True
        return await finish_trajectory(trajectory, rubric, rollout_start_time)
    else:
        rubric.no_content = False

    thinking = re.search(r"<think>(.*?)</think>", content)
    joke_continuation = re.search(
        r"<joke_continuation>(.*?)</joke_continuation>", content
    )

    if thinking is None:
        rubric.successfully_parsed_thinking = False
    else:
        rubric.successfully_parsed_thinking = True
        rubric.thinking_length = len(thinking.group(1).strip())

    if joke_continuation is None:
        rubric.successfully_parsed_joke = False
        return await finish_trajectory(trajectory, rubric, rollout_start_time)
    else:
        rubric.successfully_parsed_joke = True
        rubric.joke_length = len(joke_continuation.group(1).strip())

    joke_content = joke_continuation.group(1).strip()

    async with trajectory.track_duration("determine_if_completion_matches_prefix"):
        rubric.matches_prefix = await determine_if_completion_matches_prefix(
            joke_content, scenario.title
        )

    async with trajectory.track_duration("score_joke"):
        rm_score = await score_joke(scenario.title, joke_content)

    rubric.rm_score = rm_score

    trajectory.reward = calculate_reward(model.config, rubric, trajectory)

    return await finish_trajectory(trajectory, rubric, rollout_start_time)


if __name__ == "__main__":
    from dataset import get_jokes
    from dotenv import load_dotenv
    import asyncio
    import yaml

    load_dotenv()

    test_joke = get_jokes(split="test", limit=1)[0]
    traj = asyncio.run(
        rollout(
            art.Model(
                name="gpt-4o",
                project="email_agent",
                inference_model_name="gemini/gemini-2.5-pro-preview-03-25",
                config=PolicyConfig(),
            ),
            test_joke,
        )
    )
    print(yaml.dump(traj.for_logging()))
