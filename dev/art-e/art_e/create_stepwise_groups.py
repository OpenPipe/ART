import litellm
from art_e.project_types import ProjectPolicyConfig
from litellm.types.utils import ModelResponse
from tqdm.asyncio import tqdm

import art
from art.rewards import ruler_score_group
from art.trajectories import Trajectory, TrajectoryGroup
from art.types import Messages, Tools
from art.utils.limit_concurrency import limit_concurrency
from art.utils.litellm import convert_litellm_choice_to_openai


@limit_concurrency(20)
async def rate_limited_ruler_score_group(
    group: TrajectoryGroup, model: art.Model[ProjectPolicyConfig]
) -> TrajectoryGroup | None:
    """
    Rate-limit the ruler score group.
    """
    assert model.config.ruler_judge_model is not None

    return await ruler_score_group(
        group, model.config.ruler_judge_model, swallow_exceptions=True
    )


async def create_group_from_prefix(
    prefix: Messages,
    tools: Tools | None,
    model: art.Model[ProjectPolicyConfig],
) -> TrajectoryGroup | None:
    """
    Create a group from a prefix.
    """

    completions = await litellm.acompletion(
        **model.litellm_completion_params(),
        messages=prefix,
        max_completion_tokens=model.config.max_tokens,
        tools=tools,
        n=4,
    )

    assert isinstance(completions, ModelResponse)

    # Convert choices to OpenAI format immediately to avoid pickling issues
    openai_choices = [
        convert_litellm_choice_to_openai(choice) for choice in completions.choices
    ]

    group = TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=prefix + [openai_choice],
                tools=tools,
                reward=0,
            )
            for openai_choice in openai_choices
        ]
    )

    scored_group = await rate_limited_ruler_score_group(group, model)
    return scored_group


async def create_stepwise_groups(
    model: art.Model[ProjectPolicyConfig],
    trajectory: Trajectory,
) -> list[TrajectoryGroup]:
    """
    Create stepwise groups from a trajectory.
    """
    prefixes = []
    messages = trajectory.messages()

    # Ensure messages are clean dictionaries without any litellm objects
    clean_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Create a clean copy of the message
            clean_msg = {"role": msg["role"]}

            # Handle different message types
            if msg["role"] == "tool":
                # Tool messages need tool_call_id
                clean_msg["tool_call_id"] = msg.get("tool_call_id", "unknown")
                clean_msg["content"] = msg.get("content", "")  # type: ignore
            else:
                # Regular messages
                if "content" in msg:
                    clean_msg["content"] = msg["content"]  # type: ignore
                if "tool_calls" in msg and msg["tool_calls"]:
                    clean_msg["tool_calls"] = msg["tool_calls"]  # type: ignore

            clean_messages.append(clean_msg)
        else:
            # This shouldn't happen but let's be safe
            clean_messages.append({"role": "user", "content": str(msg)})

    for i in range(len(clean_messages)):
        if clean_messages[i]["role"] == "assistant":
            prefix = clean_messages[:i]
            prefixes.append(prefix)

    groups = await tqdm.gather(
        *(
            create_group_from_prefix(prefix, trajectory.tools, model)
            for prefix in prefixes
        )
    )

    return [group for group in groups if group is not None]


async def create_all_stepwise_groups(
    model: art.Model[ProjectPolicyConfig],
    groups: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    trajectories = [t for group in groups for t in group.trajectories]
    return await tqdm.gather(
        *(create_stepwise_groups(model, trajectory) for trajectory in trajectories)
    )
