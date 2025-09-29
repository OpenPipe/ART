import copy
from typing import Any, Dict, List, Tuple, Type, TypeVar

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, Field

import art
from tau_bench.rl_utils import update_openpipe_log
from tau_bench.types import RunConfig, SolveResult

T = TypeVar("T")


class RolloutScore(BaseModel):
    """
    Model representing the score and explanation for a single rollout.
    """

    rollout_index: int = Field(description="Index of the rollout being scored")
    explanation: str = Field(
        description="Detailed explanation of what the rollout did well and what it did poorly, as determined by grading standards"
    )
    score: float = Field(
        description="Numerical score between 0 and 1 indicating rollout quality, as determined by grading standards and explanation"
    )


class RolloutScoreLLM(BaseModel):
    """
    Model representing the score and explanation for a single rollout, for the LLM reward type.
    """

    explanation: str = Field(
        description="Detailed explanation of what the rollout did well and what it did poorly, as determined by grading standards"
    )
    score: float = Field(
        description="Numerical score between 0 and 1 indicating rollout quality, as determined by grading standards and explanation"
    )


class RolloutScores(BaseModel):
    """
    Model representing scores for a group of rollouts.
    """

    rollout_scores: List[RolloutScore] = Field(
        description="List of RolloutScore objects containing indices, explanations, and scores for each rollout"
    )


GENERAL_RM_PROMPT = """All of the rollouts below have been given the same task. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the task's intended outcome. 

Grading standards:
- A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
- A rollout that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a rollout that achieves its goal less efficiently.
- If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
- You may give some partial credit for a rollout that makes progress towards the task but does not complete it.

For each rollout, you need to output the rollout index, the explanation of the score, and the score. The score should be between 0 and 1.

"""

LLM_JUDGE_SINGLE_ROLLOUT_PROMPT = """
You are a helpful assistant that judges the quality of a customer support agent based on a single rollout of a conversation between a user and a customer support agent. You will be given the following information:
- The system prompt that was provided at the beginning of the rollout to the assistant. Understand how the agent is supposed to behave based on this prompt.
- The initial objective of the user that they set out to achieve with the assistant.
- The correct order and set of tools that the assistant could have used to achieve the user's objective. This is not necessarily the only way, but definitely a correct way.
- The rollout of the conversation between the user and the assistant.

Output a score between 0 and 1 indicating the quality of the assistant's performance. 0 is the lowest score, 1 is the highest score. A rollout that completes the task partially should get credit accordingly, based on how well it did.

Output your score and a detailed explanation of your score.
"""


def keep_only_messages(
    messages_and_choices: art.MessagesAndChoices,
) -> List[Dict[str, Any]]:
    """
    Keep only the messages from the messages_and_choices.
    """
    only_messages = []
    for message_and_choice in messages_and_choices:
        if isinstance(message_and_choice, Choice):
            only_messages.append(message_and_choice.message.model_dump())
        else:
            only_messages.append(message_and_choice)
    return only_messages


def create_and_split_messages(
    messages_and_choices: art.MessagesAndChoices,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create a system message from the first message in the group and return the remaining messages.
    """
    only_messages = keep_only_messages(messages_and_choices)
    return only_messages[0]["content"], only_messages[1:]


async def create_openai_response(
    prompt: str, judge_model: str = "o3", response_format: Type[T] = RolloutScores
) -> T | None:
    async with AsyncOpenAI() as client:
        response = await client.beta.chat.completions.parse(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        return response.choices[0].message.parsed


async def create_general_rm_trajectory_groups(
    group: art.TrajectoryGroup, config: RunConfig
) -> art.TrajectoryGroup:
    try:
        user_prompt = GENERAL_RM_PROMPT

        system_message, remaining_messages = create_and_split_messages(
            group.trajectories[0].messages_and_choices
        )
        user_prompt += f"Here is the system prompt that was provided at the beginning of each of the rollouts:\n--- START OF SYSTEM PROMPT ---\n{system_message}\n--- END OF SYSTEM PROMPT ---\n\n Here are the tools that were available to the rollouts:\n--- START OF TOOLS ---\n{group.trajectories[0].tools}\n--- END OF TOOLS ---\n\n Here are the rollouts to evaluate:"
        for idx, trajectory in enumerate(group.trajectories):
            user_prompt += f"\n\n--- ROLLOUT {idx} ---\n"
            _, messages = create_and_split_messages(trajectory.messages_and_choices)
            # Format conversation
            user_prompt = add_messages_to_prompt(user_prompt, messages)

        if config.judge_model.startswith("o3") or config.judge_model.startswith(
            "o4-mini"
        ):
            response = await create_openai_response(user_prompt, config.judge_model)
        else:
            raise ValueError(f"General RM model {config.judge_model} not supported")

        assert response is not None
        assert len(response.rollout_scores) == len(group.trajectories)

        new_trajectories = []
        for idx, trajectory in enumerate(group.trajectories):
            new_trajectory = copy.deepcopy(trajectory)
            if new_trajectory.reward == -1:
                new_trajectory.metadata["judge_explanation"] = "Max token trajectory"
            else:
                new_trajectory.metrics["outcome_correct"] = new_trajectory.reward
                new_trajectory.reward = response.rollout_scores[idx].score
                new_trajectory.metadata["judge_explanation"] = response.rollout_scores[
                    idx
                ].explanation
            try:
                await update_openpipe_log(new_trajectory)
            except Exception as e:
                print(f"Error updating openpipe log: {e}")
            new_trajectories.append(new_trajectory)

        return art.TrajectoryGroup(new_trajectories)
    except Exception as e:
        print(f"Error creating general RM trajectory groups: {e}")
        return group


def add_messages_to_prompt(user_prompt: str, messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls and len(tool_calls) > 0:
            tool_call_str = ""
            for tool_call in tool_calls:
                tool_call_str += f"TOOL CALL: {tool_call['function']['name']}: {tool_call['function']['arguments']}\n"
            user_prompt += f"{role.upper()}: {tool_call_str}\n"
        else:
            user_prompt += f"{role.upper()}: {content}\n"
    return user_prompt


def create_correct_order_and_set_of_tools(actions: List[Dict[str, Any]]) -> str:
    """
    Create a string representation of the correct order and set of tools that the assistant could have used to achieve the user's objective.
    """
    resp = ""
    for action in actions:
        resp += f"TOOL CALL: {action['name']}: {action['kwargs']}\n"
    return resp


async def calculate_reward(result: SolveResult, config: RunConfig) -> Tuple[float, str]:
    # If the agent was forced to stop, it should get a score of -1, regardless of the reward type
    if result.info["forced_stop"]:
        return -1, "Max token trajectory"

    if config.reward_type == "general_rm":
        # general rm reward is calculated in reference to the other rollouts in the group
        return result.reward, "pending_general_rm_reward"

    reward = 0.0

    # add the verifiable reward from the environment directly
    reward += result.reward

    if config.reward_type == "real":
        return reward, "real_reward"

    if config.reward_type == "real+llm":
        try:
            user_objective = result.info["task"]["instruction"]
            correct_order_and_set_of_tools = create_correct_order_and_set_of_tools(
                result.info["task"]["actions"]
            )
            user_prompt = LLM_JUDGE_SINGLE_ROLLOUT_PROMPT
            user_prompt += f"\n\nSystem Prompt:\n--- START OF SYSTEM PROMPT ---\n{result.messages[0]['content']}\n--- END OF SYSTEM PROMPT ---\n\n"
            user_prompt += f"\n User Objective:\n--- START OF USER OBJECTIVE ---\n{user_objective}\n--- END OF USER OBJECTIVE ---\n\n"
            user_prompt += f"\n Correct Order and Set of Tools:\n--- START OF CORRECT TOOLS ---\n{correct_order_and_set_of_tools}\n--- END OF CORRECT TOOLS ---\n\n"
            user_prompt += "\n Rollout:\n--- START OF ROLLOUT ---\n"
            user_prompt = add_messages_to_prompt(user_prompt, result.messages)
            user_prompt += "\n--- END OF ROLLOUT ---\n\n"
            response = await create_openai_response(
                user_prompt, config.judge_model, response_format=RolloutScoreLLM
            )
            assert response is not None
            reward += response.score
            return reward, response.explanation
        except Exception as e:
            print(f"Error calculating LLM reward: {e}")
            return reward, "real_reward, error_calculating_llm_reward: " + str(e)

    raise ValueError(f"Invalid reward type: {config.reward_type}")
