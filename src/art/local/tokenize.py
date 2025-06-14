from dataclasses import dataclass
from itertools import takewhile
import math
import random
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import cast, Generator

from ..trajectories import Trajectory, TrajectoryGroup


@dataclass
class TokenizedResult:
    trajectory: Trajectory
    advantage: float
    chat: str
    tokens: list[str]
    token_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    logprobs: list[float]
    prompt_id: int = 0
    prompt_length: int = 0

    def without_prompt(self) -> "TokenizedResult":
        return TokenizedResult(
            trajectory=self.trajectory,
            advantage=self.advantage,
            chat=self.chat,
            tokens=self.tokens[self.prompt_length :],
            token_ids=self.token_ids[self.prompt_length :],
            input_pos=self.input_pos[self.prompt_length :],
            assistant_mask=self.assistant_mask[self.prompt_length :],
            logprobs=self.logprobs[self.prompt_length :],
            prompt_id=self.prompt_id,
            prompt_length=0,
        )


def tokenize_trajectory_groups(
    tokenizer: "PreTrainedTokenizerBase",
    trajectory_groups: list[TrajectoryGroup],
    allow_training_without_logprobs: bool,
) -> Generator["TokenizedResult", None, None]:
    for group in trajectory_groups:
        if not group:
            continue
        results: list[TokenizedResult] = []
        # Calculate GRPO group mean and standard deviation
        reward_mean = sum(trajectory.reward for trajectory in group) / len(group)
        reward_std = math.sqrt(
            sum((trajectory.reward - reward_mean) ** 2 for trajectory in group)
            / len(group)
        )
        for trajectory in group:
            # Calculate GRPO advantage for this trajectory
            advantage = (trajectory.reward - reward_mean) / (reward_std + 1e-6)
            # Skip trajectories with no advantage
            if advantage == 0:
                continue
            if result := tokenize_trajectory(
                tokenizer,
                trajectory,
                advantage,
                allow_training_without_logprobs,
            ):
                results.append(result)
        # Choose a random prompt id
        prompt_id = random.randint(-(2**63), 2**63 - 1)
        # Find the longest shared prefix
        # TODO: Potentially support multiple prompts per group
        # Initial thought is to sort the results by token_ids and then
        # successively group prompts with the same prefix.
        prompt_length = len(
            list(
                takewhile(
                    lambda x: len(set(x)) == 1,
                    zip(*(r.token_ids for r in results)),
                )
            )
        )
        # Set the prompt id and length
        for result in results:
            result.prompt_id = prompt_id
            result.prompt_length = prompt_length
        random.shuffle(results)
        yield from results


def tokenize_trajectory(
    tokenizer: "PreTrainedTokenizerBase",
    trajectory: Trajectory,
    advantage: float,
    allow_training_without_logprobs: bool,
) -> TokenizedResult | None:
    """
    Tokenizes a trajectory and returns a TokenizedResult.
    """
    # Find the index of the last assistant message
    last_assistant_index = -1
    for i, message_or_choice in enumerate(trajectory.messages_and_choices):
        if (
            isinstance(message_or_choice, dict)
            and message_or_choice["role"] == "assistant"
            and allow_training_without_logprobs
        ):
            last_assistant_index = i
        elif not isinstance(message_or_choice, dict) and (
            message_or_choice.logprobs or allow_training_without_logprobs
        ):
            last_assistant_index = i
    # If there are no trainable assistant messages, return None
    if last_assistant_index == -1:
        return None
    trajectory = trajectory.model_copy(
        update={
            "messages_and_choices": trajectory.messages_and_choices[
                : last_assistant_index + 1
            ]
        }
    )
    chat = cast(
        str,
        tokenizer.apply_chat_template(
            cast(list[dict], trajectory.messages()),
            tools=trajectory.tools,  # type: ignore
            tokenize=False,
        ),
    )
    original_token_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            cast(list[dict], trajectory.messages()),
            tools=trajectory.tools,  # type: ignore
        ),
    )
    sentinal_token_id = max(
        set(range(cast(int, tokenizer.vocab_size))) - set(original_token_ids)
    )
    sentinal_token = tokenizer.decode(sentinal_token_id)
    result = cast(
        dict,
        tokenizer.apply_chat_template(
            cast(
                list[dict],
                [
                    (
                        message_or_choice
                        if isinstance(message_or_choice, dict)
                        else {
                            "role": "assistant",
                            "content": sentinal_token,
                        }
                    )
                    for message_or_choice in trajectory.messages_and_choices
                ],
            ),
            tools=trajectory.tools,  # type: ignore
            return_dict=True,
            return_assistant_token_mask=allow_training_without_logprobs,
        ),
    )
    token_ids: list[int] = result["input_ids"]
    assistant_mask: list[int] = (
        result["attention_mask"]
        if allow_training_without_logprobs
        else [0] * len(token_ids)
    )
    logprobs = [float("nan")] * len(token_ids)
    for message_or_choice in trajectory.messages_and_choices:
        if isinstance(message_or_choice, dict):
            continue
        choice = message_or_choice
        assert (
            choice.logprobs or allow_training_without_logprobs
        ), "Chat completion choices must have logprobs"
        if not choice.logprobs:
            continue
        token_logprobs = choice.logprobs.content or choice.logprobs.refusal or []
        sentinal_index = token_ids.index(sentinal_token_id)
        token_ids[sentinal_index : sentinal_index + 1] = (
            int(token_logprob.token.split(":")[1]) for token_logprob in token_logprobs
        )
        logprobs[sentinal_index : sentinal_index + 1] = (
            token_logprob.logprob for token_logprob in token_logprobs
        )
        assistant_mask[sentinal_index : sentinal_index + 1] = [1] * len(token_logprobs)
    return TokenizedResult(
        trajectory=trajectory,
        advantage=advantage,
        chat=chat,
        tokens=[tokenizer.decode(token_id) for token_id in token_ids],
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
    )
