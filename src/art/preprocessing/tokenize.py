import math
import random
from dataclasses import dataclass
from itertools import takewhile
from typing import Generator, cast

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..trajectories import History, TrajectoryGroup, get_messages


@dataclass
class TokenizedResult:
    advantage: float
    chat: str
    tokens: list[str]
    token_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    logprobs: list[float]
    weight: float = 0.0
    prompt_id: int = 0
    prompt_length: int = 0

    def without_prompt(self) -> "TokenizedResult":
        return TokenizedResult(
            advantage=self.advantage,
            chat=self.chat,
            tokens=self.tokens[self.prompt_length :],
            token_ids=self.token_ids[self.prompt_length :],
            input_pos=self.input_pos[self.prompt_length :],
            assistant_mask=self.assistant_mask[self.prompt_length :],
            logprobs=self.logprobs[self.prompt_length :],
            weight=self.weight,
            prompt_id=self.prompt_id,
            prompt_length=0,
        )


def tokenize_trajectory_groups(
    tokenizer: "PreTrainedTokenizerBase",
    trajectory_groups: list[TrajectoryGroup],
    allow_training_without_logprobs: bool,
    scale_rewards: bool,
    shuffle_group_trajectories: bool = True,
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
            advantage = trajectory.reward - reward_mean
            if scale_rewards:
                advantage /= reward_std + 1e-6
            # Skip trajectories with no advantage
            if advantage == 0:
                continue
            trajectory_results: list[TokenizedResult] = []
            for history in [
                History(
                    messages_and_choices=trajectory.messages_and_choices,
                    tools=trajectory.tools,
                ),
                *trajectory.additional_histories,
            ]:
                if result := tokenize_trajectory(
                    tokenizer,
                    history,
                    advantage,
                    allow_training_without_logprobs,
                ):
                    trajectory_results.append(result)
            weight = 1 / (
                sum(sum(result.assistant_mask) for result in trajectory_results) + 1e-6
            )
            for result in trajectory_results:
                result.weight = weight
            results.extend(trajectory_results)
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
        if shuffle_group_trajectories:
            random.shuffle(results)
        yield from results


def tokenize_trajectory(
    tokenizer: "PreTrainedTokenizerBase",
    history: History,
    advantage: float,
    allow_training_without_logprobs: bool,
) -> TokenizedResult | None:
    """
    Tokenizes a trajectory and returns a TokenizedResult.
    """
    # Find the index of the last assistant message
    last_assistant_index = -1
    for i, message_or_choice in enumerate(history.messages_and_choices):
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
    messages_and_choices = history.messages_and_choices[: last_assistant_index + 1]
    messages = get_messages(messages_and_choices)
    chat = cast(
        str,
        tokenizer.apply_chat_template(
            cast(list[dict], messages),
            tools=history.tools,  # type: ignore
            tokenize=False,
        ),
    )
    original_token_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            cast(list[dict], messages),
            tools=history.tools,  # type: ignore
        ),
    )
    sentinal_token_id = max(
        set(range(cast(int, tokenizer.vocab_size))) - set(original_token_ids)
    )
    sentinal_token = tokenizer.decode(sentinal_token_id)
    token_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            cast(
                list[dict],
                [
                    (
                        message_or_choice
                        if isinstance(message_or_choice, dict)
                        and not message_or_choice["role"] == "assistant"
                        else {
                            "role": "assistant",
                            "content": sentinal_token,
                        }
                    )
                    for message_or_choice in messages_and_choices
                ],
            ),
            tools=history.tools,  # type: ignore
        ),
    )
    assistant_mask: list[int] = [0] * len(token_ids)
    logprobs = [float("nan")] * len(token_ids)
    for message_or_choice in messages_and_choices:
        if (
            isinstance(message_or_choice, dict)
            and not message_or_choice["role"] == "assistant"
        ):
            continue
        start = token_ids.index(sentinal_token_id)
        end = start + 1
        if isinstance(message_or_choice, dict):
            content = message_or_choice.get("content")
            assert isinstance(content, str)
            content_token_ids = tokenizer.encode(
                content,
                add_special_tokens=False,
            )
            token_ids[start:end] = content_token_ids
            logprobs[start:end] = [float("nan")] * len(content_token_ids)
            assistant_mask[start:end] = [1] * len(content_token_ids)
        else:
            choice = message_or_choice
            assert choice.logprobs or allow_training_without_logprobs, (
                "Chat completion choices must have logprobs"
            )
            if not choice.logprobs:
                continue
            token_logprobs = choice.logprobs.content or choice.logprobs.refusal or []
            if (
                bytes(token_logprobs[0].bytes or []).decode("utf-8")
                == "<think>"
                == tokenizer.decode(token_ids[start - 4])
            ):
                start -= 4
            token_ids[start:end] = (
                int(token_logprob.token.split(":")[1])
                for token_logprob in token_logprobs
            )
            logprobs[start:end] = (
                token_logprob.logprob for token_logprob in token_logprobs
            )
            assistant_mask[start:end] = [1] * len(token_logprobs)
    return TokenizedResult(
        advantage=advantage,
        chat=chat,
        tokens=[tokenizer.decode(token_id) for token_id in token_ids],
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
    )
