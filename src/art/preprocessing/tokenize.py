from dataclasses import dataclass
from itertools import takewhile
import math
import random
from typing import Any, Generator, cast

from PIL import Image
import torch
from transformers.image_processing_utils import BaseImageProcessor
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
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
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
            pixel_values=None,
            image_grid_thw=None,
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
    image_processor: BaseImageProcessor | None = None,
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
                    image_processor,
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
        first_non_nan_index = min(
            (
                next(
                    (i for i, lp in enumerate(r.logprobs) if not math.isnan(lp)),
                    len(r.logprobs),
                )
                for r in results
            ),
            default=0,
        )
        prompt_length = max(min(prompt_length, first_non_nan_index) - 1, 0)
        # Set the prompt id and length
        for result in results:
            result.prompt_id = prompt_id
            result.prompt_length = prompt_length
        if shuffle_group_trajectories:
            random.shuffle(results)
        yield from results


def tokenize_trajectory(
    tokenizer: "PreTrainedTokenizerBase",
    image_processor: BaseImageProcessor | None,
    history: History,
    advantage: float,
    allow_training_without_logprobs: bool,
) -> TokenizedResult | None:
    """
    Tokenizes a trajectory and returns a TokenizedResult.
    """
    # Find the index of the last assistant message
    last_assistant_index = -1
    for i, message in enumerate(history.messages_and_choices):
        if (
            isinstance(message, dict)
            and message["role"] == "assistant"
            and allow_training_without_logprobs
        ):
            last_assistant_index = i
        elif not isinstance(message, dict) and (
            message.logprobs or allow_training_without_logprobs
        ):
            last_assistant_index = i
    # If there are no trainable assistant messages, return None
    if last_assistant_index == -1:
        return None
    messages_and_choices = history.messages_and_choices[: last_assistant_index + 1]
    messages = get_messages(messages_and_choices)
    tools: Any = (
        [{"type": "function", "function": tool} for tool in history.tools]
        if history.tools is not None
        else None
    )
    chat = cast(
        str,
        tokenizer.apply_chat_template(
            cast(list[dict], messages),
            tools=tools,  # type: ignore
            continue_final_message=True,
            tokenize=False,
        ),
    )
    original_token_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            cast(list[dict], messages),
            tools=tools,  # type: ignore
            continue_final_message=True,
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
            tools=tools,  # type: ignore
            continue_final_message=True,
        ),
    )
    assistant_mask: list[int] = [0] * len(token_ids)
    logprobs = [float("nan")] * len(token_ids)
    for message in messages_and_choices:
        if isinstance(message, dict) and not message["role"] == "assistant":
            continue
        start = token_ids.index(sentinal_token_id)
        end = start + 1
        try:
            end_token_id = token_ids[end]
        except IndexError:
            end_token_id = None
        if isinstance(message, dict):
            content = message.get("content")
            assert isinstance(content, str)
            content_token_ids = tokenizer.encode(
                content,
                add_special_tokens=False,
            )
            token_ids[start:end] = content_token_ids
            logprobs[start:end] = [float("nan")] * len(content_token_ids)
            assistant_mask[start:end] = [1] * len(content_token_ids)
        else:
            choice = message
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
            try:
                token_ids[start:end] = (
                    int(token_logprob.token.split(":")[1])
                    for token_logprob in token_logprobs
                )
            except (IndexError, ValueError):
                token_ids[start:end] = [  # type: ignore
                    token_id if token_id is not None else tokenizer.eos_token_id
                    for token_id in tokenizer.convert_tokens_to_ids(
                        [
                            token_logprob.token or tokenizer.eos_token
                            for token_logprob in token_logprobs
                        ]
                    )  # type: ignore
                ]
            logprobs[start:end] = (
                token_logprob.logprob for token_logprob in token_logprobs
            )
            assistant_mask[start:end] = [1] * len(token_logprobs)
            if token_ids[start + len(token_logprobs) - 1] == end_token_id:
                token_ids.pop(start + len(token_logprobs))
                logprobs.pop(start + len(token_logprobs))
                assistant_mask.pop(start + len(token_logprobs))
    if image_processor:
        images: list[Image.Image] = []
        for message in messages_and_choices:
            if (
                isinstance(message, dict)
                and message["role"] == "user"
                and isinstance(message["content"], (list, tuple))
            ):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        image_url = content["image_url"]["url"].removeprefix("file://")
                        images.append(Image.open(image_url))
        image_token_id = cast(
            int,
            getattr(image_processor, "image_token_id", None)
            or tokenizer.convert_tokens_to_ids(  # type: ignore
                getattr(image_processor, "image_token", "<|image_pad|>")
            ),
        )
        if images:
            result = image_processor(images=images)
            offset = 0
            for num_image_tokens in (
                image_grid_thw.prod().item()
                // (getattr(image_processor, "merge_size", 1) ** 2)
                for image_grid_thw in result["image_grid_thw"]
            ):
                start = token_ids.index(image_token_id, offset)
                offset = start + num_image_tokens
                end = start + 1
                token_ids[start:end] = [image_token_id] * num_image_tokens
                logprobs[start:end] = [float("nan")] * num_image_tokens
                assistant_mask[start:end] = [0] * num_image_tokens
            pixel_values = result["pixel_values"]
            image_grid_thw = result["image_grid_thw"]
        else:
            pixel_values = None
            image_grid_thw = None
    else:
        pixel_values = None
        image_grid_thw = None
    return TokenizedResult(
        advantage=advantage,
        chat=chat,
        tokens=[tokenizer.decode(token_id) for token_id in token_ids],
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
