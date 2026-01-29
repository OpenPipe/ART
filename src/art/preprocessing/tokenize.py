# ruff: noqa: I001
# Import order is intentional - unsloth MUST be imported before transformers
import math
import random
from dataclasses import dataclass
from itertools import takewhile
from typing import Any, Generator, cast

import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..trajectories import History, Trajectory, TrajectoryGroup, get_messages


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


@dataclass
class SFTBatch:
    """A batch of tokenized trajectories for supervised fine-tuning.
    Attributes:
        trajectory_tensors: List of tensor dictionaries, one per trajectory.
                           Each dict contains 'input_ids', 'attention_mask', and 'labels'.
        learning_rate: Learning rate to use for this batch.
        num_trajectories: Number of trajectories in this batch.
        num_trainable_tokens: Total number of tokens being trained on (labels != -100).
    """

    trajectory_tensors: list[dict[str, torch.Tensor]]
    learning_rate: float
    num_trajectories: int
    num_trainable_tokens: int


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
            message.logprobs or allow_training_without_logprobs  # ty:ignore[possibly-missing-attribute]
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
            tools=tools,
            continue_final_message=True,
            tokenize=False,
        ),
    )
    original_token_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            cast(list[dict], messages),
            tools=tools,
            continue_final_message=True,
        ),
    )
    sentinal_token_id = max(
        set(range(cast(int, tokenizer.vocab_size))) - set(original_token_ids)
    )
    sentinal_token = tokenizer.decode(sentinal_token_id)
    token_template_messages: list[dict[str, Any]] = []
    for original, message in zip(messages_and_choices, messages):
        trainable_assistant = (
            not isinstance(original, dict) and original.logprobs is not None
        ) or (
            allow_training_without_logprobs
            and isinstance(original, dict)
            and original.get("role") == "assistant"
        )
        if trainable_assistant:
            token_template_messages.append(
                {
                    "role": "assistant",
                    "content": sentinal_token,
                    **(
                        {"tool_calls": message.get("tool_calls")}
                        if message.get("tool_calls")
                        else {}
                    ),
                }
            )
        else:
            token_template_messages.append(cast(dict[str, Any], message))
    token_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            cast(list[dict], token_template_messages),
            tools=tools,
            continue_final_message=True,
        ),
    )
    assistant_mask: list[int] = [0] * len(token_ids)
    logprobs = [float("nan")] * len(token_ids)
    for message in messages_and_choices:
        if isinstance(message, dict):
            if message["role"] != "assistant":
                continue
            if not allow_training_without_logprobs:
                continue
        elif message.logprobs is None and not allow_training_without_logprobs:  # ty:ignore[possibly-missing-attribute]
            continue
        start = token_ids.index(sentinal_token_id)
        end = start + 1
        try:
            end_token_id = token_ids[end]
        except IndexError:
            end_token_id = None
        if isinstance(message, dict):
            if message.get("tool_calls"):
                raise ValueError(
                    "Assistant message has tool_calls but is being tokenized "
                    "via tokenizer.encode(content). This path ignores tool calls."
                )
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
            assert choice.logprobs or allow_training_without_logprobs, (  # ty:ignore[possibly-missing-attribute]
                "Chat completion choices must have logprobs"
            )
            if not choice.logprobs:  # ty:ignore[possibly-missing-attribute]
                continue
            token_logprobs = choice.logprobs.content or choice.logprobs.refusal or []  # ty:ignore[possibly-missing-attribute]
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
                token_ids[start:end] = [
                    token_id if token_id is not None else tokenizer.eos_token_id
                    for token_id in tokenizer.convert_tokens_to_ids(
                        [
                            token_logprob.token or tokenizer.eos_token
                            for token_logprob in token_logprobs
                        ]
                    )
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
            or tokenizer.convert_tokens_to_ids(
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


def tokenize_sft_batches(
    trajectories: list[Trajectory],
    batch_size: int,
    learning_rates: list[float],
    tokenizer: PreTrainedTokenizerBase,
    instruction_part: str,
    response_part: str,
) -> Generator[SFTBatch, None, None]:
    """
    Tokenize trajectories into batches for supervised fine-tuning.
    Args:
        trajectories: Flat list of trajectories
        batch_size: Number of trajectories per batch
        learning_rates: Learning rate for each batch
        tokenizer: Tokenizer to use for encoding
        instruction_part: Instruction template part (e.g., "User:")
        response_part: Response template part (e.g., "Assistant:")
    Yields:
        SFTBatch object containing:
            - trajectory_tensors: List of tensors for each trajectory
            - learning_rate: Learning rate for this batch
            - num_trajectories: Number of trajectories in this batch
            - num_trainable_tokens: Total number of trainable tokens
    """
    # Import Unsloth Zoo utility for training on responses only
    # Source: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/dataset_utils.py
    # This function handles edge cases with tokenization (newlines, spaces, etc.)
    from unsloth_zoo.dataset_utils import train_on_responses_only

    # Validate inputs
    num_trajectories = len(trajectories)
    num_learning_rates = len(learning_rates)
    expected_num_batches = math.ceil(num_trajectories / batch_size)

    if num_learning_rates != expected_num_batches:
        raise ValueError(
            f"Mismatch between trajectories and learning_rates: "
            f"{num_trajectories} trajectories with batch_size={batch_size} "
            f"yields {expected_num_batches} batches, but got {num_learning_rates} learning_rates"
        )

    # Handle missing pad_token_id (common for LLaMA and similar models)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    _train_on_responses_only = train_on_responses_only(
        trainer=None,
        instruction_part=instruction_part,
        response_part=response_part,
        force_match=False,
        tokenizer=tokenizer,
        return_function=True,
    )

    # TODO Process input_ids in batch for better efficiency
    for batch_idx, lr in enumerate(learning_rates):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        trajectory_batch = trajectories[start_idx:end_idx]

        # First pass: tokenize all trajectories
        tokenized_trajectories = []
        for trajectory in trajectory_batch:
            messages = trajectory.messages_and_choices
            tools = trajectory.tools

            # Single-step tokenization: apply_chat_template with tokenize=True
            input_ids = cast(
                list[int],
                tokenizer.apply_chat_template(
                    cast(Any, messages),
                    tools=cast(Any, tools),
                    tokenize=True,
                    add_generation_prompt=False,
                ),
            )

            # Create attention mask (all 1s - no padding yet)
            attention_mask = [1] * len(input_ids)

            labels = _train_on_responses_only({"input_ids": [input_ids]})["labels"][0]

            tokenized_trajectories.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        # Find max length in this batch for padding
        max_seq_length = max(len(t["input_ids"]) for t in tokenized_trajectories)

        # Second pass: pad all trajectories to max_seq_length
        trajectory_tensors = []
        for tokenized in tokenized_trajectories:
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = tokenized["labels"]

            # Pad to max_seq_length
            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length

            trajectory_tensor = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            }

            trajectory_tensors.append(trajectory_tensor)

        # Calculate total trainable tokens (labels != -100)
        num_trainable_tokens = sum(
            (tensor_dict["labels"] != -100).sum().item()
            for tensor_dict in trajectory_tensors
        )

        yield SFTBatch(
            trajectory_tensors=trajectory_tensors,
            learning_rate=lr,
            num_trajectories=len(trajectory_tensors),
            num_trainable_tokens=num_trainable_tokens,
        )
