from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from itertools import takewhile
import json
import math
import random
from typing import Any, Generator, cast

from openai.types.chat.chat_completion import Choice
from PIL import Image
import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from ..trajectories import History, Trajectory, TrajectoryGroup, get_messages

ChatTemplateTool = dict[Any, Any] | Callable[..., Any]


def _normalize_tools_for_chat_template(tools: Any) -> list[ChatTemplateTool] | None:
    if tools is None:
        return None
    normalized_tools: list[ChatTemplateTool] = []
    for tool in tools:
        if callable(tool):
            normalized_tools.append(tool)
        elif isinstance(tool, dict) and "type" in tool:
            normalized_tools.append(cast(dict[Any, Any], tool))
        else:
            normalized_tools.append({"type": "function", "function": tool})
    return normalized_tools


def _normalize_tool_call_arguments_for_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chat_template = tokenizer.chat_template
    assert isinstance(chat_template, str)
    if "tool_call.arguments|items" not in chat_template:
        return messages

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            normalized_messages.append(message)
            continue

        assert isinstance(tool_calls, list)
        normalized_tool_calls = []
        for tool_call in tool_calls:
            assert isinstance(tool_call, dict)
            function = tool_call["function"]
            assert isinstance(function, dict)
            arguments_json = function["arguments"]
            assert isinstance(arguments_json, str)
            arguments = json.loads(arguments_json)
            assert isinstance(arguments, dict)
            normalized_tool_calls.append(
                {**tool_call, "function": {**function, "arguments": arguments}}
            )
        normalized_messages.append({**message, "tool_calls": normalized_tool_calls})

    return normalized_messages


@dataclass
class TokenizedResult:
    advantage: float
    chat: str
    token_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    logprobs: list[float]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    trajectory: Trajectory
    choice_offsets: list[int]
    extra_logprobs: dict[str, list[float]]
    _tokenizer: "PreTrainedTokenizerBase" = field(repr=False, compare=False)
    weight: float = 0.0
    prompt_id: int = 0
    prompt_length: int = 0

    @cached_property
    def tokens(self) -> list[str]:
        return [
            cast(str, self._tokenizer.decode(token_id)) for token_id in self.token_ids
        ]

    def without_prompt(self) -> "TokenizedResult":
        return TokenizedResult(
            advantage=self.advantage,
            chat=self.chat,
            token_ids=self.token_ids[self.prompt_length :],
            input_pos=self.input_pos[self.prompt_length :],
            assistant_mask=self.assistant_mask[self.prompt_length :],
            logprobs=self.logprobs[self.prompt_length :],
            pixel_values=None,
            image_grid_thw=None,
            trajectory=self.trajectory,
            choice_offsets=self.choice_offsets,
            extra_logprobs={
                key: values[self.prompt_length :]
                for key, values in self.extra_logprobs.items()
            },
            _tokenizer=self._tokenizer,
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
        num_tokens: Total number of non-padding tokens (attention_mask != 0).
        num_trainable_tokens: Total number of tokens being trained on (labels != -100).
    """

    trajectory_tensors: list[dict[str, torch.Tensor]]
    learning_rate: float
    num_trajectories: int
    num_tokens: int
    num_trainable_tokens: int


def _apply_chat_template_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    output = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(output, BatchEncoding):
        output = output["input_ids"]
    if isinstance(output, torch.Tensor):
        output = output.tolist()
    assert isinstance(output, list)
    if output and isinstance(output[0], list):
        assert len(output) == 1
        output = output[0]
    return cast(list[int], output)


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
                    trajectory,
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
    trajectory: Trajectory,
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
        elif isinstance(message, Choice) and (
            message.logprobs or allow_training_without_logprobs
        ):
            last_assistant_index = i
    # If there are no trainable assistant messages, return None
    if last_assistant_index == -1:
        return None
    messages_and_choices = history.messages_and_choices[: last_assistant_index + 1]
    messages = cast(list[dict[str, Any]], get_messages(messages_and_choices))
    # Qwen3.5's chat template uses `tool_call.arguments|items`, so it needs a
    # mapping here instead of the OpenAI JSON string.
    messages = _normalize_tool_call_arguments_for_chat_template(tokenizer, messages)
    tools = _normalize_tools_for_chat_template(history.tools)
    chat = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            continue_final_message=True,
            tokenize=False,
        ),
    )
    original_token_ids = _apply_chat_template_token_ids(
        tokenizer,
        messages,
        tools=tools,
        continue_final_message=True,
    )
    sentinel_token_id = max(set(range(tokenizer.vocab_size)) - set(original_token_ids))
    sentinel_token = tokenizer.decode(sentinel_token_id)
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
                    "content": sentinel_token,
                    **(
                        {"tool_calls": message.get("tool_calls")}
                        if message.get("tool_calls")
                        else {}
                    ),
                }
            )
        else:
            token_template_messages.append(cast(dict[str, Any], message))
    token_ids = _apply_chat_template_token_ids(
        tokenizer,
        token_template_messages,
        tools=tools,
        continue_final_message=True,
    )
    assistant_mask: list[int] = [0] * len(token_ids)
    logprobs = [float("nan")] * len(token_ids)
    choice_offsets, choice_token_logprobs = [], []

    for message in messages_and_choices:
        if isinstance(message, dict):
            if message["role"] != "assistant":
                continue
            if not allow_training_without_logprobs:
                continue
        elif message.logprobs is None and not allow_training_without_logprobs:  # ty:ignore[possibly-missing-attribute]
            continue
        start = token_ids.index(sentinel_token_id)
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
            assert isinstance(content, str), (
                "Trajectories must have a 'content' field of type str"
            )
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
            if token_logprobs and (
                bytes(token_logprobs[0].bytes or []).decode("utf-8")
                == "<think>"
                == tokenizer.decode(token_ids[start - 4])
            ):
                start -= 4
            choice_offsets.append(start)
            choice_token_logprobs.append(token_logprobs)
            try:
                token_ids[start:end] = (
                    int(token_logprob.token.split(":")[1])
                    for token_logprob in token_logprobs
                )
            except (IndexError, ValueError):
                token_ids[start:end] = [
                    token_id if token_id is not None else tokenizer.eos_token_id
                    for token_id in cast(
                        list[int],
                        tokenizer.convert_tokens_to_ids(
                            [
                                token_logprob.token or tokenizer.eos_token
                                for token_logprob in token_logprobs
                            ]
                        ),
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
    extra_logprobs: dict[str, list[float]] = {}
    for start, token_logprobs in zip(choice_offsets, choice_token_logprobs):
        for i, token_logprob in enumerate(token_logprobs):
            token_extra_logprobs = (token_logprob.model_extra or {}).get(
                "extra_logprobs"
            )
            if not isinstance(token_extra_logprobs, dict):
                continue
            for key, value in token_extra_logprobs.items():
                extra_logprobs.setdefault(key, [float("nan")] * len(token_ids))[
                    start + i
                ] = float("nan") if value is None else float(value)
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
                for values in extra_logprobs.values():
                    values[start:end] = [float("nan")] * num_image_tokens
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
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        trajectory=trajectory,
        choice_offsets=choice_offsets,
        extra_logprobs=extra_logprobs,
        _tokenizer=tokenizer,
    )


def tokenize_sft_batch(
    trajectory_batch: list[Trajectory],
    learning_rate: float,
    tokenizer: PreTrainedTokenizerBase,
    instruction_part: str,
    response_part: str,
) -> SFTBatch:
    """Tokenize a single batch of trajectories for SFT.

    Args:
        trajectory_batch: List of trajectories in this batch
        learning_rate: Learning rate for this batch
        tokenizer: Tokenizer to use for encoding
        instruction_part: Instruction template part (e.g., "<|im_start|>user")
        response_part: Response template part (e.g., "<|im_start|>assistant")

    Returns:
        SFTBatch object for this batch
    """
    import unsloth  # noqa: F401 - Must be imported first to set UNSLOTH_IS_PRESENT env var
    from unsloth_zoo.dataset_utils import train_on_responses_only

    train_on_responses_only_fn = train_on_responses_only(
        trainer=None,
        instruction_part=instruction_part,
        response_part=response_part,
        force_match=False,
        tokenizer=tokenizer,
        return_function=True,
    )
    # Tokenize all trajectories (no padding — each keeps its natural length)
    trajectory_tensors = []
    num_tokens = 0
    num_trainable_tokens = 0
    for trajectory in trajectory_batch:
        messages = trajectory.messages_and_choices
        tools = trajectory.tools

        # Single-step tokenization: apply_chat_template with tokenize=True
        input_ids = _apply_chat_template_token_ids(
            tokenizer,
            cast(Any, messages),
            tools=cast(Any, tools),
            tokenize=True,
            add_generation_prompt=False,
        )

        attention_mask = [1] * len(input_ids)

        labels = train_on_responses_only_fn({"input_ids": [input_ids]})["labels"][0]

        trajectory_tensors.append(
            {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            }
        )
        num_tokens += sum(attention_mask)
        num_trainable_tokens += sum(1 for l in labels if l != -100)

    return SFTBatch(
        trajectory_tensors=trajectory_tensors,
        learning_rate=learning_rate,
        num_trajectories=len(trajectory_tensors),
        num_tokens=num_tokens,
        num_trainable_tokens=num_trainable_tokens,
    )
