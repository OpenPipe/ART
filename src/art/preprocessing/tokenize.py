from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from itertools import takewhile
import json
import math
import random
from typing import TYPE_CHECKING, Any, Generator, Literal, cast

from openai.types.chat.chat_completion import Choice
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor

from ..trajectories import History, Trajectory, TrajectoryGroup, get_messages
from ..types import MessagesAndChoices
from ..utils.chat_template import (
    default_chat_template_kwargs_for_tokenizer,
    merge_chat_template_kwargs,
)
from .moe_routing import (
    MoeRoutingAlignmentStats,
    TokenRoute,
    align_choice_routes_to_tokenized_result,
)
from .response_masking import response_only_labels, token_ids_for_template_part
from .vllm_tokens import choice_vllm_token_metadata

ChatTemplateTool = dict[Any, Any] | Callable[..., Any]
ChatTemplateToolSchemaFormat = Literal["default", "vllm_openai"]


def _chat_template_kwargs(
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    return merge_chat_template_kwargs(
        default_chat_template_kwargs_for_tokenizer(tokenizer),
        chat_template_kwargs,
    )


def _normalize_tool_for_vllm_openai(tool: ChatTemplateTool) -> ChatTemplateTool:
    if callable(tool) or not isinstance(tool, dict):
        return tool
    if tool.get("type") != "function":
        return tool
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool

    ordered_function = {
        key: function[key]
        for key in ("name", "description", "parameters", "strict")
        if key in function
    }
    ordered_function.update(
        {key: value for key, value in function.items() if key not in ordered_function}
    )
    ordered_tool = {"type": "function", "function": ordered_function}
    ordered_tool.update(
        {key: value for key, value in tool.items() if key not in ordered_tool}
    )
    return ordered_tool


def _normalize_tools_for_chat_template(
    tools: Any,
    tool_schema_format: ChatTemplateToolSchemaFormat = "default",
) -> list[ChatTemplateTool] | None:
    if tools is None:
        return None
    if tool_schema_format not in ("default", "vllm_openai"):
        raise ValueError(
            f"Unknown chat template tool schema format: {tool_schema_format}"
        )
    normalized_tools: list[ChatTemplateTool] = []
    for tool in tools:
        if callable(tool):
            normalized_tool = tool
        elif isinstance(tool, dict) and "type" in tool:
            normalized_tool = cast(dict[Any, Any], tool)
        else:
            normalized_tool = {"type": "function", "function": tool}
        if tool_schema_format == "vllm_openai":
            normalized_tool = _normalize_tool_for_vllm_openai(normalized_tool)
        normalized_tools.append(normalized_tool)
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


def _messages_for_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages_and_choices: MessagesAndChoices,
    *,
    final_trainable_choice_index: int | None = None,
) -> list[dict[str, Any]]:
    messages = cast(list[dict[str, Any]], get_messages(messages_and_choices))
    if (
        final_trainable_choice_index is not None
        and 0 <= final_trainable_choice_index < len(messages)
    ):
        message = messages[final_trainable_choice_index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            messages[final_trainable_choice_index] = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
    return _normalize_tool_call_arguments_for_chat_template(tokenizer, messages)


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
    moe_routed_experts: list[TokenRoute | None] | None = None
    moe_routing_alignment_stats: MoeRoutingAlignmentStats | None = None
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
            moe_routed_experts=(
                self.moe_routed_experts[self.prompt_length :]
                if self.moe_routed_experts is not None
                else None
            ),
            moe_routing_alignment_stats=self.moe_routing_alignment_stats,
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
        num_dropped_trajectories: Number of overlength trajectories dropped while tokenizing.
    """

    trajectory_tensors: list[dict[str, torch.Tensor]]
    learning_rate: float
    num_trajectories: int
    num_tokens: int
    num_trainable_tokens: int
    num_dropped_trajectories: int = 0


def _validate_max_seq_length(max_seq_length: int | None) -> None:
    if max_seq_length is None:
        return
    if max_seq_length < 1:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")


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


def _choice_logprobs(
    choice: Choice,
    *,
    token_count: int,
    allow_training_without_logprobs: bool,
) -> tuple[list[float], list[Any]]:
    if choice.logprobs is None:
        if allow_training_without_logprobs:
            return [float("nan")] * token_count, []
        raise RuntimeError("Trainable vLLM Choice is missing logprobs")
    token_logprobs = choice.logprobs.content or choice.logprobs.refusal or []
    if len(token_logprobs) != token_count:
        raise RuntimeError(
            "Choice logprob length does not match vLLM completion token ids: "
            f"{len(token_logprobs)} != {token_count}"
        )
    return [float(token_logprob.logprob) for token_logprob in token_logprobs], list(
        token_logprobs
    )


def _choice_extra_logprobs(
    *,
    token_count: int,
    choice_offsets: list[int],
    choice_token_logprobs: list[list[Any]],
) -> dict[str, list[float]]:
    extra_logprobs: dict[str, list[float]] = {}
    for start, token_logprobs in zip(choice_offsets, choice_token_logprobs):
        for i, token_logprob in enumerate(token_logprobs):
            token_extra_logprobs = (token_logprob.model_extra or {}).get(
                "extra_logprobs"
            )
            if not isinstance(token_extra_logprobs, dict):
                continue
            for key, value in token_extra_logprobs.items():
                extra_logprobs.setdefault(key, [float("nan")] * token_count)[
                    start + i
                ] = float("nan") if value is None else float(value)
    return extra_logprobs


def _tokenized_result_from_vllm_choices(
    *,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    assistant_mask: list[int],
    logprobs: list[float],
    choices: list[Choice],
    choice_offsets: list[int],
    choice_token_lengths: list[int],
    choice_token_logprobs: list[list[Any]],
    advantage: float,
    trajectory: Trajectory,
) -> TokenizedResult:
    moe_routed_experts, moe_routing_alignment_stats = (
        align_choice_routes_to_tokenized_result(
            token_ids=token_ids,
            choices=choices,
            choice_offsets=choice_offsets,
            choice_token_lengths=choice_token_lengths,
        )
    )
    return TokenizedResult(
        advantage=advantage,
        chat="",
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        pixel_values=None,
        image_grid_thw=None,
        trajectory=trajectory,
        choice_offsets=choice_offsets,
        extra_logprobs=_choice_extra_logprobs(
            token_count=len(token_ids),
            choice_offsets=choice_offsets,
            choice_token_logprobs=choice_token_logprobs,
        ),
        moe_routed_experts=moe_routed_experts,
        moe_routing_alignment_stats=moe_routing_alignment_stats,
        _tokenizer=tokenizer,
    )


def tokenize_vllm_trajectory_histories(
    *,
    tokenizer: PreTrainedTokenizerBase,
    histories: list[History],
    advantage: float,
    allow_training_without_logprobs: bool,
    trajectory: Trajectory,
) -> list[TokenizedResult]:
    results: list[TokenizedResult] = []
    token_ids: list[int] = []
    assistant_mask: list[int] = []
    logprobs: list[float] = []
    choices: list[Choice] = []
    choice_offsets: list[int] = []
    choice_token_lengths: list[int] = []
    choice_token_logprobs: list[list[Any]] = []

    def flush() -> None:
        nonlocal token_ids, assistant_mask, logprobs, choices
        nonlocal choice_offsets, choice_token_lengths, choice_token_logprobs
        if not choices:
            return
        results.append(
            _tokenized_result_from_vllm_choices(
                tokenizer=tokenizer,
                token_ids=token_ids,
                assistant_mask=assistant_mask,
                logprobs=logprobs,
                choices=choices,
                choice_offsets=choice_offsets,
                choice_token_lengths=choice_token_lengths,
                choice_token_logprobs=choice_token_logprobs,
                advantage=advantage,
                trajectory=trajectory,
            )
        )
        token_ids = []
        assistant_mask = []
        logprobs = []
        choices = []
        choice_offsets = []
        choice_token_lengths = []
        choice_token_logprobs = []

    for history in histories:
        for choice in (
            item
            for item in history.messages_and_choices
            if isinstance(item, Choice)
            and (item.logprobs is not None or allow_training_without_logprobs)
        ):
            metadata = choice_vllm_token_metadata(choice)
            if metadata is None:
                raise RuntimeError(
                    "Trainable Choice is missing vLLM prompt_token_ids/token_ids. "
                    "Use a vLLM endpoint with return_token_ids enabled."
                )
            prompt_token_ids, completion_token_ids = metadata
            completion_logprobs, token_logprobs = _choice_logprobs(
                choice,
                token_count=len(completion_token_ids),
                allow_training_without_logprobs=allow_training_without_logprobs,
            )
            if not token_ids:
                token_ids.extend(prompt_token_ids)
                assistant_mask.extend([0] * len(prompt_token_ids))
                logprobs.extend([float("nan")] * len(prompt_token_ids))
            elif (
                len(prompt_token_ids) >= len(token_ids)
                and prompt_token_ids[: len(token_ids)] == token_ids
            ):
                suffix = prompt_token_ids[len(token_ids) :]
                token_ids.extend(suffix)
                assistant_mask.extend([0] * len(suffix))
                logprobs.extend([float("nan")] * len(suffix))
            else:
                flush()
                token_ids.extend(prompt_token_ids)
                assistant_mask.extend([0] * len(prompt_token_ids))
                logprobs.extend([float("nan")] * len(prompt_token_ids))

            choice_offsets.append(len(token_ids))
            choice_token_lengths.append(len(completion_token_ids))
            choice_token_logprobs.append(token_logprobs)
            choices.append(choice)
            token_ids.extend(completion_token_ids)
            assistant_mask.extend([1] * len(completion_token_ids))
            logprobs.extend(completion_logprobs)
    flush()
    return results


def tokenize_trajectory_groups(
    tokenizer: "PreTrainedTokenizerBase",
    trajectory_groups: list[TrajectoryGroup],
    allow_training_without_logprobs: bool,
    scale_rewards: bool,
    shuffle_group_trajectories: bool = True,
    drop_zero_advantage_trajectories: bool = True,
    image_processor: BaseImageProcessor | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template_tool_schema_format: ChatTemplateToolSchemaFormat = "default",
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
            if advantage == 0 and drop_zero_advantage_trajectories:
                continue
            trajectory_results = tokenize_vllm_trajectory_histories(
                tokenizer=tokenizer,
                histories=[
                    History(
                        messages_and_choices=trajectory.messages_and_choices,
                        tools=trajectory.tools,
                    ),
                    *trajectory.additional_histories,
                ],
                advantage=advantage,
                allow_training_without_logprobs=allow_training_without_logprobs,
                trajectory=trajectory,
            )
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
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template_tool_schema_format: ChatTemplateToolSchemaFormat = "default",
) -> TokenizedResult | None:
    """
    Tokenizes a trajectory and returns a TokenizedResult.
    """
    del image_processor, chat_template_kwargs, chat_template_tool_schema_format
    results = tokenize_vllm_trajectory_histories(
        tokenizer=tokenizer,
        histories=[history],
        advantage=advantage,
        allow_training_without_logprobs=allow_training_without_logprobs,
        trajectory=trajectory,
    )
    if not results:
        return None
    if len(results) > 1:
        raise RuntimeError(
            "History produced multiple non-append-only vLLM token sequences; "
            "use tokenize_vllm_trajectory_histories to preserve split histories."
        )
    return results[0]


def tokenize_sft_batch(
    trajectory_batch: list[Trajectory],
    learning_rate: float,
    tokenizer: PreTrainedTokenizerBase,
    instruction_part: str,
    response_part: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template_tool_schema_format: ChatTemplateToolSchemaFormat = "default",
    max_seq_length: int | None = None,
) -> SFTBatch:
    """Tokenize a single batch of trajectories for SFT.

    Args:
        trajectory_batch: List of trajectories in this batch
        learning_rate: Learning rate for this batch
        tokenizer: Tokenizer to use for encoding
        instruction_part: Instruction template part (e.g., "<|im_start|>user")
        response_part: Response template part (e.g., "<|im_start|>assistant")
        max_seq_length: Optional maximum tokenized trajectory length. Trajectories
            longer than this limit are dropped before tensors are created.

    Returns:
        SFTBatch object for this batch
    """
    _validate_max_seq_length(max_seq_length)

    instruction_ids = token_ids_for_template_part(tokenizer, instruction_part)
    response_ids = token_ids_for_template_part(tokenizer, response_part)
    # Tokenize all trajectories (no padding — each keeps its natural length)
    trajectory_tensors = []
    num_tokens = 0
    num_trainable_tokens = 0
    num_dropped_trajectories = 0
    for trajectory in trajectory_batch:
        messages = _messages_for_chat_template(
            tokenizer,
            trajectory.messages_and_choices,
        )
        tools = _normalize_tools_for_chat_template(
            trajectory.tools,
            tool_schema_format=chat_template_tool_schema_format,
        )
        template_kwargs = _chat_template_kwargs(tokenizer, chat_template_kwargs)

        # Single-step tokenization: apply_chat_template with tokenize=True
        input_ids = _apply_chat_template_token_ids(
            tokenizer,
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
        if max_seq_length is not None and len(input_ids) > max_seq_length:
            num_dropped_trajectories += 1
            continue

        attention_mask = [1] * len(input_ids)

        labels = response_only_labels(
            input_ids,
            instruction_ids=instruction_ids,
            response_ids=response_ids,
        )

        trajectory_tensors.append(
            {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            }
        )
        num_tokens += sum(attention_mask)
        num_trainable_tokens += sum(1 for l in labels if l != -100)

    if num_dropped_trajectories:
        print(
            "WARNING: Dropped "
            f"{num_dropped_trajectories}/{len(trajectory_batch)} SFT trajectories "
            f"because they exceed max_seq_length={max_seq_length}."
        )

    return SFTBatch(
        trajectory_tensors=trajectory_tensors,
        learning_rate=learning_rate,
        num_trajectories=len(trajectory_tensors),
        num_tokens=num_tokens,
        num_trainable_tokens=num_trainable_tokens,
        num_dropped_trajectories=num_dropped_trajectories,
    )
