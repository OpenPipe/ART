from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

import art
from art.local import LocalBackend
from art.local.backend import _tokenizer_cache_key
from art.preprocessing.pack import PackedTensors
from art.preprocessing.tokenize import (
    TokenizedResult,
    _apply_chat_template_token_ids,
    _messages_for_chat_template,
    tokenize_trajectory,
    tokenize_trajectory_groups,
)
from art.trajectories import History
from tests.support.chat_template_conformance_cases import (
    build_chat_template_conformance_inputs,
)


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _artifact_dir(base_model: str) -> Path:
    root = Path(__file__).resolve().parents[4] / ".local" / "model_support_validation"
    path = root / _slugify(base_model) / "chat_template_rollout"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _history(trajectory: art.Trajectory) -> History:
    return History(
        messages_and_choices=trajectory.messages_and_choices,
        tools=trajectory.tools,
    )


def _pack_trajectory_group(
    backend: LocalBackend,
    model: art.TrainableModel,
    trajectory_group: art.TrajectoryGroup,
) -> PackedTensors:
    packed_tensors = backend._get_packed_tensors(
        model,
        [trajectory_group],
        advantage_balance=0.0,
        allow_training_without_logprobs=False,
        scale_rewards=True,
        plot_tensors=False,
        packed_sequence_length=512,
        logprob_calculation_chunk_size=256,
    )
    if packed_tensors is None:
        raise RuntimeError("chat template conformance produced no packed tensors")
    return packed_tensors


def _assistant_prefix_tokens(
    result: TokenizedResult,
    *,
    choice_index: int = 0,
) -> list[int]:
    if not result.choice_offsets:
        raise RuntimeError("Expected at least one trainable assistant span")
    return result.token_ids[: result.choice_offsets[choice_index]]


class ChatTemplateScenarioReport(BaseModel):
    name: str
    entrypoint: str
    passed: bool
    assistant_token_count: int = 0
    packed_num_sequences: int = 0
    packed_sequence_length: int = 0
    result_count: int = 0
    num_tokens: int = 0
    num_trainable_tokens: int = 0
    mutation_changed_prompt: bool = False
    expected_error_substring: str | None = None
    observed_error: str | None = None


class ChatTemplateRolloutReport(BaseModel):
    base_model: str
    output_dir: str
    passed: bool
    scenario_count: int
    failed_scenarios: list[str] = Field(default_factory=list)
    scenarios: list[ChatTemplateScenarioReport] = Field(default_factory=list)


def run_chat_template_rollout(base_model: str) -> ChatTemplateRolloutReport:
    output_dir = _artifact_dir(base_model)
    backend = LocalBackend(path=str(output_dir))
    model = art.TrainableModel(
        name="model-support-chat-template",
        project="model-support-validation",
        base_model=base_model,
        _internal_config={"init_args": {"max_seq_length": 2048}},
    )
    internal_config = model._internal_config
    assert internal_config is not None
    tokenizer_key = _tokenizer_cache_key(base_model, internal_config)
    tokenizer = backend._tokenizers.get(tokenizer_key)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = backend._configure_training_tokenizer(
            AutoTokenizer.from_pretrained(base_model),
            model=model,
            internal_config=internal_config,
        )
        backend._tokenizers[tokenizer_key] = tokenizer

    inputs = build_chat_template_conformance_inputs(tokenizer)
    scenarios: list[ChatTemplateScenarioReport] = []

    text_pack = _pack_trajectory_group(backend, model, inputs.text_pack_group)
    scenarios.append(
        ChatTemplateScenarioReport(
            name="rl_text_pack",
            entrypoint="LocalBackend._get_packed_tensors",
            passed=int(text_pack["assistant_mask"].sum().item()) > 0,
            assistant_token_count=int(text_pack["assistant_mask"].sum().item()),
            packed_num_sequences=int(text_pack["tokens"].shape[0]),
            packed_sequence_length=int(text_pack["tokens"].shape[1]),
        )
    )

    non_final_tool_call_base = tokenize_trajectory(
        tokenizer=tokenizer,
        image_processor=None,
        history=_history(inputs.non_final_tool_call_base),
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=inputs.non_final_tool_call_base,
    )
    non_final_tool_call_mutated = tokenize_trajectory(
        tokenizer=tokenizer,
        image_processor=None,
        history=_history(inputs.non_final_tool_call_mutated),
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=inputs.non_final_tool_call_mutated,
    )
    if non_final_tool_call_base is None or non_final_tool_call_mutated is None:
        raise RuntimeError("tool-call tokenization produced no trainable tokens")
    if (
        len(non_final_tool_call_base.choice_offsets) < 2
        or len(non_final_tool_call_mutated.choice_offsets) < 2
    ):
        raise RuntimeError("expected non-final tool call and final assistant answer")
    non_final_tool_call_prefix_changed = _assistant_prefix_tokens(
        non_final_tool_call_base,
        choice_index=-1,
    ) != _assistant_prefix_tokens(
        non_final_tool_call_mutated,
        choice_index=-1,
    )
    scenarios.append(
        ChatTemplateScenarioReport(
            name="rl_non_final_tool_call_prefill_mutation",
            entrypoint="tokenize_trajectory",
            passed=non_final_tool_call_prefix_changed
            and int(sum(non_final_tool_call_base.assistant_mask)) > 0,
            assistant_token_count=int(sum(non_final_tool_call_base.assistant_mask)),
            mutation_changed_prompt=non_final_tool_call_prefix_changed,
        )
    )

    tool_conversation_pack = _pack_trajectory_group(
        backend,
        model,
        inputs.tool_conversation_group,
    )
    scenarios.append(
        ChatTemplateScenarioReport(
            name="rl_tool_conversation_pack",
            entrypoint="LocalBackend._get_packed_tensors",
            passed=int(tool_conversation_pack["assistant_mask"].sum().item()) > 0,
            assistant_token_count=int(
                tool_conversation_pack["assistant_mask"].sum().item()
            ),
            packed_num_sequences=int(tool_conversation_pack["tokens"].shape[0]),
            packed_sequence_length=int(tool_conversation_pack["tokens"].shape[1]),
        )
    )

    additional_history_results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [inputs.additional_histories_group],
            allow_training_without_logprobs=False,
            scale_rewards=True,
        )
    )
    additional_histories_pack = _pack_trajectory_group(
        backend,
        model,
        inputs.additional_histories_group,
    )
    scenarios.append(
        ChatTemplateScenarioReport(
            name="additional_histories_pack",
            entrypoint="tokenize_trajectory_groups + LocalBackend._get_packed_tensors",
            passed=len(additional_history_results) >= 4
            and int(additional_histories_pack["assistant_mask"].sum().item()) > 0,
            assistant_token_count=int(
                additional_histories_pack["assistant_mask"].sum().item()
            ),
            packed_num_sequences=int(additional_histories_pack["tokens"].shape[0]),
            packed_sequence_length=int(additional_histories_pack["tokens"].shape[1]),
            result_count=len(additional_history_results),
        )
    )

    full_conversation_messages = _messages_for_chat_template(
        tokenizer,
        inputs.sft_tool_conversation.messages_and_choices,
    )
    full_conversation_mutated_messages = _messages_for_chat_template(
        tokenizer,
        inputs.sft_tool_conversation_mutated.messages_and_choices,
    )
    full_conversation_input_ids = _apply_chat_template_token_ids(
        tokenizer,
        full_conversation_messages,
        tools=inputs.sft_tool_conversation.tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    full_conversation_mutated_input_ids = _apply_chat_template_token_ids(
        tokenizer,
        full_conversation_mutated_messages,
        tools=inputs.sft_tool_conversation_mutated.tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    scenarios.append(
        ChatTemplateScenarioReport(
            name="full_conversation_token_mutation",
            entrypoint="_apply_chat_template_token_ids",
            passed=full_conversation_input_ids != full_conversation_mutated_input_ids
            and len(full_conversation_input_ids) > 0,
            num_tokens=len(full_conversation_input_ids),
            mutation_changed_prompt=(
                full_conversation_input_ids != full_conversation_mutated_input_ids
            ),
        )
    )

    unsupported_result = tokenize_trajectory(
        tokenizer=tokenizer,
        image_processor=None,
        history=_history(inputs.unsupported_assistant_tool_calls),
        advantage=1.0,
        allow_training_without_logprobs=True,
        trajectory=inputs.unsupported_assistant_tool_calls,
    )
    scenarios.append(
        ChatTemplateScenarioReport(
            name="rl_dict_assistant_tool_calls_without_choice_is_not_trainable",
            entrypoint="tokenize_trajectory",
            passed=unsupported_result is None,
            result_count=int(unsupported_result is not None),
            assistant_token_count=(
                0
                if unsupported_result is None
                else int(sum(unsupported_result.assistant_mask))
            ),
        )
    )

    failed_scenarios = [scenario.name for scenario in scenarios if not scenario.passed]
    report = ChatTemplateRolloutReport(
        base_model=base_model,
        output_dir=str(output_dir),
        passed=not failed_scenarios,
        scenario_count=len(scenarios),
        failed_scenarios=failed_scenarios,
        scenarios=scenarios,
    )
    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report
