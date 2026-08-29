import asyncio
from types import SimpleNamespace

from art_vllm_runtime import policy_spans
import numpy as np
import pytest


def _identity(version: int) -> dict[str, object]:
    return {
        "generation_id": f"generation-{version}",
        "policy_version": version,
        "lora_slot": "run:active",
        "update_seq": version,
    }


def test_prompt_spans_follow_executed_prefill_chunks() -> None:
    request = SimpleNamespace(lora_int_id=7)
    policy_spans._WORKER_LORA_POLICY_BY_ID[7] = _identity(3)
    batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_computed_tokens_cpu=np.array([0, 99]),
        num_prompt_tokens=np.array([7, 99]),
        req_id_to_index={"request": 0},
        request_lora_mapping=np.array([7]),
        lora_id_to_lora_request={7: request},
    )
    runner = SimpleNamespace(input_batch=batch)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request": 3})

    context = policy_spans._policy_context_from_runner(runner, scheduler_output)
    output = SimpleNamespace(req_ids=["request"], sampled_token_ids=[[]])
    policy_spans._attach_policy_spans_to_model_output(output, context)

    assert output.art_prompt_policy_token_spans == {
        "request": [
            {
                "start_token": 1,
                "end_token": 4,
                **_identity(3),
                policy_spans._CACHED_PROMPT_POLICY_SPAN_FIELD: False,
            }
        ]
    }

    split_batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_scheduled_tokens=np.array([3]),
        num_computed_tokens_np=np.array([0]),
        prefill_len_np=np.array([7]),
    )
    split_runner = SimpleNamespace(
        input_batch=split_batch,
        lora_state=SimpleNamespace(lora_requests={"request": request}),
    )
    assert (
        policy_spans._policy_context_from_runner(
            split_runner, SimpleNamespace(num_scheduled_tokens={"dummy": 64})
        )
        == context
    )


def test_prompt_accumulator_keeps_real_boundaries_and_flushes_with_output() -> None:
    accumulated: list[dict[str, object]] = []
    policy_spans._append_absolute_prompt_spans(
        accumulated,
        [{"start_token": 1, "end_token": 4, **_identity(2)}],
    )
    policy_spans._append_absolute_prompt_spans(
        accumulated,
        [{"start_token": 4, "end_token": 7, **_identity(3)}],
    )
    policy_spans._append_absolute_prompt_spans(
        accumulated,
        [
            {
                "start_token": 1,
                "end_token": 7,
                **_identity(3),
                policy_spans._CACHED_PROMPT_POLICY_SPAN_FIELD: True,
            }
        ],
    )
    request = SimpleNamespace(num_prompt_tokens=7)
    setattr(request, policy_spans.ART_PROMPT_POLICY_TOKEN_SPANS_FIELD, accumulated)

    policy_spans._flush_complete_prompt_spans(request, None)
    assert hasattr(request, policy_spans.ART_PROMPT_POLICY_TOKEN_SPANS_FIELD)

    output = SimpleNamespace()
    policy_spans._flush_complete_prompt_spans(request, output)
    assert output.art_prompt_policy_token_spans == [
        {"start_token": 1, "end_token": 4, **_identity(2)},
        {"start_token": 4, "end_token": 7, **_identity(3)},
    ]
    assert not hasattr(request, policy_spans.ART_PROMPT_POLICY_TOKEN_SPANS_FIELD)


def test_cached_prompt_gets_one_synthetic_span() -> None:
    output = SimpleNamespace(req_ids=["request"], sampled_token_ids=[[42]])
    context = {
        "request": {
            **_identity(5),
            "prompt_span": (8, 8),
            "prompt_tokens": 8,
        }
    }

    policy_spans._attach_policy_spans_to_model_output(output, context)

    span = output.art_prompt_policy_token_spans["request"][0]
    assert (span["start_token"], span["end_token"]) == (1, 8)
    assert span[policy_spans._CACHED_PROMPT_POLICY_SPAN_FIELD] is True


def test_generation_transition_splits_same_policy_version() -> None:
    accumulated: list[dict[str, object]] = []
    first = {**_identity(3), "generation_id": "generation-a"}
    second = {**_identity(3), "generation_id": "generation-b"}

    policy_spans._append_absolute_prompt_spans(
        accumulated, [{"start_token": 1, "end_token": 4, **first}]
    )
    policy_spans._append_absolute_prompt_spans(
        accumulated, [{"start_token": 4, "end_token": 7, **second}]
    )

    assert [span["generation_id"] for span in accumulated] == [
        "generation-a",
        "generation-b",
    ]


def test_update_requires_exact_current_generation() -> None:
    async def exercise() -> None:
        coordinator = policy_spans.LoraUpdateCoordinator()
        initial = policy_spans.PolicyLoRARequest(
            lora_name="run:active",
            lora_int_id=1,
            lora_path="/generation-a",
            generation_id="generation-a",
            policy_version=3,
            update_seq=1,
        )
        await coordinator.declare_initial("run:active", initial)

        with pytest.raises(RuntimeError, match="expected 'generation-stale'"):
            await coordinator.begin_update(
                "run:active", expected_generation_id="generation-stale"
            )
        sequence = await coordinator.begin_update(
            "run:active", expected_generation_id="generation-a"
        )
        assert sequence == 2
        await coordinator.cancel_update("run:active", sequence)

    asyncio.run(exercise())


def test_publication_initializes_an_empty_mutable_slot_once() -> None:
    async def exercise() -> None:
        coordinator = policy_spans.LoraUpdateCoordinator()
        slot = "run:active"
        sequence = await coordinator.begin_publication(
            slot, expected_generation_id=None
        )
        initial = policy_spans.PolicyLoRARequest(
            lora_name=slot,
            lora_int_id=1,
            lora_path="/generation-a",
            generation_id="generation-a",
            policy_version=1,
            update_seq=sequence,
        )
        await coordinator.commit_update(slot, initial)

        async with coordinator.admission(slot) as admitted:
            assert admitted == initial
        with pytest.raises(RuntimeError, match="already active"):
            await coordinator.begin_publication(slot, expected_generation_id=None)

    asyncio.run(exercise())
