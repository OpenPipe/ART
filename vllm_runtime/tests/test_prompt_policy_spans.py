import json
from threading import Event, Lock
import time
from types import SimpleNamespace

from art_vllm_runtime import policy_spans
import numpy as np
import pytest


def _identity(version: int, generation: str | None = None) -> dict[str, object]:
    return {
        "generation_id": generation or f"generation-{version}",
        "policy_version": version,
        "lora_slot": "run:active",
        "update_seq": version,
    }


def _step_span(start: int, end: int, version: int) -> dict[str, object]:
    return {"start_token": start, "end_token": end, **_identity(version)}


def test_prefill_only_execution_retains_prompt_provenance() -> None:
    request = SimpleNamespace(lora_int_id=7)
    policy_spans._WORKER_LORA_POLICY_BY_ID[7] = _identity(3)
    batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_computed_tokens_cpu=np.array([0]),
        num_prompt_tokens=np.array([7]),
        req_id_to_index={"request": 0},
        request_lora_mapping=np.array([7]),
        lora_id_to_lora_request={7: request},
    )
    context = policy_spans._policy_context_from_runner(
        SimpleNamespace(input_batch=batch),
        SimpleNamespace(num_scheduled_tokens={"request": 3}),
    )
    output = SimpleNamespace(req_ids=["request"], sampled_token_ids=[[]])

    policy_spans._attach_policy_spans_to_model_output(output, context)

    assert not hasattr(output, policy_spans.ART_POLICY_TOKEN_SPANS_FIELD)
    assert output.art_prompt_policy_token_spans["request"] == [
        {
            "start_token": 1,
            "end_token": 4,
            **_identity(3),
            policy_spans._CACHED_PROMPT_POLICY_SPAN_FIELD: False,
        }
    ]


def test_multichunk_prefill_preserves_exact_update_boundary() -> None:
    accumulated: list[dict[str, object]] = []
    policy_spans._append_absolute_prompt_spans(accumulated, [_step_span(1, 4, 2)])
    policy_spans._append_absolute_prompt_spans(accumulated, [_step_span(4, 7, 3)])

    assert [(span["start_token"], span["end_token"]) for span in accumulated] == [
        (1, 4),
        (4, 7),
    ]
    assert [span["generation_id"] for span in accumulated] == [
        "generation-2",
        "generation-3",
    ]

    request = SimpleNamespace(num_prompt_tokens=7)
    setattr(request, policy_spans.ART_PROMPT_POLICY_TOKEN_SPANS_FIELD, accumulated)
    output = SimpleNamespace()
    policy_spans._flush_complete_prompt_spans(request, output)
    assert output.art_prompt_policy_token_spans == accumulated


def test_decode_update_creates_exact_completion_boundary() -> None:
    state = SimpleNamespace(
        request_id="request",
        detokenizer=SimpleNamespace(num_output_tokens=lambda: 1),
    )
    previous = policy_spans._CURRENT_ENGINE_POLICY_SPANS
    try:
        policy_spans._CURRENT_ENGINE_POLICY_SPANS = {"request": [_step_span(0, 1, 2)]}
        policy_spans._append_current_policy_spans(state, 1)
        state.detokenizer = SimpleNamespace(num_output_tokens=lambda: 2)
        policy_spans._CURRENT_ENGINE_POLICY_SPANS = {"request": [_step_span(0, 1, 3)]}
        policy_spans._append_current_policy_spans(state, 1)
    finally:
        policy_spans._CURRENT_ENGINE_POLICY_SPANS = previous

    spans = state.art_policy_token_spans
    assert [(span["start_token"], span["end_token"]) for span in spans] == [
        (0, 1),
        (1, 2),
    ]
    assert [span["generation_id"] for span in spans] == [
        "generation-2",
        "generation-3",
    ]


def test_prefix_cache_and_preemption_do_not_replay_stale_prompt_spans() -> None:
    cached = {
        **_step_span(1, 8, 4),
        policy_spans._CACHED_PROMPT_POLICY_SPAN_FIELD: True,
    }
    accumulated: list[dict[str, object]] = []
    policy_spans._append_absolute_prompt_spans(accumulated, [cached])
    policy_spans._append_absolute_prompt_spans(accumulated, [cached])
    assert accumulated == [_step_span(1, 8, 4)]

    accumulated.clear()  # Scheduler preemption discards the prior attempt.
    policy_spans._append_absolute_prompt_spans(accumulated, [_step_span(1, 8, 5)])
    assert accumulated == [_step_span(1, 8, 5)]


def test_nonstreaming_choices_keep_independent_prompt_and_completion_spans() -> None:
    outputs = [
        SimpleNamespace(
            index=index,
            art_prompt_policy_token_spans=[_step_span(1, 4, index + 1)],
            art_policy_token_spans=[_step_span(0, 2, index + 1)],
        )
        for index in (0, 1)
    ]
    result = SimpleNamespace(outputs=outputs)

    assert policy_spans._policy_spans_by_choice_from_final_output(
        result, policy_spans.ART_PROMPT_POLICY_TOKEN_SPANS_FIELD
    ) == {0: [_step_span(1, 4, 1)], 1: [_step_span(1, 4, 2)]}
    assert policy_spans._policy_spans_by_choice_from_final_output(
        result, policy_spans.ART_POLICY_TOKEN_SPANS_FIELD
    ) == {0: [_step_span(0, 2, 1)], 1: [_step_span(0, 2, 2)]}


def test_streaming_choices_emit_independent_spans() -> None:
    prompt = {0: [_step_span(1, 4, 1)], 1: [_step_span(1, 4, 2)]}
    completion = {0: [_step_span(0, 1, 1)], 1: [_step_span(0, 1, 2)]}
    sent: set[int] = set()
    frame = (
        "data: "
        + json.dumps(
            {
                "model": "internal",
                "choices": [
                    {"index": 0, "token_ids": [10]},
                    {"index": 1, "token_ids": [20]},
                ],
            }
        )
        + "\n\n"
    )

    result = policy_spans._attach_policy_spans_to_stream_frame(
        frame,
        prompt_by_choice=prompt,
        completion_by_choice=completion,
        prompt_sent=sent,
        model_name="run:active",
    )
    payload = json.loads(result.removeprefix("data: "))

    assert payload["model"] == "run:active"
    assert payload["choices"][0]["prompt_policy_token_spans"] == [_step_span(1, 4, 1)]
    assert payload["choices"][1]["policy_token_spans"] == [_step_span(0, 1, 2)]
    assert prompt == completion == {}


def _lora_request(
    slot: str, generation: str, version: int, *, update_seq: int
) -> policy_spans.PolicyLoRARequest:
    return policy_spans.PolicyLoRARequest(
        lora_name=slot,
        lora_int_id=version,
        lora_path=f"/{generation}",
        generation_id=generation,
        policy_version=version,
        update_seq=update_seq,
    )


def test_same_slot_update_uses_generation_cas() -> None:
    async def exercise() -> None:
        coordinator = policy_spans.LoraUpdateCoordinator()
        initial = _lora_request("run:active", "generation-1", 1, update_seq=1)
        await coordinator.declare_initial(initial.lora_name, initial)

        with pytest.raises(RuntimeError, match="generation-stale"):
            await coordinator.begin_update(
                initial.lora_name, expected_generation_id="generation-stale"
            )
        sequence = await coordinator.begin_update(
            initial.lora_name, expected_generation_id="generation-1"
        )
        assert sequence == 2
        await coordinator.cancel_update(initial.lora_name, sequence)

    import asyncio

    asyncio.run(exercise())


def test_different_slots_prepare_concurrently() -> None:
    started = {"slot-a": Event(), "slot-b": Event()}
    release = Event()
    calls: list[str] = []
    calls_lock = Lock()

    class Manager:
        def _load_adapter(self, request: object) -> object:
            slot = str(request.lora_name)  # type: ignore[attr-defined]
            with calls_lock:
                calls.append(slot)
            started[slot].set()
            assert release.wait(timeout=1)
            return SimpleNamespace(id=request.lora_int_id)  # type: ignore[attr-defined]

    worker = SimpleNamespace(model_runner=SimpleNamespace(lora_manager=Manager()))
    requests = {
        slot: _lora_request(slot, f"generation-{slot}", index, update_seq=0)
        for index, slot in enumerate(("slot-a", "slot-b"), start=1)
    }
    try:
        for slot, request in requests.items():
            policy_spans._prepare_worker_lora(
                worker,
                f"operation-{slot}",
                policy_spans.policy_lora_request_payload(request),
            )
        assert all(event.wait(timeout=1) for event in started.values())
        assert set(calls) == set(requests)
    finally:
        release.set()
        deadline = time.monotonic() + 1
        while policy_spans._PREPARED_LORA_FUTURES and time.monotonic() < deadline:
            time.sleep(0.001)
        for slot in requests:
            policy_spans._abort_worker_lora(f"operation-{slot}")


def test_worker_commit_requires_completed_preparation() -> None:
    request = _lora_request("slot-a", "generation-a2", 1, update_seq=2)
    worker = SimpleNamespace(model_runner=SimpleNamespace(lora_manager=object()))

    with pytest.raises(RuntimeError, match="not ready"):
        policy_spans._commit_worker_lora(
            worker,
            "missing-operation",
            policy_spans.policy_lora_request_payload(request),
        )


def _scheduler_request(request: policy_spans.PolicyLoRARequest) -> SimpleNamespace:
    value = SimpleNamespace(
        request_id=request.lora_name,
        lora_request=request,
        num_computed_tokens=0,
        num_preemptions=0,
        output_token_ids=[],
        block_hashes=[],
        cache_salt=None,
    )
    value.update_block_hashes = lambda: None
    return value


def test_two_slot_commits_do_not_mix_scheduler_state() -> None:
    old_a = _lora_request("slot-a", "generation-a1", 1, update_seq=1)
    old_b = _lora_request("slot-b", "generation-b1", 2, update_seq=1)
    requests = {
        "slot-a": _scheduler_request(old_a),
        "slot-b": _scheduler_request(old_b),
    }

    class Core:
        scheduler = SimpleNamespace(requests=requests)

        def collective_rpc(self, method: str, *, args: tuple[object, ...]):
            assert method == "art_commit_prepared_lora_policy"
            payload = args[1]
            current = {
                key: payload[key]  # type: ignore[index]
                for key in (
                    "generation_id",
                    "policy_version",
                    "lora_name",
                    "lora_path",
                    "update_seq",
                )
            }
            current["lora_slot"] = current.pop("lora_name")
            previous_request = requests[current["lora_slot"]].lora_request
            previous = {
                "generation_id": previous_request.generation_id,
                "policy_version": previous_request.policy_version,
                "lora_slot": previous_request.lora_name,
                "lora_path": previous_request.lora_path,
                "update_seq": previous_request.update_seq,
            }
            return [{"loaded": True, "previous": previous, "current": current}]

        def pause_scheduler(self, _mode: str, _abort: bool) -> None:
            raise AssertionError("successful commits must not pause the scheduler")

    core = Core()
    new_a = _lora_request("slot-a", "generation-a2", 1, update_seq=2)
    new_b = _lora_request("slot-b", "generation-b2", 2, update_seq=2)

    policy_spans._commit_prepared_policy_lora_update(
        core, "operation-a", policy_spans.policy_lora_request_payload(new_a)
    )
    assert requests["slot-a"].lora_request.generation_id == "generation-a2"
    assert requests["slot-b"].lora_request.generation_id == "generation-b1"
    policy_spans._commit_prepared_policy_lora_update(
        core, "operation-b", policy_spans.policy_lora_request_payload(new_b)
    )
    assert requests["slot-b"].lora_request.generation_id == "generation-b2"


def test_partial_worker_commit_poisons_scheduler() -> None:
    request = _lora_request("slot-a", "generation-a2", 1, update_seq=2)

    class Core:
        scheduler = SimpleNamespace(requests={})

        def __init__(self) -> None:
            self.pauses: list[tuple[str, bool]] = []

        def collective_rpc(self, _method: str, *, args: tuple[object, ...]):
            raise RuntimeError("worker collective failed")

        def pause_scheduler(self, mode: str, abort: bool) -> None:
            self.pauses.append((mode, abort))

    core = Core()
    with pytest.raises(RuntimeError, match="worker collective failed"):
        policy_spans._commit_prepared_policy_lora_update(
            core, "operation-a", policy_spans.policy_lora_request_payload(request)
        )
    assert core.pauses == [("abort", True)]
