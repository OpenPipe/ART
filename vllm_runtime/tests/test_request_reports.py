from types import SimpleNamespace

from art_vllm_runtime.request_reports import (
    observe_policy_transition,
    observe_preemption,
    request_runtime_report,
    request_snapshot,
)


class _KvCache:
    def __init__(self) -> None:
        self.blocks = [[7, 8], [9]]

    def get_block_ids(self, _request_id: str):
        return self.blocks


def _request(*, computed: int, output: int = 0):
    return SimpleNamespace(
        request_id="a" * 64,
        prompt_token_ids=tuple(range(8)),
        num_prompt_tokens=8,
        num_computed_tokens=computed,
        output_token_ids=[42] * output,
        num_preemptions=0,
        lora_request=SimpleNamespace(
            lora_name="run:active",
            generation_id="generation-1",
            policy_version=1,
            update_seq=1,
        ),
    )


def test_policy_transition_reports_phase_and_exact_kv_snapshots() -> None:
    request = _request(computed=4)
    scheduler = SimpleNamespace(
        requests={request.request_id: request}, kv_cache_manager=_KvCache()
    )
    before = request_snapshot(scheduler, request)
    request.lora_request = SimpleNamespace(
        lora_name="run:active",
        generation_id="generation-2",
        policy_version=2,
        update_seq=2,
    )

    observe_policy_transition(
        scheduler,
        request,
        before=before,
        previous_policy={
            "lora_slot": "run:active",
            "generation_id": "generation-1",
            "policy_version": 1,
            "update_seq": 1,
        },
        next_policy={
            "lora_slot": "run:active",
            "generation_id": "generation-2",
            "policy_version": 2,
            "update_seq": 2,
        },
    )

    report = request_runtime_report(scheduler, request.request_id)["report"]
    assert report["events"] == [
        {
            "kind": "policy_transition",
            "phase": "prefill",
            "computed_tokens": 4,
            "prompt_tokens": 8,
            "output_tokens": 0,
            "preemptions": 0,
            "previous_policy": {
                "lora_slot": "run:active",
                "generation_id": "generation-1",
                "policy_version": 1,
                "update_seq": 1,
            },
            "next_policy": {
                "lora_slot": "run:active",
                "generation_id": "generation-2",
                "policy_version": 2,
                "update_seq": 2,
            },
            "kv_before": before["kv"],
            "kv_after": before["kv"],
            "ordinal": 0,
        }
    ]


def test_preemption_report_retains_bounded_before_and_after_state() -> None:
    request = _request(computed=10, output=2)
    scheduler = SimpleNamespace(
        requests={request.request_id: request}, kv_cache_manager=_KvCache()
    )
    before = request_snapshot(scheduler, request)
    request.num_preemptions = 1
    request.num_computed_tokens = 0
    scheduler.kv_cache_manager.blocks = [[], []]

    observe_preemption(scheduler, request, before=before)

    result = request_runtime_report(scheduler, request.request_id)
    event = result["report"]["events"][0]
    assert event["kind"] == "preemption"
    assert event["before"]["phase"] == "decode"
    assert event["after"]["phase"] == "queued"
    assert event["after"]["preemptions"] == 1
    assert result["registry"] == {
        "capacity": 4096,
        "max_events_per_request": 32,
        "retained_requests": 1,
        "active_requests": 1,
        "terminal_requests": 0,
        "unretained_active_requests": 0,
        "evicted_active_reports": 0,
    }
