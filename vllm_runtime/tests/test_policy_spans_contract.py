import asyncio
from types import SimpleNamespace

from art_vllm_runtime import policy_spans
import pytest


@pytest.mark.asyncio
async def test_holder_update_is_atomic_and_failure_blocks_new_admission() -> None:
    coordinator = policy_spans.LoraUpdateCoordinator()
    policy_1 = SimpleNamespace(lora_name="active", policy_version=1, update_seq=1)
    await coordinator.declare_initial("active", policy_1)

    existing = await coordinator.acquire("active")
    assert existing.lora_request is policy_1
    update_2 = asyncio.create_task(coordinator.begin_update("active"))
    await asyncio.sleep(0)
    assert not update_2.done()

    await existing.release()
    update_seq_2 = await update_2
    assert update_seq_2 == 2
    await coordinator.fail_update("active", update_seq_2)

    blocked_admission = asyncio.create_task(coordinator.acquire("active"))
    await asyncio.sleep(0)
    assert not blocked_admission.done()

    update_seq_3 = await coordinator.begin_update("active")
    policy_3 = SimpleNamespace(
        lora_name="active", policy_version=3, update_seq=update_seq_3
    )
    await coordinator.commit_update("active", policy_3)
    recovered = await blocked_admission
    assert recovered.lora_request is policy_3
    await recovered.release()


def test_prompt_and_decode_spans_report_the_executing_forward_policy() -> None:
    output = SimpleNamespace(req_ids=["request"], sampled_token_ids=[[42]])
    context = {
        "request": {
            "policy_version": 7,
            "lora_slot": "active",
            "update_seq": 8,
            "prompt_span": (1, 5),
            "prompt_tokens": 5,
        }
    }

    policy_spans._attach_policy_spans_to_model_output(output, context)

    assert output.art_policy_token_spans == {
        "request": [
            {
                "start_token": 0,
                "end_token": 1,
                "policy_version": 7,
                "lora_slot": "active",
                "update_seq": 8,
            }
        ]
    }
    assert output.art_prompt_policy_token_spans == {
        "request": [
            {
                "start_token": 1,
                "end_token": 5,
                "policy_version": 7,
                "lora_slot": "active",
                "update_seq": 8,
                policy_spans._CACHED_PROMPT_POLICY_SPAN_FIELD: False,
            }
        ]
    }


def test_chunked_prefill_keeps_policy_boundaries() -> None:
    accumulated: list[dict[str, object]] = []
    policy_spans._append_absolute_prompt_spans(
        accumulated,
        [
            {
                "start_token": 1,
                "end_token": 3,
                "policy_version": 1,
                "lora_slot": "active",
                "update_seq": 1,
            }
        ],
    )
    policy_spans._append_absolute_prompt_spans(
        accumulated,
        [
            {
                "start_token": 3,
                "end_token": 5,
                "policy_version": 2,
                "lora_slot": "active",
                "update_seq": 2,
            }
        ],
    )

    assert [(span["start_token"], span["end_token"]) for span in accumulated] == [
        (1, 3),
        (3, 5),
    ]
    assert [span["policy_version"] for span in accumulated] == [1, 2]


def test_new_requests_get_policy_generation_specific_cache_identity() -> None:
    first = {"cache_salt": "caller"}
    second = {"cache_salt": "caller"}

    policy_spans._set_policy_cache_salt(
        first,
        lora_slot="active",
        policy_version=1,
        update_seq=1,
    )
    policy_spans._set_policy_cache_salt(
        second,
        lora_slot="active",
        policy_version=2,
        update_seq=2,
    )

    assert first["cache_salt"] != second["cache_salt"]
    assert str(first["cache_salt"]).startswith("caller|")
    assert str(second["cache_salt"]).startswith("caller|")
