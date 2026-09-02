import asyncio
import time
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


def test_preflight_keeps_target_admissions_open_until_publication() -> None:
    async def exercise() -> None:
        coordinator = policy_spans.LoraUpdateCoordinator()
        slot = "run:active"
        initial = policy_spans.PolicyLoRARequest(
            lora_name=slot,
            lora_int_id=1,
            lora_path="/generation-a",
            generation_id="generation-a",
            policy_version=1,
            update_seq=1,
        )
        await coordinator.declare_initial(slot, initial)
        ticket = await coordinator.acquire(slot)

        await coordinator.preflight_publication(
            slot, expected_generation_id="generation-a"
        )
        publication = asyncio.create_task(
            coordinator.begin_publication(slot, expected_generation_id="generation-a")
        )
        await asyncio.sleep(0)
        assert not publication.done()

        await ticket.release()
        sequence = await publication
        assert sequence == 2
        await coordinator.cancel_update(slot, sequence)

    asyncio.run(exercise())


def test_prepared_lora_uses_native_cpu_loader_before_install(tmp_path) -> None:
    class AdapterManager:
        capacity = 2

        def __init__(self) -> None:
            self.adapters = {7: SimpleNamespace(id=7, generation="old")}
            self.active: list[int] = []

        def __len__(self) -> int:
            return len(self.adapters)

        def remove_adapter(self, adapter_id: int) -> bool:
            return self.adapters.pop(adapter_id, None) is not None

        def add_adapter(self, adapter: object) -> bool:
            adapter_id = int(adapter.id)  # type: ignore[attr-defined]
            self.adapters[adapter_id] = adapter
            return True

        def activate_adapter(self, adapter_id: int) -> bool:
            self.active.append(adapter_id)
            return True

    class Manager:
        def __init__(self) -> None:
            self._adapter_manager = AdapterManager()
            self.loader_thread = ""

        def _load_adapter(self, request: object) -> object:
            from threading import current_thread

            self.loader_thread = current_thread().name
            return SimpleNamespace(
                id=request.lora_int_id,  # type: ignore[attr-defined]
                generation="new",
            )

        def pin_adapter(self, adapter_id: int) -> bool:
            return adapter_id in self._adapter_manager.adapters

    manager = Manager()
    worker = SimpleNamespace(model_runner=SimpleNamespace(lora_manager=manager))
    operation_id = "prepared-worker-test"
    adapter_path = tmp_path / "generation-b"
    adapter_path.mkdir()
    (adapter_path / "adapter_model.safetensors").write_bytes(b"m" * 4096)
    (adapter_path / "adapter_config.json").write_bytes(b"c" * 512)
    staged = {
        "path": str(adapter_path),
        "source_identity": "generation-b",
        "layout": "peft_safetensors_v1",
        "model_bytes": 4096,
        "config_bytes": 512,
    }
    prepare_request = policy_spans.PolicyLoRARequest(
        lora_name="run:active",
        lora_int_id=7,
        lora_path=staged["path"],
        generation_id="generation-b",
        policy_version=2,
        update_seq=0,
    )
    final_request = policy_spans.PolicyLoRARequest(
        **{
            **policy_spans.policy_lora_request_payload(prepare_request),
            "update_seq": 2,
        }
    )
    policy_spans._WORKER_LORA_POLICY_BY_ID[7] = {
        "generation_id": "generation-a",
        "policy_version": 1,
        "lora_slot": "run:active",
        "lora_path": "/adapter/generation-a",
        "update_seq": 1,
    }

    try:
        state = policy_spans._prepare_worker_lora(
            worker,
            operation_id,
            policy_spans.policy_lora_request_payload(prepare_request),
            staged,
        )
        assert state["state"] in {"preparing", "ready"}
        deadline = time.monotonic() + 1
        while True:
            state = policy_spans._worker_lora_status(
                operation_id,
                policy_spans.policy_lora_request_payload(prepare_request),
                staged,
            )
            if state == {"state": "ready"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.001)

        changed_payload = policy_spans.policy_lora_request_payload(prepare_request)
        changed_payload["is_3d_lora_weight"] = True
        with pytest.raises(RuntimeError, match="changed identity"):
            policy_spans._worker_lora_status(
                operation_id,
                changed_payload,
                staged,
            )

        acknowledgement = policy_spans._commit_worker_lora(
            worker,
            operation_id,
            policy_spans.policy_lora_request_payload(final_request),
            staged,
        )
        assert manager.loader_thread.startswith("art-lora-prepare")
        assert manager._adapter_manager.adapters[7].generation == "new"
        assert manager._adapter_manager.active == [7]
        assert acknowledgement["current"]["generation_id"] == "generation-b"
        assert acknowledgement["current"]["update_seq"] == 2

        changed_staged = {**staged, "model_bytes": 4095}
        with pytest.raises(RuntimeError, match="changed its byte layout"):
            policy_spans._prepare_worker_lora(
                worker,
                "prepared-worker-size-conflict",
                policy_spans.policy_lora_request_payload(prepare_request),
                changed_staged,
            )
    finally:
        policy_spans._abort_worker_lora(operation_id)
        policy_spans._WORKER_LORA_POLICY_BY_ID.pop(7, None)


def test_engine_commit_is_one_queue_serialized_descriptor_only_transaction() -> None:
    request = policy_spans.PolicyLoRARequest(
        lora_name="run:active",
        lora_int_id=7,
        lora_path="/adapter/generation-b",
        generation_id="generation-b",
        policy_version=2,
        update_seq=2,
    )
    payload = policy_spans.policy_lora_request_payload(request)
    staged = {
        "path": request.lora_path,
        "source_identity": request.generation_id,
        "layout": "peft_safetensors_v1",
        "model_bytes": 4096,
        "config_bytes": 512,
    }
    previous = {
        "generation_id": "generation-a",
        "policy_version": 1,
        "lora_slot": request.lora_name,
        "lora_path": "/adapter/generation-a",
        "update_seq": 1,
    }
    current = {
        "generation_id": request.generation_id,
        "policy_version": request.policy_version,
        "lora_slot": request.lora_name,
        "lora_path": request.lora_path,
        "update_seq": request.update_seq,
    }

    class Core:
        def __init__(self, *, fail: bool = False) -> None:
            self.scheduler = SimpleNamespace(requests={})
            self.pauses: list[tuple[str, bool]] = []
            self.rpc: tuple[str, tuple[object, ...]] | None = None
            self.fail = fail

        def pause_scheduler(self, mode: str, abort: bool) -> None:
            self.pauses.append((mode, abort))

        def collective_rpc(
            self, method: str, *, args: tuple[object, ...]
        ) -> list[dict[str, object]]:
            self.rpc = (method, args)
            if self.fail:
                raise RuntimeError("collective failed")
            return [{"loaded": True, "previous": previous, "current": current}]

    core = Core()
    result = policy_spans._commit_staged_policy_lora_update(
        core, "operation-2", payload, staged
    )

    assert core.pauses == []
    assert core.rpc == (
        "art_commit_staged_lora_policy",
        ("operation-2", payload, staged),
    )
    assert result == {
        "workers": 1,
        "cache_transition": {"updated_requests": 0, "continued_requests": 0},
    }

    failed = Core(fail=True)
    with pytest.raises(RuntimeError, match="collective failed"):
        policy_spans._commit_staged_policy_lora_update(
            failed, "operation-3", payload, staged
        )
    assert failed.pauses == [("abort", True)]
