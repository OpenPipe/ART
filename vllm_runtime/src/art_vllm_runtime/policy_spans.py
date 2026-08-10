"""Policy-token span tracking for ART's owned vLLM runtime.

The hot path intentionally uses plain dict/list payloads. ART validates them
with Pydantic after the OpenAI response crosses back into the training process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import importlib
import re
import sys
from typing import Any, AsyncIterator

import msgspec
import numpy as np
import torch
from vllm.lora.request import LoRARequest

POLICY_TOKEN_SPANS_FIELD = "policy_token_spans"
ART_POLICY_TOKEN_SPANS_FIELD = "art_policy_token_spans"
_PARENT_POLICY_TOKEN_SPANS_FIELD = "_art_policy_token_spans_by_choice"

_CURRENT_ENGINE_POLICY_SPANS: dict[str, list[dict[str, Any]]] = {}
_WORKER_LORA_POLICY_BY_ID: dict[int, dict[str, Any]] = {}
_POLICY_CACHE_SALT_PREFIX = "art_policy_cache_salt="
_POLICY_CACHE_SALT_MARKER = f"|{_POLICY_CACHE_SALT_PREFIX}"
_POLICY_CACHE_SALT_VERSION = "v1:"
_LORA_UPDATE_COORDINATOR_FIELD = "_art_lora_update_coordinator"
_EXECUTING_POLICY_CONTEXT_FIELD = "_art_executing_policy_context"
_POLICY_EXECUTION_MARKER_FIELD = "_art_policy_execution_marker"
_POLICY_HISTORY_BASE_FIELD = "_art_policy_history_before_current"


class _RequestAdmissionLease:
    __slots__ = ("closed", "context", "lora_request", "lora_slot", "request_id")

    def __init__(self) -> None:
        self.closed = False
        self.context: Any | None = None
        self.lora_request: Any | None = None
        self.lora_slot: str | None = None
        self.request_id: str | None = None


_REQUEST_ADMISSION_LEASE: ContextVar[_RequestAdmissionLease | None] = ContextVar(
    "art_request_admission_lease", default=None
)

_MODEL_RUNNER_OUTPUT_MODULES = (
    "vllm.v1.outputs",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.async_utils",
    "vllm.v1.worker.gpu_worker",
    "vllm.v1.core.sched.scheduler",
)

_GPU_MODEL_RUNNER_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
)


class PolicyLoRARequest(LoRARequest, omit_defaults=True, array_like=True):  # type: ignore[call-arg]
    """LoRA request carrying ART's exact executing-policy identity."""

    policy_version: int = 0
    update_seq: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.policy_version < 0 or self.update_seq < 0:
            raise ValueError("policy_version and update_seq must be non-negative")


def patch_policy_token_spans() -> None:
    _patch_model_runner_output_type()
    _patch_engine_core_output_type()
    _patch_worker_policy_span_capture()
    _patch_scheduler_policy_span_transport()
    _patch_output_processor_policy_span_accumulation()
    _patch_openai_response_policy_spans()
    _patch_lora_alias_resolution()
    _patch_engine_request_admission()
    _patch_load_inplace_storage()
    _patch_policy_lora_update_rpc()


class _SlotAdmissionState:
    __slots__ = (
        "condition",
        "active_admissions",
        "blocked",
        "lora_request",
        "next_update_seq",
        "pending_update_seq",
        "poisoned",
        "update_active",
    )

    def __init__(self) -> None:
        self.condition = asyncio.Condition()
        self.active_admissions = 0
        self.blocked = False
        self.lora_request: Any | None = None
        self.next_update_seq = 1
        self.pending_update_seq: int | None = None
        self.poisoned = False
        self.update_active = False


class LoraUpdateCoordinator:
    """Linearizes request admission with mutable LoRA slot updates."""

    def __init__(self) -> None:
        self._states: dict[str, _SlotAdmissionState] = {}

    def _state(self, lora_slot: str) -> _SlotAdmissionState:
        return self._states.setdefault(lora_slot, _SlotAdmissionState())

    @asynccontextmanager
    async def admission(self, lora_slot: str) -> AsyncIterator[Any | None]:
        state = self._state(lora_slot)
        async with state.condition:
            await state.condition.wait_for(lambda: not state.blocked)
            state.active_admissions += 1
        try:
            yield state.lora_request
        finally:
            async with state.condition:
                state.active_admissions -= 1
                state.condition.notify_all()

    async def declare_initial(
        self, lora_slot: str, lora_request: PolicyLoRARequest
    ) -> None:
        state = self._state(lora_slot)
        async with state.condition:
            if (
                state.active_admissions
                or state.update_active
                or state.lora_request is not None
            ):
                raise RuntimeError(f"LoRA slot {lora_slot!r} is already active")
            if lora_request.update_seq <= 0:
                raise ValueError(
                    "initial mutable LoRA policy requires a positive sequence"
                )
            if lora_request.lora_name != lora_slot:
                raise ValueError("initial LoRA policy does not match its slot")
            state.lora_request = lora_request
            state.next_update_seq = lora_request.update_seq + 1

    async def begin_update(self, lora_slot: str) -> int:
        state = self._state(lora_slot)
        async with state.condition:
            await state.condition.wait_for(lambda: not state.update_active)
            state.update_active = True
            state.blocked = True
            update_seq = state.next_update_seq
            state.next_update_seq += 1
            state.pending_update_seq = update_seq
            try:
                await state.condition.wait_for(lambda: state.active_admissions == 0)
            except BaseException:
                state.update_active = False
                state.blocked = state.poisoned
                state.pending_update_seq = None
                state.condition.notify_all()
                raise
            return update_seq

    async def commit_update(
        self,
        lora_slot: str,
        lora_request: PolicyLoRARequest,
    ) -> None:
        state = self._state(lora_slot)
        async with state.condition:
            self._require_pending(state, lora_slot, lora_request.update_seq)
            state.lora_request = lora_request
            state.update_active = False
            state.blocked = False
            state.poisoned = False
            state.pending_update_seq = None
            state.condition.notify_all()

    async def cancel_update(self, lora_slot: str, update_seq: int) -> None:
        state = self._state(lora_slot)
        async with state.condition:
            self._require_pending(state, lora_slot, update_seq)
            state.update_active = False
            state.blocked = state.poisoned
            state.pending_update_seq = None
            state.condition.notify_all()

    async def fail_update(self, lora_slot: str, update_seq: int) -> None:
        state = self._state(lora_slot)
        async with state.condition:
            self._require_pending(state, lora_slot, update_seq)
            state.update_active = False
            # A worker may already hold new weights. This slot stays poisoned.
            state.blocked = True
            state.poisoned = True
            state.pending_update_seq = None
            state.condition.notify_all()

    @staticmethod
    def _require_pending(
        state: _SlotAdmissionState, lora_slot: str, update_seq: int
    ) -> None:
        if not state.update_active or state.pending_update_seq != update_seq:
            raise RuntimeError(
                f"LoRA slot {lora_slot!r} has no update {update_seq} in progress"
            )


def lora_update_coordinator(models: Any, engine_client: Any) -> LoraUpdateCoordinator:
    coordinator = getattr(models, _LORA_UPDATE_COORDINATOR_FIELD, None)
    if coordinator is None:
        coordinator = getattr(engine_client, _LORA_UPDATE_COORDINATOR_FIELD, None)
    if coordinator is None:
        coordinator = LoraUpdateCoordinator()
    setattr(models, _LORA_UPDATE_COORDINATOR_FIELD, coordinator)
    setattr(engine_client, _LORA_UPDATE_COORDINATOR_FIELD, coordinator)
    return coordinator


async def declare_initial_lora_policy(
    models: Any,
    engine_client: Any,
    *,
    lora_slot: str,
    policy_version: int,
) -> None:
    loaded = models.lora_requests.get(lora_slot)
    if loaded is None:
        raise RuntimeError(f"Initial LoRA slot {lora_slot!r} is not loaded")
    request = PolicyLoRARequest(
        lora_name=loaded.lora_name,
        lora_int_id=loaded.lora_int_id,
        lora_path=loaded.lora_path,
        base_model_name=loaded.base_model_name,
        tensorizer_config_dict=loaded.tensorizer_config_dict,
        is_3d_lora_weight=loaded.is_3d_lora_weight,
        policy_version=policy_version,
        update_seq=1,
    )
    report = await engine_client.engine_core.call_utility_async(
        "art_declare_loaded_lora_policy", policy_lora_request_payload(request)
    )
    if int(report.get("workers", 0)) <= 0:
        raise RuntimeError("Initial LoRA policy declaration reached no workers")
    models.lora_requests[lora_slot] = request
    publish_lora_slot_policy(
        models,
        lora_slot=lora_slot,
        policy_version=policy_version,
        update_seq=request.update_seq,
    )
    await lora_update_coordinator(models, engine_client).declare_initial(
        lora_slot, request
    )


def _patch_model_runner_output_type() -> None:
    import vllm.v1.outputs as outputs_mod

    if getattr(outputs_mod, "_art_policy_token_spans_model_runner_patched", False):
        return

    BaseModelRunnerOutput = outputs_mod.ModelRunnerOutput

    @dataclass
    class ModelRunnerOutput(BaseModelRunnerOutput):  # type: ignore[misc, valid-type]
        # This object crosses the worker->scheduler process boundary. Dynamic
        # attrs are not part of that transport contract, so ART span metadata
        # must be a declared field.
        art_policy_token_spans: dict[str, list[dict[str, Any]]] | None = None

    ModelRunnerOutput.__module__ = outputs_mod.__name__
    ModelRunnerOutput.__qualname__ = "ModelRunnerOutput"
    outputs_mod.ModelRunnerOutput = ModelRunnerOutput
    outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}
    )
    for module_name in _MODEL_RUNNER_OUTPUT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "ModelRunnerOutput", ModelRunnerOutput)
            if hasattr(module, "EMPTY_MODEL_RUNNER_OUTPUT"):
                setattr(
                    module,
                    "EMPTY_MODEL_RUNNER_OUTPUT",
                    outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT,
                )
    setattr(outputs_mod, "_art_policy_token_spans_model_runner_patched", True)


def register_lora_alias(
    models: Any,
    *,
    public_model_name: str,
    lora_slot: str,
) -> None:
    aliases = getattr(models, "_art_lora_aliases", None)
    if aliases is None:
        aliases = {}
        setattr(models, "_art_lora_aliases", aliases)
    aliases[public_model_name] = lora_slot


def publish_lora_slot_policy(
    models: Any,
    *,
    lora_slot: str,
    policy_version: int,
    update_seq: int,
) -> None:
    identities = getattr(models, "_art_lora_slot_policy_identities", None)
    if identities is None:
        identities = {}
        setattr(models, "_art_lora_slot_policy_identities", identities)
    identities[lora_slot] = (int(policy_version), int(update_seq))


def _resolve_lora_alias(models: Any, model_name: str | None) -> Any | None:
    if not model_name:
        return None
    slot = getattr(models, "_art_lora_aliases", {}).get(model_name)
    if not slot:
        return None
    return models.lora_requests.get(slot)


def _slot_policy_identity(models: Any, lora_slot: str) -> tuple[int, int] | None:
    identity = getattr(models, "_art_lora_slot_policy_identities", {}).get(lora_slot)
    if identity is None:
        return None
    policy_version, update_seq = identity
    return int(policy_version), int(update_seq)


def _strip_policy_cache_salt(cache_salt: str | None) -> str | None:
    if not cache_salt:
        return None
    if cache_salt.startswith(_POLICY_CACHE_SALT_PREFIX):
        return None
    base, marker, _policy = cache_salt.partition(_POLICY_CACHE_SALT_MARKER)
    if marker:
        return base or None
    return cache_salt


def _policy_history_from_cache_salt(cache_salt: str | None) -> str | None:
    if not cache_salt:
        return None
    if cache_salt.startswith(_POLICY_CACHE_SALT_PREFIX):
        value = cache_salt.removeprefix(_POLICY_CACHE_SALT_PREFIX)
    else:
        _base, marker, value = cache_salt.partition(_POLICY_CACHE_SALT_MARKER)
        if not marker:
            return None
    if not value.startswith(_POLICY_CACHE_SALT_VERSION):
        raise RuntimeError("Unsupported ART policy cache-salt format")
    digest = value.removeprefix(_POLICY_CACHE_SALT_VERSION)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("Malformed ART policy cache-salt digest")
    return digest


def _extend_policy_history(
    previous_digest: str | None,
    *,
    lora_slot: str,
    policy_version: int,
    update_seq: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"art-policy-history-v1\0")
    if previous_digest is not None:
        digest.update(bytes.fromhex(previous_digest))
    digest.update(lora_slot.encode())
    digest.update(b"\0")
    digest.update(str(policy_version).encode())
    digest.update(b"\0")
    digest.update(str(update_seq).encode())
    return digest.hexdigest()


def _policy_cache_salt(
    *,
    history_digest: str,
    user_cache_salt: str | None,
) -> str:
    policy_salt = f"{_POLICY_CACHE_SALT_VERSION}{history_digest}"
    if user_cache_salt:
        return f"{user_cache_salt}{_POLICY_CACHE_SALT_MARKER}{policy_salt}"
    return f"{_POLICY_CACHE_SALT_PREFIX}{policy_salt}"


def _set_policy_cache_salt(
    request: Any,
    *,
    lora_slot: str,
    policy_version: int,
    update_seq: int,
    previous_digest: str | None = None,
) -> None:
    current_salt = (
        request.get("cache_salt")
        if isinstance(request, dict)
        else getattr(request, "cache_salt", None)
    )
    user_cache_salt = _strip_policy_cache_salt(current_salt)
    cache_salt = _policy_cache_salt(
        history_digest=_extend_policy_history(
            previous_digest,
            lora_slot=lora_slot,
            policy_version=policy_version,
            update_seq=update_seq,
        ),
        user_cache_salt=user_cache_salt,
    )
    if isinstance(request, dict):
        request["cache_salt"] = cache_salt
    else:
        request.cache_salt = cache_salt


def _apply_lora_alias_policy_cache_salt(
    models: Any,
    request: Any,
    lora_request: Any,
) -> None:
    lora_slot = str(lora_request.lora_name)
    identity = _slot_policy_identity(models, lora_slot)
    if identity is None:
        return
    policy_version, update_seq = identity
    _set_policy_cache_salt(
        request,
        lora_slot=lora_slot,
        policy_version=policy_version,
        update_seq=update_seq,
    )


def _set_pydantic_extra(model: Any, key: str, value: Any) -> None:
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        extra[key] = value
        return
    setattr(model, key, value)


def _patch_engine_core_output_type() -> None:
    import vllm.v1.engine as engine_mod
    from vllm.v1.metrics.stats import PrefillStats, SchedulerStats
    from vllm.v1.outputs import LogprobsLists, LogprobsTensors
    from vllm.v1.serial_utils import UtilityResult

    if getattr(engine_mod, "_art_policy_token_spans_patched", False):
        return

    FinishReason = engine_mod.FinishReason
    EngineCoreEvent = engine_mod.EngineCoreEvent

    class EngineCoreOutput(  # type: ignore[call-arg]
        msgspec.Struct,
        array_like=True,
        omit_defaults=True,
        gc=False,
    ):
        request_id: str
        new_token_ids: list[int]

        new_logprobs: LogprobsLists | None = None
        new_prompt_logprobs_tensors: LogprobsTensors | None = None

        pooling_output: torch.Tensor | None = None

        finish_reason: FinishReason | None = None
        stop_reason: int | str | None = None
        events: list[EngineCoreEvent] | None = None
        kv_transfer_params: dict[str, Any] | None = None

        trace_headers: Mapping[str, str] | None = None

        prefill_stats: PrefillStats | None = None

        routed_experts: np.ndarray | None = None
        num_nans_in_logits: int = 0
        art_policy_token_spans: list[dict[str, Any]] | None = None

        @property
        def finished(self) -> bool:
            return self.finish_reason is not None

    class UtilityOutput(  # type: ignore[call-arg]
        msgspec.Struct,
        array_like=True,
        gc=False,
    ):
        call_id: int
        failure_message: str | None = None
        result: UtilityResult | None = None

    class EngineCoreOutputs(  # type: ignore[call-arg]
        msgspec.Struct,
        array_like=True,
        omit_defaults=True,
        gc=False,
    ):
        engine_index: int = 0
        outputs: list[EngineCoreOutput] = []
        scheduler_stats: SchedulerStats | None = None
        timestamp: float = 0.0
        utility_output: UtilityOutput | None = None
        finished_requests: set[str] | None = None
        wave_complete: int | None = None
        start_wave: int | None = None

    EngineCoreOutput.__module__ = engine_mod.__name__
    UtilityOutput.__module__ = engine_mod.__name__
    EngineCoreOutputs.__module__ = engine_mod.__name__
    engine_mod.EngineCoreOutput = EngineCoreOutput
    engine_mod.UtilityOutput = UtilityOutput
    engine_mod.EngineCoreOutputs = EngineCoreOutputs
    for module_name in (
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.engine.core",
        "vllm.v1.engine.output_processor",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "EngineCoreOutput", EngineCoreOutput)
            setattr(module, "EngineCoreOutputs", EngineCoreOutputs)
            if hasattr(module, "UtilityOutput"):
                setattr(module, "UtilityOutput", UtilityOutput)
    setattr(engine_mod, "_art_policy_token_spans_patched", True)


def _patch_worker_policy_span_capture() -> None:
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.v1.worker.gpu.async_utils import AsyncOutput

    original_add_adapter = LRUCacheWorkerLoRAManager.add_adapter
    if not getattr(original_add_adapter, "__art_policy_spans_patched__", False):

        def add_adapter(self: Any, lora_request: Any) -> bool:
            already_loaded = lora_request.lora_int_id in self.list_adapters()
            loaded = original_add_adapter(self, lora_request)
            if lora_request.load_inplace or not already_loaded:
                _record_worker_lora_policy(lora_request)
            return loaded

        add_adapter.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        LRUCacheWorkerLoRAManager.add_adapter = add_adapter  # type: ignore[method-assign]

    for module_name in _GPU_MODEL_RUNNER_MODULES:
        module = importlib.import_module(module_name)
        gpu_model_runner_cls = module.GPUModelRunner

        original_execute_model = gpu_model_runner_cls.execute_model
        if not getattr(original_execute_model, "__art_policy_spans_patched__", False):

            def make_execute_model(original: Any):
                def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
                    output = original(self, *args, **kwargs)
                    # The input batch is current only after execute_model, and the
                    # next serial worker RPC may replace this adapter before sampling.
                    context = _policy_context_from_runner(self)
                    if getattr(self, "execute_model_state", None) is not None:
                        setattr(self, _EXECUTING_POLICY_CONTEXT_FIELD, context)
                    elif context and hasattr(output, "req_ids"):
                        _attach_policy_spans_to_model_output(output, context)
                    return output

                return execute_model

            execute_model = make_execute_model(original_execute_model)
            execute_model.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
            gpu_model_runner_cls.execute_model = execute_model  # type: ignore[method-assign]

        original_sample_tokens = gpu_model_runner_cls.sample_tokens
        if getattr(original_sample_tokens, "__art_policy_spans_patched__", False):
            continue

        def make_sample_tokens(original: Any):
            def sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
                context = getattr(self, _EXECUTING_POLICY_CONTEXT_FIELD, None)
                try:
                    output = original(self, *args, **kwargs)
                finally:
                    if hasattr(self, _EXECUTING_POLICY_CONTEXT_FIELD):
                        delattr(self, _EXECUTING_POLICY_CONTEXT_FIELD)
                if context and output is not None:
                    if hasattr(output, "get_output"):
                        _attach_policy_span_context_to_sample_output(output, context)
                    else:
                        _attach_policy_spans_to_model_output(output, context)
                return output

            return sample_tokens

        sample_tokens = make_sample_tokens(original_sample_tokens)

        sample_tokens.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        gpu_model_runner_cls.sample_tokens = sample_tokens  # type: ignore[method-assign]

    async_output_classes = [AsyncOutput]
    active_runner = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    if active_runner is not None and hasattr(
        active_runner, "AsyncGPUModelRunnerOutput"
    ):
        async_output_classes.append(active_runner.AsyncGPUModelRunnerOutput)

    for async_output_cls in async_output_classes:
        original_get_output = async_output_cls.get_output
        if getattr(original_get_output, "__art_policy_spans_patched__", False):
            continue

        def make_get_output(original: Any):
            def get_output(self: Any) -> Any:
                output = original(self)
                context = _policy_span_context_from_sample_output(self)
                if context:
                    _attach_policy_spans_to_model_output(output, context)
                return output

            return get_output

        get_output = make_get_output(original_get_output)

        get_output.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        async_output_cls.get_output = get_output  # type: ignore[method-assign]


def _patch_scheduler_policy_span_transport() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original_update = Scheduler.update_from_output
    if getattr(original_update, "__art_policy_spans_patched__", False):
        return

    def update_from_output(self: Any, scheduler_output: Any, model_runner_output: Any):
        outputs_by_client = original_update(self, scheduler_output, model_runner_output)
        spans_by_req = getattr(model_runner_output, ART_POLICY_TOKEN_SPANS_FIELD, None)
        if not spans_by_req:
            return outputs_by_client
        for client_outputs in outputs_by_client.values():
            for output in client_outputs.outputs:
                spans = spans_by_req.get(output.request_id)
                if not spans:
                    continue
                output.art_policy_token_spans = _trim_step_spans(
                    spans, len(output.new_token_ids)
                )
        return outputs_by_client

    update_from_output.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    Scheduler.update_from_output = update_from_output  # type: ignore[method-assign]


def _patch_output_processor_policy_span_accumulation() -> None:
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState

    original_process_outputs = OutputProcessor.process_outputs
    if not getattr(original_process_outputs, "__art_policy_spans_patched__", False):

        def process_outputs(
            self: Any, engine_core_outputs: list[Any], *args: Any, **kwargs: Any
        ):
            global _CURRENT_ENGINE_POLICY_SPANS
            previous = _CURRENT_ENGINE_POLICY_SPANS
            _CURRENT_ENGINE_POLICY_SPANS = _engine_core_policy_spans_by_request(
                engine_core_outputs
            )
            try:
                return original_process_outputs(
                    self, engine_core_outputs, *args, **kwargs
                )
            finally:
                _CURRENT_ENGINE_POLICY_SPANS = previous

        process_outputs.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        OutputProcessor.process_outputs = process_outputs  # type: ignore[method-assign]

    original_make = RequestState.make_request_output
    if getattr(original_make, "__art_policy_spans_patched__", False):
        return

    def make_request_output(
        self: Any,
        new_token_ids: list[int],
        *args: Any,
        **kwargs: Any,
    ):
        _append_current_policy_spans(self, len(new_token_ids))
        spans = getattr(self, ART_POLICY_TOKEN_SPANS_FIELD, None)
        parent_req = getattr(self, "parent_req", None)
        if spans and parent_req is not None:
            spans_by_choice = getattr(
                parent_req, _PARENT_POLICY_TOKEN_SPANS_FIELD, None
            )
            if spans_by_choice is None:
                spans_by_choice = {}
                setattr(parent_req, _PARENT_POLICY_TOKEN_SPANS_FIELD, spans_by_choice)
            spans_by_choice[int(getattr(self, "request_index", 0))] = [
                dict(span) for span in spans
            ]

        request_output = original_make(self, new_token_ids, *args, **kwargs)
        if request_output is not None and hasattr(request_output, "outputs"):
            spans_by_choice = (
                getattr(parent_req, _PARENT_POLICY_TOKEN_SPANS_FIELD, {})
                if parent_req is not None
                else {int(getattr(self, "request_index", 0)): spans}
            )
            _record_request_output_spans(request_output, spans_by_choice)
        return request_output

    make_request_output.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    RequestState.make_request_output = make_request_output  # type: ignore[method-assign]


def _patch_openai_response_policy_spans() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    original_full = OpenAIServingChat.chat_completion_full_generator
    if getattr(original_full, "__art_policy_spans_patched__", False):
        return

    async def chat_completion_full_generator(
        self: Any,
        request: Any,
        result_generator: Any,
        request_id: str,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        final_res = None

        async def tracked_result_generator():
            nonlocal final_res
            async for res in result_generator:
                final_res = res
                yield res

        response = await original_full(
            self,
            request,
            tracked_result_generator(),
            request_id,
            model_name,
            *args,
            **kwargs,
        )
        if isinstance(response, ChatCompletionResponse):
            spans_by_choice = _policy_spans_by_choice_from_final_output(final_res)
            for choice in response.choices:
                spans = spans_by_choice.get(choice.index)
                if spans:
                    _set_pydantic_extra(choice, POLICY_TOKEN_SPANS_FIELD, spans)
            if _resolve_lora_alias(self.models, getattr(request, "model", None)):
                response.model = request.model
        return response

    chat_completion_full_generator.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator  # type: ignore[method-assign]


def _patch_lora_alias_resolution() -> None:
    try:
        module = importlib.import_module("vllm.entrypoints.openai.engine.serving")
        serving_base = module.OpenAIServing
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.entrypoints.openai.engine.serving":
            raise
        module = importlib.import_module("vllm.entrypoints.generate.base.serving")
        serving_base = module.GenerateBaseServing

    original_init = serving_base.__init__
    if not getattr(original_init, "__art_lora_update_patched__", False):

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            lora_update_coordinator(self.models, self.engine_client)

        __init__.__art_lora_update_patched__ = True  # type: ignore[attr-defined]
        serving_base.__init__ = __init__

    original_check = serving_base._check_model
    if not getattr(original_check, "__art_policy_spans_patched__", False):

        async def _check_model(self: Any, request: Any) -> Any:
            lora_request = _resolve_lora_alias(
                self.models, getattr(request, "model", None)
            )
            if lora_request is not None:
                from art_vllm_runtime.metrics import record_policy_cache_salt_audit

                _apply_lora_alias_policy_cache_salt(self.models, request, lora_request)
                record_policy_cache_salt_audit(
                    lora_request=True,
                    salted=bool(getattr(request, "cache_salt", None)),
                )
                return None
            return await original_check(self, request)

        _check_model.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        serving_base._check_model = _check_model

    original_maybe = serving_base._maybe_get_adapters
    if getattr(original_maybe, "__art_policy_spans_patched__", False):
        return

    def _maybe_get_adapters(
        self: Any,
        request: Any,
        supports_default_mm_loras: bool = False,
    ) -> Any:
        lora_request = _resolve_lora_alias(self.models, getattr(request, "model", None))
        if lora_request is not None:
            _apply_lora_alias_policy_cache_salt(self.models, request, lora_request)
            return lora_request
        return original_maybe(
            self,
            request,
            supports_default_mm_loras=supports_default_mm_loras,
        )

    _maybe_get_adapters.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    serving_base._maybe_get_adapters = _maybe_get_adapters


def _patch_engine_request_admission() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM

    # Long-prompt input processing is policy-independent; only serialize the
    # complete final fanout and engine enqueue with in-place weight updates.
    original_add_request = AsyncLLM.add_request
    original_enqueue = AsyncLLM._add_request
    if getattr(original_add_request, "__art_lora_update_patched__", False):
        return

    @wraps(original_add_request)
    async def add_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        lease = _RequestAdmissionLease()
        token = _REQUEST_ADMISSION_LEASE.set(lease)
        try:
            return await original_add_request(self, *args, **kwargs)
        except BaseException:
            if lease.request_id is not None:
                await asyncio.shield(self.abort(lease.request_id, internal=True))
            raise
        finally:
            lease.closed = True
            try:
                if lease.context is not None:
                    await asyncio.shield(lease.context.__aexit__(None, None, None))
            finally:
                _REQUEST_ADMISSION_LEASE.reset(token)

    async def _add_request(
        self: Any,
        request: Any,
        prompt: str | None,
        parent_req: Any,
        index: int,
        queue: Any,
    ) -> Any:
        lease = _REQUEST_ADMISSION_LEASE.get()
        if lease is not None and not lease.closed:
            request_id = parent_req.request_id if parent_req else request.request_id
            if lease.request_id not in (None, request_id):
                raise RuntimeError("One admission lease received multiple requests")
            lease.request_id = request_id
        lora_request = request.lora_request
        coordinator = getattr(self, _LORA_UPDATE_COORDINATOR_FIELD, None)
        if coordinator is None or lora_request is None:
            return await original_enqueue(
                self, request, prompt, parent_req, index, queue
            )
        lora_slot = str(lora_request.lora_name)
        if lease is None or lease.closed:
            async with coordinator.admission(lora_slot) as current_lora_request:
                _bind_admitted_lora(request, lora_slot, current_lora_request)
                return await original_enqueue(
                    self, request, prompt, parent_req, index, queue
                )
        if lease.lora_slot is None:
            context = coordinator.admission(lora_slot)
            lease.lora_request = await context.__aenter__()
            lease.context = context
            lease.lora_slot = lora_slot
        elif lease.lora_slot != lora_slot:
            raise RuntimeError("One request fanout resolved to multiple LoRA slots")
        _bind_admitted_lora(request, lora_slot, lease.lora_request)
        return await original_enqueue(self, request, prompt, parent_req, index, queue)

    add_request.__art_lora_update_patched__ = True  # type: ignore[attr-defined]
    _add_request.__art_lora_update_patched__ = True  # type: ignore[attr-defined]
    AsyncLLM.add_request = add_request  # type: ignore[method-assign]
    AsyncLLM._add_request = _add_request  # type: ignore[method-assign]


def _bind_admitted_lora(
    request: Any,
    lora_slot: str,
    lora_request: Any | None,
) -> None:
    if lora_request is None:
        if lora_slot.endswith(":active"):
            raise RuntimeError(
                f"Mutable LoRA slot {lora_slot!r} has no declared policy identity"
            )
        lora_request = request.lora_request
    if isinstance(lora_request, PolicyLoRARequest):
        request.lora_request = lora_request
        _set_policy_cache_salt(
            request,
            lora_slot=lora_slot,
            policy_version=lora_request.policy_version,
            update_seq=lora_request.update_seq,
        )


def _patch_load_inplace_storage() -> None:
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels

    original = OpenAIServingModels.load_lora_adapter
    if getattr(original, "__art_policy_spans_patched__", False):
        return

    async def load_lora_adapter(
        self: Any,
        request: Any,
        base_model_name: str | None = None,
    ) -> Any:
        if request.load_inplace and request.lora_name in self.lora_requests:
            raise RuntimeError(
                "Existing LoRA slots must be updated through /art/in_flight_lora_update"
            )
        result = await original(self, request, base_model_name=base_model_name)
        lora_request = self.lora_requests.get(request.lora_name)
        if lora_request is not None and lora_request.load_inplace:
            self.lora_requests[request.lora_name] = _normalized_lora_request(
                lora_request
            )
        return result

    load_lora_adapter.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingModels.load_lora_adapter = load_lora_adapter  # type: ignore[method-assign]


def _patch_policy_lora_update_rpc() -> None:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.worker_base import WorkerBase

    if not hasattr(WorkerBase, "art_load_lora_policy"):

        def art_load_lora_policy(self: Any, payload: dict[str, Any]) -> dict[str, Any]:
            lora_request = _policy_lora_request_from_payload(payload)
            previous = _WORKER_LORA_POLICY_BY_ID.get(lora_request.lora_int_id)
            loaded = self.add_lora(lora_request)
            if not self.pin_lora(lora_request.lora_int_id):
                raise RuntimeError("Loaded policy LoRA could not be pinned")
            current = _record_worker_lora_policy(lora_request)
            return {
                "loaded": bool(loaded),
                "previous": None if previous is None else dict(previous),
                "current": dict(current),
            }

        WorkerBase.art_load_lora_policy = art_load_lora_policy  # type: ignore[attr-defined]

    if not hasattr(WorkerBase, "art_declare_loaded_lora_policy"):

        def art_declare_loaded_lora_policy(
            self: Any, payload: dict[str, Any]
        ) -> dict[str, Any]:
            lora_request = _policy_lora_request_from_payload(
                payload, load_inplace=False
            )
            previous = _WORKER_LORA_POLICY_BY_ID.get(lora_request.lora_int_id)
            if lora_request.lora_int_id not in self.list_loras() or previous is None:
                raise RuntimeError(
                    f"LoRA {lora_request.lora_int_id} is not loaded on this worker"
                )
            for field in ("lora_slot", "lora_path"):
                expected = getattr(
                    lora_request,
                    "lora_name" if field == "lora_slot" else "lora_path",
                )
                if previous[field] != expected:
                    raise RuntimeError(
                        f"Loaded LoRA {field} is {previous[field]!r}, expected {expected!r}"
                    )
            if not self.pin_lora(lora_request.lora_int_id):
                raise RuntimeError("Initial policy LoRA could not be pinned")
            current = _record_worker_lora_policy(lora_request)
            return {
                "loaded": True,
                "previous": dict(previous),
                "current": dict(current),
            }

        WorkerBase.art_declare_loaded_lora_policy = art_declare_loaded_lora_policy  # type: ignore[attr-defined]

    if not hasattr(EngineCore, "art_apply_lora_policy_update"):

        def art_apply_lora_policy_update(
            self: Any, payload: dict[str, Any]
        ) -> dict[str, int]:
            return _apply_policy_lora_update(self, payload)

        EngineCore.art_apply_lora_policy_update = art_apply_lora_policy_update  # type: ignore[attr-defined]

    if not hasattr(EngineCore, "art_declare_loaded_lora_policy"):

        def art_declare_loaded_lora_policy(
            self: Any, payload: dict[str, Any]
        ) -> dict[str, int]:
            request = _policy_lora_request_from_payload(payload, load_inplace=False)
            acknowledgements = self.collective_rpc(
                "art_declare_loaded_lora_policy", args=(payload,)
            )
            _validate_worker_lora_update(request, acknowledgements)
            return {"workers": len(acknowledgements)}

        EngineCore.art_declare_loaded_lora_policy = art_declare_loaded_lora_policy  # type: ignore[attr-defined]


def _apply_policy_lora_update(
    engine_core: Any, payload: dict[str, Any]
) -> dict[str, int]:
    if not engine_core.is_scheduler_paused():
        raise RuntimeError("Policy LoRA updates require a paused scheduler")
    lora_request = _policy_lora_request_from_payload(payload)
    try:
        acknowledgements = engine_core.collective_rpc(
            "art_load_lora_policy", args=(payload,)
        )
        previous = _validate_worker_lora_update(lora_request, acknowledgements)
        started = {
            request.request_id
            for request in engine_core.scheduler.requests.values()
            if _request_uses_lora_slot(request, lora_request.lora_name)
            and _request_has_executed(request)
        }
        engine_core._reset_caches(
            reset_running_requests=True,
            reset_connector=True,
        )
        return _transition_scheduler_policy_history(
            engine_core.scheduler,
            lora_request=_policy_lora_request_from_payload(payload, load_inplace=False),
            previous_policy=previous,
            started_request_ids=started,
        )
    except BaseException:
        # Never let a core that may have partially changed workers schedule again.
        engine_core.pause_scheduler("abort", True)
        raise


def _transition_scheduler_policy_history(
    scheduler: Any,
    *,
    lora_request: PolicyLoRARequest,
    previous_policy: Mapping[str, Any] | None,
    started_request_ids: set[str],
) -> dict[str, int]:
    updated = 0
    continued = 0
    for request in scheduler.requests.values():
        if not _request_uses_lora_slot(request, lora_request.lora_name):
            continue
        previous_digest = getattr(request, _POLICY_HISTORY_BASE_FIELD, None)
        if request.request_id in started_request_ids:
            continued += 1
            previous_digest = _policy_history_from_cache_salt(request.cache_salt)
            if previous_digest is None:
                if previous_policy is None:
                    raise RuntimeError(
                        f"Started request {request.request_id!r} has no policy identity"
                    )
                if int(previous_policy["update_seq"]) != 0:
                    raise RuntimeError(
                        f"Started request {request.request_id!r} lost policy history"
                    )
                previous_digest = _extend_policy_history(
                    None,
                    lora_slot=str(previous_policy["lora_slot"]),
                    policy_version=int(previous_policy["policy_version"]),
                    update_seq=0,
                )
        request.lora_request = lora_request
        setattr(request, _POLICY_HISTORY_BASE_FIELD, previous_digest)
        _set_policy_cache_salt(
            request,
            lora_slot=lora_request.lora_name,
            policy_version=lora_request.policy_version,
            update_seq=lora_request.update_seq,
            previous_digest=previous_digest,
        )
        request.block_hashes.clear()
        request.update_block_hashes()
        setattr(
            request,
            _POLICY_EXECUTION_MARKER_FIELD,
            (
                int(getattr(request, "num_preemptions", 0) or 0),
                len(getattr(request, "output_token_ids", ())),
            ),
        )
        updated += 1
    return {
        "updated_requests": updated,
        "continued_requests": continued,
    }


def _policy_lora_request_from_payload(
    payload: Mapping[str, Any], *, load_inplace: bool = True
) -> PolicyLoRARequest:
    return PolicyLoRARequest(
        lora_name=str(payload["lora_name"]),
        lora_int_id=int(payload["lora_int_id"]),
        lora_path=str(payload["lora_path"]),
        base_model_name=payload.get("base_model_name"),
        tensorizer_config_dict=payload.get("tensorizer_config_dict"),
        load_inplace=load_inplace,
        is_3d_lora_weight=bool(payload.get("is_3d_lora_weight", False)),
        policy_version=int(payload["policy_version"]),
        update_seq=int(payload["update_seq"]),
    )


def policy_lora_request_payload(lora_request: PolicyLoRARequest) -> dict[str, Any]:
    return {
        "lora_name": lora_request.lora_name,
        "lora_int_id": lora_request.lora_int_id,
        "lora_path": lora_request.lora_path,
        "base_model_name": lora_request.base_model_name,
        "tensorizer_config_dict": lora_request.tensorizer_config_dict,
        "is_3d_lora_weight": lora_request.is_3d_lora_weight,
        "policy_version": lora_request.policy_version,
        "update_seq": lora_request.update_seq,
    }


def _normalized_lora_request(lora_request: Any) -> LoRARequest:
    request_type = (
        PolicyLoRARequest
        if isinstance(lora_request, PolicyLoRARequest)
        else LoRARequest
    )
    policy_fields = (
        {
            "policy_version": lora_request.policy_version,
            "update_seq": lora_request.update_seq,
        }
        if request_type is PolicyLoRARequest
        else {}
    )
    return request_type(
        lora_name=lora_request.lora_name,
        lora_int_id=lora_request.lora_int_id,
        lora_path=lora_request.lora_path,
        base_model_name=lora_request.base_model_name,
        tensorizer_config_dict=lora_request.tensorizer_config_dict,
        load_inplace=False,
        is_3d_lora_weight=lora_request.is_3d_lora_weight,
        **policy_fields,
    )


def _validate_worker_lora_update(
    lora_request: PolicyLoRARequest,
    acknowledgements: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not acknowledgements:
        raise RuntimeError("Policy LoRA update returned no worker acknowledgements")
    expected = {
        "policy_version": lora_request.policy_version,
        "lora_slot": lora_request.lora_name,
        "lora_path": lora_request.lora_path,
        "update_seq": lora_request.update_seq,
    }
    previous: Mapping[str, Any] | None = None
    previous_set = False
    for rank, acknowledgement in enumerate(acknowledgements):
        if not acknowledgement.get("loaded"):
            raise RuntimeError(f"Worker rank {rank} did not load the policy LoRA")
        current = acknowledgement.get("current")
        if current != expected:
            raise RuntimeError(
                f"Worker rank {rank} acknowledged {current!r}, expected {expected!r}"
            )
        rank_previous = acknowledgement.get("previous")
        if previous_set and rank_previous != previous:
            raise RuntimeError("Policy LoRA workers started from different policies")
        previous = rank_previous
        previous_set = True
    return previous


def _request_uses_lora_slot(request: Any, lora_slot: str) -> bool:
    lora_request = getattr(request, "lora_request", None)
    return lora_request is not None and str(lora_request.lora_name) == lora_slot


def _request_has_executed(request: Any) -> bool:
    preemptions = int(getattr(request, "num_preemptions", 0) or 0)
    output_tokens = len(getattr(request, "output_token_ids", ()))
    marker = getattr(request, _POLICY_EXECUTION_MARKER_FIELD, None)
    if marker is not None:
        baseline_preemptions, baseline_output_tokens = marker
        return bool(
            int(getattr(request, "num_computed_tokens", 0) or 0)
            or preemptions > baseline_preemptions
            or output_tokens > baseline_output_tokens
        )
    return bool(
        int(getattr(request, "num_computed_tokens", 0) or 0)
        or output_tokens
        or preemptions
    )


def _policy_context_from_runner(runner: Any) -> dict[str, dict[str, Any]]:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        state = getattr(runner, "execute_model_state", None)
        input_batch = getattr(state, "input_batch", None)
    if input_batch is None:
        return {}
    lora_state = getattr(runner, "lora_state", None)
    context: dict[str, dict[str, Any]] = {}
    for req_id in input_batch.req_ids:
        lora_request = _lora_request_for_input_batch_req(input_batch, req_id)
        if lora_request is None and lora_state is not None:
            lora_request = getattr(lora_state, "lora_requests", {}).get(req_id)
        context[req_id] = _policy_metadata_for_lora_request(lora_request)
    return context


def _lora_request_for_input_batch_req(input_batch: Any, req_id: str) -> Any | None:
    req_index = getattr(input_batch, "req_id_to_index", {}).get(req_id)
    request_lora_mapping = getattr(input_batch, "request_lora_mapping", None)
    if req_index is None or request_lora_mapping is None:
        return None
    lora_id = int(request_lora_mapping[req_index])
    if lora_id <= 0:
        return None
    return getattr(input_batch, "lora_id_to_lora_request", {}).get(lora_id)


def _policy_metadata_for_lora_request(lora_request: Any | None) -> dict[str, Any]:
    if lora_request is None:
        return {"policy_version": 0, "lora_slot": "base", "update_seq": 0}
    state = _WORKER_LORA_POLICY_BY_ID.get(lora_request.lora_int_id)
    if state is None:
        state = _record_worker_lora_policy(lora_request)
    if state["lora_slot"].endswith(":active") and state["update_seq"] == 0:
        raise RuntimeError(
            f"Mutable LoRA slot {state['lora_slot']!r} has no declared policy identity"
        )
    return state


def _record_worker_lora_policy(lora_request: Any) -> dict[str, Any]:
    policy_version = getattr(lora_request, "policy_version", None)
    update_seq = getattr(lora_request, "update_seq", None)
    if policy_version is None:
        policy_version = _immutable_policy_version_from_lora_name(
            str(lora_request.lora_name)
        )
        update_seq = int(policy_version or 0)
    state = {
        "policy_version": int(policy_version or 0),
        "lora_slot": str(lora_request.lora_name),
        "lora_path": str(lora_request.lora_path),
        "update_seq": int(update_seq or 0),
    }
    _WORKER_LORA_POLICY_BY_ID[int(lora_request.lora_int_id)] = state
    return state


def get_worker_lora_states(lora_ids: set[int]) -> tuple[dict[str, Any], ...]:
    states = []
    for lora_id in sorted(lora_ids):
        state = _WORKER_LORA_POLICY_BY_ID.get(lora_id)
        if state is None:
            raise RuntimeError(f"loaded LoRA {lora_id} has no ART worker state")
        states.append(
            {
                "lora_id": lora_id,
                "lora_name": state["lora_slot"],
                "lora_path": state["lora_path"],
                "policy_version": state["policy_version"],
                "update_seq": state["update_seq"],
            }
        )
    return tuple(states)


def _immutable_policy_version_from_lora_name(lora_name: str) -> int | None:
    match = re.search(r"@(\d+)$", lora_name)
    return int(match.group(1)) if match else None


def _attach_policy_spans_to_model_output(
    output: Any, context: dict[str, dict[str, Any]]
) -> None:
    spans_by_req: dict[str, list[dict[str, Any]]] = {}
    for req_id, token_ids in zip(output.req_ids, output.sampled_token_ids or ()):
        num_tokens = len(token_ids)
        if num_tokens <= 0:
            continue
        metadata = context.get(req_id)
        if not metadata:
            continue
        spans_by_req[req_id] = [
            {
                "start_token": 0,
                "end_token": num_tokens,
                "policy_version": metadata["policy_version"],
                "lora_slot": metadata["lora_slot"],
                "update_seq": metadata["update_seq"],
            }
        ]
    if spans_by_req:
        setattr(output, ART_POLICY_TOKEN_SPANS_FIELD, spans_by_req)


def _attach_policy_span_context_to_sample_output(
    output: Any, context: dict[str, dict[str, Any]]
) -> None:
    setattr(output, "_art_policy_span_context", context)
    for field in ("model_runner_output", "_model_runner_output"):
        target = getattr(output, field, None)
        if target is not None:
            setattr(target, "_art_policy_span_context", context)


def _policy_span_context_from_sample_output(output: Any) -> dict[str, dict[str, Any]]:
    context = getattr(output, "_art_policy_span_context", None)
    if isinstance(context, dict):
        return context
    for field in ("model_runner_output", "_model_runner_output"):
        target = getattr(output, field, None)
        context = getattr(target, "_art_policy_span_context", None)
        if isinstance(context, dict):
            return context
    return {}


def _engine_core_policy_spans_by_request(
    engine_core_outputs: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    spans_by_request: dict[str, list[dict[str, Any]]] = {}
    for item in engine_core_outputs:
        outputs = getattr(item, "outputs", None)
        if outputs is None:
            outputs = (item,)
        for output in outputs:
            spans = getattr(output, ART_POLICY_TOKEN_SPANS_FIELD, None)
            if spans:
                spans_by_request[output.request_id] = spans
    return spans_by_request


def _trim_step_spans(
    spans: list[dict[str, Any]], token_count: int
) -> list[dict[str, Any]]:
    if token_count <= 0:
        return []
    trimmed: list[dict[str, Any]] = []
    for span in spans:
        start = min(max(int(span["start_token"]), 0), token_count)
        end = min(max(int(span["end_token"]), start), token_count)
        if end <= start:
            continue
        current = {**span, "start_token": start, "end_token": end}
        if trimmed and _can_merge_spans(trimmed[-1], current):
            trimmed[-1]["end_token"] = end
        else:
            trimmed.append(current)
    return trimmed


def _append_current_policy_spans(req_state: Any, token_count: int) -> None:
    step_spans = _CURRENT_ENGINE_POLICY_SPANS.get(req_state.request_id)
    if not step_spans or token_count <= 0:
        return
    detokenizer = getattr(req_state, "detokenizer", None)
    output_tokens = detokenizer.num_output_tokens() if detokenizer is not None else 0
    offset = max(output_tokens - token_count, 0)
    accumulated = getattr(req_state, ART_POLICY_TOKEN_SPANS_FIELD, None)
    if accumulated is None:
        accumulated = []
        setattr(req_state, ART_POLICY_TOKEN_SPANS_FIELD, accumulated)
    for span in _trim_step_spans(step_spans, token_count):
        current = {
            **span,
            "start_token": offset + int(span["start_token"]),
            "end_token": offset + int(span["end_token"]),
        }
        if accumulated and _can_merge_spans(accumulated[-1], current):
            accumulated[-1]["end_token"] = current["end_token"]
        else:
            accumulated.append(current)


def _can_merge_spans(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("end_token") == right.get("start_token")
        and left.get("policy_version") == right.get("policy_version")
        and left.get("lora_slot") == right.get("lora_slot")
        and left.get("update_seq") == right.get("update_seq")
    )


def _record_request_output_spans(
    request_output: Any,
    spans_by_choice: Mapping[int, list[dict[str, Any]] | None],
) -> None:
    for output in request_output.outputs:
        spans = spans_by_choice.get(int(output.index))
        if not spans:
            continue
        copied = [dict(span) for span in spans]
        setattr(output, ART_POLICY_TOKEN_SPANS_FIELD, copied)


def _policy_spans_by_choice_from_final_output(
    final_res: Any,
) -> dict[int, list[dict[str, Any]]]:
    outputs = getattr(final_res, "outputs", None)
    if not outputs:
        return {}
    spans_by_choice: dict[int, list[dict[str, Any]]] = {}
    for output in outputs:
        spans = getattr(output, ART_POLICY_TOKEN_SPANS_FIELD, None)
        if spans:
            spans_by_choice[int(output.index)] = [dict(span) for span in spans]
    return spans_by_choice
