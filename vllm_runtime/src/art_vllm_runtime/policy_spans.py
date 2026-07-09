"""Policy-token span tracking for ART's owned vLLM runtime.

The hot path intentionally uses plain dict/list payloads. ART validates them
with Pydantic after the OpenAI response crosses back into the training process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import re
import sys
from typing import Any

import msgspec
import numpy as np
import torch

POLICY_TOKEN_SPANS_FIELD = "policy_token_spans"
ART_POLICY_TOKEN_SPANS_FIELD = "art_policy_token_spans"

_CURRENT_ENGINE_POLICY_SPANS: dict[str, list[dict[str, Any]]] = {}
_COMPLETION_POLICY_SPANS_BY_REQUEST: dict[str, dict[int, list[dict[str, Any]]]] = {}
_WORKER_LORA_POLICY_BY_ID: dict[int, dict[str, Any]] = {}
_WORKER_LORA_UPDATE_SEQ = 0
_POLICY_CACHE_SALT_PREFIX = "art_policy_cache_salt="
_POLICY_CACHE_SALT_MARKER = f"|{_POLICY_CACHE_SALT_PREFIX}"

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


def patch_policy_token_spans() -> None:
    _patch_model_runner_output_type()
    _patch_engine_core_output_type()
    _patch_worker_policy_span_capture()
    _patch_scheduler_policy_span_transport()
    _patch_output_processor_policy_span_accumulation()
    _patch_openai_response_policy_spans()
    _patch_lora_alias_resolution()
    _patch_load_inplace_storage()
    _patch_engine_waiting_cache_salt_utility()


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
    policy_version: int | None,
) -> None:
    aliases = getattr(models, "_art_lora_aliases", None)
    if aliases is None:
        aliases = {}
        setattr(models, "_art_lora_aliases", aliases)
    aliases[public_model_name] = lora_slot

    versions = getattr(models, "_art_lora_alias_policy_versions", None)
    if versions is None:
        versions = {}
        setattr(models, "_art_lora_alias_policy_versions", versions)
    if policy_version is not None:
        versions[public_model_name] = policy_version


def _resolve_lora_alias(models: Any, model_name: str | None) -> Any | None:
    if not model_name:
        return None
    slot = getattr(models, "_art_lora_aliases", {}).get(model_name)
    if not slot:
        return None
    return models.lora_requests.get(slot)


def _alias_policy_version(models: Any, model_name: str | None) -> int | None:
    if not model_name:
        return None
    version = getattr(models, "_art_lora_alias_policy_versions", {}).get(model_name)
    if version is None:
        return None
    return int(version)


def _strip_policy_cache_salt(cache_salt: str | None) -> str | None:
    if not cache_salt:
        return None
    if cache_salt.startswith(_POLICY_CACHE_SALT_PREFIX):
        return None
    base, marker, _policy = cache_salt.partition(_POLICY_CACHE_SALT_MARKER)
    if marker:
        return base or None
    return cache_salt


def _policy_cache_salt(
    *,
    lora_slot: str,
    policy_version: int,
    user_cache_salt: str | None,
) -> str:
    policy_salt = f"{lora_slot}:{policy_version}"
    if user_cache_salt:
        return f"{user_cache_salt}{_POLICY_CACHE_SALT_MARKER}{policy_salt}"
    return f"{_POLICY_CACHE_SALT_PREFIX}{policy_salt}"


def _set_policy_cache_salt(
    request: Any,
    *,
    lora_slot: str,
    policy_version: int,
) -> None:
    user_cache_salt = _strip_policy_cache_salt(getattr(request, "cache_salt", None))
    request.cache_salt = _policy_cache_salt(
        lora_slot=lora_slot,
        policy_version=policy_version,
        user_cache_salt=user_cache_salt,
    )


def _apply_lora_alias_policy_cache_salt(
    models: Any,
    request: Any,
    lora_request: Any,
) -> None:
    policy_version = _alias_policy_version(models, getattr(request, "model", None))
    if policy_version is None:
        return
    _set_policy_cache_salt(
        request,
        lora_slot=str(lora_request.lora_name),
        policy_version=policy_version,
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
        original_sample_tokens = gpu_model_runner_cls.sample_tokens
        if getattr(original_sample_tokens, "__art_policy_spans_patched__", False):
            continue

        def make_sample_tokens(original: Any):
            def sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
                context = _policy_context_from_runner(self)
                output = original(self, *args, **kwargs)
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
        request_output = original_make(self, new_token_ids, *args, **kwargs)
        if request_output is not None and hasattr(request_output, "outputs"):
            spans = getattr(self, ART_POLICY_TOKEN_SPANS_FIELD, None)
            if spans:
                _record_request_output_spans(
                    request_output,
                    request_index=getattr(self, "request_index", 0),
                    spans=spans,
                )
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
            fallback_spans = _COMPLETION_POLICY_SPANS_BY_REQUEST.pop(request_id, {})
            if not spans_by_choice:
                spans_by_choice = fallback_spans
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
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    original_check = OpenAIServing._check_model
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
        OpenAIServing._check_model = _check_model  # type: ignore[method-assign]

    original_maybe = OpenAIServing._maybe_get_adapters
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
    OpenAIServing._maybe_get_adapters = _maybe_get_adapters  # type: ignore[method-assign]


def _patch_load_inplace_storage() -> None:
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.lora.request import LoRARequest

    original = OpenAIServingModels.load_lora_adapter
    if getattr(original, "__art_policy_spans_patched__", False):
        return

    async def load_lora_adapter(
        self: Any,
        request: Any,
        base_model_name: str | None = None,
    ) -> Any:
        result = await original(self, request, base_model_name=base_model_name)
        lora_request = self.lora_requests.get(request.lora_name)
        if lora_request is not None and lora_request.load_inplace:
            normalized = LoRARequest(
                lora_name=lora_request.lora_name,
                lora_int_id=lora_request.lora_int_id,
                lora_path=lora_request.lora_path,
                base_model_name=lora_request.base_model_name,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                load_inplace=False,
                is_3d_lora_weight=lora_request.is_3d_lora_weight,
            )
            self.lora_requests[request.lora_name] = normalized
        return result

    load_lora_adapter.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingModels.load_lora_adapter = load_lora_adapter  # type: ignore[method-assign]


def _patch_engine_waiting_cache_salt_utility() -> None:
    from vllm.v1.engine.core import EngineCore

    if hasattr(EngineCore, "art_update_waiting_lora_cache_salt"):
        return

    def art_update_waiting_lora_cache_salt(
        self: Any,
        lora_slot: str,
        policy_version: int,
    ) -> dict[str, int]:
        return _update_waiting_lora_cache_salt(
            self.scheduler,
            lora_slot=str(lora_slot),
            policy_version=int(policy_version),
        )

    EngineCore.art_update_waiting_lora_cache_salt = art_update_waiting_lora_cache_salt  # type: ignore[attr-defined]


def _update_waiting_lora_cache_salt(
    scheduler: Any,
    *,
    lora_slot: str,
    policy_version: int,
) -> dict[str, int]:
    updated = 0
    skipped_started = 0
    for queue_name in ("waiting", "skipped_waiting"):
        queue = getattr(scheduler, queue_name, None)
        if queue is None:
            continue
        for request in list(queue):
            lora_request = getattr(request, "lora_request", None)
            if lora_request is None or str(lora_request.lora_name) != lora_slot:
                continue
            if int(getattr(request, "num_computed_tokens", 0) or 0) != 0:
                skipped_started += 1
                continue
            _set_policy_cache_salt(
                request,
                lora_slot=lora_slot,
                policy_version=policy_version,
            )
            request.block_hashes.clear()
            request.update_block_hashes()
            updated += 1
    return {
        "updated_waiting_requests": updated,
        "skipped_started_waiting_requests": skipped_started,
    }


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
    return state


def _record_worker_lora_policy(lora_request: Any) -> dict[str, Any]:
    global _WORKER_LORA_UPDATE_SEQ
    _WORKER_LORA_UPDATE_SEQ += 1
    policy_version = _policy_version_from_lora_request(lora_request)
    state = {
        "policy_version": int(policy_version or 0),
        "lora_slot": str(lora_request.lora_name),
        "update_seq": _WORKER_LORA_UPDATE_SEQ,
    }
    _WORKER_LORA_POLICY_BY_ID[int(lora_request.lora_int_id)] = state
    return state


def _policy_version_from_lora_request(lora_request: Any) -> int | None:
    for pattern, value in (
        (r"@(\d+)$", getattr(lora_request, "lora_name", "")),
        (
            r"^(?:step[_-]?)?(\d+)$",
            getattr(lora_request, "lora_path", "").rstrip("/").split("/")[-1],
        ),
    ):
        match = re.search(pattern, value)
        if match:
            return int(match.group(1))
    return None


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
    *,
    request_index: int,
    spans: list[dict[str, Any]],
) -> None:
    by_choice = _COMPLETION_POLICY_SPANS_BY_REQUEST.setdefault(
        request_output.request_id, {}
    )
    for output in request_output.outputs:
        if getattr(output, "index", request_index) != request_index:
            continue
        copied = [dict(span) for span in spans]
        setattr(output, ART_POLICY_TOKEN_SPANS_FIELD, copied)
        by_choice[output.index] = copied


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
