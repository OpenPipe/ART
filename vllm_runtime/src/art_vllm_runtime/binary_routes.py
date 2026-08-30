from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import struct
from typing import Any

import numpy as np

MAGIC = b"ARTRTE2\0"
HEADER = struct.Struct("<8sQII")
ROUTE_HEADER = struct.Struct("<IB3xQQQ")
PIPELINE_ROUTES_ENV = "ART_VLLM_PIPELINE_ROUTES_PROTOCOL"
PIPELINE_ROUTES_PROTOCOL = "1"
ROUTE_REQUEST_XARG = "art_return_routed_experts"
ROUTE_CACHE_SALT = "art-routes=ARTRTE2"
_REGISTERED_NUM_EXPERTS: int | None = None
_REGISTERED_PADDING_LAYERS: tuple[int, ...] | None = None


class _CapturedRoutes(dict[int, np.ndarray]):
    def __init__(self, *, num_experts: int, padding_layers: tuple[int, ...]) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.padding_layers = padding_layers


_CAPTURE: ContextVar[_CapturedRoutes | None] = ContextVar(
    "art_binary_routed_experts", default=None
)


@contextmanager
def capture_routed_experts() -> Iterator[_CapturedRoutes]:
    if _REGISTERED_NUM_EXPERTS is None or _REGISTERED_PADDING_LAYERS is None:
        raise RuntimeError("vLLM did not register an exact MoE route layout")
    routes = _CapturedRoutes(
        num_experts=_REGISTERED_NUM_EXPERTS,
        padding_layers=_REGISTERED_PADDING_LAYERS,
    )
    token = _CAPTURE.set(routes)
    try:
        yield routes
    finally:
        _CAPTURE.reset(token)


def mark_route_request(request: Any) -> None:
    xargs = dict(getattr(request, "vllm_xargs", None) or {})
    xargs[ROUTE_REQUEST_XARG] = 1
    request.vllm_xargs = xargs
    route_salt = ROUTE_CACHE_SALT
    cache_salt = getattr(request, "cache_salt", None)
    if not cache_salt:
        request.cache_salt = route_salt
    elif cache_salt != route_salt and not cache_salt.endswith(f"|{route_salt}"):
        request.cache_salt = f"{cache_salt}|{route_salt}"


def _sampling_requests_routes(sampling_params: Any) -> bool:
    extra_args = getattr(sampling_params, "extra_args", None)
    return isinstance(extra_args, dict) and extra_args.get(ROUTE_REQUEST_XARG) == 1


def _scheduled_request_routes(runner: Any, scheduler_output: Any) -> bool:
    new_params = {
        item.req_id: item.sampling_params
        for item in scheduler_output.scheduled_new_reqs
    }
    for req_id in scheduler_output.num_scheduled_tokens:
        sampling_params = new_params.get(req_id)
        if sampling_params is None:
            state = runner.requests.get(req_id)
            sampling_params = getattr(state, "sampling_params", None)
        if _sampling_requests_routes(sampling_params):
            return True
    return False


def encode_routed_experts_response(
    json_body: bytes,
    routes: dict[int, np.ndarray],
    *,
    num_experts: int | None = None,
) -> bytes:
    response, _ = encode_routed_experts_response_parts(
        json_body,
        routes,
        num_experts=num_experts,
    )
    return response


def encode_routed_experts_response_parts(
    json_body: bytes,
    routes: dict[int, np.ndarray],
    *,
    num_experts: int | None = None,
) -> tuple[bytes, bytes]:
    """Encode the response and return the exact concatenated route payload."""

    num_experts = int(num_experts or getattr(routes, "num_experts", 0))
    chunks: list[bytes | memoryview] = [
        HEADER.pack(MAGIC, len(json_body), len(routes), num_experts),
        json_body,
    ]
    route_chunks: list[memoryview] = []
    for choice_index, dtype_code, array in _encoded_route_arrays(
        routes, num_experts=num_experts
    ):
        route_bytes = memoryview(array).cast("B")
        if array.ndim != 3:
            raise RuntimeError(f"Routed experts must have rank 3, got {array.shape}")
        chunks.extend(
            (
                ROUTE_HEADER.pack(choice_index, dtype_code, *array.shape),
                route_bytes,
            )
        )
        route_chunks.append(route_bytes)
    return b"".join(chunks), b"".join(route_chunks)


def _encoded_route_arrays(
    routes: dict[int, np.ndarray], *, num_experts: int
) -> Iterator[tuple[int, int, np.ndarray]]:
    dtype = _route_dtype(num_experts)
    for choice_index, array in sorted(routes.items()):
        if array.ndim != 3:
            raise RuntimeError(f"Routed experts must have rank 3, got {array.shape}")
        if dtype == np.dtype(np.uint8):
            dtype_code = 1
        else:
            dtype_code = 2
            array = array.astype("<u2", copy=False)
        if array.dtype != dtype:
            raise RuntimeError(
                f"vLLM routed experts for {num_experts} experts must use "
                f"{dtype}, got {array.dtype}"
            )
        array = np.ascontiguousarray(array)
        _resolve_padding_routes(
            array, padding_layers=getattr(routes, "padding_layers", ())
        )
        _validate_route_ids(array, num_experts=num_experts)
        yield choice_index, dtype_code, array


def _route_dtype(num_experts: int) -> np.dtype[Any]:
    if not 1 <= num_experts <= 65_536:
        raise RuntimeError(
            f"ART routed experts require num_experts in [1, 65536], got {num_experts}"
        )
    return np.dtype(np.uint8 if num_experts <= 256 else np.uint16)


def _validate_route_ids(array: np.ndarray, *, num_experts: int) -> None:
    if array.shape[-1] > num_experts:
        raise RuntimeError("Routed-expert top-k exceeds exact expert count")
    flat = array.reshape(-1, array.shape[-1])
    for start in range(0, len(flat), 1 << 20):
        rows = np.sort(flat[start : start + (1 << 20)], axis=1)
        if rows.size and int(rows.max()) >= num_experts:
            raise RuntimeError("Routed expert id is outside the exact model range")
        if rows.shape[1] > 1 and bool(np.any(rows[:, 1:] == rows[:, :-1])):
            raise RuntimeError("Routed expert ids must be distinct per token and layer")


def _resolve_padding_routes(
    array: np.ndarray, *, padding_layers: tuple[int, ...]
) -> None:
    if not padding_layers:
        return
    if padding_layers[-1] >= array.shape[1]:
        raise RuntimeError(
            "Routed-expert response has fewer layers than the registered model"
        )
    padding = array[:, padding_layers, :]
    if padding.size and bool(np.any(padding)):
        raise RuntimeError("Non-routed layer contained captured expert ids")
    array[:, padding_layers, :] = np.arange(array.shape[-1], dtype=array.dtype)


def _model_padding_layers(model_config: Any) -> tuple[int, ...]:
    config = getattr(model_config, "hf_text_config", None)
    if config is None:
        config = getattr(model_config, "hf_config", model_config)
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    layer_types = getattr(config, "mlp_layer_types", None)
    if layer_types is not None:
        if len(layer_types) != num_layers:
            raise RuntimeError("mlp_layer_types does not match num_hidden_layers")
        if not set(layer_types).issubset({"dense", "sparse", "moe", "hash_moe"}):
            raise RuntimeError(f"Unsupported MoE layer types: {set(layer_types)}")
        return tuple(i for i, kind in enumerate(layer_types) if kind == "dense")
    first_dense = int(getattr(config, "first_k_dense_replace", 0))
    if not 0 <= first_dense <= num_layers:
        raise RuntimeError("first_k_dense_replace is outside the model layer range")
    return tuple(range(first_dense))


def _normalize_route_topk(model_config: Any) -> None:
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None) or getattr(
        hf_config, "text_config", hf_config
    )
    configs = (model_config, hf_config, text_config)
    values = {
        int(value)
        for config in configs
        if config is not None
        for name in (
            "num_experts_per_tok",
            "experts_per_token",
            "top_k_experts",
        )
        if (value := getattr(config, name, None)) is not None and int(value) > 0
    }
    if len(values) != 1:
        raise RuntimeError(f"Model configs disagree on MoE route top-k: {values}")
    if text_config is None:
        raise RuntimeError("Unable to find the model's text config for route capture")
    text_config.num_experts_per_tok = values.pop()


def _register_model_route_layout(model_config: Any) -> None:
    global _REGISTERED_NUM_EXPERTS, _REGISTERED_PADDING_LAYERS
    _normalize_route_topk(model_config)
    getter = getattr(model_config, "get_num_experts", None)
    if callable(getter):
        num_experts = int(getter())
    else:
        configs = [
            model_config,
            getattr(model_config, "hf_config", None),
            getattr(getattr(model_config, "hf_config", None), "text_config", None),
        ]
        values = {
            int(value)
            for config in configs
            if config is not None
            for name in ("num_experts", "n_routed_experts", "num_local_experts")
            if (value := getattr(config, name, None)) is not None and int(value) > 0
        }
        if not values:
            raise RuntimeError("Unable to find the model's exact MoE expert count")
        if len(values) != 1:
            raise RuntimeError(f"Model configs disagree on MoE expert count: {values}")
        num_experts = values.pop()
    _route_dtype(num_experts)
    padding_layers = _model_padding_layers(model_config)
    if _REGISTERED_NUM_EXPERTS not in {None, num_experts}:
        raise RuntimeError(
            "One vLLM process cannot capture routes for different expert counts"
        )
    if _REGISTERED_PADDING_LAYERS not in {None, padding_layers}:
        raise RuntimeError("One vLLM process cannot capture different MoE layouts")
    _REGISTERED_NUM_EXPERTS = num_experts
    _REGISTERED_PADDING_LAYERS = padding_layers


def patch_binary_routed_experts_response() -> None:
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    original = OpenAIServingChat.chat_completion_full_generator
    if getattr(original, "__art_binary_routes_patched__", False):
        return

    @wraps(original)
    async def patched(
        self: Any,
        request: Any,
        result_generator: AsyncIterator[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        capture = _CAPTURE.get()

        async def stripped_results() -> AsyncIterator[Any]:
            async for result in result_generator:
                for output in result.outputs:
                    if output.routed_experts is not None:
                        if capture is not None:
                            capture[int(output.index)] = output.routed_experts
                        output.routed_experts = None
                yield result

        return await original(self, request, stripped_results(), *args, **kwargs)

    patched.__art_binary_routes_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_full_generator = patched


def patch_request_scoped_routed_experts() -> None:
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_capture = RoutedExpertsCapturer.capture
    if not getattr(original_capture, "__art_request_scoped__", False):

        @wraps(original_capture)
        def capture(self: Any, *args: Any, **kwargs: Any) -> None:
            if getattr(self, "_art_capture_enabled", True):
                original_capture(self, *args, **kwargs)

        capture.__art_request_scoped__ = True  # type: ignore[attr-defined]
        RoutedExpertsCapturer.capture = capture

    original_execute = GPUModelRunner.execute_model
    if getattr(original_execute, "__art_request_scoped_routes__", False):
        return

    @wraps(original_execute)
    def execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        capturer = getattr(self, "routed_experts_capturer", None)
        if capturer is None:
            return original_execute(self, scheduler_output, *args, **kwargs)
        enabled = _scheduled_request_routes(self, scheduler_output)
        capturer._art_capture_enabled = enabled
        self.routed_experts_initialized = enabled
        return original_execute(self, scheduler_output, *args, **kwargs)

    execute.__art_request_scoped_routes__ = True  # type: ignore[attr-defined]
    GPUModelRunner.execute_model = execute


def patch_pipeline_routed_experts() -> None:
    """Reduce disjoint PP-stage routes onto vLLM's output rank."""
    import torch
    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_execute = GPUModelRunner.execute_model
    if getattr(original_execute, "__art_pipeline_routes_patched__", False):
        return
    original_sample = GPUModelRunner.sample_tokens
    enabled = os.environ.get(PIPELINE_ROUTES_ENV) == PIPELINE_ROUTES_PROTOCOL

    @wraps(original_execute)
    def execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        if enabled and self.routed_experts_initialized:
            self._art_pipeline_route_tokens = int(
                scheduler_output.total_num_scheduled_tokens
            )
        else:
            self._art_pipeline_route_tokens = 0
        return original_execute(self, scheduler_output, *args, **kwargs)

    @wraps(original_sample)
    def sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not enabled:
            return original_sample(self, *args, **kwargs)
        num_tokens = int(getattr(self, "_art_pipeline_route_tokens", 0))
        self._art_pipeline_route_tokens = 0
        if not num_tokens:
            return original_sample(self, *args, **kwargs)
        pp = get_pp_group()
        if pp.world_size <= 1:
            raise RuntimeError("pipeline route capture requires PP > 1")
        if get_tp_group().rank_in_group == 0:
            if not getattr(self, "_art_pipeline_routes_ready", False):
                initialized = bool(self.routed_experts_initialized)
                buffer = (
                    self.routed_experts_capturer.get_device_buffer()
                    if initialized
                    else None
                )
                local = torch.tensor(
                    [
                        int(PIPELINE_ROUTES_PROTOCOL),
                        int(initialized),
                        buffer.ndim if buffer is not None else 0,
                        buffer.shape[0] if buffer is not None else 0,
                        buffer.shape[1] if buffer is not None else 0,
                        buffer.shape[2] if buffer is not None else 0,
                        int(buffer is not None and buffer.dtype == torch.int32),
                    ],
                    dtype=torch.int64,
                    device=buffer.device if buffer is not None else self.device,
                )
                states = [torch.empty_like(local) for _ in range(pp.world_size)]
                torch.distributed.all_gather(states, local, group=pp.device_group)
                values = [state.tolist() for state in states]
                if any(value != values[0] for value in values[1:]):
                    raise RuntimeError(
                        f"pipeline routed-expert workers disagree: {values}"
                    )
                if (
                    values[0][1] != 1
                    or values[0][2] != 3
                    or min(values[0][3:6]) <= 0
                    or values[0][6] != 1
                ):
                    raise RuntimeError(
                        f"pipeline routed-expert capturer is invalid: {values}"
                    )
                self._art_pipeline_routes_ready = True
            routes = self.routed_experts_capturer.get_device_buffer()[:num_tokens]
            torch.distributed.reduce(
                routes,
                dst=pp.last_rank,
                op=torch.distributed.ReduceOp.SUM,
                group=pp.device_group,
            )
        return original_sample(self, *args, **kwargs)

    execute.__art_pipeline_routes_patched__ = True  # type: ignore[attr-defined]
    GPUModelRunner.execute_model = execute
    GPUModelRunner.sample_tokens = sample


def patch_pipeline_routed_experts_validation() -> None:
    """Allow the supported V1 PP aggregation through repeated validation."""
    from vllm.config import VllmConfig

    original = VllmConfig.__post_init__
    if getattr(original, "__art_pipeline_routes_patched__", False):
        return

    @wraps(original)
    def post_init(self: Any) -> None:
        model = self.model_config
        if model is not None and model.enable_return_routed_experts:
            _register_model_route_layout(model)
        pipeline_capture = (
            os.environ.get(PIPELINE_ROUTES_ENV) == PIPELINE_ROUTES_PROTOCOL
            and model is not None
            and model.enable_return_routed_experts
            and self.parallel_config.pipeline_parallel_size > 1
        )
        if not pipeline_capture:
            return original(self)
        transfer = self.kv_transfer_config
        if transfer is not None and transfer.is_kv_transfer_instance:
            raise ValueError(
                "pipeline routed-expert capture is incompatible with KV connectors"
            )
        model.enable_return_routed_experts = False
        try:
            original(self)
        finally:
            model.enable_return_routed_experts = True

    post_init.__art_pipeline_routes_patched__ = True  # type: ignore[attr-defined]
    VllmConfig.__post_init__ = post_init
