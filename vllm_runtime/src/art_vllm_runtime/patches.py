"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
import inspect
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_vllm_runtime_patches() -> None:
    patch_transformers_v5_compat()
    subclass_chat_completion_request()
    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    patch_nccl_unique_id_bootstrap()
    patch_layerwise_reload_shadow_attrs()
    patch_routed_experts_prefix_cache_sidecar()


def patch_transformers_v5_compat() -> None:
    _patch_rope_validation_ignore_keys()
    _patch_qwen3_vl_moe_tie_word_embeddings()


def _patch_rope_validation_ignore_keys() -> None:
    from transformers.configuration_utils import PretrainedConfig

    original = PretrainedConfig.convert_rope_params_to_dict
    if getattr(original, "__art_patched__", False):
        return

    def patched(self: Any, ignore_keys_at_rope_validation: Any = None, **kwargs: Any):
        if ignore_keys_at_rope_validation is not None:
            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation)
        return original(
            self,
            ignore_keys_at_rope_validation=ignore_keys_at_rope_validation,
            **kwargs,
        )

    patched.__art_patched__ = True  # type: ignore[attr-defined]
    PretrainedConfig.convert_rope_params_to_dict = patched  # type: ignore[method-assign]


def _patch_qwen3_vl_moe_tie_word_embeddings() -> None:
    from transformers import Qwen3VLMoeTextConfig

    setattr(Qwen3VLMoeTextConfig, "tie_word_embeddings", False)


def subclass_chat_completion_request() -> None:
    from vllm.entrypoints.openai.chat_completion import protocol

    if getattr(protocol, "_art_chat_completion_request_patched", False):
        return

    class ChatCompletionRequest(protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0
            self.return_token_ids = True

    protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]
    setattr(protocol, "_art_chat_completion_request_patched", True)


def patch_listen_for_disconnect() -> None:
    import vllm.entrypoints.utils

    if getattr(vllm.entrypoints.utils, "_art_listen_for_disconnect_patched", False):
        return

    async def patched_listen_for_disconnect(request: Any) -> None:
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    break
        except UnboundLocalError:
            pass

    vllm.entrypoints.utils.listen_for_disconnect = patched_listen_for_disconnect  # ty:ignore[invalid-assignment]
    setattr(vllm.entrypoints.utils, "_art_listen_for_disconnect_patched", True)


def patch_tool_parser_manager() -> None:
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    original = ToolParserManager.get_tool_parser
    if getattr(original, "__art_patched__", False):
        return

    def patched_get_tool_parser(name: str) -> type:
        tool_parser_class = original(name)
        current = tool_parser_class.extract_tool_calls_streaming
        if getattr(current, "__art_patched__", False):
            return tool_parser_class

        def patch(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return current(*args, **kwargs) or DeltaMessage()

        patch.__art_patched__ = True  # type: ignore[attr-defined]
        tool_parser_class.extract_tool_calls_streaming = patch  # ty:ignore[invalid-assignment]
        return tool_parser_class

    patched_get_tool_parser.__art_patched__ = True  # type: ignore[attr-defined]
    ToolParserManager.get_tool_parser = patched_get_tool_parser  # ty:ignore[invalid-assignment]


def _restore_nccl_unique_id_payload(
    payload: object,
    template: object | None,
) -> object:
    from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId

    if not isinstance(payload, (bytes, bytearray)) or not isinstance(
        template, ncclUniqueId
    ):
        return payload
    raw = bytes(payload)
    assert len(raw) == ctypes.sizeof(ncclUniqueId)
    unique_id = ncclUniqueId()
    ctypes.memmove(ctypes.byref(unique_id), raw, len(raw))
    return unique_id


def _normalize_nccl_comm_init_rank_unique_id(library: Any, unique_id: object) -> object:
    if isinstance(unique_id, (bytes, bytearray)):
        return library.unique_id_from_bytes(bytes(unique_id))
    return unique_id


def patch_nccl_unique_id_bootstrap() -> None:
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.distributed.utils import StatelessProcessGroup

    original_broadcast = StatelessProcessGroup.broadcast_obj
    if not getattr(original_broadcast, "__art_patched__", False):

        def patched_broadcast(self: Any, obj: Any | None, src: int) -> Any:
            return _restore_nccl_unique_id_payload(
                original_broadcast(self, obj, src), obj
            )

        patched_broadcast.__art_patched__ = True  # type: ignore[attr-defined]
        StatelessProcessGroup.broadcast_obj = patched_broadcast  # type: ignore[method-assign]

    original_comm_init_rank = NCCLLibrary.ncclCommInitRank
    if getattr(original_comm_init_rank, "__art_patched__", False):
        return

    def patched_comm_init_rank(
        self: Any,
        world_size: int,
        unique_id: object,
        rank: int,
    ) -> Any:
        unique_id = _normalize_nccl_comm_init_rank_unique_id(self, unique_id)
        return original_comm_init_rank(self, world_size, unique_id, rank)

    patched_comm_init_rank.__art_patched__ = True  # type: ignore[attr-defined]
    NCCLLibrary.ncclCommInitRank = patched_comm_init_rank  # type: ignore[method-assign]


def _drop_reload_shadow_attrs(layer: Any, names: Any) -> None:
    for name in names:
        if (
            name in getattr(layer, "__dict__", {})
            and name not in layer._parameters
            and name not in layer._buffers
            and name not in layer._modules
        ):
            delattr(layer, name)


def patch_layerwise_reload_shadow_attrs() -> None:
    """Allow vLLM layerwise reload to restore processed DSV4 MegaMoE params.

    DeepSeek V4 MegaMoE drops loader-side Parameters after transforming them for
    DeepGEMM. Some vLLM builds leave same-name plain attributes behind; PyTorch
    then rejects register_parameter during the next checkpoint-format reload.
    """
    from vllm.model_executor.model_loader.reload import layerwise, meta

    if getattr(meta, "_art_reload_shadow_attrs_patched", False):
        return

    original_restore_layer_on_meta = meta.restore_layer_on_meta
    original_place_kernel_tensors = layerwise._place_kernel_tensors

    def restore_layer_on_meta(layer: Any, info: Any) -> None:
        restore_params, restore_buffers = info.restore_metadata
        _drop_reload_shadow_attrs(layer, tuple(restore_params) + tuple(restore_buffers))
        return original_restore_layer_on_meta(layer, info)

    def _place_kernel_tensors(layer: Any, info: Any) -> None:
        assert info.kernel_tensors is not None
        parameters, buffers = info.kernel_tensors
        _drop_reload_shadow_attrs(layer, tuple(parameters) + tuple(buffers))
        return original_place_kernel_tensors(layer, info)

    restore_layer_on_meta.__art_patched__ = True  # type: ignore[attr-defined]
    _place_kernel_tensors.__art_patched__ = True  # type: ignore[attr-defined]
    meta.restore_layer_on_meta = restore_layer_on_meta  # type: ignore[method-assign]
    layerwise.restore_layer_on_meta = restore_layer_on_meta  # type: ignore[method-assign]
    layerwise._place_kernel_tensors = _place_kernel_tensors  # type: ignore[method-assign]
    setattr(meta, "_art_reload_shadow_attrs_patched", True)


def _lora_cache_key(lora_request: Any) -> tuple[Any, ...]:
    if lora_request is None:
        return ()
    return (
        getattr(lora_request, "adapter_id", None),
        getattr(lora_request, "name", None),
        getattr(lora_request, "path", None),
    )


def _request_token_ids(req_state: Any) -> list[int] | None:
    prompt_token_ids = getattr(req_state, "prompt_token_ids", None)
    if prompt_token_ids is None:
        return None
    return list(prompt_token_ids) + list(getattr(req_state, "output_token_ids", ()))


def _route_block_key(
    token_ids: list[int],
    end: int,
    lora_key: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (lora_key, tuple(token_ids[:end]))


def _runner_block_size(runner: Any) -> int:
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if groups and len(groups) == 1:
        return int(groups[0].kv_cache_spec.block_size)
    return int(getattr(runner.cache_config, "block_size", 16))


def _request_snapshots(
    runner: Any, ordered: dict[str, int]
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for req_id in ordered:
        req_state = runner.requests.get(req_id)
        if req_state is None:
            continue
        token_ids = _request_token_ids(req_state)
        if token_ids is None:
            continue
        snapshots[req_id] = {
            "token_ids": token_ids,
            "lora_key": _lora_cache_key(getattr(req_state, "lora_request", None)),
            "num_computed_tokens": int(getattr(req_state, "num_computed_tokens", 0)),
        }
    return snapshots


def patch_routed_experts_prefix_cache_sidecar() -> None:
    from vllm.model_executor.layers.fused_moe import routed_experts_capturer

    if getattr(routed_experts_capturer, "_art_prefix_route_sidecar_patched", False):
        return

    host_cls = routed_experts_capturer._RoutedExpertsHostCache
    capturer_cls = routed_experts_capturer._RoutedExpertsCapturerReal

    original_host_init = host_cls.__init__
    original_get_or_grow_buffer = host_cls.get_or_grow_buffer
    original_free_request = host_cls.free_request
    original_scatter_to_host = capturer_cls._scatter_to_host
    original_get_routed_experts = capturer_cls.get_routed_experts
    original_issue_routing_d2h_copy = routed_experts_capturer.issue_routing_d2h_copy

    def host_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_host_init(self, *args, **kwargs)
        self._art_req_filled_masks: dict[str, np.ndarray] = {}
        self._art_prefix_route_blocks: dict[tuple[Any, ...], np.ndarray] = {}
        self._art_prefix_route_waiters: dict[
            tuple[Any, ...], list[tuple[str, int, int]]
        ] = {}
        self._art_prefix_route_needs_by_req: dict[str, set[tuple[Any, ...]]] = {}
        self._art_prefix_route_hydrated_tokens = 0
        self._art_prefix_route_cache_misses = 0
        self._art_prefix_route_cache_conflicts = 0

    def get_or_grow_buffer(self: Any, req_id: str, max_pos: int) -> np.ndarray:
        buf = original_get_or_grow_buffer(self, req_id, max_pos)
        mask = self._art_req_filled_masks.get(req_id)
        if mask is None:
            self._art_req_filled_masks[req_id] = np.zeros(buf.shape[0], dtype=np.bool_)
        elif mask.shape[0] < buf.shape[0]:
            new_mask = np.zeros(buf.shape[0], dtype=np.bool_)
            new_mask[: mask.shape[0]] = mask
            self._art_req_filled_masks[req_id] = new_mask
        return buf

    def free_request(self: Any, req_id: str) -> None:
        original_free_request(self, req_id)
        self._art_req_filled_masks.pop(req_id, None)
        for key in self._art_prefix_route_needs_by_req.pop(req_id, set()):
            waiters = self._art_prefix_route_waiters.get(key)
            if waiters is None:
                continue
            waiters = [waiter for waiter in waiters if waiter[0] != req_id]
            if waiters:
                self._art_prefix_route_waiters[key] = waiters
            else:
                self._art_prefix_route_waiters.pop(key, None)

    def mark_filled(self: Any, req_id: str, positions: np.ndarray) -> None:
        if positions.size == 0:
            return
        self.get_or_grow_buffer(req_id, int(positions.max()))
        self._art_req_filled_masks[req_id][positions] = True

    def require_filled(self: Any, req_id: str, seqlen: int) -> None:
        mask = self._art_req_filled_masks.get(req_id)
        if mask is None or mask.shape[0] < seqlen or not bool(mask[:seqlen].all()):
            available = (
                mask[:seqlen] if mask is not None else np.zeros(0, dtype=np.bool_)
            )
            missing = np.flatnonzero(~available)[:16].tolist()
            raise RuntimeError(
                "Routed expert capture is incomplete for request "
                f"{req_id}: seqlen={seqlen}, first_missing_positions={missing}"
            )

    def fill_prefix_block(
        self: Any,
        req_id: str,
        start: int,
        end: int,
        value: np.ndarray,
        key: tuple[Any, ...] | None = None,
    ) -> bool:
        buf = self.get_or_grow_buffer(req_id, end - 1)
        mask = self._art_req_filled_masks[req_id]
        if bool(mask[start:end].all()):
            if key is not None:
                needs = self._art_prefix_route_needs_by_req.get(req_id)
                if needs is not None:
                    needs.discard(key)
                    if not needs:
                        self._art_prefix_route_needs_by_req.pop(req_id, None)
            return False
        buf[start:end] = value
        mask[start:end] = True
        self.update_filled_len(req_id, end - 1)
        if key is not None:
            needs = self._art_prefix_route_needs_by_req.get(req_id)
            if needs is not None:
                needs.discard(key)
                if not needs:
                    self._art_prefix_route_needs_by_req.pop(req_id, None)
        return True

    def store_prefix_block(
        self: Any,
        key: tuple[Any, ...],
        value: np.ndarray,
    ) -> None:
        existing = self._art_prefix_route_blocks.get(key)
        if existing is None:
            existing = value.copy()
            self._art_prefix_route_blocks[key] = existing
        elif not np.array_equal(existing, value):
            self._art_prefix_route_cache_conflicts += 1
        hydrated = 0
        for req_id, start, end in self._art_prefix_route_waiters.pop(key, []):
            if self._art_fill_prefix_block(req_id, start, end, existing, key):
                hydrated += end - start
        if hydrated:
            self._art_prefix_route_hydrated_tokens += hydrated
            logger.info(
                "Hydrated %s routed-expert prefix-cache tokens from materialized "
                "route block",
                hydrated,
            )

    def store_prefix_blocks(
        self: Any,
        req_id: str,
        token_ids: list[int],
        lora_key: tuple[Any, ...],
        block_size: int,
        max_pos_exclusive: int,
    ) -> None:
        if block_size <= 0:
            return
        upper = min(max_pos_exclusive, len(token_ids))
        upper -= upper % block_size
        if upper <= 0:
            return
        buf = self.get_buffer(req_id)
        mask = self._art_req_filled_masks.get(req_id)
        if buf is None or mask is None:
            return
        for end in range(block_size, upper + 1, block_size):
            start = end - block_size
            if end > mask.shape[0] or not bool(mask[start:end].all()):
                continue
            key = _route_block_key(token_ids, end, lora_key)
            value = buf[start:end].copy()
            self._art_store_prefix_block(key, value)

    def need_cached_prefix(
        self: Any,
        req_id: str,
        token_ids: list[int],
        lora_key: tuple[Any, ...],
        cached_len: int,
        block_size: int,
    ) -> None:
        if block_size <= 0 or cached_len <= 0:
            return
        upper = min(cached_len, len(token_ids))
        upper -= upper % block_size
        if upper <= 0:
            return
        hydrated = 0
        for end in range(block_size, upper + 1, block_size):
            start = end - block_size
            mask = self._art_req_filled_masks.get(req_id)
            if (
                mask is not None
                and end <= mask.shape[0]
                and bool(mask[start:end].all())
            ):
                continue
            key = _route_block_key(token_ids, end, lora_key)
            value = self._art_prefix_route_blocks.get(key)
            if value is None:
                needs = self._art_prefix_route_needs_by_req.setdefault(req_id, set())
                if key not in needs:
                    self._art_prefix_route_waiters.setdefault(key, []).append(
                        (req_id, start, end)
                    )
                    needs.add(key)
                    self._art_prefix_route_cache_misses += block_size
                continue
            if self._art_fill_prefix_block(req_id, start, end, value, key):
                hydrated += block_size
        if hydrated:
            self._art_prefix_route_hydrated_tokens += hydrated
            logger.info(
                "Hydrated %s routed-expert prefix-cache tokens for request %s",
                hydrated,
                req_id,
            )

    def require_no_unmet_prefix_route_needs(self: Any, req_id: str) -> None:
        needs = self._art_prefix_route_needs_by_req.get(req_id)
        if needs:
            raise RuntimeError(
                "Routed expert capture is missing materialized prefix-cache "
                f"route blocks for request {req_id}: unmet_blocks={len(needs)}"
            )

    def scatter_to_host(self: Any) -> None:
        positions = self._pending_positions.copy()
        scheduled = dict(self._pending_num_scheduled or {})
        metadata = getattr(self, "_art_pending_route_metadata", None)
        original_scatter_to_host(self)
        host_cache = self.host_cache
        if host_cache is None:
            return
        block_size = int((metadata or {}).get("block_size", 0))
        snapshots = (metadata or {}).get("snapshots", {})
        offset = 0
        for req_id, n_tokens in scheduled.items():
            pos = positions[offset : offset + n_tokens]
            host_cache._art_mark_filled(req_id, pos)
            snapshot = snapshots.get(req_id)
            if snapshot is not None and pos.size:
                host_cache._art_store_prefix_blocks(
                    req_id,
                    snapshot["token_ids"],
                    snapshot["lora_key"],
                    block_size,
                    int(pos.max()) + 1,
                )
            offset += n_tokens
        self._art_pending_route_metadata = None

    def get_routed_experts(
        self: Any,
        req_id: str,
        seqlen: int | None = None,
        free_slot: bool = True,
    ) -> np.ndarray | None:
        if self.host_cache is not None:
            filled = self.host_cache.get_filled_len(req_id)
            effective_len = min(filled, seqlen) if seqlen is not None else filled
            if effective_len > 0:
                self.host_cache._art_require_no_unmet_prefix_route_needs(req_id)
                self.host_cache._art_require_filled(req_id, effective_len)
        return original_get_routed_experts(self, req_id, seqlen, free_slot)

    def issue_routing_d2h_copy(
        input_batch_req_ids: list[str],
        num_scheduled_tokens: dict[str, int],
        positions: Any,
        positions_cpu: Any,
    ) -> None:
        capturer = routed_experts_capturer.get_global_experts_capturer()
        host_cache = capturer.get_host_cache() if capturer is not None else None
        frame = inspect.currentframe()
        runner = frame.f_back.f_locals.get("self") if frame and frame.f_back else None
        ordered = {
            req_id: num_scheduled_tokens[req_id]
            for req_id in input_batch_req_ids
            if req_id in num_scheduled_tokens
        }
        metadata: dict[str, Any] | None = None
        if host_cache is not None and runner is not None:
            block_size = _runner_block_size(runner)
            snapshots = _request_snapshots(runner, ordered)
            for req_id, snapshot in snapshots.items():
                host_cache._art_need_cached_prefix(
                    req_id,
                    snapshot["token_ids"],
                    snapshot["lora_key"],
                    snapshot["num_computed_tokens"],
                    block_size,
                )
            metadata = {"block_size": block_size, "snapshots": snapshots}
        original_issue_routing_d2h_copy(
            input_batch_req_ids,
            num_scheduled_tokens,
            positions,
            positions_cpu,
        )
        if capturer is not None and metadata is not None and sum(ordered.values()) > 0:
            capturer._art_pending_route_metadata = metadata

    host_cls.__init__ = host_init  # type: ignore[method-assign]
    host_cls.get_or_grow_buffer = get_or_grow_buffer  # type: ignore[method-assign]
    host_cls.free_request = free_request  # type: ignore[method-assign]
    host_cls._art_mark_filled = mark_filled  # type: ignore[attr-defined]
    host_cls._art_require_filled = require_filled  # type: ignore[attr-defined]
    host_cls._art_fill_prefix_block = fill_prefix_block  # type: ignore[attr-defined]
    host_cls._art_store_prefix_block = store_prefix_block  # type: ignore[attr-defined]
    host_cls._art_store_prefix_blocks = store_prefix_blocks  # type: ignore[attr-defined]
    host_cls._art_need_cached_prefix = need_cached_prefix  # type: ignore[attr-defined]
    host_cls._art_require_no_unmet_prefix_route_needs = (  # type: ignore[attr-defined]
        require_no_unmet_prefix_route_needs
    )
    capturer_cls._scatter_to_host = scatter_to_host  # type: ignore[method-assign]
    capturer_cls.get_routed_experts = get_routed_experts  # type: ignore[method-assign]
    from vllm.v1.worker import gpu_model_runner

    gpu_model_runner_issue_routing_d2h_copy = getattr(
        gpu_model_runner, "issue_routing_d2h_copy", None
    )
    if gpu_model_runner_issue_routing_d2h_copy is not original_issue_routing_d2h_copy:
        raise RuntimeError(
            "ART routed-expert prefix-cache patch expected "
            "vllm.v1.worker.gpu_model_runner.issue_routing_d2h_copy to reference "
            "vllm.model_executor.layers.fused_moe.routed_experts_capturer."
            "issue_routing_d2h_copy. vLLM internals changed; update the patch."
        )

    routed_experts_capturer.issue_routing_d2h_copy = issue_routing_d2h_copy
    gpu_model_runner.issue_routing_d2h_copy = issue_routing_d2h_copy
    setattr(routed_experts_capturer, "_art_prefix_route_sidecar_patched", True)
