"""Exact post-profile admission geometry from vLLM's scheduler state."""

from typing import Any


def _profiled_kv_geometry(kv_cache_config: Any, *, block_tokens: int) -> dict[str, int]:
    num_blocks = int(kv_cache_config.num_blocks)
    tensors = tuple(kv_cache_config.kv_cache_tensors)
    if not kv_cache_config.kv_cache_groups or not tensors:
        raise RuntimeError("ART paired inference requires a profiled KV cache")
    allocated_bytes = sum(int(tensor.size) for tensor in tensors)
    if (
        min(num_blocks, block_tokens, allocated_bytes) <= 0
        or allocated_bytes % num_blocks
    ):
        raise RuntimeError("vLLM returned invalid KV cache geometry")
    block_bytes = allocated_bytes // num_blocks
    return {
        "kv_block_size": block_tokens,
        "kv_block_bytes_per_rank": block_bytes,
        "kv_capacity_blocks_per_rank": num_blocks,
        "kv_capacity_bytes_per_rank": allocated_bytes,
    }


def patch_runtime_geometry() -> None:
    from vllm.v1.engine.core import EngineCore

    if hasattr(EngineCore, "art_runtime_geometry"):
        return

    def art_runtime_geometry(self: Any) -> dict[str, int]:
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            raise RuntimeError("ART paired inference requires a profiled KV cache")
        return _profiled_kv_geometry(
            kv_cache_config,
            block_tokens=int(self.vllm_config.cache_config.block_size),
        )

    EngineCore.art_runtime_geometry = art_runtime_geometry  # type: ignore[attr-defined]
