"""Exact post-profile admission geometry from vLLM's scheduler state."""

from typing import Any


def patch_runtime_geometry() -> None:
    from vllm.v1.engine.core import EngineCore

    if hasattr(EngineCore, "art_runtime_geometry"):
        return

    def art_runtime_geometry(self: Any) -> dict[str, int]:
        from vllm.v1.core.kv_cache_utils import _pool_bytes_per_block

        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None or not kv_cache_config.kv_cache_groups:
            raise RuntimeError("ART paired inference requires a profiled KV cache")
        num_blocks = int(kv_cache_config.num_blocks)
        block_tokens = int(self.vllm_config.cache_config.block_size)
        block_bytes = int(
            _pool_bytes_per_block(
                self.vllm_config,
                kv_cache_config.kv_cache_groups,
            )
        )
        if min(num_blocks, block_tokens, block_bytes) <= 0:
            raise RuntimeError("vLLM returned invalid KV cache geometry")
        return {
            "kv_block_size": block_tokens,
            "kv_block_bytes_per_rank": block_bytes,
            "kv_capacity_blocks_per_rank": num_blocks,
            "kv_capacity_bytes_per_rank": num_blocks * block_bytes,
        }

    EngineCore.art_runtime_geometry = art_runtime_geometry  # type: ignore[attr-defined]
