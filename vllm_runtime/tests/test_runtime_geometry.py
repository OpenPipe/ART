from types import SimpleNamespace

from art_vllm_runtime.runtime_geometry import _profiled_kv_geometry


def test_profiled_kv_geometry_uses_finalized_tensor_allocations() -> None:
    config = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=(object(), object()),
        kv_cache_tensors=(
            SimpleNamespace(size=8 * 12),
            SimpleNamespace(size=8 * 20),
        ),
    )

    assert _profiled_kv_geometry(config, block_tokens=256) == {
        "kv_block_size": 256,
        "kv_block_bytes_per_rank": 32,
        "kv_capacity_blocks_per_rank": 8,
        "kv_capacity_bytes_per_rank": 256,
    }
