from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .builder import build_dense_reference_mask, build_prefix_tree_attention_spec
    from .layout_index import TokenLayoutIndex
    from .types import (
        ArtContextParallelState,
        AttnMaskKind,
        AttnSlice,
        ContextParallelConfig,
        DispatchedPackedTensors,
        FlexMaskSpec,
        PackedBatchAttentionSpec,
        PackedRowAttentionSpec,
        ParallelTopology,
        PreparedMegatronBatch,
        TokenRange,
    )

__all__ = [
    "ArtContextParallelState",
    "AttnMaskKind",
    "AttnSlice",
    "DispatchedPackedTensors",
    "FlexMaskSpec",
    "PackedBatchAttentionSpec",
    "PackedRowAttentionSpec",
    "ParallelTopology",
    "PreparedMegatronBatch",
    "ContextParallelConfig",
    "TokenRange",
    "TokenLayoutIndex",
    "build_dense_reference_mask",
    "build_prefix_tree_attention_spec",
]

_EXPORT_MODULES = {
    "TokenLayoutIndex": ".layout_index",
    "build_dense_reference_mask": ".builder",
    "build_prefix_tree_attention_spec": ".builder",
    "ArtContextParallelState": ".types",
    "AttnMaskKind": ".types",
    "AttnSlice": ".types",
    "DispatchedPackedTensors": ".types",
    "FlexMaskSpec": ".types",
    "PackedBatchAttentionSpec": ".types",
    "PackedRowAttentionSpec": ".types",
    "ParallelTopology": ".types",
    "PreparedMegatronBatch": ".types",
    "ContextParallelConfig": ".types",
    "TokenRange": ".types",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
