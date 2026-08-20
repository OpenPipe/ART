"""Shared metadata and planning primitives for linear-recurrent layers."""

from .buckets import (
    RecurrentSegmentBucketPlan,
    build_recurrent_bucket_plan,
    build_recurrent_segment_bucket_plan,
    build_recurrent_tree_bucket_plans,
    move_recurrent_segment_bucket_plans,
)
from .contract import (
    ChainCostSpec,
    HeadShardedFullTreePlannerConfig,
    LinearRecurrentContract,
    PartitionKind,
    ProjectedStreamSpec,
    RecurrentStateSpec,
    TokenShardedChainSpec,
)
from .prefix_tree import (
    RecurrentPackedExecutionSpec,
    RecurrentSegmentSpec,
    parse_recurrent_prefix_tree_segments,
)

__all__ = [
    "ChainCostSpec",
    "HeadShardedFullTreePlannerConfig",
    "LinearRecurrentContract",
    "PartitionKind",
    "ProjectedStreamSpec",
    "RecurrentPackedExecutionSpec",
    "RecurrentSegmentBucketPlan",
    "RecurrentSegmentSpec",
    "RecurrentStateSpec",
    "TokenShardedChainSpec",
    "build_recurrent_bucket_plan",
    "build_recurrent_segment_bucket_plan",
    "build_recurrent_tree_bucket_plans",
    "move_recurrent_segment_bucket_plans",
    "parse_recurrent_prefix_tree_segments",
]
