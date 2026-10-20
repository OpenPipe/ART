"""Normalize, cache, and union independent per-root GDN decisions.

Kernel buckets may batch independent roots, but never connect their states.
CP collectives still span the replica, with zero lengths on inactive ranks.
"""

from dataclasses import replace
from functools import lru_cache

from art.megatron.context_parallel.layout_index import TokenLayoutIndex

from .gdn_prefix_tree import (
    GdnGlobalExecutionDecision,
    GdnPackedExecutionSpec,
    build_gdn_global_execution_decision,
)


@lru_cache(maxsize=1024)
def _root_decision(spec, layout, config):
    return build_gdn_global_execution_decision(
        spec,
        cp_size=len(layout.token_counts_by_rank),
        attention_token_layout_index=layout,
        planner_config=replace(config, root_closed=False),
    )


def build_root_closed_gdn_decision(spec, *, cp_size, source_layout, planner_config):
    if spec.batch_size != 1:
        raise ValueError("root-closed GDN currently requires one packed row")
    nodes_by_root = {}
    root_by_node = []
    for i, parent in enumerate(spec.tree_parent_indices):
        root = i if parent < 0 else root_by_node[parent]
        root_by_node.append(root)
        nodes_by_root.setdefault(root, []).append(i)
    depth_count = max(spec.tree_depths, default=0) + 1
    loads = [0] * cp_size
    ranges = [[] for _ in loads]
    local = [[[] for _ in range(depth_count)] for _ in loads]
    chains = [[] for _ in range(depth_count)]
    owners = [-1] * len(spec.tree_segments)
    chained = [False] * len(owners)
    has_children = [False] * len(owners)
    cross = 0
    for parent in spec.tree_parent_indices:
        if parent >= 0:
            has_children[parent] = True
    for nodes in nodes_by_root.values():
        segments = tuple(spec.tree_segments[i] for i in nodes)
        start, end = min(s.start for s in segments), max(s.end for s in segments)
        if sum(s.length for s in segments) != end - start:
            raise ValueError("root segments must be contiguous in root-closed packing")
        mapping = {old: new for new, old in enumerate(nodes)}
        normalized = GdnPackedExecutionSpec(
            batch_size=1,
            sequence_length=end - start,
            valid_lengths=(end - start,),
            tree_segments=tuple(
                replace(
                    s,
                    family_index=j,
                    group_id=j,
                    parent_id=j
                    if spec.tree_parent_indices[nodes[j]] < 0
                    else mapping[spec.tree_parent_indices[nodes[j]]],
                    start=s.start - start,
                    end=s.end - start,
                )
                for j, s in enumerate(segments)
            ),
            tree_parent_indices=tuple(
                -1
                if spec.tree_parent_indices[i] < 0
                else mapping[spec.tree_parent_indices[i]]
                for i in nodes
            ),
            tree_depths=tuple(spec.tree_depths[i] for i in nodes),
        )
        active, root_ranges, counts = [], [], []
        for rank, rr in enumerate(source_layout.ownership_ranges_by_rank):
            overlaps = [
                (max(a, start) - start, min(b, end) - start)
                for a, b, _ in rr
                if a < end and b > start
            ]
            if not overlaps:
                continue
            active.append(rank)
            pos, pieces = 0, []
            for a, b in overlaps:
                pieces.append((a, b, pos))
                pos += b - a
            root_ranges.append(tuple(pieces))
            counts.append(pos)
        if not active:
            raise ValueError("root has no owning rank")
        decision = _root_decision(
            normalized,
            TokenLayoutIndex(tuple(root_ranges), tuple(counts)),
            planner_config,
        )
        cross += decision.cross_rank_token_count
        for j, old in enumerate(nodes):
            owners[old] = (
                active[decision.owner_by_node[j]]
                if decision.owner_by_node[j] >= 0
                else -1
            )
            chained[old] = decision.chained_nodes[j]
        for r, global_rank in enumerate(active):
            base = loads[global_rank]
            ranges[global_rank].extend(
                (a + start, b + start, p + base)
                for a, b, p in decision.gdn_ranges_by_rank_by_position[r]
            )
            loads[global_rank] += decision.gdn_token_counts_by_rank[r]
            for depth in range(decision.depth_count):
                local[global_rank][depth].extend(
                    segments[s.family_index]
                    for s in decision.segments_by_rank_depth[r][depth]
                )
        for depth in range(decision.depth_count):
            chains[depth].extend(
                segments[s.family_index]
                for s in decision.chain_segments_by_depth[depth]
            )
    return GdnGlobalExecutionDecision(
        cp_size=cp_size,
        source_layout=source_layout,
        depth_count=depth_count,
        gdn_token_counts_by_rank=tuple(loads),
        owner_by_node=tuple(owners),
        chained_nodes=tuple(chained),
        tree_has_children=tuple(has_children),
        gdn_ranges_by_rank_by_position=tuple(tuple(r) for r in ranges),
        gdn_ranges_by_rank_by_source=tuple(tuple(sorted(r)) for r in ranges),
        segments_by_rank_depth=tuple(tuple(tuple(s) for s in d) for d in local),
        chain_segments_by_depth=tuple(tuple(s) for s in chains),
        cross_rank_token_count=cross,
    )
