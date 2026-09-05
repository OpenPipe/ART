"""The vectorized CP plan evaluator reproduces the scalar reference exactly.

``_evaluate_plan`` prices one chunk assignment (per-rank forward and backward
time) inside the context-parallel assignment search. It was rewritten from
per-rank torch mask operations into numpy mask algebra over all ranks, sources
and waves at once (about six times cheaper per evaluation); the scalar cost
formulas and stage simulations are shared. The pre-rewrite implementation is
kept here as the reference: every randomized instance below must price
identically, floats included.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch

pytest.importorskip("megatron.core")
from art.megatron.context_parallel import runtime as rt  # noqa: E402
from art.megatron.context_parallel.runtime import (  # noqa: E402
    _comm_cost_ms,
    _simulate_backward_time_ms,
    _simulate_forward_time_ms,
    _stage_cost_ms,
)
from art.megatron.context_parallel.types import (  # noqa: E402
    ContextParallelConfig,
    TokenRange,
)

_CHUNK_MASK_STATS_TORCH_THRESHOLD = 1024


def _reference_chunk_mask_stats(
    *,
    chunk_lengths: tuple[int, ...],
    chunk_mask: torch.Tensor,
    chunk_lengths_tensor: torch.Tensor | None = None,
) -> tuple[int, int]:
    if (
        chunk_lengths_tensor is not None
        and len(chunk_lengths) >= _CHUNK_MASK_STATS_TORCH_THRESHOLD
    ):
        if int(chunk_mask.numel()) == 0 or not bool(chunk_mask.any().item()):
            return 0, 0
        token_count = int(chunk_lengths_tensor[chunk_mask].sum().item())
        run_starts = chunk_mask.clone()
        run_starts[1:] = torch.logical_and(
            run_starts[1:], torch.logical_not(chunk_mask[:-1])
        )
        range_count = int(run_starts.sum().item())
        return token_count, range_count
    token_count = 0
    range_count = 0
    in_run = False
    for is_set, length in zip(chunk_mask.tolist(), chunk_lengths, strict=True):
        if bool(is_set):
            token_count += int(length)
            if not in_run:
                range_count += 1
                in_run = True
            continue
        in_run = False
    return token_count, range_count


def _reference_evaluate_plan(
    *,
    chunk_ranges: tuple[TokenRange, ...],
    pair_matrix: list[list[int]] | torch.Tensor,
    owners: tuple[int, ...],
    wave_assignment: tuple[int, ...],
    cp_size: int,
    config: ContextParallelConfig,
    pair_positive: torch.Tensor | None = None,
    chunk_lengths: tuple[int, ...] | None = None,
    chunk_lengths_tensor: torch.Tensor | None = None,
) -> dict[str, Any]:
    rank_scores: list[float] = []
    rank_forward_ms: list[float] = []
    rank_backward_ms: list[float] = []
    chunk_count = len(chunk_ranges)
    wave_count = max(wave_assignment, default=0) + 1 if wave_assignment else 0
    pair_counts = (
        pair_matrix
        if isinstance(pair_matrix, torch.Tensor) and pair_matrix.dtype == torch.int64
        else torch.as_tensor(pair_matrix, dtype=torch.int64)
    )
    if pair_positive is None:
        pair_positive = pair_counts > 0
    if chunk_lengths is None:
        chunk_lengths = tuple(int(range_.size()) for range_ in chunk_ranges)
    if (
        chunk_lengths_tensor is None
        and len(chunk_lengths) >= _CHUNK_MASK_STATS_TORCH_THRESHOLD
    ):
        chunk_lengths_tensor = torch.tensor(chunk_lengths, dtype=torch.int64)
    owners_tensor = torch.tensor(owners, dtype=torch.int64)
    wave_tensor = torch.tensor(
        wave_assignment,
        dtype=torch.int64,
    )
    owner_masks = [owners_tensor == rank for rank in range(cp_size)]
    owner_indices = [
        torch.nonzero(owner_mask, as_tuple=False).flatten()
        for owner_mask in owner_masks
    ]
    empty_pair_counts = pair_counts.new_zeros((0, chunk_count))
    empty_pair_positive = pair_positive.new_zeros((0, chunk_count))
    pair_counts_by_rank_rows = [
        (
            empty_pair_counts
            if int(owner_index.numel()) == 0
            else pair_counts.index_select(0, owner_index)
        )
        for owner_index in owner_indices
    ]
    pair_positive_by_rank_rows = [
        (
            empty_pair_positive
            if int(owner_index.numel()) == 0
            else pair_positive.index_select(0, owner_index)
        )
        for owner_index in owner_indices
    ]
    pair_positive_by_rank_cols = [
        (
            torch.zeros(chunk_count, dtype=torch.bool)
            if int(rank_rows.numel()) == 0
            else rank_rows.any(dim=0)
        )
        for rank_rows in pair_positive_by_rank_rows
    ]
    wave_masks = [wave_tensor == wave_index for wave_index in range(wave_count)]

    for rank in range(cp_size):
        owned_q_mask = owner_masks[rank]
        owned_q_indices = owner_indices[rank]
        owned_pair_counts = pair_counts_by_rank_rows[rank]
        owned_pair_positive = pair_positive_by_rank_rows[rank]
        owned_positive_cols = pair_positive_by_rank_cols[rank]

        local_pairs = (
            0
            if int(owned_q_indices.numel()) == 0
            else int(owned_pair_counts.index_select(1, owned_q_indices).sum().item())
        )
        local_q_mask = torch.zeros(chunk_count, dtype=torch.bool)
        if int(owned_q_indices.numel()) > 0:
            touched_local_q = owned_pair_positive.index_select(1, owned_q_indices).any(
                dim=1
            )
            if bool(touched_local_q.any().item()):
                local_q_mask[owned_q_indices[touched_local_q]] = True
        local_k_mask = owned_q_mask & owned_positive_cols
        local_q_tokens, local_q_range_count = _reference_chunk_mask_stats(
            chunk_lengths=chunk_lengths,
            chunk_mask=local_q_mask,
            chunk_lengths_tensor=chunk_lengths_tensor,
        )
        local_k_tokens, local_k_range_count = _reference_chunk_mask_stats(
            chunk_lengths=chunk_lengths,
            chunk_mask=local_k_mask,
            chunk_lengths_tensor=chunk_lengths_tensor,
        )
        local_stage_ms = _stage_cost_ms(
            pair_count=local_pairs,
            q_tokens=local_q_tokens,
            k_tokens=local_k_tokens,
            q_range_count=local_q_range_count,
            k_range_count=local_k_range_count,
            config=config,
            backward=False,
            local=True,
        )
        local_backward_ms = _stage_cost_ms(
            pair_count=local_pairs,
            q_tokens=local_q_tokens,
            k_tokens=local_k_tokens,
            q_range_count=local_q_range_count,
            k_range_count=local_k_range_count,
            config=config,
            backward=True,
            local=True,
        )

        remote_stage_ms: list[float] = []
        remote_fetch_ms: list[float] = []
        remote_backward_ms: list[float] = []
        remote_reduce_ms: list[float] = []
        for wave_index in range(wave_count):
            request_tokens_by_source = [0 for _ in range(cp_size)]
            request_range_counts_by_source = [0 for _ in range(cp_size)]
            request_pairs = 0
            touched_q_mask = torch.zeros(chunk_count, dtype=torch.bool)
            for source_rank in range(cp_size):
                if source_rank == rank:
                    continue
                touched_source_mask = (
                    owner_masks[source_rank]
                    & wave_masks[wave_index]
                    & owned_positive_cols
                )
                (
                    request_tokens_by_source[source_rank],
                    request_range_counts_by_source[source_rank],
                ) = _reference_chunk_mask_stats(
                    chunk_lengths=chunk_lengths,
                    chunk_mask=touched_source_mask,
                    chunk_lengths_tensor=chunk_lengths_tensor,
                )
                if request_tokens_by_source[source_rank] <= 0:
                    continue
                touched_source_indices = torch.nonzero(
                    touched_source_mask,
                    as_tuple=False,
                ).flatten()
                request_pairs += int(
                    owned_pair_counts.index_select(1, touched_source_indices)
                    .sum()
                    .item()
                )
                touched_remote_q = owned_pair_positive.index_select(
                    1,
                    touched_source_indices,
                ).any(dim=1)
                if bool(touched_remote_q.any().item()):
                    touched_q_mask[owned_q_indices[touched_remote_q]] = True
            recv_tokens = sum(request_tokens_by_source)
            recv_range_count = sum(request_range_counts_by_source)
            if request_pairs <= 0 and recv_tokens <= 0 and recv_range_count <= 0:
                continue

            send_tokens_by_peer = [0 for _ in range(cp_size)]
            send_range_counts_by_peer = [0 for _ in range(cp_size)]
            aggregate_send_mask = torch.zeros(chunk_count, dtype=torch.bool)
            owned_wave_mask = owned_q_mask & wave_masks[wave_index]
            if bool(owned_wave_mask.any().item()):
                for peer_rank in range(cp_size):
                    if peer_rank == rank:
                        continue
                    send_mask = owned_wave_mask & pair_positive_by_rank_cols[peer_rank]
                    (
                        send_tokens_by_peer[peer_rank],
                        send_range_counts_by_peer[peer_rank],
                    ) = _reference_chunk_mask_stats(
                        chunk_lengths=chunk_lengths,
                        chunk_mask=send_mask,
                        chunk_lengths_tensor=chunk_lengths_tensor,
                    )
                    if send_tokens_by_peer[peer_rank] > 0:
                        aggregate_send_mask |= send_mask
            (
                send_tokens_by_peer[rank],
                send_range_counts_by_peer[rank],
            ) = _reference_chunk_mask_stats(
                chunk_lengths=chunk_lengths,
                chunk_mask=aggregate_send_mask,
                chunk_lengths_tensor=chunk_lengths_tensor,
            )

            send_tokens = sum(send_tokens_by_peer)
            q_tokens, q_range_count = _reference_chunk_mask_stats(
                chunk_lengths=chunk_lengths,
                chunk_mask=touched_q_mask,
                chunk_lengths_tensor=chunk_lengths_tensor,
            )
            remote_stage_ms.append(
                _stage_cost_ms(
                    pair_count=request_pairs,
                    q_tokens=q_tokens,
                    k_tokens=recv_tokens,
                    q_range_count=q_range_count,
                    k_range_count=recv_range_count,
                    config=config,
                    backward=False,
                    local=False,
                )
            )
            remote_backward_ms.append(
                _stage_cost_ms(
                    pair_count=request_pairs,
                    q_tokens=q_tokens,
                    k_tokens=recv_tokens,
                    q_range_count=q_range_count,
                    k_range_count=recv_range_count,
                    config=config,
                    backward=True,
                    local=False,
                )
            )
            remote_fetch_ms.append(
                _comm_cost_ms(
                    tokens=max(send_tokens, recv_tokens),
                    range_count=max(sum(send_range_counts_by_peer), recv_range_count),
                    config=config,
                    backward=False,
                )
            )
            remote_reduce_ms.append(
                _comm_cost_ms(
                    tokens=max(send_tokens, recv_tokens),
                    range_count=max(sum(send_range_counts_by_peer), recv_range_count),
                    config=config,
                    backward=True,
                )
            )

        forward_ms = _simulate_forward_time_ms(
            local_stage_ms=local_stage_ms if local_pairs > 0 else 0.0,
            remote_stage_ms=tuple(remote_stage_ms),
            remote_fetch_ms=tuple(remote_fetch_ms),
        )
        backward_ms = _simulate_backward_time_ms(
            local_stage_ms=local_backward_ms if local_pairs > 0 else 0.0,
            remote_stage_ms=tuple(remote_backward_ms),
            remote_reduce_ms=tuple(remote_reduce_ms),
        )
        rank_forward_ms.append(float(forward_ms))
        rank_backward_ms.append(float(backward_ms))
        rank_scores.append(float(forward_ms + backward_ms))
    return {
        "score": max(rank_scores, default=0.0),
        "rank_scores": tuple(rank_scores),
        "rank_forward_ms": tuple(rank_forward_ms),
        "rank_backward_ms": tuple(rank_backward_ms),
    }


def _random_instance(seed: int, *, chunk_count: int | None = None):
    rng = random.Random(seed)
    cp_size = rng.choice((1, 2, 4, 8))
    n = rng.randint(0, 48) if chunk_count is None else chunk_count
    chunk_size = 512
    lengths = [chunk_size] * n
    if n:
        lengths[-1] = rng.randint(1, chunk_size)
    ranges, start = [], 0
    for length in lengths:
        ranges.append(TokenRange(start=start, end=start + length))
        start += length
    pairs = [[0] * n for _ in range(n)]
    density = rng.choice((0.15, 0.4, 0.8))
    for i in range(n):
        if rng.random() < 0.1:
            continue  # a q chunk that attends nothing (e.g. padding)
        for j in range(i + 1):
            if rng.random() < density or j == i:
                pairs[i][j] = rng.randint(1, chunk_size * chunk_size)
    owners = tuple(rng.randrange(cp_size) for _ in range(n))
    wave_count = rng.randint(1, 4)
    waves = rt._wave_assignment(chunk_count=n, wave_count=wave_count)
    return tuple(ranges), pairs, owners, waves, cp_size


def _both(
    ranges, pairs, owners, waves, cp_size
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = ContextParallelConfig()
    program = rt._pair_program(pairs, chunk_ranges=ranges)
    new = rt._evaluate_plan(
        program=program,
        owners=owners,
        wave_assignment=waves,
        cp_size=cp_size,
        config=config,
    )
    reference = _reference_evaluate_plan(
        chunk_ranges=ranges,
        pair_matrix=torch.as_tensor(pairs, dtype=torch.int64).reshape(
            len(ranges), len(ranges)
        ),
        owners=owners,
        wave_assignment=waves,
        cp_size=cp_size,
        config=config,
    )
    return new, reference


@pytest.mark.parametrize("seed", range(120))
def test_vectorized_evaluator_matches_scalar_reference(seed: int) -> None:
    new, reference = _both(*_random_instance(seed))
    assert new == reference


def test_vectorized_evaluator_matches_reference_on_a_long_row() -> None:
    # The reference switches to its torch path beyond 1,024 chunks.
    new, reference = _both(*_random_instance(7, chunk_count=1_100))
    assert new == reference


@pytest.mark.parametrize("seed", range(200, 230))
def test_batched_evaluation_matches_one_by_one(seed: int) -> None:
    import numpy as np

    ranges, pairs, _owners, waves, cp_size = _random_instance(seed)
    rng = random.Random(seed)
    batch = [
        tuple(rng.randrange(cp_size) for _ in range(len(ranges))) for _ in range(7)
    ]
    program = rt._pair_program(pairs, chunk_ranges=ranges)
    config = ContextParallelConfig()
    together = rt._evaluate_plans(
        program=program,
        owners_batch=np.asarray(batch, dtype=np.int64).reshape(len(batch), len(ranges)),
        wave_assignment=waves,
        cp_size=cp_size,
        config=config,
    )
    for owners, evaluation in zip(batch, together, strict=True):
        assert evaluation == _reference_evaluate_plan(
            chunk_ranges=ranges,
            pair_matrix=torch.as_tensor(pairs, dtype=torch.int64).reshape(
                len(ranges), len(ranges)
            ),
            owners=owners,
            wave_assignment=waves,
            cp_size=cp_size,
            config=config,
        )


def test_empty_row_prices_to_zero() -> None:
    new, reference = _both((), [], (), (), 4)
    assert new == reference
    assert new["score"] == 0.0 and len(new["rank_scores"]) == 4


def test_search_is_unchanged_by_the_rewrite() -> None:
    """The assignment search only consumes evaluations; on a fixed program its
    result (owners, waves, score) is a pure function of them."""

    ranges, pairs, _owners, _waves, cp_size = _random_instance(11, chunk_count=40)
    q_weights = [float(sum(row)) for row in pairs]
    owners, waves, evaluation = rt._search_generic_chunk_assignment(
        chunk_ranges=ranges,
        pair_matrix=torch.as_tensor(pairs, dtype=torch.int64),
        q_weights=q_weights,
        cp_size=max(cp_size, 2),
        config=ContextParallelConfig(),
    )
    _new, reference = _both(ranges, pairs, owners, waves, max(cp_size, 2))
    assert evaluation == reference
