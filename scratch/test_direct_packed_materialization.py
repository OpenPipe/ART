from __future__ import annotations

import math
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import pytest
import torch

from art.distributed.data_plane import (
    PackedBatchLeaseError,
    SharedMemoryPackedBatchStore,
    _flatten_packed_tensors,
    packed_plan_storage_byte_count,
)
from art.preprocessing.pack import (
    materialize_packed_tensors,
    materialize_packed_tensors_into,
    prepare_packed_tensors_from_tokenized_datums,
    prepare_packed_tensors_from_tokenized_results,
)
from art.preprocessing.tokenize import TokenizedResult
from art.training.tokenized import TokenizedDatum, TokenizedMoeRoutes
from art.trajectories import Trajectory


class _Decoder:
    def decode(self, token_id: int) -> str:
        return str(token_id)


def _rl_plan():
    prefix = [10, 11, 12, 13]
    results = []
    for branch, (advantage, weight) in enumerate(((2.0, 2.0), (-1.0, 1.0))):
        token_ids = [*prefix, 20 + branch * 2, 21 + branch * 2]
        results.append(
            TokenizedResult(
                advantage=advantage,
                chat="",
                token_ids=token_ids,
                input_pos=list(range(len(token_ids))),
                assistant_mask=[0, 0, 0, 0, 1, 1],
                logprobs=[math.nan] * 4 + [-0.1, -0.2],
                pixel_values=torch.tensor([[1.0, 2.0, 3.0]]),
                image_grid_thw=torch.tensor([[1, 2, 3]]),
                trajectory=Trajectory(),
                choice_offsets=[4],
                extra_logprobs={},
                _tokenizer=_Decoder(),
                weight=weight,
                prompt_id=7,
                prompt_length=4,
            )
        )
    return prepare_packed_tensors_from_tokenized_results(
        results,
        seq_len=16,
        truncate_long_results=False,
        pack_results=True,
        min_prefix_tree_shared_segment_length=1,
    )


def _tokenized_moe_plan():
    common_routes = np.array(
        [
            [[0, 1], [2, 3]],
            [[1, 2], [3, 4]],
            [[2, 3], [4, 5]],
        ],
        dtype=np.uint16,
    )
    datums = []
    for branch in range(2):
        tokens = (1, 2, 3, 10 + branch, 20 + branch)
        targets = tuple((token + 30, token + 60) for token in tokens)
        routes = np.concatenate(
            (
                common_routes,
                np.full((2, 2, 2), 255 + branch, dtype=np.uint16),
            )
        )
        datums.append(
            TokenizedDatum(
                input_tokens=tokens,
                target_tokens=targets,
                weights=((0.0, 0.0),) * 3 + ((1.0, 0.5), (0.25, 1.0)),
                packing_group_id=9,
                moe_routes=TokenizedMoeRoutes(
                    num_experts=257,
                    dtype="uint16",
                    shape=(5, 2, 2),
                    data=(routes.tobytes(),),
                ),
            )
        )
    return prepare_packed_tensors_from_tokenized_datums(
        datums,
        loss="cross_entropy",
        seq_len=16,
        min_prefix_tree_shared_segment_length=1,
    )


def _assert_flat_equal(expected, actual) -> None:
    expected_flat = dict(_flatten_packed_tensors(expected)[0])
    actual_flat = dict(_flatten_packed_tensors(actual)[0])
    assert actual_flat.keys() == expected_flat.keys()
    for name in expected_flat:
        torch.testing.assert_close(
            actual_flat[name], expected_flat[name], rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("plan_factory", [_rl_plan, _tokenized_moe_plan])
def test_direct_materialization_matches_owned_allocation(plan_factory) -> None:
    plan = plan_factory()
    expected = materialize_packed_tensors(plan, advantage_balance=0.25)
    storage_bytes = packed_plan_storage_byte_count(plan)
    store = SharedMemoryPackedBatchStore(
        owner_actor_id="test-owner", capacity_bytes=storage_bytes
    )
    writer = store.reserve_plan(plan, batch_id="batch")
    try:
        assert writer.storage_byte_count == storage_bytes
        reserved_stats = store.stats()
        assert reserved_stats.capacity_bytes == storage_bytes
        assert (
            reserved_stats.reserved_bytes == reserved_stats.peak_bytes == storage_bytes
        )
        assert reserved_stats.used_bytes == reserved_stats.created_bytes == 0
        assert reserved_stats.batches == reserved_stats.leases == 0
        tensors = writer.tensors
        assert tensors is not None
        writer.begin()
        materialized = False
        try:
            counts = materialize_packed_tensors_into(
                plan, tensors, advantage_balance=0.25
            )
            materialized = True
        finally:
            writer.finish(success=materialized)
        assert counts.trainable_assistant_tokens == int(
            expected["assistant_mask"].sum()
        )
        assert counts.shifted_loss_bearing_tokens == int(
            expected["assistant_mask"][:, 1:].sum()
        )
        assert counts.non_padding_tokens == int((expected["group_ids"] != -1).sum())
        assert counts.padding_tokens == expected["tokens"].numel() - int(
            (expected["group_ids"] != -1).sum()
        )
        assert store.stats().batches == 0
        ref = store.commit_plan(writer)
        assert ref.storage_byte_count == storage_bytes
        assert ref.prefix_tree_packing_stats is not None
        assert (
            ref.prefix_tree_packing_stats.model_dump()
            == expected["prefix_tree_packing_stats"]
        )
        expected_output_map = expected.get("tokenized_output_map")
        assert (
            None if expected_output_map is None else expected_output_map.model_dump()
        ) == (
            None
            if ref.tokenized_output_map is None
            else ref.tokenized_output_map.model_dump()
        )
        with store.map(ref) as mapped:
            _assert_flat_equal(expected, mapped.tensors)
        stats = store.stats()
        assert stats.used_bytes == stats.created_bytes == storage_bytes
        assert stats.reserved_bytes == stats.copied_bytes == stats.copy_count == 0
        assert stats.batches == stats.leases == 1
    finally:
        store.close()


def test_partial_materialization_failure_never_publishes_a_lease() -> None:
    plan = _rl_plan()
    storage_bytes = packed_plan_storage_byte_count(plan)
    store = SharedMemoryPackedBatchStore(
        owner_actor_id="test-owner", capacity_bytes=storage_bytes
    )
    writer = store.reserve_plan(plan, batch_id="partial")
    name = writer.shared_memory_name
    tensors = writer.tensors
    assert tensors is not None
    writer.begin()
    try:
        with pytest.raises(RuntimeError, match="injected materialization failure"):
            with patch(
                "art.preprocessing.pack._materialize_packed_row_tensor_list",
                side_effect=RuntimeError("injected materialization failure"),
            ):
                materialize_packed_tensors_into(plan, tensors)
    finally:
        writer.finish(success=False)
    assert not writer.ready
    with pytest.raises(PackedBatchLeaseError):
        store.commit_plan(writer)
    store.abort(writer.reservation.reservation_id)

    stats = store.stats()
    assert stats.used_bytes == stats.reserved_bytes == stats.created_bytes == 0
    assert stats.copied_bytes == stats.copy_count == stats.batches == stats.leases == 0
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
    store.close()
