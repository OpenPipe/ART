from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("megatron")

from art.megatron.context_parallel.types import (
    DkvReducePlan,
    KvFetchPlan,
    StagePlan,
    TokenRange,
)
from art.megatron.glm52 import cp_stage


def test_fused_local_stage_owns_remote_fetch_and_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = StagePlan(
        stage_index=0,
        source_rank=0,
        is_local_stage=True,
        slices=(),
        owner_local_q_ranges=(TokenRange(0, 2),),
        owner_local_k_ranges=(TokenRange(0, 2),),
        q_len=2,
        k_len=4,
        fused_remote_k_len=2,
        kv_fetch_plan=KvFetchPlan((0, 0), (0, 2), ((), ())),
        dkv_reduce_plan=DkvReducePlan((0, 2), (0, 0), ((), ())),
    )
    empty_remote = StagePlan(
        stage_index=1,
        source_rank=1,
        is_local_stage=False,
        slices=(),
        owner_local_q_ranges=(),
        owner_local_k_ranges=(),
        q_len=0,
        k_len=0,
    )
    state = cast(
        Any,
        SimpleNamespace(
            rank_plan=SimpleNamespace(stage_plans=(local, empty_remote)),
            cp_group=object(),
            execution_cache=SimpleNamespace(range_meta={}),
        ),
    )
    remote_rows = torch.tensor([[3.0], [4.0]])
    fetch_work = MagicMock()
    fetch_work.wait_post_process.return_value = remote_rows
    monkeypatch.setattr(
        cp_stage._COMMUNICATOR,
        "launch_tensor_fetch",
        MagicMock(return_value=fetch_work),
    )

    local_rows = torch.tensor([[1.0], [2.0]])
    fetches = cp_stage.launch_remote_stage_fetches(local_rows, state)

    assert set(fetches) == {0}
    torch.testing.assert_close(
        cp_stage.stage_kv_rows(local_rows, local, state, fetches),
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    assert cp_stage.stage_kv_rows(local_rows, empty_remote, state, fetches).shape == (
        0,
        1,
    )

    local_reduce = MagicMock()
    reduce_work = object()
    remote_reduce = MagicMock(return_value=reduce_work)
    monkeypatch.setattr(cp_stage, "range_reduce_sum_", local_reduce)
    monkeypatch.setattr(cp_stage, "launch_remote_stage_reduce", remote_reduce)
    target = torch.zeros_like(local_rows)
    stage_grad = torch.tensor([[10.0], [20.0], [30.0], [40.0]])

    assert (
        cp_stage.reduce_local_stage_rows_(target, stage_grad, local, state)
        is reduce_work
    )
    torch.testing.assert_close(local_reduce.call_args.args[0], stage_grad[:2])
    torch.testing.assert_close(remote_reduce.call_args.args[0], stage_grad[2:])
