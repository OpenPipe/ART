"""Constructor topology contract for TrainerRank.

Written before lifting the TP>1 refusal (test-first). TrainerRank never used
the MCore pipeline schedule, so PP>1 stays refused; tensor parallelism is
executed by machinery that pre-dates the automatic planner (vocab-parallel
head, sequence-parallel gather, TP padding of packed batches, sharded LoRA
gradient reduction) and is admitted again, with the planner's memory profile
keyed by topology so TP=2 calibrates itself online.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    TrainerRank,
    TrainerRankRuntimeSupportError,
)

if TYPE_CHECKING:
    from art.megatron.train import TrainingRuntime


class _FakeGPT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))
        self.config = SimpleNamespace(hidden_size=8, num_layers=4, padded_vocab_size=32)
        self.decoder = object()

    def _preprocess(self, *args: object, **kwargs: object) -> None:
        return None


def _runtime(*, tp: int = 1, pp: int = 1, chunks: int = 1) -> "TrainingRuntime":
    return SimpleNamespace(
        model=[_FakeGPT() for _ in range(chunks)],
        optimizer=None,
        provider=SimpleNamespace(
            hidden_size=8,
            num_layers=4,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
        ),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
    )  # type: ignore


@pytest.mark.parametrize("tp", (1, 2, 4))
def test_trainer_rank_accepts_tensor_parallel_runtimes(tp: int) -> None:
    rank = TrainerRank(_runtime(tp=tp))

    assert rank.last_forward_telemetry is not None


def test_trainer_rank_still_refuses_pipeline_parallel_runtimes() -> None:
    with pytest.raises(TrainerRankRuntimeSupportError, match="PP=1"):
        TrainerRank(_runtime(pp=2))
    with pytest.raises(TrainerRankRuntimeSupportError, match="one local model chunk"):
        TrainerRank(_runtime(chunks=2))


def test_packed_tokens_count_per_group_tensor_parallel_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution pads every group to a TP multiple; admission must count that.

    Two groups (grad and no_grad) of 5 and 7 tokens are physically 6 + 8 at
    TP=2. The plan, the cheap bounds, the exact estimate and the split lower
    bound must all agree on the physical count; at TP=1 they equal the sum.
    """

    rank = TrainerRank(_runtime(tp=2))  # type: ignore[arg-type]
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        lambda *, packed_tokens, **_kwargs: packed_tokens,
    )
    first = torch.tensor([11, 12, 13, 14, 15], dtype=torch.long)
    second = torch.tensor([21, 22, 23, 24, 25, 26, 27], dtype=torch.long)
    requests = [
        ForwardInput(input_tokens=first, target_tokens=first),
        ForwardInput(input_tokens=second, target_tokens=second, no_grad=True),
    ]

    plan = rank._plan_flat_forward(requests)
    assert sorted(int(g.packed.tokens.numel()) for g in plan.groups) == [5, 7]
    assert plan.packed_tokens == 12  # TP=1 without Megatron: unpadded sum

    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 2, 1, 1))
    plan = rank._plan_flat_forward(requests)
    assert plan.packed_tokens == 6 + 8
    cheap = rank._estimate_flat_forward(requests)
    exact = rank._estimate_flat_forward(requests, exact=True)
    assert cheap is not None and exact is not None
    assert cheap[0] == exact[0] == plan.packed_tokens
    rows = tuple(request.input_tokens for request in requests)
    from art.trainer_rank._impl import Unset

    lower = rank._split_chunk_lower_cost(requests, rows, checkpoint=Unset)
    assert lower.required == 6 + 8
