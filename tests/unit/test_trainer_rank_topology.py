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

from art.trainer_rank import TrainerRank, TrainerRankRuntimeSupportError

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
