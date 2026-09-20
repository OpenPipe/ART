from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, cast, overload

import torch
import torch.distributed as dist

from . import _impl
from ._checkpoint import CheckpointManifest, materialize_lora, validate_checkpoint
from ._heads import ModuleHandle
from ._options import (
    ForwardOptions,
    ImportanceSamplingGradientCorrection,
    ResolvedForwardOptions,
    resolve_forward_options,
)
from ._options import _Unset as _Unset

AdapterSelection = _impl.AdapterSelection
AdamParams = _impl.AdamParams
AnyForwardInput = _impl.AnyForwardInput
AnyForwardOutput = _impl.AnyForwardOutput
ForwardInput = _impl.ForwardInput
ForwardInputs = _impl.ForwardInputs
ForwardOutput = _impl.ForwardOutput
ForwardOutputs = _impl.ForwardOutputs
MicroBatch = _impl.MicroBatch
MicroBatchStats = _impl.MicroBatchStats
TopK = _impl.TopK
LogprobsT = TypeVar("LogprobsT", bound=torch.Tensor | None, covariant=True)
TopKT = TypeVar("TopKT", bound=TopK | None, covariant=True)
LogitsT = TypeVar("LogitsT", bound=torch.Tensor | None, covariant=True)
HiddenStatesT = TypeVar("HiddenStatesT", bound=torch.Tensor | None, covariant=True)
TrainerRankMemoryError = _impl.TrainerRankMemoryError
TrainerRankPartialExecutionError = _impl.TrainerRankPartialExecutionError
TrainerRankRuntimeSupportError = _impl.TrainerRankRuntimeSupportError
TrainerRankSlotStateError = _impl.TrainerRankSlotStateError
Unset = _impl.Unset
MaterializedCheckpoint = _impl.MaterializedCheckpoint
PushedCheckpoint = _impl.PushedCheckpoint

if TYPE_CHECKING:
    from art.megatron.train import TrainingRuntime

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


for _public_type in (
    AdamParams,
    ForwardInput,
    ForwardOptions,
    ForwardOutput,
    ImportanceSamplingGradientCorrection,
    ResolvedForwardOptions,
    MicroBatch,
    MicroBatchStats,
    TopK,
    TrainerRankMemoryError,
    TrainerRankPartialExecutionError,
    TrainerRankRuntimeSupportError,
    TrainerRankSlotStateError,
    MaterializedCheckpoint,
    PushedCheckpoint,
):
    _public_type.__module__ = __name__
del _public_type


class TrainerRank(_impl.TrainerRank):
    """Execute TrainerRank forwards using automatic, data-dependent planning.

    The constructor accepts the training runtime and optional forward policy.
    Prefix sharing and microbatch width are data-dependent planner decisions;
    output-head chunking and memory margins are internal calibrated policy.
    None are user tuning parameters. Requires PP=1 (TrainerRank does not use
    the MCore pipeline schedule); PP>1 raises ``TrainerRankRuntimeSupportError``
    at construction. Tensor parallelism is supported; the planner's memory
    profile is keyed by topology and calibrates itself online.
    """

    @property
    def hidden_size(self) -> int:
        """Width of the returned hidden states."""
        return self._hidden_size

    # Keep ModuleHandle available to runtime type-hint resolution.
    def module(
        self,
        name: str,
        factory: Callable[[], ModuleT],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> ModuleHandle:
        """Retrieve a live module whose calls capture immutable checkpoint weights."""
        return super().module(name, factory, checkpoint=checkpoint)


from ._commands import (
    RankCallbackResult,
    TrainerRankZero,
    get_rank_callback_metadata,
    rank_callback_leader,
    run_rank_callback,
    run_rank_callback_stream,
)

__all__ = [
    "RankCallbackResult",
    "TrainerRankZero",
    "get_rank_callback_metadata",
    "rank_callback_leader",
    "run_rank_callback",
    "run_rank_callback_stream",
    "AdapterSelection",
    "AdamParams",
    "CheckpointManifest",
    "ForwardInput",
    "ForwardOptions",
    "ImportanceSamplingGradientCorrection",
    "ResolvedForwardOptions",
    "resolve_forward_options",
    "ForwardOutput",
    "MicroBatch",
    "MicroBatchStats",
    "MaterializedCheckpoint",
    "ModuleHandle",
    "materialize_lora",
    "TopK",
    "TrainerRank",
    "TrainerRankMemoryError",
    "TrainerRankPartialExecutionError",
    "TrainerRankRuntimeSupportError",
    "PushedCheckpoint",
    "TrainerRankSlotStateError",
    "Unset",
    "validate_checkpoint",
]
