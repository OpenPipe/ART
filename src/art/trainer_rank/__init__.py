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

    def __init__(
        self, runtime: TrainingRuntime, *, options: ForwardOptions | None = None
    ) -> None:
        super().__init__(runtime, options=options)

    @property
    def hidden_size(self) -> int:
        """Width of the returned hidden states."""
        return self._hidden_size

    def zero_grad(self) -> None:
        super().zero_grad()

    def module(
        self,
        name: str,
        factory: Callable[[], ModuleT],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> ModuleHandle:
        """Retrieve a live module whose calls capture immutable checkpoint weights."""
        return super().module(name, factory, checkpoint=checkpoint)

    def parameter(
        self,
        name: str,
        factory: Callable[[], torch.Tensor | torch.nn.Parameter],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> torch.nn.Parameter:
        """Register or retrieve a checkpoint-owned trainable tensor."""
        return super().parameter(name, factory, checkpoint=checkpoint)

    def buffer(
        self,
        name: str,
        factory: Callable[[], torch.Tensor],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> torch.Tensor:
        """Register or retrieve a checkpoint-owned persistent buffer."""
        return super().buffer(name, factory, checkpoint=checkpoint)

    def prefetch_checkpoints(
        self,
        *checkpoints: str | MaterializedCheckpoint,
    ) -> asyncio.Task[None]:
        return super().prefetch_checkpoints(*checkpoints)

    def load_checkpoint(self, checkpoint: str | MaterializedCheckpoint | None) -> None:
        super().load_checkpoint(checkpoint)

    def snapshot_checkpoint(self, source: str, destination: str) -> bool:
        """Clone a loaded checkpoint into a forward-only resident snapshot."""
        return super().snapshot_checkpoint(source, destination)

    def push_checkpoint(
        self, checkpoint: str | MaterializedCheckpoint | None
    ) -> PushedCheckpoint:
        return super().push_checkpoint(checkpoint)

    def pop_checkpoint(self) -> None:
        super().pop_checkpoint()

    def save_checkpoint(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        super().save_checkpoint(output_dir, checkpoint_path)

    def prepare_checkpoint_save(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        super().prepare_checkpoint_save(output_dir, checkpoint_path)

    def finish_checkpoint_save(self, output_dir: str) -> None:
        super().finish_checkpoint_save(output_dir)

    def abort_checkpoint_save(self, output_dir: str) -> None:
        super().abort_checkpoint_save(output_dir)

    def export_lora(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> int:
        return super().export_lora(output_dir, checkpoint_path)

    @overload
    def forward_batches(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
            ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
        ]
    ]: ...

    @overload
    def forward_batches(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
            Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        ]
    ]: ...

    @overload
    def forward_batches(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            Sequence[Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]],
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]],
        ]
    ]: ...

    @overload
    def forward_batches(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            Sequence[
                Sequence[
                    Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ],
            Sequence[
                Sequence[
                    Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ],
        ]
    ]: ...

    def forward_batches(
        self,
        inputs: Iterable[ForwardInputs],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        """Forward replicated inputs in adaptive data-parallel microbatches.

        Per-input checkpoints and `no_grad` values override the method defaults.
        `no_grad=None` inherits the ambient PyTorch grad mode; `True` disables
        grads and `False` enables them.
        Input and target tensors may be on a different device from the trainer;
        ART moves its packed model inputs and labels internally without mutating
        the caller-owned `ForwardInput` objects.

        Per-position outputs contain the full flattened input sequence in source
        order, including with context parallelism. Logical callbacks execute
        once per DP rank; use `backward(loss)` to route cotangents to internal
        TP/CP participants. Direct physical callers must invoke matching
        forwards and backwards on their TP/CP peers. `reduce` combines only
        distinct data-parallel batches.

        Empty local microbatches are skipped unless `yield_empty=True`. Every
        rank must use the same setting. When a wave skips ranks, TrainerRank
        collective methods raise if called from its loop body; fully populated
        waves permit them. Use `yield_empty=True` for per-wave collectives,
        including reductions on ranks with no outputs. Exhaust or close a retained
        iterator before making collective calls after an early exit. Guards apply
        on the iterator's thread; raw torch.distributed calls are not guarded.
        Collective calls must still match across ranks.
        """
        forward = cast(
            Callable[..., Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]],
            super().forward_batches,
        )
        return forward(
            inputs,
            options=options,
            checkpoint=checkpoint,
            no_grad=no_grad,
            yield_empty=yield_empty,
        )

    @overload
    def forward(
        self,
        inputs: ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
    ]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
    ]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ]
    ]: ...

    def forward(
        self,
        inputs: ForwardInputs,
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> ForwardOutputs:
        """Forward inputs already local to this data-parallel rank.

        Outputs contain full sequences in source order on every TP/CP rank,
        with the same loss and reduction contract as `forward_batches`.

        Per-input checkpoints and `no_grad` values override the method defaults.
        `no_grad=None` inherits the ambient PyTorch grad mode; `True` disables
        grads and `False` enables them.
        Input and target tensors may be on a different device from the trainer;
        ART moves its packed model inputs and labels internally without mutating
        the caller-owned `ForwardInput` objects.
        """
        forward = cast(
            Callable[..., ForwardOutputs],
            super().forward,
        )
        return forward(inputs, options=options, checkpoint=checkpoint, no_grad=no_grad)

    def reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> None:
        """Reduce in place over data-parallel batches, excluding TP/CP replicas."""
        super().reduce(tensor, op=op)

    def optim_step(
        self,
        *,
        params: AdamParams | Mapping[str, AdamParams],
        scale_grads: float | Mapping[str, float] = 1.0,
        checkpoints: Sequence[str] | None = None,
        on_live_graphs: Literal["allow", "error"] = "allow",
    ) -> dict[str, float]:
        """Step checkpoint slots that have accumulated gradients.

        A mapping assigns independent optimizer parameters to each checkpoint;
        ``scale_grads`` may likewise map checkpoints to gradient scales. Mapping
        keys select the checkpoints when ``checkpoints`` is omitted, and all
        explicitly supplied checkpoint sets must match. Each checkpoint's gradient
        norm is clipped independently. If any selected norm is nonfinite, no
        selected checkpoint is updated.

        Retained forwards use immutable checkpoint versions and may be consumed
        after this step within their captured `max_gradient_staleness` policy.
        Pass `on_live_graphs="error"` to additionally refuse updates while a
        selected checkpoint still has a live forward graph on any rank.
        """
        return super().optim_step(
            params=params,
            scale_grads=scale_grads,
            checkpoints=checkpoints,
            on_live_graphs=on_live_graphs,
        )


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
