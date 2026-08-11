from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, cast, overload

import torch
import torch.distributed as dist

from . import _impl  # noqa: E402

AdapterSelection = _impl.AdapterSelection
AdamParams = _impl.AdamParams
AnyForwardInput = _impl.AnyForwardInput
AnyForwardOutput = _impl.AnyForwardOutput
ForwardInput = _impl.ForwardInput
ForwardInputs = _impl.ForwardInputs
ForwardOutput = _impl.ForwardOutput
ForwardOutputs = _impl.ForwardOutputs
HiddenStatesT = _impl.HiddenStatesT
LogitsT = _impl.LogitsT
LogprobsT = _impl.LogprobsT
MicroBatch = _impl.MicroBatch
MicroBatchStats = _impl.MicroBatchStats
TopK = _impl.TopK
TopKT = _impl.TopKT
TrainerRankMemoryError = _impl.TrainerRankMemoryError
TrainerRankSlotStateError = _impl.TrainerRankSlotStateError
Unset = _impl.Unset
PushedCheckpoint = _impl.PushedCheckpoint

if TYPE_CHECKING:
    from art.megatron.train import TrainingRuntime


for _public_type in (
    AdamParams,
    ForwardInput,
    ForwardOutput,
    MicroBatch,
    MicroBatchStats,
    TopK,
    TrainerRankMemoryError,
    PushedCheckpoint,
    TrainerRankSlotStateError,
):
    _public_type.__module__ = __name__
del _public_type


class TrainerRank(_impl.TrainerRank):
    def __init__(
        self,
        runtime: TrainingRuntime,
        *,
        head_chunk_tokens: int = 512,
        shared_prefix_max_depth: int = 1,
        memory_safety_factor: float = 1.10,
        memory_reserve_fraction: float = 0.03,
    ) -> None:
        super().__init__(
            runtime,
            head_chunk_tokens=head_chunk_tokens,
            shared_prefix_max_depth=shared_prefix_max_depth,
            memory_safety_factor=memory_safety_factor,
            memory_reserve_fraction=memory_reserve_fraction,
        )

    def zero_grad(self) -> None:
        super().zero_grad()

    def prefetch_checkpoints(self, *paths: str) -> asyncio.Task[None]:
        return super().prefetch_checkpoints(*paths)

    def load_checkpoint(self, path: str | None) -> asyncio.Task[None]:
        return super().load_checkpoint(path)

    def push_checkpoint(self, path: str | None) -> PushedCheckpoint:
        return super().push_checkpoint(path)

    def pop_checkpoint(self) -> None:
        super().pop_checkpoint()

    def save_checkpoint(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        super().save_checkpoint(output_dir, checkpoint_path)

    def export_lora(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> int:
        return super().export_lora(output_dir, checkpoint_path)

    @overload
    def forward_micro_batches(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
    ) -> Iterator[
        MicroBatch[
            ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
            ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
        ]
    ]: ...

    @overload
    def forward_micro_batches(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
    ) -> Iterator[
        MicroBatch[
            Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
            Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        ]
    ]: ...

    @overload
    def forward_micro_batches(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
    ) -> Iterator[
        MicroBatch[
            Sequence[Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]],
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]],
        ]
    ]: ...

    @overload
    def forward_micro_batches(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
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

    def forward_micro_batches(
        self, inputs: Iterable[ForwardInputs]
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        forward = cast(
            Callable[
                [Iterable[ForwardInputs]],
                Iterator[MicroBatch[ForwardInputs, ForwardOutputs]],
            ],
            super().forward_micro_batches,
        )
        return forward(inputs)

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
    ) -> Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
    ) -> Sequence[
        Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
    ]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
    ) -> Sequence[
        Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
    ]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
    ) -> Sequence[
        Sequence[
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ]
    ]: ...

    def dp_rank_forward(self, inputs: ForwardInputs) -> ForwardOutputs:
        forward = cast(
            Callable[[ForwardInputs], ForwardOutputs],
            super().dp_rank_forward,
        )
        return forward(inputs)

    def dp_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> None:
        super().dp_reduce(tensor, op=op)

    def optim_step(
        self,
        *,
        params: AdamParams,
        scale_grads: float = 1.0,
        checkpoints: Sequence[str] | None = None,
    ) -> dict[str, float]:
        return super().optim_step(
            params=params,
            scale_grads=scale_grads,
            checkpoints=checkpoints,
        )


__all__ = [
    "AdapterSelection",
    "AdamParams",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "MicroBatchStats",
    "TopK",
    "TrainerRank",
    "TrainerRankMemoryError",
    "PushedCheckpoint",
    "TrainerRankSlotStateError",
    "Unset",
]
