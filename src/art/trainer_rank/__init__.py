from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, cast, overload

import torch
import torch.distributed as dist


class TrainerRankOptimizerLayout(TypedDict):
    parallel: tuple[int, int, int, int, int, int, int, int]
    parameters: tuple[
        tuple[tuple[int, ...], str, str, bool, int | None, str, tuple[int, ...]],
        ...,
    ]


class TrainerRankOptimizerState(TypedDict):
    format_version: Literal[1]
    layout: TrainerRankOptimizerLayout
    master_params: tuple[torch.Tensor, ...]
    optimizer: dict[str, object]


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
_PushedSlot = _impl._PushedSlot

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
    TrainerRankOptimizerLayout,
    TrainerRankOptimizerState,
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

    def set_checkpoint(self, name: str | None) -> None:
        super().set_checkpoint(name)

    def set_lora(self, name: str | None) -> None:
        super().set_lora(name)

    def push_checkpoint(self, name: str | None) -> _PushedSlot:
        return super().push_checkpoint(name)

    def push_lora(self, name: str | None) -> _PushedSlot:
        return super().push_lora(name)

    def pop_pushed_lora_or_checkpoint(self) -> None:
        super().pop_pushed_lora_or_checkpoint()

    def load_checkpoint_slot(
        self,
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
        *,
        optimizer_state: TrainerRankOptimizerState | None = None,
        alpha: float | None = None,
        adapter_config: Mapping[str, object] | None = None,
    ) -> int:
        return super().load_checkpoint_slot(
            name,
            adapter_model,
            optimizer_state=optimizer_state,
            alpha=alpha,
            adapter_config=adapter_config,
        )

    def checkpoint_slot_optimizer_state(
        self, name: str
    ) -> TrainerRankOptimizerState | None:
        return super().checkpoint_slot_optimizer_state(name)

    def save_checkpoint_slot_lora(self, name: str, output_dir: str) -> None:
        """Collectively publish a trained checkpoint slot as a vLLM LoRA."""
        super().save_checkpoint_slot_lora(name, output_dir)

    def load_lora_slot(
        self,
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
        *,
        alpha: float | None = None,
    ) -> int:
        return super().load_lora_slot(name, adapter_model, alpha=alpha)

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
    "TrainerRankOptimizerLayout",
    "TrainerRankOptimizerState",
    "TrainerRankSlotStateError",
    "Unset",
]
