from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, cast, overload

import torch
import torch.distributed as dist

from . import _impl
from ._checkpoint import CheckpointManifest, materialize_lora, validate_checkpoint

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
MaterializedCheckpoint = _impl.MaterializedCheckpoint
PushedCheckpoint = _impl.PushedCheckpoint

if TYPE_CHECKING:
    from art.megatron.train import TrainingRuntime

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


for _public_type in (
    AdamParams,
    ForwardInput,
    ForwardOutput,
    MicroBatch,
    MicroBatchStats,
    TopK,
    TrainerRankMemoryError,
    TrainerRankSlotStateError,
    MaterializedCheckpoint,
    PushedCheckpoint,
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

    def module(
        self,
        name: str,
        factory: Callable[[], ModuleT],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> ModuleT:
        """Register or retrieve a checkpoint-owned PyTorch module."""
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

    def load_checkpoint(
        self, checkpoint: str | MaterializedCheckpoint | None
    ) -> asyncio.Task[None]:
        return super().load_checkpoint(checkpoint)

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

    def checkpoint_slot_tensor_owners(self, name: str) -> tuple[tuple[str, int], ...]:
        self.checkpoint_slot_parameters(name)
        from art.megatron.weights.lora_publish import collect_local_lora_entries

        _tensors, metadata = collect_local_lora_entries(
            self.runtime.model,
            {},
            owner_rank=dist.get_rank() if dist.is_initialized() else 0,
            slot_ref=self._slot_ref(name),
        )
        return tuple(
            sorted(
                {
                    (item.key, int(item.manifest.get("shard_rank", 0)))
                    for item in metadata
                }
            )
        )

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
    def forward_micro_batches(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
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
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
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
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
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
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
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
        self,
        inputs: Iterable[ForwardInputs],
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        """Forward replicated inputs in adaptive data-parallel microbatches.

        Per-input checkpoints override `checkpoint`. `no_grad=None` inherits the
        ambient PyTorch grad mode; `True` disables grads and `False` enables them.
        Input and target tensors may be on a different device from the trainer;
        ART moves its packed model inputs and labels internally without mutating
        the caller-owned `ForwardInput` objects.
        """
        forward = cast(
            Callable[..., Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]],
            super().forward_micro_batches,
        )
        return forward(inputs, checkpoint=checkpoint, no_grad=no_grad)

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
    ]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
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
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ]
    ]: ...

    def dp_rank_forward(
        self,
        inputs: ForwardInputs,
        *,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> ForwardOutputs:
        """Forward inputs already local to this data-parallel rank.

        Per-input checkpoints override `checkpoint`. `no_grad=None` inherits the
        ambient PyTorch grad mode; `True` disables grads and `False` enables them.
        Input and target tensors may be on a different device from the trainer;
        ART moves its packed model inputs and labels internally without mutating
        the caller-owned `ForwardInput` objects.
        """
        forward = cast(
            Callable[..., ForwardOutputs],
            super().dp_rank_forward,
        )
        return forward(inputs, checkpoint=checkpoint, no_grad=no_grad)

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
        on_live_graphs: Literal["allow", "error"] = "allow",
    ) -> dict[str, float]:
        """Step checkpoint slots that have accumulated gradients.

        By default, caller-retained forward graphs do not block the step. ART does
        not detach or free those graphs, and backward through one after the step is
        unsafe: it may fail PyTorch's version checks or recompute against updated
        checkpoint-slot weights. Pass `on_live_graphs="error"` to raise before
        mutating any selected slot when a live graph remains on any rank.
        """
        return super().optim_step(
            params=params,
            scale_grads=scale_grads,
            checkpoints=checkpoints,
            on_live_graphs=on_live_graphs,
        )

    def checkpoint_slot_parameters(self, name: str) -> tuple[torch.nn.Parameter, ...]:
        return super().checkpoint_slot_parameters(name)

    def checkpoint_slot_optimizer_tensors(self, name: str) -> tuple[torch.Tensor, ...]:
        return super().checkpoint_slot_optimizer_tensors(name)

    def prepare_checkpoint_slot_optimizer(
        self, name: str, params: AdamParams
    ) -> tuple[torch.Tensor, ...]:
        return super().prepare_checkpoint_slot_optimizer(name, params)

    def clear_checkpoint_slot_grads(self, name: str) -> None:
        super().clear_checkpoint_slot_grads(name)

    def release_checkpoint_slot(self, name: str) -> None:
        super().release_checkpoint_slot(name)

    def reduce_checkpoint_slot_grads(
        self,
        name: str,
        gradients: Sequence[torch.Tensor],
        *,
        scale_grads: float,
    ) -> tuple[torch.Tensor, ...]:
        return super().reduce_checkpoint_slot_grads(
            name, gradients, scale_grads=scale_grads
        )

    def optim_step_reduced(
        self,
        name: str,
        *,
        params: AdamParams,
        gradients: Sequence[torch.Tensor],
        step_flags: Sequence[bool],
    ) -> dict[str, float]:
        return super().optim_step_reduced(
            name,
            params=params,
            gradients=gradients,
            step_flags=step_flags,
        )


__all__ = [
    "AdapterSelection",
    "AdamParams",
    "CheckpointManifest",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "MicroBatchStats",
    "MaterializedCheckpoint",
    "materialize_lora",
    "TopK",
    "TrainerRank",
    "TrainerRankMemoryError",
    "PushedCheckpoint",
    "TrainerRankSlotStateError",
    "Unset",
    "validate_checkpoint",
]
