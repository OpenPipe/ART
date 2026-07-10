from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)
import weakref

import torch
import torch.distributed as dist

from art.megatron.prefix_tree_packing import (
    PrefixTreePack,
    _local_position_pairs,
    estimate_prefix_tree_packed_tokens,
    prefix_tree_pack,
)

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.packed_seq_params import PackedSeqParams

    from art.megatron.context_parallel.types import (
        ArtContextParallelState,
        ParallelTopology,
    )
    from art.megatron.lora import LoRASlotRef
    from art.megatron.prefix_tree_state import PrefixTreeAttentionState
    from art.megatron.train import TrainingRuntime


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    grad_clip_norm: float = 0.1


@dataclass(frozen=True)
class TopK:
    logprobs: torch.Tensor
    tokens: torch.Tensor


LogprobsT = TypeVar("LogprobsT", bound=torch.Tensor | None, covariant=True)
TopKT = TypeVar("TopKT", bound=TopK | None, covariant=True)
LogitsT = TypeVar("LogitsT", bound=torch.Tensor | None, covariant=True)
HiddenStatesT = TypeVar("HiddenStatesT", bound=torch.Tensor | None, covariant=True)
T = TypeVar("T")

_MEMORY_PROFILE_TRUST_GROWTH = 8


class _Unset:
    pass


Unset = _Unset()
type AdapterSelection = str | None | _Unset


@dataclass(frozen=True)
class _LocalLoRASlotRef:
    kind: Literal["checkpoint", "lora"]
    name: str | None


@dataclass(frozen=True)
class ForwardOutput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    target_logprobs: LogprobsT
    top_k: TopKT
    logits: LogitsT
    hidden_states: HiddenStatesT


@dataclass(slots=True)
class ForwardInput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    input_tokens: torch.Tensor
    target_tokens: torch.Tensor | None = None
    top_k: int | None = None
    logits: bool = False
    hidden_states: bool = False
    checkpoint: AdapterSelection = Unset
    lora: AdapterSelection = Unset

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, None, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, None, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, TopK, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, None, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, None, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, TopK, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, None, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, None, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, TopK, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, TopK, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, None, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, TopK, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, TopK, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, None, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[None, TopK, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[True],
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor, TopK, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None]": ...

    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> Any:
        return object.__new__(cls)

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.checkpoint is not Unset and self.lora is not Unset:
            raise ValueError("ForwardInput cannot set both checkpoint and lora")


type AnyForwardInput = ForwardInput[
    torch.Tensor | None,
    TopK | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
type AnyForwardOutput = ForwardOutput[
    torch.Tensor | None,
    TopK | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
type ForwardInputs = AnyForwardInput | Iterable["ForwardInputs"]
type ForwardOutputs = AnyForwardOutput | Sequence["ForwardOutputs"]
ForwardInputsT = TypeVar("ForwardInputsT", bound=ForwardInputs)
ForwardOutputsT = TypeVar("ForwardOutputsT", bound=ForwardOutputs)
MicroBatchInputsT = TypeVar("MicroBatchInputsT", bound=ForwardInputs, covariant=True)
MicroBatchOutputsT = TypeVar("MicroBatchOutputsT", bound=ForwardOutputs, covariant=True)


@dataclass(frozen=True)
class MicroBatch(Generic[MicroBatchInputsT, MicroBatchOutputsT]):
    inputs: Sequence[MicroBatchInputsT]
    outputs: Sequence[MicroBatchOutputsT]
    indices: Sequence[int]
    stats: "MicroBatchStats"

    def select(self, xs: Sequence[T]) -> Sequence[T]:
        return [xs[i] for i in self.indices]


@dataclass(frozen=True)
class MicroBatchStats:
    global_start: int
    global_stop: int
    global_count: int
    local_count: int
    packed_tokens: int
    logical_tokens: int
    estimated_required_bytes: int
    available_bytes: int
    rejected_candidates: int
    cold_start: bool


class TrainerRankMemoryError(RuntimeError):
    pass


class TrainerRankSlotStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class _MemoryCheck:
    estimated_required_bytes: int
    available_bytes: int
    fits: bool


@dataclass(frozen=True)
class _MemoryProfile:
    bytes_per_token: float
    packed_tokens: int


@dataclass(frozen=True)
class _CandidateMicroBatch(Generic[ForwardInputsT]):
    inputs: Sequence[ForwardInputsT]
    indices: tuple[int, ...]
    plan: "_FlatForwardPlan"
    check: _MemoryCheck
    stats_global_count: int
    rejected_candidates: int
    cold_start: bool


class _SlotGraphSentinel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        marker: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(marker)
        return tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None]:
        return cast(torch.Tensor, grad_outputs[0]), None


@dataclass(frozen=True)
class _DynamicOptimizer:
    optimizer: torch.optim.Optimizer
    master_params: tuple[torch.nn.Parameter, ...]


@dataclass(frozen=True)
class _PushedSlot:
    trainer: "TrainerRank"
    ref: "LoRASlotRef"

    def __enter__(self) -> "_PushedSlot":
        return self

    def __exit__(self, *args: object) -> bool:
        if not self.trainer._slot_stack or self.trainer._slot_stack[-1] != self.ref:
            raise RuntimeError(
                "Pushed LoRA/checkpoint stack changed before context exit"
            )
        self.trainer.pop_pushed_lora_or_checkpoint()
        return False


@dataclass(frozen=True)
class _ForwardItem:
    request: AnyForwardInput
    input_ids: torch.Tensor
    labels: torch.Tensor | None


@dataclass(frozen=True)
class _PreparedPackedForward:
    tokens: torch.Tensor
    position_ids: torch.Tensor
    attention_state: "PrefixTreeAttentionState | ArtContextParallelState"
    packed_seq_params: "PackedSeqParams | None"
    positions_by_item: tuple[torch.Tensor, ...]
    source_positions_by_item: tuple[torch.Tensor, ...]


type _RowMatch = tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]


@dataclass(frozen=True)
class _MemorySignature:
    topology: tuple[int, int, int, int]
    shared_prefix_max_depth: int
    slot_group_count: int
    request_mix: tuple[str, ...]


@dataclass(frozen=True)
class _ForwardGroupPlan:
    slot_ref: "LoRASlotRef | None"
    request_indices: tuple[int, ...]
    items: tuple[_ForwardItem, ...]
    packed: PrefixTreePack


@dataclass(frozen=True)
class _FlatForwardPlan:
    request_count: int
    groups: tuple[_ForwardGroupPlan, ...]
    packed_tokens: int
    logical_tokens: int
    output_bytes: int
    signature: _MemorySignature


class TrainerRank:
    def __init__(
        self,
        runtime: TrainingRuntime,
        *,
        head_chunk_tokens: int = 512,
        shared_prefix_max_depth: int = 1,
        memory_safety_factor: float = 1.10,
        memory_reserve_fraction: float = 0.03,
    ) -> None:
        if head_chunk_tokens < 1:
            raise ValueError("head_chunk_tokens must be >= 1")
        if shared_prefix_max_depth < 0:
            raise ValueError("shared_prefix_max_depth must be >= 0")
        if memory_safety_factor < 1.0:
            raise ValueError("memory_safety_factor must be >= 1.0")
        if not (0.0 <= memory_reserve_fraction < 1.0):
            raise ValueError("memory_reserve_fraction must be in [0, 1)")
        self.runtime: TrainingRuntime = runtime
        self.head_chunk_tokens = head_chunk_tokens
        self.shared_prefix_max_depth = shared_prefix_max_depth
        self.memory_safety_factor = memory_safety_factor
        self.memory_reserve_fraction = memory_reserve_fraction
        self.device = next(runtime.model[0].parameters()).device
        self._param_dtype_size = _dtype_size(next(runtime.model[0].parameters()).dtype)
        try:
            metadata_model = _language_model(runtime.model[0])
        except RuntimeError:
            metadata_model = None
        self._hidden_size = _hidden_size(metadata_model, runtime.provider)
        self._padded_vocab_size = (
            None if metadata_model is None else _padded_vocab_size(metadata_model)
        )
        self._num_layers = int(
            getattr(getattr(metadata_model, "config", None), "num_layers", 0)
            or getattr(runtime.provider, "num_layers", 1)
            or 1
        )
        self._default_slot_ref: LoRASlotRef | None = None
        self._slot_stack: list[LoRASlotRef] = []
        self._dynamic_optimizers: dict[str, _DynamicOptimizer] = {}
        self._checkpoint_slot_params_by_name: dict[
            str, tuple[torch.nn.Parameter, ...]
        ] = {}
        self._pending_slot_graphs: dict[
            LoRASlotRef, list[weakref.ReferenceType[torch.Tensor]]
        ] = {}
        self._memory_profiles: dict[_MemorySignature, _MemoryProfile] = {}
        self._last_global_micro_batch_size: int | None = None
        self.zero_grad()

    def zero_grad(self) -> None:
        for chunk in self.runtime.model:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()
        optimizer = self.runtime.optimizer
        if optimizer is not None:
            optimizer.zero_grad()
        for params in self._checkpoint_slot_params_by_name.values():
            for param in params:
                param.grad = None
        self._prune_slot_graphs()

    def set_checkpoint(self, name: str | None) -> None:
        self._set_default_slot(self._slot_ref("checkpoint", name))

    def set_lora(self, name: str | None) -> None:
        self._set_default_slot(self._slot_ref("lora", name))

    def push_checkpoint(self, name: str | None) -> _PushedSlot:
        ref = self._slot_ref("checkpoint", name)
        self._slot_stack.append(ref)
        return _PushedSlot(self, ref)

    def push_lora(self, name: str | None) -> _PushedSlot:
        ref = self._slot_ref("lora", name)
        self._slot_stack.append(ref)
        return _PushedSlot(self, ref)

    def pop_pushed_lora_or_checkpoint(self) -> None:
        if not self._slot_stack:
            raise RuntimeError("No pushed LoRA or checkpoint to pop")
        self._slot_stack.pop()

    def load_checkpoint_slot(
        self,
        name: str,
        adapter_model: dict[str, torch.Tensor],
        *,
        optimizer_state: Mapping[str, object] | None = None,
        alpha: float | None = None,
    ) -> int:
        loaded = self._load_slot(
            "checkpoint", name, adapter_model, trainable=True, alpha=alpha
        )
        self._checkpoint_slot_params_by_name[name] = (
            self._validate_dynamic_slot_consistency("checkpoint", name, loaded)
        )
        if optimizer_state is None:
            self._dynamic_optimizers.pop(name, None)
        else:
            self._dynamic_optimizers[name] = self._restore_dynamic_optimizer(
                name, optimizer_state
            )
        return loaded

    def checkpoint_slot_optimizer_state(self, name: str) -> dict[str, object] | None:
        if name not in self._checkpoint_slot_params_by_name:
            raise ValueError(f"Unknown checkpoint slot: {name!r}")
        dynamic = self._dynamic_optimizers.get(name)
        if dynamic is None:
            return None
        return {
            "format_version": 1,
            "layout": self._dynamic_optimizer_layout(name),
            "master_params": tuple(
                param.detach().cpu().clone() for param in dynamic.master_params
            ),
            "optimizer": _state_to_cpu(dynamic.optimizer.state_dict()),
        }

    def load_lora_slot(
        self,
        name: str,
        adapter_model: dict[str, torch.Tensor],
        *,
        alpha: float | None = None,
    ) -> int:
        loaded = self._load_slot(
            "lora", name, adapter_model, trainable=False, alpha=alpha
        )
        self._validate_dynamic_slot_consistency("lora", name, loaded)
        return loaded

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
        self,
        inputs: Iterable[ForwardInputs],
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        items = [_materialize(item) for item in inputs]
        requests = list(_flatten(items))
        for _, indices in self._group_active_request_indices(requests):
            for index in indices:
                self._forward_item(requests[index])
        self._validate_replicated_top_level_count(len(items))
        start = 0
        while start < len(items):
            candidate = self._select_next_micro_batch(items, start)
            flat_outputs = iter(
                self._run_flat_plan_with_memory_tracking(
                    candidate.plan,
                    context="forward_micro_batches",
                )
            )
            outputs = [_unflatten(item, flat_outputs) for item in candidate.inputs]
            stop = start + candidate.stats_global_count
            if stop < len(items):
                self._last_global_micro_batch_size = max(
                    self._last_global_micro_batch_size or 0,
                    candidate.stats_global_count,
                )
            yield MicroBatch(
                inputs=candidate.inputs,
                outputs=outputs,
                indices=candidate.indices,
                stats=MicroBatchStats(
                    global_start=start,
                    global_stop=stop,
                    global_count=candidate.stats_global_count,
                    local_count=len(candidate.inputs),
                    packed_tokens=candidate.plan.packed_tokens,
                    logical_tokens=candidate.plan.logical_tokens,
                    estimated_required_bytes=candidate.check.estimated_required_bytes,
                    available_bytes=candidate.check.available_bytes,
                    rejected_candidates=candidate.rejected_candidates,
                    cold_start=candidate.cold_start,
                ),
            )
            start = stop

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
        materialized = _materialize(inputs)
        plan = self._plan_flat_forward(list(_flatten(materialized)))
        check = self._memory_check(plan)
        if not check.fits:
            self._raise_memory_error(
                plan,
                check,
                context="dp_rank_forward",
                message="forward is predicted to exceed available memory",
            )
        outputs = iter(
            self._run_flat_plan_with_memory_tracking(
                plan,
                context="dp_rank_forward",
            )
        )
        return _unflatten(materialized, outputs)

    def dp_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> None:
        from megatron.core import parallel_state as ps

        dist.all_reduce(
            tensor,
            op=op,
            group=ps.get_data_parallel_group(with_context_parallel=True),
        )

    def optim_step(
        self,
        *,
        params: AdamParams,
        scale_grads: float = 1.0,
        checkpoints: Sequence[str] | None = None,
    ) -> dict[str, float]:
        selected_checkpoints = self._selected_dynamic_checkpoints(checkpoints)
        return self._dynamic_optim_step(
            selected_checkpoints,
            params=params,
            scale_grads=scale_grads,
        )

    def _load_slot(
        self,
        kind: Literal["checkpoint", "lora"],
        name: str,
        adapter_model: dict[str, torch.Tensor],
        *,
        trainable: bool,
        alpha: float | None,
    ) -> int:
        if self._slot_stack:
            raise RuntimeError("Cannot load a LoRA/checkpoint while a slot is pushed")
        adapter_model = self._prepare_adapter_model(kind, name, adapter_model)
        from art.megatron.lora import LORA_ALPHA, load_lora_slot_into_model

        ref = self._slot_ref(kind, name)
        self._guard_slot_can_load(ref)
        return load_lora_slot_into_model(
            self.runtime.model,
            ref,
            adapter_model,
            alpha=LORA_ALPHA if alpha is None else alpha,
            requires_grad=trainable,
        )

    def _prepare_adapter_model(
        self,
        kind: Literal["checkpoint", "lora"],
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        templates = self._local_lora_adapter_templates()
        keys = set(adapter_model)
        expected = set(templates)
        if dist.is_available() and dist.is_initialized():
            gathered: list[set[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, expected)
            expected = set().union(*(value for value in gathered if value is not None))
        if unknown := sorted(keys - expected):
            preview = ", ".join(repr(key) for key in unknown[:8])
            more = "" if len(unknown) <= 8 else f", ... +{len(unknown) - 8} more"
            raise ValueError(
                f"Adapter for {kind} slot {name!r} contains keys that do not match "
                f"installed LoRA wrapper sites: {preview}{more}. Configure the "
                "Megatron runtime with matching LoRA target modules before loading."
            )
        return {
            key: (
                tensor.to(
                    device=templates[key].device,
                    dtype=templates[key].dtype,
                    non_blocking=True,
                )
                if key in templates
                else tensor
            )
            for key, tensor in adapter_model.items()
        }

    def _local_lora_adapter_templates(self) -> dict[str, torch.Tensor]:
        templates: dict[str, torch.Tensor] = {}
        for chunk in self.runtime.model:
            for module in chunk.modules():
                expected_weight_keys = getattr(module, "_expected_weight_keys", None)
                if not callable(expected_weight_keys):
                    continue
                for suffix, parameter_name in (
                    ("lora_A", "A_T"),
                    ("lora_B", "B_T"),
                ):
                    parameter = getattr(module, parameter_name, None)
                    if not isinstance(parameter, torch.Tensor):
                        continue
                    templates.update(
                        (str(key), parameter) for key in expected_weight_keys(suffix)
                    )
        return templates

    def _set_default_slot(self, ref: "LoRASlotRef") -> None:
        if self._slot_stack:
            raise RuntimeError("Cannot set a LoRA/checkpoint while a slot is pushed")
        self._default_slot_ref = ref

    @staticmethod
    def _slot_ref(
        kind: Literal["checkpoint", "lora"], name: str | None
    ) -> "LoRASlotRef":
        try:
            from art.megatron.lora import LoRASlotRef
        except ModuleNotFoundError as exc:
            if exc.name is None or not exc.name.startswith("megatron"):
                raise

            return cast("LoRASlotRef", _LocalLoRASlotRef(kind=kind, name=name))

        return LoRASlotRef(kind=kind, name=name)

    def _validate_dynamic_slot_consistency(
        self,
        kind: Literal["checkpoint", "lora"],
        name: str,
        loaded_sites: int,
    ) -> tuple[torch.nn.Parameter, ...]:
        from art.megatron.lora import iter_lora_slot_parameters

        ref = self._slot_ref(kind, name)
        params = tuple(iter_lora_slot_parameters(self.runtime.model, ref))
        if not (dist.is_available() and dist.is_initialized()):
            return params

        signature = tuple(
            (
                tuple(param.shape),
                str(param.dtype),
                bool(getattr(param, "allreduce", True)),
                str(getattr(param, "grad_sync_domain", "tp_default")),
                str(getattr(param, "grad_sync_op", "none")),
            )
            for param in params
        )
        local = (int(loaded_sites), signature)
        gathered: list[tuple[int, object] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        ranks = [state for state in gathered if state is not None]
        if all(state == ranks[0] for state in ranks[1:]):
            return params
        raise RuntimeError(
            f"Dynamic LoRA slot {kind}:{name} is not loaded consistently across "
            "distributed ranks. This usually means a sharded/exported LoRA state "
            "dict was passed directly to TrainerRank; gather or materialize the "
            "full adapter state before loading a dynamic slot. "
            f"Loaded-site counts by rank: {[state[0] for state in ranks]}."
        )

    def _resolve_slot_ref(self, request: AnyForwardInput) -> "LoRASlotRef | None":
        if request.checkpoint is not Unset:
            return self._slot_ref("checkpoint", cast(str | None, request.checkpoint))
        if request.lora is not Unset:
            return self._slot_ref("lora", cast(str | None, request.lora))
        if self._slot_stack:
            return self._slot_stack[-1]
        if self._default_slot_ref is not None:
            return self._default_slot_ref
        return self._slot_ref("checkpoint", None)

    def _selected_dynamic_checkpoints(
        self,
        checkpoints: Sequence[str] | None,
    ) -> tuple[str, ...]:
        loaded = set(self._checkpoint_slot_params_by_name)
        if not loaded:
            raise TrainerRankSlotStateError(
                "TrainerRank.optim_step requires a loaded checkpoint slot. Call "
                "load_checkpoint_slot(...) and run backward on outputs produced by "
                "that slot before stepping."
            )
        requested = (
            tuple(sorted(loaded))
            if checkpoints is None
            else tuple(dict.fromkeys(checkpoints))
        )
        if not requested:
            raise TrainerRankSlotStateError(
                "TrainerRank.optim_step(checkpoints=...) received no checkpoint "
                "names. Pass at least one loaded checkpoint slot."
            )
        if unknown := set(requested) - loaded:
            raise ValueError(f"Unknown checkpoint slots: {sorted(unknown)}")
        flags = self._checkpoint_grad_flags(requested)
        selected = tuple(
            name for name, has_grad in zip(requested, flags, strict=True) if has_grad
        )
        if checkpoints is None:
            if selected:
                return selected
            raise TrainerRankSlotStateError(
                "TrainerRank.optim_step found loaded checkpoint slots, but none "
                "have gradients on any rank. Call loss.backward() first."
            )
        if missing := [
            name
            for name, has_grad in zip(requested, flags, strict=True)
            if not has_grad
        ]:
            raise TrainerRankSlotStateError(
                "TrainerRank.optim_step was asked to step checkpoint slots with no "
                f"gradients on any rank: {missing}. Call loss.backward() for those "
                "slots first, or omit them from checkpoints=[...]."
            )
        return selected

    def _checkpoint_grad_flags(self, names: Sequence[str]) -> tuple[bool, ...]:
        flags = torch.tensor(
            [
                any(
                    param.grad is not None
                    for param in self._checkpoint_slot_params_by_name[name]
                )
                for name in names
            ],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        return tuple(bool(flag) for flag in flags.tolist())

    def _dynamic_optim_step(
        self,
        checkpoint_names: Sequence[str],
        *,
        params: AdamParams,
        scale_grads: float,
    ) -> dict[str, float]:
        selected = []
        for name in checkpoint_names:
            self._guard_checkpoint_can_step(name)
            slot_params = self._checkpoint_slot_params_by_name[name]
            slot_grads = self._reduce_dynamic_grads(
                slot_params, scale_grads=scale_grads
            )
            selected.append((name, slot_params, slot_grads))

        all_params = tuple(
            param for _, slot_params, _ in selected for param in slot_params
        )
        all_grads = tuple(grad for _, _, slot_grads in selected for grad in slot_grads)
        grad_norm = _distributed_grad_norm(all_params, all_grads)
        if not torch.isfinite(torch.tensor(grad_norm)):
            self.zero_grad()
            return {
                "learning_rate": float(params.learning_rate),
                "grad_norm": float(grad_norm),
                "update_successful": 0.0,
                "num_zeros_in_grad": 0.0,
            }
        clip = (
            min(1.0, params.grad_clip_norm / (grad_norm + 1.0e-6))
            if params.grad_clip_norm > 0.0
            else 1.0
        )
        for name, model_params, grads in selected:
            dynamic = self._dynamic_optimizer(name, params)
            for master, grad in zip(dynamic.master_params, grads, strict=True):
                master.grad = grad.mul(clip)
            dynamic.optimizer.step()
            dynamic.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for model, master in zip(
                    model_params, dynamic.master_params, strict=True
                ):
                    model.copy_(master)
                    model.grad = None
            self._prune_slot_graphs(self._slot_ref("checkpoint", name))
        return {
            "learning_rate": float(params.learning_rate),
            "grad_norm": float(grad_norm),
            "update_successful": 1.0,
            "num_zeros_in_grad": 0.0,
        }

    def _dynamic_optimizer(
        self,
        name: str,
        params: AdamParams,
    ) -> _DynamicOptimizer:
        dynamic = self._dynamic_optimizers.get(name)
        if dynamic is None:
            dynamic = self._new_dynamic_optimizer(name, params)
            self._dynamic_optimizers[name] = dynamic
            return dynamic
        for group in dynamic.optimizer.param_groups:
            group["lr"] = params.learning_rate
            group["betas"] = (params.beta1, params.beta2)
            group["weight_decay"] = params.weight_decay
        return dynamic

    def _new_dynamic_optimizer(
        self,
        name: str,
        params: AdamParams,
        *,
        master_params: Sequence[torch.Tensor] | None = None,
    ) -> _DynamicOptimizer:
        model_params = self._checkpoint_slot_params_by_name[name]
        sources = model_params if master_params is None else tuple(master_params)
        if len(sources) != len(model_params) or any(
            not isinstance(source, torch.Tensor) for source in sources
        ):
            raise TrainerRankSlotStateError(
                f"Optimizer state for checkpoint slot {name!r} has "
                f"{len(sources)} master parameters; expected {len(model_params)}."
            )
        masters = tuple(
            torch.nn.Parameter(
                source.detach().to(device=model.device, dtype=torch.float32).clone()
            )
            for model, source in zip(
                model_params,
                sources,
                strict=True,
            )
        )
        optimizer = torch.optim.AdamW(
            masters,
            lr=params.learning_rate,
            betas=(params.beta1, params.beta2),
            weight_decay=params.weight_decay,
        )
        return _DynamicOptimizer(optimizer, masters)

    def _restore_dynamic_optimizer(
        self,
        name: str,
        state: Mapping[str, object],
    ) -> _DynamicOptimizer:
        if state.get("format_version") != 1:
            raise TrainerRankSlotStateError(
                f"Unsupported optimizer state format for checkpoint slot {name!r}."
            )
        if state.get("layout") != self._dynamic_optimizer_layout(name):
            raise TrainerRankSlotStateError(
                f"Optimizer state for checkpoint slot {name!r} was saved for a "
                "different topology or parameter layout. Save and restore one "
                "optimizer shard per TrainerRank with matching TP/EP/ETP ranks."
            )
        master_params = state.get("master_params")
        optimizer_state = state.get("optimizer")
        if not isinstance(master_params, Sequence) or not isinstance(
            optimizer_state, Mapping
        ):
            raise TrainerRankSlotStateError(
                f"Optimizer state for checkpoint slot {name!r} is incomplete."
            )
        dynamic = self._new_dynamic_optimizer(
            name,
            AdamParams(learning_rate=0.0),
            master_params=cast(Sequence[torch.Tensor], master_params),
        )
        try:
            dynamic.optimizer.load_state_dict(
                {str(key): value for key, value in optimizer_state.items()}
            )
        except ValueError as exc:
            raise TrainerRankSlotStateError(
                f"Optimizer state for checkpoint slot {name!r} does not match the "
                "loaded slot parameter groups."
            ) from exc
        for param in dynamic.master_params:
            for state_name, value in dynamic.optimizer.state.get(param, {}).items():
                if (
                    isinstance(value, torch.Tensor)
                    and int(value.ndim) > 0
                    and tuple(value.shape) != tuple(param.shape)
                ):
                    raise TrainerRankSlotStateError(
                        f"Optimizer state {state_name!r} for checkpoint slot "
                        f"{name!r} has shape {tuple(value.shape)}, but the loaded "
                        f"slot parameter has shape {tuple(param.shape)}."
                    )
        return dynamic

    def _reduce_dynamic_grads(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        scale_grads: float,
    ) -> tuple[torch.Tensor, ...]:
        from megatron.core import parallel_state as ps

        from art.megatron.training.finalize_grads import (
            coalesced_all_reduce,
            tensor_parallel_grad_sync,
        )

        buckets: dict[
            tuple[int, str, torch.dtype, torch.device],
            tuple[object, dist.ReduceOp.RedOpType, list[torch.Tensor]],
        ] = {}

        def add(group: object, op: dist.ReduceOp.RedOpType, grad: torch.Tensor) -> None:
            key = (id(group), str(op), grad.dtype, grad.device)
            buckets.setdefault(key, (group, op, []))[2].append(grad)

        grads = tuple(
            (
                torch.zeros_like(param, dtype=torch.float32)
                if param.grad is None
                else param.grad.detach().float().mul(scale_grads)
            )
            for param in params
        )
        for param, grad in zip(params, grads, strict=True):
            if bool(getattr(param, "allreduce", True)):
                group = ps.get_data_parallel_group(with_context_parallel=True)
            else:
                group = ps.get_expert_data_parallel_group()
            if group is not None and group.size() > 1:
                add(group, dist.ReduceOp.SUM, grad)

            sync = tensor_parallel_grad_sync(param, name="dynamic LoRA")
            if sync is not None:
                group, reduce_op = sync
                add(group, reduce_op, grad)

        for group, op, bucket_grads in buckets.values():
            coalesced_all_reduce(bucket_grads, group=group, op=op)
        return grads

    def _dynamic_optimizer_layout(self, name: str) -> dict[str, object]:
        return {
            "parallel": _parallel_optimizer_coordinates(),
            "parameters": tuple(
                (
                    tuple(param.shape),
                    str(param.dtype),
                    str(getattr(param, "lora_shard_domain", "tp")),
                    bool(getattr(param, "lora_tp_sharded", False)),
                    getattr(param, "lora_tp_shard_dim", None),
                    str(getattr(param, "lora_tp_shard_strategy", "uniform")),
                    tuple(getattr(param, "lora_tp_component_sizes", ())),
                )
                for param in self._checkpoint_slot_params_by_name[name]
            ),
        }

    def _select_next_micro_batch(
        self,
        items: Sequence[ForwardInputsT],
        start: int,
    ) -> _CandidateMicroBatch[ForwardInputsT]:
        dp_rank, dp_size = self._dp_rank_and_size()
        remaining = len(items) - start
        min_width = min(dp_size, remaining)
        if min_width <= 0:
            raise RuntimeError("cannot select an empty microbatch window")
        base_granularity = 1 if remaining < 64 else 8 if remaining < 256 else 32
        granularity = max(
            1,
            ((base_granularity + dp_size - 1) // dp_size) * dp_size,
        )

        def normalize(width: int) -> int:
            width = max(min_width, min(width, remaining))
            if width in (min_width, remaining) or granularity <= 1:
                return width
            if width < granularity:
                return width
            return max(min_width, (width // granularity) * granularity)

        def local_slice(width: int) -> tuple[tuple[int, ...], list[ForwardInputsT]]:
            stop = start + width
            indices = tuple(range(start + dp_rank, stop, dp_size))
            return indices, [items[index] for index in indices]

        estimates: dict[int, tuple[_MemoryCheck, bool, bool] | None] = {}
        plans: dict[int, _FlatForwardPlan] = {}

        def estimate(width: int) -> tuple[_MemoryCheck, bool, bool] | None:
            width = normalize(width)
            if width in estimates:
                return estimates[width]
            indices, local_inputs = local_slice(width)
            values = self._estimate_flat_forward(list(_flatten(local_inputs)))
            if not self._all_ranks_true(values is not None):
                estimates[width] = None
                return None
            assert values is not None
            packed_tokens, output_bytes, signature = values
            result = (
                self._memory_check_required(
                    self._estimate_required_memory_bytes_from_values(
                        packed_tokens=packed_tokens,
                        output_bytes=output_bytes,
                        signature=signature,
                    ),
                    sync_across_dp=True,
                ),
                self._all_ranks_have_memory_profile(
                    packed_tokens=packed_tokens,
                    signature=signature,
                ),
                self._all_ranks_true(signature in self._memory_profiles),
            )
            estimates[width] = result
            return result

        rejected_widths: set[int] = set()

        def fits(width: int) -> tuple[bool, bool]:
            width = normalize(width)
            result = estimate(width)
            if result is None:
                plan = materialize(width)
                check = self._memory_check(plan, sync_across_dp=True)
                trusted = self._all_ranks_have_memory_profile(
                    packed_tokens=plan.packed_tokens,
                    signature=plan.signature,
                )
                profiled = self._all_ranks_true(plan.signature in self._memory_profiles)
            else:
                check, trusted, profiled = result
            if not check.fits:
                rejected_widths.add(width)
            return check.fits and (trusted or not profiled), trusted

        def materialize(width: int) -> _FlatForwardPlan:
            width = normalize(width)
            plan = plans.get(width)
            if plan is None:
                _, local_inputs = local_slice(width)
                plan = self._plan_flat_forward(list(_flatten(local_inputs)))
                plans[width] = plan
            return plan

        def candidate(width: int) -> _CandidateMicroBatch[ForwardInputsT]:
            width = normalize(width)
            indices, local_inputs = local_slice(width)
            plan = materialize(width)
            estimated = estimates.get(width)
            check = (
                estimated[0]
                if estimated is not None
                else self._memory_check(plan, sync_across_dp=True)
            )
            cold_start = not self._all_ranks_have_memory_profile(
                packed_tokens=plan.packed_tokens,
                signature=plan.signature,
            )
            return _CandidateMicroBatch(
                inputs=local_inputs,
                indices=indices,
                plan=plan,
                check=check,
                stats_global_count=width,
                rejected_candidates=len(rejected_widths),
                cold_start=cold_start,
            )

        first_estimate = estimate(min_width)
        if first_estimate is None or not (first_estimate[0].fits and first_estimate[1]):
            first = candidate(min_width)
            if not first.check.fits:
                self._raise_memory_error(
                    first.plan,
                    first.check,
                    context="forward_micro_batches",
                    message="smallest DP microbatch is predicted to exceed available memory",
                )
            if first.cold_start:
                return first

        best = min_width
        failed: int | None = None
        width = normalize(self._last_global_micro_batch_size or min_width)
        if width > best:
            fit, trusted = fits(width)
            if fit:
                best = width
                if not trusted:
                    return candidate(best)
            else:
                failed = width

        while failed is None and best < remaining:
            width = normalize(max(best + 1, best * 2))
            if width == best:
                break
            fit, trusted = fits(width)
            if fit:
                best = width
                if not trusted:
                    break
            else:
                failed = width

        if failed is not None:
            while failed - best > 1:
                width = normalize((best + failed) // 2)
                if width in (best, failed):
                    break
                if fits(width)[0]:
                    best = width
                else:
                    failed = width

        return candidate(best)

    def _validate_replicated_top_level_count(self, count: int) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        counts = [0 for _ in range(dist.get_world_size())]
        dist.all_gather_object(counts, int(count))
        if len(set(counts)) == 1:
            return
        raise ValueError(
            "forward_micro_batches requires the same top-level input count on every "
            "distributed rank. Pass already-DP-local inputs to dp_rank_forward instead. "
            f"Observed counts by rank: {counts}."
        )

    def _dp_rank_and_size(self) -> tuple[int, int]:
        try:
            from megatron.core import parallel_state as ps

            return int(ps.get_data_parallel_rank()), int(
                ps.get_data_parallel_world_size()
            )
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return 0, 1

    def _plan_flat_forward(
        self, requests: Sequence[AnyForwardInput]
    ) -> _FlatForwardPlan:
        plans: list[_ForwardGroupPlan] = []
        output_bytes = self._estimate_group_request_output_bytes(requests)
        logical_tokens = sum(int(request.input_tokens.numel()) for request in requests)
        groups = self._group_active_request_indices(requests)
        for slot_ref, group_indices in groups:
            items = tuple(
                self._forward_item(requests[index]) for index in group_indices
            )
            packed = prefix_tree_pack(
                (item.input_ids for item in items),
                max_depth=self.shared_prefix_max_depth,
            )
            plans.append(
                _ForwardGroupPlan(
                    slot_ref=slot_ref,
                    request_indices=tuple(group_indices),
                    items=items,
                    packed=packed,
                )
            )

        return _FlatForwardPlan(
            request_count=len(requests),
            groups=tuple(plans),
            packed_tokens=sum(int(plan.packed.tokens.numel()) for plan in plans),
            logical_tokens=logical_tokens,
            output_bytes=output_bytes,
            signature=self._memory_signature_from_requests(
                requests,
                slot_group_count=len(plans),
            ),
        )

    def _estimate_flat_forward(
        self, requests: Sequence[AnyForwardInput]
    ) -> tuple[int, int, _MemorySignature] | None:
        groups = self._group_active_request_indices(requests)
        packed_tokens = 0
        for _, group_indices in groups:
            group_packed_tokens = estimate_prefix_tree_packed_tokens(
                (requests[index].input_tokens.reshape(-1) for index in group_indices),
                max_depth=self.shared_prefix_max_depth,
            )
            if group_packed_tokens is None:
                return None
            packed_tokens += group_packed_tokens

        return (
            packed_tokens,
            self._estimate_group_request_output_bytes(requests),
            self._memory_signature_from_requests(
                requests,
                slot_group_count=len(groups),
            ),
        )

    def _group_active_request_indices(
        self,
        requests: Sequence[AnyForwardInput],
    ) -> tuple[tuple["LoRASlotRef | None", tuple[int, ...]], ...]:
        groups: dict[LoRASlotRef | None, list[int]] = {}
        for index, request in enumerate(requests):
            if (
                request.target_tokens is not None
                or request.logits
                or request.top_k is not None
                or request.hidden_states
            ):
                groups.setdefault(self._resolve_slot_ref(request), []).append(index)
        return tuple((slot_ref, tuple(indices)) for slot_ref, indices in groups.items())

    def _run_flat_plan_with_memory_tracking(
        self,
        plan: _FlatForwardPlan,
        *,
        context: str,
    ) -> list[AnyForwardOutput]:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            baseline = int(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            baseline = 0
        try:
            outputs = self._execute_flat_plan(plan)
        except torch.cuda.OutOfMemoryError as exc:
            check = self._memory_check(plan)
            self._raise_memory_error(
                plan,
                check,
                context=context,
                message="CUDA OOM occurred despite the planner estimate",
            )
            raise AssertionError("unreachable") from exc
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
            self._update_memory_profile(plan, max(0, peak - baseline))
        return outputs

    def _execute_flat_plan(self, plan: _FlatForwardPlan) -> list[AnyForwardOutput]:
        outputs = [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ]
        for group in plan.groups:
            from art.megatron.lora import use_lora_slot

            with use_lora_slot(group.slot_ref):
                prepared = self._prepare_packed_forward(group.packed)
                item_outputs = self._forward_packed(group.items, prepared)
            item_outputs = self._track_slot_graph_outputs(group.slot_ref, item_outputs)
            for index, output in zip(group.request_indices, item_outputs, strict=True):
                outputs[index] = output
        return outputs

    def _track_slot_graph_outputs(
        self,
        ref: "LoRASlotRef | None",
        outputs: Sequence[AnyForwardOutput],
    ) -> list[AnyForwardOutput]:
        if ref is None or ref.name is None:
            return list(outputs)

        marker: torch.Tensor | None = None

        def track(tensor: torch.Tensor | None) -> torch.Tensor | None:
            nonlocal marker
            if tensor is None or not tensor.requires_grad:
                return tensor
            if marker is None:
                marker = tensor.new_empty(0)
            return cast(torch.Tensor, _SlotGraphSentinel.apply(tensor, marker))

        tracked_outputs = [
            ForwardOutput(
                target_logprobs=track(output.target_logprobs),
                top_k=(
                    None
                    if output.top_k is None
                    else TopK(
                        logprobs=cast(torch.Tensor, track(output.top_k.logprobs)),
                        tokens=output.top_k.tokens,
                    )
                ),
                logits=track(output.logits),
                hidden_states=track(output.hidden_states),
            )
            for output in outputs
        ]
        if marker is not None:
            self._slot_graphs().setdefault(ref, []).append(weakref.ref(marker))
        return tracked_outputs

    def _slot_graphs(
        self,
    ) -> dict["LoRASlotRef", list[weakref.ReferenceType[torch.Tensor]]]:
        graphs = getattr(self, "_pending_slot_graphs", None)
        if graphs is None:
            graphs = {}
            self._pending_slot_graphs = graphs
        return graphs

    def _prune_slot_graphs(self, ref: "LoRASlotRef | None" = None) -> None:
        graphs = self._slot_graphs()
        refs = tuple(graphs) if ref is None else (ref,)
        for current in refs:
            live = [
                marker for marker in graphs.get(current, ()) if marker() is not None
            ]
            if live:
                graphs[current] = live
            else:
                graphs.pop(current, None)

    def _has_live_slot_graph(self, ref: "LoRASlotRef") -> bool:
        self._prune_slot_graphs(ref)
        return bool(self._slot_graphs().get(ref))

    def _guard_slot_can_load(self, ref: "LoRASlotRef") -> None:
        if not self._has_live_slot_graph(ref):
            return
        raise TrainerRankSlotStateError(
            f"Cannot load {ref.kind} slot {ref.name!r} while outputs from an "
            "earlier forward using that slot still have a live backward graph. "
            "Activation checkpoint recompute resolves slots by name, so replacing "
            "the slot before backward can compute gradients with different LoRA "
            "weights than the original forward. Finish backward first; if the "
            "forward was abandoned, release all references to its outputs; or load "
            "the new weights under a different slot name."
        )

    def _guard_checkpoint_can_step(self, name: str) -> None:
        ref = self._slot_ref("checkpoint", name)
        if not self._has_live_slot_graph(ref):
            return
        raise TrainerRankSlotStateError(
            f"Cannot optim_step checkpoint slot {name!r} while outputs from an "
            "earlier forward using that slot have not been backpropagated. Call "
            "loss.backward() without retaining the graph before optim_step(); if "
            "the forward was abandoned, release all references to its outputs."
        )

    def _estimate_group_request_output_bytes(
        self,
        requests: Sequence[AnyForwardInput],
    ) -> int:
        total = 0
        for request in requests:
            seq_len = int(request.input_tokens.numel())
            if request.target_tokens is not None:
                total += int(request.target_tokens.numel()) * _dtype_size(torch.float32)
            if request.top_k is not None:
                total += (
                    seq_len
                    * int(request.top_k)
                    * (_dtype_size(torch.float32) + _dtype_size(torch.long))
                )
            if request.logits:
                if self._padded_vocab_size is None:
                    raise RuntimeError("logits output memory requires a GPT model")
                total += seq_len * self._padded_vocab_size * self._param_dtype_size
            if request.hidden_states:
                total += seq_len * self._hidden_size * self._param_dtype_size
        return total

    def _memory_signature_from_requests(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        slot_group_count: int,
    ) -> _MemorySignature:
        return _MemorySignature(
            topology=self._topology_key(),
            shared_prefix_max_depth=self.shared_prefix_max_depth,
            slot_group_count=slot_group_count,
            request_mix=tuple(
                sorted({_request_mix_key(request) for request in requests})
            ),
        )

    def _topology_key(self) -> tuple[int, int, int, int]:
        try:
            topology = self._topology()
            return cast(
                tuple[int, int, int, int],
                tuple(
                    int(getattr(topology, name)) for name in ("dp", "tp", "cp", "pp")
                ),
            )
        except (AssertionError, AttributeError, ImportError, RuntimeError, ValueError):
            return (1, 1, 1, 1)

    def _memory_check(
        self,
        forward: _FlatForwardPlan,
        *,
        sync_across_dp: bool = False,
    ) -> _MemoryCheck:
        return self._memory_check_required(
            self._estimate_required_memory_bytes_from_values(
                packed_tokens=forward.packed_tokens,
                output_bytes=forward.output_bytes,
                signature=forward.signature,
            ),
            sync_across_dp=sync_across_dp,
        )

    def _memory_check_required(
        self,
        required: int,
        *,
        sync_across_dp: bool = False,
    ) -> _MemoryCheck:
        available = self._available_memory_bytes()
        if dist.is_available() and dist.is_initialized():
            group = None if sync_across_dp else self._forward_memory_group()
            values = torch.tensor(
                [float(required), float(available)],
                device=self.device if self.device.type == "cuda" else "cpu",
                dtype=torch.float64,
            )
            dist.all_reduce(values[0], op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(values[1], op=dist.ReduceOp.MIN, group=group)
            required = int(values[0].item())
            available = int(values[1].item())
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=available,
            fits=required <= available,
        )

    @staticmethod
    def _forward_memory_group() -> object | None:
        try:
            from megatron.core import parallel_state as ps

            return ps.get_tensor_and_context_parallel_group(check_initialized=False)
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return None

    def _raise_memory_error(
        self,
        plan: _FlatForwardPlan,
        check: _MemoryCheck,
        *,
        context: str,
        message: str,
    ) -> None:
        raise TrainerRankMemoryError(
            f"{context}: {message}. "
            f"packed_tokens={plan.packed_tokens} "
            f"logical_tokens={plan.logical_tokens} "
            f"output_gb={plan.output_bytes / 1024**3:.3f} "
            f"estimated_required_gb={check.estimated_required_bytes / 1024**3:.3f} "
            f"available_gb={check.available_bytes / 1024**3:.3f}. "
            "Use smaller top-level items, reduce output requests, or call "
            "dp_rank_forward with already-DP-local smaller inputs."
        )

    def _estimate_required_memory_bytes_from_values(
        self,
        *,
        packed_tokens: int,
        output_bytes: int,
        signature: _MemorySignature,
    ) -> int:
        if packed_tokens <= 0:
            return output_bytes
        profiled = self._memory_profiles.get(signature)
        activation_factor = max(4, min(16, self._num_layers // 4 + 4))
        static_compute = (
            packed_tokens
            * self._hidden_size
            * self._param_dtype_size
            * activation_factor
        )
        if (
            profiled is None
            or profiled.packed_tokens * _MEMORY_PROFILE_TRUST_GROWTH < packed_tokens
        ):
            compute = static_compute
        else:
            compute = max(static_compute, int(profiled.bytes_per_token * packed_tokens))
        return int((output_bytes + compute) * self.memory_safety_factor)

    def _available_memory_bytes(self) -> int:
        if not (torch.cuda.is_available() and self.device.type == "cuda"):
            return 1 << 60
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        reusable_reserved = max(0, reserved - allocated)
        reserve = int(total * self.memory_reserve_fraction)
        return max(0, int(free) + reusable_reserved - reserve)

    def _all_ranks_have_memory_profile(
        self,
        *,
        packed_tokens: int,
        signature: _MemorySignature,
    ) -> bool:
        profile = self._memory_profiles.get(signature)
        local = packed_tokens <= 0 or (
            profile is not None
            and profile.packed_tokens * _MEMORY_PROFILE_TRUST_GROWTH >= packed_tokens
        )
        return self._all_ranks_true(local)

    def _all_ranks_true(self, local: bool) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return local
        value = torch.tensor(
            int(local),
            device=self.device if self.device.type == "cuda" else "cpu",
            dtype=torch.int32,
        )
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def _update_memory_profile(
        self, plan: _FlatForwardPlan, peak_delta_bytes: int
    ) -> None:
        if plan.packed_tokens <= 0:
            return
        compute_delta = max(0, peak_delta_bytes - plan.output_bytes)
        bytes_per_token = compute_delta / max(1, plan.packed_tokens)
        previous = self._memory_profiles.get(plan.signature)
        self._memory_profiles[plan.signature] = _MemoryProfile(
            bytes_per_token=max(
                bytes_per_token,
                0.0 if previous is None else previous.bytes_per_token,
            ),
            packed_tokens=max(
                plan.packed_tokens,
                0 if previous is None else previous.packed_tokens,
            ),
        )

    def _forward_item(self, request: AnyForwardInput) -> _ForwardItem:
        if request.top_k is not None:
            _validate_top_k(request.top_k, _language_model(self.runtime.model[0]))
        input_ids = request.input_tokens.reshape(-1).to(dtype=torch.long)
        if int(input_ids.numel()) == 0:
            raise ValueError("input_tokens must not be empty")
        labels = None
        if request.target_tokens is not None:
            labels = request.target_tokens.to(dtype=torch.long)
            if int(labels.numel()) == 0:
                raise ValueError("target_tokens must not be empty")
            input_shape = tuple(request.input_tokens.shape)
            if tuple(labels.shape) == input_shape:
                labels = labels.reshape(-1)
            elif (
                labels.ndim > request.input_tokens.ndim
                and tuple(labels.shape[: request.input_tokens.ndim]) == input_shape
            ):
                labels = labels.reshape(
                    int(input_ids.numel()), *labels.shape[request.input_tokens.ndim :]
                )
            elif labels.ndim < 1 or int(labels.shape[0]) != int(input_ids.numel()):
                raise ValueError(
                    "target_tokens must match input_tokens or add trailing target "
                    f"dimensions: input_tokens={input_shape} "
                    f"target_tokens={tuple(labels.shape)}"
                )
        return _ForwardItem(request=request, input_ids=input_ids, labels=labels)

    def _forward_packed(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
    ) -> list[AnyForwardOutput]:
        hidden_by_row = self._gather_sequence_parallel_hidden(
            self._decoder_hidden(prepared)
        )
        return self._project_head(items, prepared, hidden_by_row)

    def _decoder_hidden(
        self,
        prepared: _PreparedPackedForward,
    ) -> torch.Tensor:
        from art.megatron.train import _placeholder_attention_mask

        handler = self.runtime.model_support_handler
        model = _language_model(self.runtime.model[0])
        attention_mask = _placeholder_attention_mask(self.device)
        forward_kwargs = handler.get_forward_kwargs(
            self.runtime.model[0],
            attention_bias=prepared.attention_state,
        )
        extra_block_kwargs = cast(
            dict[str, object] | None,
            forward_kwargs.pop("extra_block_kwargs", None),
        )
        preprocessed = model._preprocess(
            input_ids=prepared.tokens,
            position_ids=prepared.position_ids,
            packed_seq_params=cast("PackedSeqParams", prepared.packed_seq_params),
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preprocessed[:6]
        rotary_pos_cos_sin = preprocessed[6] if len(preprocessed) == 7 else None
        return cast(
            torch.Tensor,
            model.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                packed_seq_params=prepared.packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                **(extra_block_kwargs or {}),
            ),
        )

    def _project_head(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
        hidden_by_row: torch.Tensor,
    ) -> list[AnyForwardOutput]:
        model = _language_model(self.runtime.model[0])
        output_weight = (
            model.shared_embedding_or_output_weight()
            if bool(model.share_embeddings_and_output_weights)
            else None
        )
        device = hidden_by_row.device
        target_logprobs = [None for _ in items]
        logits: list[torch.Tensor | None] = [None for _ in items]
        top_k: list[TopK | None] = [None for _ in items]
        label_rows: list[torch.Tensor | None] = [None for _ in items]
        projected_rows: list[torch.Tensor] = []

        for index, (item, positions_cpu) in enumerate(
            zip(items, prepared.positions_by_item, strict=True)
        ):
            positions = positions_cpu.to(device=device)
            if item.request.logits or item.request.top_k is not None:
                projected_rows.append(positions)
            if item.labels is not None:
                source_positions = prepared.source_positions_by_item[index].to(device)
                labels = item.labels.to(device=device).index_select(0, source_positions)
                label_rows[index] = labels
                target_logprobs[index] = torch.zeros(
                    tuple(labels.shape),
                    device=device,
                    dtype=torch.float32,
                )
                if item.request.top_k is None and not item.request.logits:
                    if int(labels.shape[0]):
                        valid = labels != -100
                        if labels.ndim > 1:
                            valid = valid.reshape(int(labels.shape[0]), -1).any(dim=1)
                        valid_offsets = torch.nonzero(valid, as_tuple=False).reshape(-1)
                        if int(valid_offsets.numel()):
                            projected_rows.append(
                                positions.index_select(0, valid_offsets)
                            )
            if item.request.logits:
                logits[index] = torch.empty(
                    (int(positions.numel()), _padded_vocab_size(model)),
                    device=hidden_by_row.device,
                    dtype=hidden_by_row.dtype,
                )
            if item.request.top_k is not None:
                shape = (int(positions.numel()), item.request.top_k)
                top_k[index] = TopK(
                    logprobs=torch.empty(shape, device=device, dtype=torch.float32),
                    tokens=torch.empty(shape, device=device, dtype=torch.long),
                )

        row_tensor = (
            torch.cat(projected_rows).unique(sorted=True)
            if projected_rows
            else torch.empty(0, dtype=torch.long, device=device)
        )
        if int(row_tensor.numel()):
            rows_cpu = row_tensor.detach().cpu()
            cpu_matches = tuple(
                _row_match(
                    positions.cpu(),
                    rows_cpu,
                    chunk_tokens=self.head_chunk_tokens,
                )
                for positions in prepared.positions_by_item
            )
            local_row_matches = tuple(
                (source.to(device), row.to(device), bounds)
                for source, row, bounds in cpu_matches
            )
            logit_rows_cpu = torch.cat(
                tuple(
                    match[1]
                    for item, match in zip(items, cpu_matches, strict=True)
                    if item.request.logits
                )
                or (torch.empty(0, dtype=torch.long),)
            ).unique(sorted=True)
            self._project_vocab_parallel(
                items,
                hidden_by_row,
                row_tensor,
                row_matches=local_row_matches,
                logit_rows=logit_rows_cpu.to(device),
                logit_bounds=_chunk_boundaries(
                    logit_rows_cpu,
                    end=int(row_tensor.numel()),
                    chunk_tokens=self.head_chunk_tokens,
                ),
                output_weight=output_weight,
                target_logprobs=target_logprobs,
                top_k=top_k,
                logits=logits,
                label_rows=label_rows,
            )

        target_logprobs, top_k = _anchor_disconnected_outputs(
            target_logprobs,
            top_k,
            hidden_by_row,
        )
        return [
            ForwardOutput(
                target_logprobs=target_logprobs[index],
                top_k=top_k[index],
                logits=logits[index],
                hidden_states=(
                    _select_positions(hidden_by_row, positions)
                    if item.request.hidden_states
                    else None
                ),
            )
            for index, (item, positions) in enumerate(
                zip(items, prepared.positions_by_item, strict=True)
            )
        ]

    def _project_vocab_parallel(
        self,
        items: Sequence[_ForwardItem],
        hidden_by_row: torch.Tensor,
        rows: torch.Tensor,
        *,
        row_matches: Sequence[_RowMatch],
        logit_rows: torch.Tensor,
        logit_bounds: tuple[int, ...],
        output_weight: torch.Tensor | None,
        target_logprobs: list[torch.Tensor | None],
        top_k: list[TopK | None],
        logits: list[torch.Tensor | None],
        label_rows: list[torch.Tensor | None],
    ) -> None:
        model = _language_model(self.runtime.model[0])
        max_top_k = max((int(item.request.top_k or 0) for item in items), default=0)
        need_log_z = any(
            item.labels is not None or item.request.top_k is not None for item in items
        )
        for chunk_index, start in enumerate(
            range(0, int(rows.numel()), self.head_chunk_tokens)
        ):
            chunk_rows = rows[start : start + self.head_chunk_tokens]
            local_logits = self._local_logits_from_hidden_rows(
                model,
                _select_positions(hidden_by_row, chunk_rows),
                output_weight=output_weight,
            )
            log_z: torch.Tensor | None = None
            local_topk: tuple[torch.Tensor, torch.Tensor] | None = None
            if need_log_z:
                topk_stats = _try_triton_local_topk_stats(local_logits, k=max_top_k)
                logsumexp_stats = (
                    cast(
                        tuple[torch.Tensor, torch.Tensor] | None,
                        _try_triton_stats("local_logsumexp_stats", local_logits),
                    )
                    if topk_stats is None
                    else None
                )
                stats = topk_stats if topk_stats is not None else logsumexp_stats
                if stats is not None:
                    local_max, local_sum = stats[:2]
                    local_max = local_max.detach()
                    global_max = _all_reduce_tensor_parallel_max(local_max)
                    global_sum = _all_reduce_tensor_parallel_sum(
                        local_sum * torch.exp(local_max - global_max)
                    )
                    log_z = global_max + torch.log(global_sum)
                else:
                    log_z = _vocab_parallel_log_z(local_logits)

                if topk_stats is not None:
                    _, _, local_values, local_tokens = topk_stats
                    local_topk = (local_values, local_tokens)
                elif logsumexp_stats is not None and max_top_k > 0:
                    local_k = min(max_top_k, int(local_logits.shape[1]))
                    local_values, local_tokens = torch.topk(
                        local_logits, k=local_k, dim=-1
                    )
                    local_topk = (local_values.float(), local_tokens)

            logit_start, logit_end = logit_bounds[chunk_index : chunk_index + 2]
            logit_chunk_offsets = logit_rows[logit_start:logit_end] - start
            chunk_logits: torch.Tensor | None = None
            if int(logit_chunk_offsets.numel()):
                chunk_logits = _batch_seq_logits(
                    self._gather_tensor_parallel_logits(
                        local_logits.index_select(0, logit_chunk_offsets).unsqueeze(1)
                    ),
                    seq_len=int(logit_chunk_offsets.numel()),
                ).squeeze(0)

            for index, item in enumerate(items):
                offsets, row_offsets, bounds = row_matches[index]
                begin, finish = bounds[chunk_index : chunk_index + 2]
                offsets = offsets[begin:finish]
                chunk_offsets = row_offsets[begin:finish] - start
                if int(offsets.numel()) == 0:
                    continue
                item_logits = logits[index]
                if item_logits is not None:
                    if chunk_logits is None:
                        raise RuntimeError("logits output requires gathered logits")
                    item_logits[offsets] = chunk_logits.index_select(
                        0,
                        torch.searchsorted(logit_chunk_offsets, chunk_offsets),
                    )
                labels = label_rows[index]
                item_logprobs = target_logprobs[index]
                if item_logprobs is not None and labels is not None:
                    if log_z is None:
                        raise RuntimeError("target logprobs require logsumexp")
                    selected_log_z = log_z.index_select(0, chunk_offsets)
                    item_logprobs[offsets] = _vocab_parallel_target_logprobs(
                        local_logits,
                        labels.index_select(0, offsets),
                        selected_log_z,
                        row_offsets=chunk_offsets,
                    )
                k = item.request.top_k
                if k is not None:
                    if log_z is None:
                        raise RuntimeError("top_k requires logsumexp")
                    selected_log_z = log_z.index_select(0, chunk_offsets)
                    if local_topk is not None:
                        local_values, local_tokens = local_topk
                        selected_values = local_values.index_select(0, chunk_offsets)
                        selected_tokens = local_tokens.index_select(0, chunk_offsets)
                    else:
                        selected_logits = local_logits.index_select(0, chunk_offsets)
                        selected_values, selected_tokens = torch.topk(
                            selected_logits.float(),
                            k=min(k, int(selected_logits.shape[1])),
                            dim=-1,
                        )
                    values = _vocab_parallel_topk_from_local(
                        selected_values,
                        selected_tokens,
                        k=k,
                        log_z=selected_log_z,
                        vocab_start=_vocab_range(local_logits)[0],
                    )
                    current = top_k[index]
                    if current is None:
                        raise RuntimeError("top_k output was not allocated")
                    current.logprobs[offsets] = values.logprobs
                    current.tokens[offsets] = values.tokens

    def _local_logits_from_hidden_rows(
        self,
        model: "GPTModel",
        hidden: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        output_layer = model.output_layer
        sequence_parallel = bool(getattr(output_layer, "sequence_parallel", False))
        if sequence_parallel:
            output_layer.sequence_parallel = False
        try:
            logits, _ = output_layer(
                hidden.unsqueeze(1),
                weight=output_weight,
                runtime_gather_output=None,
            )
        finally:
            if sequence_parallel:
                output_layer.sequence_parallel = True
        return _batch_seq_logits(
            model._scale_logits(logits),
            seq_len=int(hidden.shape[0]),
        ).squeeze(0)

    def _gather_sequence_parallel_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as ps

        if int(ps.get_tensor_model_parallel_world_size()) <= 1:
            return hidden.squeeze(1)
        from megatron.core import tensor_parallel

        gathered = tensor_parallel.gather_from_sequence_parallel_region(
            hidden,
            tensor_parallel_output_grad=False,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        )
        return cast(torch.Tensor, gathered).squeeze(1)

    def _prepare_packed_forward(
        self,
        batch: PrefixTreePack,
    ) -> _PreparedPackedForward:
        topology = self._topology()
        batch = _pad_packed_batch(batch, multiple=int(topology.tp))
        if int(topology.cp) > 1:
            return self._prepare_context_parallel_forward(batch, topology=topology)
        from art.megatron.prefix_tree_state import create_prefix_tree_state

        handler = self.runtime.model_support_handler
        provider = self.runtime.provider
        return _PreparedPackedForward(
            tokens=batch.tokens.to(self.device),
            position_ids=batch.position_ids.to(self.device),
            attention_state=create_prefix_tree_state(
                group_ids=batch.group_ids,
                parent_ids=batch.parent_ids,
                target_device=self.device,
                build_gdn_execution_spec=handler.build_gdn_execution_spec,
                attention_head_dim=provider.kv_channels,
                attention_value_head_dim=provider.kv_channels,
            ),
            packed_seq_params=None,
            positions_by_item=batch.positions_by_sequence,
            source_positions_by_item=tuple(
                torch.arange(
                    int(positions.numel()),
                    dtype=torch.long,
                    device=positions.device,
                )
                for positions in batch.positions_by_sequence
            ),
        )

    def _prepare_context_parallel_forward(
        self,
        batch: PrefixTreePack,
        *,
        topology: "ParallelTopology",
    ) -> _PreparedPackedForward:
        from megatron.core import parallel_state as ps

        from art.megatron.context_parallel.runtime import (
            _dispatch_tensor,
            prepare_cp_micro,
        )
        from art.megatron.training.microbatches import (
            _art_flex_cp_block_mask_variants,
            _context_parallel_config_for_provider,
            _gdn_planner_config_for_provider,
        )
        from art.preprocessing.pack import PackedTensors

        assistant_mask = torch.ones_like(batch.tokens, dtype=torch.bool)
        sparse_micro: PackedTensors = {
            "tokens": batch.tokens,
            "group_ids": batch.group_ids,
            "parent_ids": batch.parent_ids,
            "input_pos": batch.position_ids,
            "assistant_mask": assistant_mask,
            "logprobs": torch.full_like(
                batch.tokens, float("nan"), dtype=torch.float32
            ),
            "advantages": torch.zeros_like(batch.tokens, dtype=torch.float32),
            "weights": assistant_mask.to(dtype=torch.float32),
            "pixel_values": [None],
            "image_grid_thw": [None],
            "moe_routing_replay": None,
        }
        handler = self.runtime.model_support_handler
        provider = self.runtime.provider
        prepared = prepare_cp_micro(
            micro=sparse_micro,
            topology=topology,
            config=_context_parallel_config_for_provider(provider, self.device),
            cp_group=ps.get_context_parallel_group(check_initialized=False),
            cp_rank=ps.get_context_parallel_rank(),
            build_gdn_execution_spec=handler.build_gdn_execution_spec,
            gdn_planner_config=_gdn_planner_config_for_provider(provider, handler),
            block_mask_variants=_art_flex_cp_block_mask_variants(provider, self.device),
            target_device=self.device,
        )
        if prepared.rank_plan is None:
            raise RuntimeError("CP forward preparation did not return a rank plan")
        local_positions = _dispatch_tensor(
            torch.arange(
                int(batch.tokens.shape[1]),
                dtype=torch.long,
            ).unsqueeze(0),
            rank_plan=prepared.rank_plan,
            pad_value=-1,
            pad_multiple=prepared.pad_multiple,
        )
        local_position_pairs = tuple(
            _local_position_pairs(local_positions, positions)
            for positions in batch.positions_by_sequence
        )
        return _PreparedPackedForward(
            tokens=prepared.tensors.tokens,
            position_ids=prepared.tensors.input_pos,
            attention_state=cast("ArtContextParallelState", prepared.attention_state),
            packed_seq_params=prepared.packed_seq_params,
            positions_by_item=tuple(pair[0] for pair in local_position_pairs),
            source_positions_by_item=tuple(pair[1] for pair in local_position_pairs),
        )

    def _topology(self) -> "ParallelTopology":
        from art.megatron.train import _infer_parallel_topology

        return _infer_parallel_topology(self.runtime.model)

    def _gather_tensor_parallel_logits(self, logits: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as ps

        if int(ps.get_tensor_model_parallel_world_size()) <= 1:
            return logits
        from megatron.core import tensor_parallel

        return cast(
            torch.Tensor,
            tensor_parallel.gather_from_tensor_model_parallel_region(logits),
        )


def _validate_top_k(top_k: int, model: "GPTModel") -> None:
    vocab_size = _padded_vocab_size(model)
    if top_k > vocab_size:
        raise ValueError(f"top_k={top_k} exceeds vocabulary size {vocab_size}")


def _request_mix_key(request: AnyForwardInput) -> str:
    parts = []
    if request.target_tokens is not None:
        target = request.target_tokens
        tail_shape = tuple(target.shape[request.input_tokens.ndim :])
        parts.append(f"target:{tail_shape or 'single'}")
    if request.top_k is not None:
        parts.append(f"topk:{int(request.top_k)}")
    if request.logits:
        parts.append("logits")
    if request.hidden_states:
        parts.append("hidden")
    return "+".join(parts) if parts else "inactive"


def _pad_packed_batch(
    batch: PrefixTreePack,
    *,
    multiple: int,
) -> PrefixTreePack:
    if multiple <= 1:
        return batch
    seq_len = int(batch.tokens.shape[1])
    pad = -seq_len % multiple
    if pad == 0:
        return batch

    device = batch.tokens.device
    next_group = (
        int(batch.group_ids.max().item()) + 1 if int(batch.group_ids.numel()) else 1
    )
    pad_group_ids = torch.arange(
        next_group,
        next_group + pad,
        dtype=batch.group_ids.dtype,
        device=device,
    ).unsqueeze(0)
    return PrefixTreePack(
        tokens=torch.cat((batch.tokens, batch.tokens.new_zeros((1, pad))), dim=1),
        group_ids=torch.cat((batch.group_ids, pad_group_ids), dim=1),
        parent_ids=torch.cat((batch.parent_ids, pad_group_ids), dim=1),
        position_ids=torch.cat(
            (batch.position_ids, batch.position_ids.new_zeros((1, pad))), dim=1
        ),
        positions_by_sequence=batch.positions_by_sequence,
    )


def _language_model(model: torch.nn.Module) -> "GPTModel":
    module: object = model
    while hasattr(module, "module"):
        module = getattr(module, "module")
    if hasattr(module, "_preprocess") and hasattr(module, "decoder"):
        return cast("GPTModel", module)
    language_model = getattr(module, "language_model", None)
    if language_model is not None:
        return cast("GPTModel", language_model)
    raise RuntimeError("expected a Megatron GPT model")


def _padded_vocab_size(model: "GPTModel") -> int:
    vocab_size = getattr(getattr(model, "config", None), "padded_vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("could not determine full padded vocabulary size")
    return int(vocab_size)


def _hidden_size(model: "GPTModel | None", provider: object) -> int:
    for source in (getattr(model, "config", None), model, provider):
        if source is None:
            continue
        hidden_size = getattr(source, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise RuntimeError("could not determine hidden size")


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _distributed_grad_norm(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[torch.Tensor],
) -> float:
    if len(params) != len(grads):
        raise ValueError("params and grads must have matching lengths")
    included = [
        grad
        for param, grad in zip(params, grads, strict=True)
        if _include_in_distributed_grad_norm(param)
    ]
    device = grads[0].device if grads else torch.device("cpu")
    squared = torch.zeros((), device=device, dtype=torch.float32)
    for grad in included:
        squared.add_(grad.float().square().sum())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return float(torch.sqrt(squared).item())


def _include_in_distributed_grad_norm(param: torch.nn.Parameter) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return True
    from megatron.core import parallel_state as ps

    replica_group = (
        ps.get_data_parallel_group(with_context_parallel=True)
        if bool(getattr(param, "allreduce", True))
        else ps.get_expert_data_parallel_group()
    )
    if replica_group is not None and replica_group.size() > 1:
        if replica_group.rank() != 0:
            return False
    if bool(getattr(param, "lora_tp_sharded", False)):
        return True
    shard_group = (
        ps.get_tensor_model_parallel_group(check_initialized=False)
        if getattr(param, "lora_shard_domain", "tp") == "tp"
        else ps.get_expert_tensor_parallel_group(check_initialized=False)
    )
    return shard_group is None or shard_group.size() <= 1 or shard_group.rank() == 0


def _parallel_optimizer_coordinates() -> tuple[int, ...]:
    if not (dist.is_available() and dist.is_initialized()):
        return (1, 0, 1, 0, 1, 0, 1, 0)
    from megatron.core import parallel_state as ps

    expert_tp_group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    return (
        int(ps.get_tensor_model_parallel_world_size()),
        int(ps.get_tensor_model_parallel_rank()),
        int(ps.get_expert_model_parallel_world_size()),
        int(ps.get_expert_model_parallel_rank()),
        1 if expert_tp_group is None else int(expert_tp_group.size()),
        0 if expert_tp_group is None else int(expert_tp_group.rank()),
        int(ps.get_pipeline_model_parallel_world_size()),
        int(ps.get_pipeline_model_parallel_rank()),
    )


def _state_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    return value


def _vocab_parallel_target_logprobs(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
    *,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    start, _ = _vocab_range(local_logits)
    flat_labels = labels.reshape(int(labels.shape[0]), -1)
    local_labels = flat_labels - start
    owns_label = (
        (flat_labels != -100)
        & (local_labels >= 0)
        & (local_labels < int(local_logits.shape[1]))
    )
    rows = row_offsets.reshape(-1, 1).expand_as(flat_labels)
    target_logits = local_logits[
        rows,
        local_labels.clamp(0, int(local_logits.shape[1]) - 1),
    ].float()
    target_logits = target_logits.masked_fill(~owns_label, 0.0).reshape(labels.shape)
    target_logits = _all_reduce_tensor_parallel_sum(target_logits)
    log_z = log_z.reshape(int(log_z.shape[0]), *((1,) * (int(labels.ndim) - 1)))
    return (target_logits.float() - log_z).masked_fill(labels == -100, 0.0)


def _anchor_disconnected_outputs(
    target_logprobs: list[torch.Tensor | None],
    top_k: list[TopK | None],
    hidden_by_row: torch.Tensor,
) -> tuple[list[torch.Tensor | None], list[TopK | None]]:
    if not hidden_by_row.requires_grad:
        return target_logprobs, top_k
    anchor: torch.Tensor | None = None

    def anchor_tensor(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal anchor
        if tensor.requires_grad:
            return tensor
        if anchor is None:
            anchor = hidden_by_row.reshape(-1)[:1].float().sum() * 0.0
        return tensor + anchor

    for index, logprobs in enumerate(target_logprobs):
        if logprobs is not None:
            target_logprobs[index] = anchor_tensor(logprobs)
    for index, item in enumerate(top_k):
        if item is not None:
            top_k[index] = TopK(anchor_tensor(item.logprobs), item.tokens)
    return target_logprobs, top_k


def _try_triton_local_topk_stats(
    local_logits: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if k <= 0 or k > int(
        os.environ.get("ART_TRAINER_RANK_TRITON_FUSED_TOPK_MAX", "10")
    ):
        return None
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        _try_triton_stats(
            "local_topk_stats",
            local_logits,
            k=min(k, int(local_logits.shape[1])),
        ),
    )


def _try_triton_stats(
    name: str,
    local_logits: torch.Tensor,
    **kwargs: object,
) -> object | None:
    if not local_logits.is_cuda:
        return None
    if os.environ.get("ART_TRAINER_RANK_TRITON_TOPK", "1").lower() in {
        "0",
        "false",
    } or int(local_logits.shape[0]) < int(
        os.environ.get("ART_TRAINER_RANK_TRITON_MIN_ROWS", "64")
    ):
        return None
    try:
        from art.trainer_rank import topk

        return getattr(topk, name)(local_logits, **kwargs)
    except Exception:
        if os.environ.get("ART_TRAINER_RANK_TRITON_TOPK", "1").lower() == "strict":
            raise
        return None


def _vocab_parallel_topk_from_local(
    local_values: torch.Tensor,
    local_tokens: torch.Tensor,
    *,
    k: int,
    log_z: torch.Tensor,
    vocab_start: int,
) -> TopK:
    local_k = min(k, int(local_values.shape[1]))
    local_values = local_values[:, :local_k]
    local_tokens = local_tokens[:, :local_k] + vocab_start

    from megatron.core import parallel_state as ps

    tp_size = int(ps.get_tensor_model_parallel_world_size())
    if tp_size <= 1:
        return TopK(
            logprobs=local_values - log_z.unsqueeze(1),
            tokens=local_tokens,
        )

    from megatron.core import tensor_parallel

    group = ps.get_tensor_model_parallel_group(check_initialized=False)
    values = cast(
        torch.Tensor,
        tensor_parallel.gather_from_tensor_model_parallel_region(
            local_values,
            group=group,
        ),
    )
    gathered_tokens = [torch.empty_like(local_tokens) for _ in range(tp_size)]
    dist.all_gather(gathered_tokens, local_tokens, group=group)
    tokens = torch.cat(gathered_tokens, dim=1)
    top_values, top_offsets = torch.topk(values, k=k, dim=-1)
    return TopK(
        logprobs=top_values - log_z.unsqueeze(1),
        tokens=tokens.gather(1, top_offsets),
    )


def _vocab_parallel_log_z(local_logits: torch.Tensor) -> torch.Tensor:
    local_logits = local_logits.float()
    local_max = local_logits.max(dim=-1).values.detach()
    global_max = _all_reduce_tensor_parallel_max(local_max)
    local_sum = _local_vocab_exp_sum(local_logits, global_max)
    global_sum = _all_reduce_tensor_parallel_sum(local_sum)
    return global_max + torch.log(global_sum)


def _local_vocab_exp_sum(
    local_logits: torch.Tensor,
    global_max: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(local_logits.float() - global_max.unsqueeze(1)).sum(dim=-1)


def _vocab_range(local_logits: torch.Tensor) -> tuple[int, int]:
    from megatron.core import parallel_state as ps

    local_size = int(local_logits.shape[1])
    rank = int(ps.get_tensor_model_parallel_rank())
    start = rank * local_size
    return start, start + local_size


def _all_reduce_tensor_parallel_sum(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as ps

    if int(ps.get_tensor_model_parallel_world_size()) <= 1:
        return tensor
    from megatron.core import tensor_parallel

    return cast(
        torch.Tensor,
        tensor_parallel.reduce_from_tensor_model_parallel_region(
            tensor,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        ),
    )


def _all_reduce_tensor_parallel_max(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as ps

    if int(ps.get_tensor_model_parallel_world_size()) <= 1:
        return tensor
    output = tensor.clone()
    dist.all_reduce(
        output,
        op=dist.ReduceOp.MAX,
        group=ps.get_tensor_model_parallel_group(check_initialized=False),
    )
    return output


def _row_match(
    positions: torch.Tensor,
    rows: torch.Tensor,
    *,
    chunk_tokens: int,
) -> _RowMatch:
    row_offsets = torch.searchsorted(rows, positions)
    in_bounds = row_offsets < int(rows.numel())
    source_offsets = torch.arange(
        int(positions.numel()), device=positions.device, dtype=torch.long
    )[in_bounds]
    row_offsets = row_offsets[in_bounds]
    keep = rows.index_select(0, row_offsets) == positions.index_select(
        0, source_offsets
    )
    source_offsets, row_offsets = source_offsets[keep], row_offsets[keep]
    if int(row_offsets.numel()) > 1:
        order = row_offsets.argsort()
        source_offsets = source_offsets.index_select(0, order)
        row_offsets = row_offsets.index_select(0, order)
    return (
        source_offsets,
        row_offsets,
        _chunk_boundaries(
            row_offsets,
            end=int(rows.numel()),
            chunk_tokens=chunk_tokens,
        ),
    )


def _chunk_boundaries(
    offsets: torch.Tensor,
    *,
    end: int,
    chunk_tokens: int,
) -> tuple[int, ...]:
    edges = torch.arange(0, end, chunk_tokens, dtype=torch.long)
    edges = torch.cat((edges, torch.tensor((end,), dtype=torch.long)))
    return tuple(torch.searchsorted(offsets, edges).tolist())


def _select_positions(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if int(positions.numel()) == 0:
        return values[:0]
    return values.index_select(0, positions.to(device=values.device))


def _batch_seq_logits(logits: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    if int(logits.ndim) != 3:
        raise RuntimeError(
            f"expected logits with shape [B, S, V] or [S, B, V], got {tuple(logits.shape)}"
        )
    if int(logits.shape[0]) == 1 and int(logits.shape[1]) == seq_len:
        return logits
    if int(logits.shape[0]) == seq_len and int(logits.shape[1]) == 1:
        return logits.transpose(0, 1).contiguous()
    raise RuntimeError(
        f"logits do not match sequence length {seq_len}: {tuple(logits.shape)}"
    )


def _materialize(inputs: ForwardInputs) -> ForwardInputs:
    if isinstance(inputs, ForwardInput):
        return inputs
    return [_materialize(item) for item in _nested_forward_children(inputs)]


def _flatten(inputs: ForwardInputs) -> Iterator[AnyForwardInput]:
    if isinstance(inputs, ForwardInput):
        yield inputs
        return
    for item in _nested_forward_children(inputs):
        yield from _flatten(item)


def _unflatten(
    template: ForwardInputs, outputs: Iterator[AnyForwardOutput]
) -> ForwardOutputs:
    if isinstance(template, ForwardInput):
        return next(outputs)
    return [_unflatten(item, outputs) for item in _nested_forward_children(template)]


def _nested_forward_children(inputs: ForwardInputs) -> Iterator[ForwardInputs]:
    if isinstance(inputs, Mapping):
        raise TypeError(
            "dict was passed directly to TrainerRank; gather or materialize the "
            "values into a list/tuple so nested forward output ordering is explicit"
        )
    if isinstance(inputs, str | bytes):
        raise TypeError(
            "TrainerRank forward inputs must be ForwardInput objects or nested "
            "iterables of ForwardInput objects, not strings"
        )
    try:
        return iter(cast(Iterable[ForwardInputs], inputs))
    except TypeError as exc:
        raise TypeError(
            "TrainerRank forward inputs must be ForwardInput objects or nested "
            "iterables of ForwardInput objects"
        ) from exc


__all__ = [
    "AdamParams",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "MicroBatchStats",
    "TopK",
    "TrainerRank",
    "TrainerRankMemoryError",
    "TrainerRankSlotStateError",
]
