from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Generic, Literal, ParamSpec, TypeVar, cast, overload

import torch
import torch.distributed as dist

from art.megatron.shared_prefix_packing import (
    SharedPrefixPack,
    _local_position_pairs,
    estimate_shared_prefix_packed_tokens,
    pack_shared_prefixes,
)

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
    from megatron.core.packed_seq_params import PackedSeqParams

    from art.megatron.context_parallel.types import (
        ArtContextParallelState,
        ParallelTopology,
    )
    from art.megatron.lora import LoRASlotRef
    from art.megatron.shared_prefix_state import SharedPrefixAttentionState
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
P = ParamSpec("P")
R = TypeVar("R")

_COMPILED_FUNCTIONS: dict[Callable[..., object], Callable[..., object]] = {}
_MEMORY_PROFILE_TRUST_GROWTH = 8


class _Unset:
    pass


Unset = _Unset()
type AdapterSelection = str | None | _Unset


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


@dataclass(frozen=True)
class MicroBatch(Generic[ForwardInputsT]):
    inputs: Sequence[ForwardInputsT]
    outputs: Sequence[ForwardOutputs]
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


class TrainerRankMemoryError(RuntimeError):
    pass


class TrainerRankSlotStateError(RuntimeError):
    pass


@dataclass
class _SlotGraphLease:
    trainer: "TrainerRank"
    ref: "LoRASlotRef"
    active: bool = True

    def release(self) -> None:
        if not self.active:
            return
        self.active = False
        self.trainer._release_slot_graph(self.ref)


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
    attention_state: "SharedPrefixAttentionState | ArtContextParallelState"
    packed_seq_params: "PackedSeqParams | None"
    positions_by_item: tuple[torch.Tensor, ...]
    source_positions_by_item: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _RowMatch:
    source_offsets: torch.Tensor
    row_offsets: torch.Tensor


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
    packed: SharedPrefixPack


@dataclass(frozen=True)
class _FlatForwardPlan:
    request_count: int
    groups: tuple[_ForwardGroupPlan, ...]
    packed_tokens: int
    logical_tokens: int
    output_bytes: int
    signature: _MemorySignature


type _AdaptivePlanCacheKey = tuple[tuple[int, ...], object, tuple[object, ...], int]


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
        self._dynamic_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._checkpoint_slot_params_by_name: dict[
            str, tuple[torch.nn.Parameter, ...]
        ] = {}
        self._pending_slot_graphs: dict[LoRASlotRef, int] = {}
        self._memory_profiles: dict[_MemorySignature, _MemoryProfile] = {}
        self._adaptive_plan_cache: dict[_AdaptivePlanCacheKey, _FlatForwardPlan] = {}
        self._adaptive_estimate_cache: dict[
            _AdaptivePlanCacheKey, tuple[int, int, _MemorySignature] | None
        ] = {}
        self._last_global_micro_batch_size: int | None = None
        self.zero_grad()

    def zero_grad(self) -> None:
        for chunk in self.runtime.model:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()
        optimizer = cast("MegatronOptimizer | None", self.runtime.optimizer)
        if optimizer is not None:
            optimizer.zero_grad()
        for params in self._checkpoint_slot_params_by_name.values():
            for param in params:
                param.grad = None
        self._pending_slot_graphs.clear()

    def _optimizer(self) -> "MegatronOptimizer":
        optimizer = cast("MegatronOptimizer | None", self.runtime.optimizer)
        if optimizer is None:
            raise RuntimeError("TrainerRank requires a runtime with an optimizer")
        return optimizer

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
        alpha: float | None = None,
    ) -> int:
        loaded = self._load_slot(
            "checkpoint", name, adapter_model, trainable=True, alpha=alpha
        )
        self._checkpoint_slot_params_by_name[name] = (
            self._validate_dynamic_slot_consistency("checkpoint", name, loaded)
        )
        self._dynamic_optimizers.pop(name, None)
        return loaded

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

    def _set_default_slot(self, ref: "LoRASlotRef") -> None:
        if self._slot_stack:
            raise RuntimeError("Cannot set a LoRA/checkpoint while a slot is pushed")
        self._default_slot_ref = ref

    @staticmethod
    def _slot_ref(
        kind: Literal["checkpoint", "lora"], name: str | None
    ) -> "LoRASlotRef":
        from art.megatron.lora import LoRASlotRef

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

        local = {
            "rank": dist.get_rank(),
            "loaded_sites": int(loaded_sites),
            "param_count": len(params),
            "numel": sum(int(param.numel()) for param in params),
            "signature": [
                (
                    tuple(int(dim) for dim in param.shape),
                    str(param.dtype),
                    bool(getattr(param, "allreduce", True)),
                    str(getattr(param, "grad_sync_domain", "tp_default")),
                    str(getattr(param, "grad_sync_op", "none")),
                )
                for param in params
            ],
        }
        gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        ranks = [rank for rank in gathered if rank is not None]
        reference = ranks[0]
        if all(
            rank["loaded_sites"] == reference["loaded_sites"]
            and rank["signature"] == reference["signature"]
            for rank in ranks
        ):
            return params

        summary = [
            {key: rank[key] for key in ("rank", "loaded_sites", "param_count", "numel")}
            for rank in ranks
        ]
        raise RuntimeError(
            f"Dynamic LoRA slot {kind}:{name} is not loaded consistently across "
            "distributed ranks. This usually means a sharded/exported LoRA state "
            "dict was passed directly to TrainerRank; gather or materialize the "
            "full adapter state before loading a dynamic slot. "
            f"Rank summary: {summary}."
        )

    def _resolve_slot_ref(self, request: AnyForwardInput) -> "LoRASlotRef | None":
        if request.checkpoint is not Unset:
            return self._slot_ref("checkpoint", cast(str | None, request.checkpoint))
        if request.lora is not Unset:
            return self._slot_ref("lora", cast(str | None, request.lora))
        if self._slot_stack:
            return self._slot_stack[-1]
        return self._default_slot_ref

    def forward_micro_batches(
        self,
        inputs: Iterable[ForwardInputsT],
    ) -> Iterator[MicroBatch[ForwardInputsT]]:
        items = list(inputs)
        self._adaptive_plan_cache.clear()
        self._adaptive_estimate_cache.clear()
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
        if selected_checkpoints:
            return self._dynamic_optim_step(
                selected_checkpoints,
                params=params,
                scale_grads=scale_grads,
            )

        from art.megatron.training.finalize_grads import (
            finalize_model_grads_extended,
            flush_param_grads_to_main_grads,
        )
        from art.megatron.training.model_chunks import as_megatron_api_chunks

        optimizer = self._optimizer()
        flush_param_grads_to_main_grads(self.runtime.model)
        finalize_model_grads_extended(
            as_megatron_api_chunks(self.runtime.model),
            num_tokens=None,
        )
        self._scale_main_grads(scale_grads)
        self._configure_optimizer(params)
        update_successful, grad_norm, num_zeros = optimizer.step()
        optimizer.zero_grad()
        self.zero_grad()
        return {
            "learning_rate": float(params.learning_rate),
            "grad_norm": float(grad_norm),
            "update_successful": float(bool(update_successful)),
            "num_zeros_in_grad": float(num_zeros or 0),
        }

    def _selected_dynamic_checkpoints(
        self,
        checkpoints: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if checkpoints is not None:
            if (
                unknown := set(checkpoints)
                - self._checkpoint_slot_params_by_name.keys()
            ):
                raise ValueError(f"Unknown checkpoint slots: {sorted(unknown)}")
            return tuple(dict.fromkeys(checkpoints))
        slots = tuple(sorted(self._checkpoint_slot_params_by_name.items()))
        if not slots:
            return ()
        has_grad = torch.tensor(
            [
                int(any(param.grad is not None for param in params))
                for _, params in slots
            ],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
        return tuple(name for (name, _), flag in zip(slots, has_grad.tolist()) if flag)

    def _dynamic_optim_step(
        self,
        checkpoint_names: Sequence[str],
        *,
        params: AdamParams,
        scale_grads: float,
    ) -> dict[str, float]:
        all_params: list[torch.nn.Parameter] = []
        for name in checkpoint_names:
            self._guard_checkpoint_can_step(name)
            slot_params = self._checkpoint_slot_params_by_name[name]
            for param in slot_params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                elif scale_grads != 1.0:
                    param.grad.mul_(scale_grads)
            self._reduce_dynamic_grads(slot_params)
            all_params.extend(slot_params)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            all_params,
            max_norm=params.grad_clip_norm,
        )
        for name in checkpoint_names:
            optimizer = self._dynamic_optimizer(name, params)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self._pending_slot_graphs.pop(self._slot_ref("checkpoint", name), None)
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
    ) -> torch.optim.Optimizer:
        optimizer = self._dynamic_optimizers.get(name)
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self._checkpoint_slot_params_by_name[name],
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                weight_decay=params.weight_decay,
            )
            self._dynamic_optimizers[name] = optimizer
            return optimizer
        for group in optimizer.param_groups:
            group["lr"] = params.learning_rate
            group["betas"] = (params.beta1, params.beta2)
            group["weight_decay"] = params.weight_decay
        return optimizer

    def _reduce_dynamic_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
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

        for param in params:
            grad = param.grad
            if grad is None:
                continue
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

        for group, op, grads in buckets.values():
            coalesced_all_reduce(grads, group=group, op=op)

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

        def clamp_width(width: int) -> int:
            return max(min_width, min(width, remaining))

        base_granularity = 1 if remaining < 64 else 8 if remaining < 256 else 32
        granularity = max(
            1,
            ((base_granularity + dp_size - 1) // dp_size) * dp_size,
        )

        def snap_width(width: int) -> int:
            width = clamp_width(width)
            if width in (min_width, remaining) or granularity <= 1:
                return width
            if width < granularity:
                return width
            return max(min_width, (width // granularity) * granularity)

        def local_slice(width: int) -> tuple[tuple[int, ...], list[ForwardInputsT]]:
            stop = start + clamp_width(width)
            indices = tuple(range(start + dp_rank, stop, dp_size))
            return indices, [items[index] for index in indices]

        def candidate(
            width: int,
            estimated_check: _MemoryCheck | None = None,
            *,
            rejected: int,
        ) -> _CandidateMicroBatch[ForwardInputsT]:
            width = clamp_width(width)
            indices, local_inputs = local_slice(width)
            plan = self._cached_adaptive_plan(indices, local_inputs)
            return _CandidateMicroBatch(
                inputs=local_inputs,
                indices=indices,
                plan=plan,
                check=estimated_check or self._memory_check(plan, sync_across_dp=True),
                stats_global_count=width,
                rejected_candidates=rejected,
                cold_start=not self._all_ranks_have_memory_profile(
                    packed_tokens=plan.packed_tokens,
                    signature=plan.signature,
                ),
            )

        def estimate(width: int) -> tuple[_MemoryCheck, bool] | None:
            indices, local_inputs = local_slice(width)
            return self._cached_adaptive_estimate(indices, local_inputs)

        def probe(width: int) -> tuple[bool, _MemoryCheck | None, bool]:
            estimated = estimate(width)
            if estimated is not None:
                check, trusted = estimated
                return trusted and check.fits, check, trusted
            item = candidate(width, rejected=0)
            return item.check.fits, item.check, not item.cold_start

        rejected = 0
        best_width = min_width
        best_check: _MemoryCheck | None = None

        def fit(width: int) -> bool:
            nonlocal best_width, best_check, rejected
            ok, check, _ = probe(width)
            if ok:
                best_width = snap_width(width)
                best_check = check
            else:
                rejected += 1
            return ok

        def search_below(failed_width: int) -> None:
            low = best_width + 1
            high = failed_width - 1
            while low <= high:
                mid = (low + high) // 2
                if fit(mid):
                    low = mid + 1
                else:
                    high = mid - 1

        first_fits, first_check, first_trusted = probe(min_width)
        if not first_fits:
            first = candidate(min_width, first_check, rejected=rejected)
            if not first.check.fits:
                self._raise_memory_error(
                    first.plan,
                    first.check,
                    context="forward_micro_batches",
                    message="smallest DP microbatch is predicted to exceed available memory",
                )
            if first.cold_start:
                return first
            best_check = first.check
        else:
            best_check = first_check

        stable_width = self._last_global_micro_batch_size
        if stable_width is not None and stable_width >= max(64, granularity * 2):
            stable_capacity = stable_width
            stable_width = clamp_width(stable_capacity)
            if fit(stable_width):
                grow_multiplier = 4 if stable_capacity < 256 else 2
                grow_capacity = min(remaining, stable_capacity * grow_multiplier)
                if remaining > grow_capacity:
                    grow_width = clamp_width(grow_capacity)
                    if grow_width > stable_width and not fit(grow_width):
                        search_below(grow_width)
                return candidate(best_width, best_check, rejected=rejected)
            search_below(stable_width)
            self._last_global_micro_batch_size = best_width
            return candidate(best_width, best_check, rejected=rejected)

        high_fail: int | None = None
        width = min(
            remaining,
            max(min_width, (self._last_global_micro_batch_size or min_width) * 2),
        )
        while width <= remaining:
            if fit(width):
                if width == remaining:
                    break
                width = min(remaining, max(width + 1, width * 2))
                continue
            high_fail = width
            break

        if high_fail is not None:
            search_below(high_fail)

        if not first_trusted and best_width == min_width and best_check is None:
            return candidate(min_width, first_check, rejected=rejected)
        return candidate(best_width, best_check, rejected=rejected)

    def _cached_adaptive_plan(
        self,
        indices: tuple[int, ...],
        local_inputs: Sequence[ForwardInputsT],
    ) -> _FlatForwardPlan:
        key = self._adaptive_cache_key(indices)
        cached = self._adaptive_plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._plan_flat_forward(list(_flatten(local_inputs)))
        self._adaptive_plan_cache[key] = plan
        return plan

    def _cached_adaptive_estimate(
        self,
        indices: tuple[int, ...],
        local_inputs: Sequence[ForwardInputsT],
    ) -> tuple[_MemoryCheck, bool] | None:
        key = self._adaptive_cache_key(indices)
        if key in self._adaptive_estimate_cache:
            estimate = self._adaptive_estimate_cache[key]
        else:
            estimate = self._estimate_flat_forward(list(_flatten(local_inputs)))
            self._adaptive_estimate_cache[key] = estimate
        if estimate is None:
            return None
        packed_tokens, output_bytes, signature = estimate
        return (
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
        )

    def _adaptive_cache_key(
        self,
        indices: tuple[int, ...],
    ) -> _AdaptivePlanCacheKey:
        return (
            indices,
            self._default_slot_ref,
            tuple(self._slot_stack),
            self.shared_prefix_max_depth,
        )

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
            packed = pack_shared_prefixes(
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
            group_packed_tokens = estimate_shared_prefix_packed_tokens(
                (requests[index].input_tokens for index in group_indices),
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
            ForwardOutput(
                target_logprobs=None,
                top_k=None,
                logits=None,
                hidden_states=None,
            )
            for _ in range(plan.request_count)
        ]
        for group in plan.groups:
            from art.megatron.lora import use_lora_slot

            with use_lora_slot(group.slot_ref):
                prepared = self._prepare_packed_forward(group.packed)
                item_outputs = self._forward_packed(group.items, prepared)
            self._track_slot_graph_outputs(group.slot_ref, item_outputs)
            for index, output in zip(group.request_indices, item_outputs, strict=True):
                outputs[index] = output
        return outputs

    def _track_slot_graph_outputs(
        self,
        ref: "LoRASlotRef | None",
        outputs: Sequence[AnyForwardOutput],
    ) -> None:
        if ref is None or ref.name is None:
            return
        tensors = [
            tensor
            for output in outputs
            for tensor in _forward_output_grad_tensors(output)
        ]
        if not tensors:
            return

        self._pending_slot_graphs[ref] = self._pending_slot_graphs.get(ref, 0) + 1
        lease = _SlotGraphLease(self, ref)

        def release(grad: torch.Tensor) -> torch.Tensor:
            lease.release()
            return grad

        for tensor in tensors:
            tensor.register_hook(release)

    def _release_slot_graph(self, ref: "LoRASlotRef") -> None:
        count = self._pending_slot_graphs.get(ref, 0)
        if count <= 1:
            self._pending_slot_graphs.pop(ref, None)
        else:
            self._pending_slot_graphs[ref] = count - 1

    def _guard_slot_can_load(self, ref: "LoRASlotRef") -> None:
        if self._pending_slot_graphs.get(ref, 0) <= 0:
            return
        raise TrainerRankSlotStateError(
            f"Cannot load {ref.kind} slot {ref.name!r} while outputs from an "
            "earlier forward using that slot still have a live backward graph. "
            "Activation checkpoint recompute resolves slots by name, so replacing "
            "the slot before backward can compute gradients with different LoRA "
            "weights than the original forward. Call loss.backward() first, or "
            "call zero_grad() if the forward was abandoned, or load the new "
            "weights under a different slot name."
        )

    def _guard_checkpoint_can_step(self, name: str) -> None:
        ref = self._slot_ref("checkpoint", name)
        if self._pending_slot_graphs.get(ref, 0) <= 0:
            return
        raise TrainerRankSlotStateError(
            f"Cannot optim_step checkpoint slot {name!r} while outputs from an "
            "earlier forward using that slot have not been backpropagated. Call "
            "loss.backward() before optim_step(), or call zero_grad() if that "
            "forward was abandoned."
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
        if dist.is_available() and dist.is_initialized():
            value = torch.tensor(
                int(local),
                device=self.device if self.device.type == "cuda" else "cpu",
                dtype=torch.int32,
            )
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
            return bool(value.item())
        return local

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

        row_tensor = (
            torch.cat(projected_rows).unique(sorted=True)
            if projected_rows
            else torch.empty(0, dtype=torch.long, device=device)
        )
        if int(row_tensor.numel()):
            local_row_matches = tuple(
                _row_match(positions.to(device=device), row_tensor)
                for positions in prepared.positions_by_item
            )
            self._project_vocab_parallel(
                items,
                hidden_by_row,
                row_tensor,
                row_matches=local_row_matches,
                item_lengths=tuple(
                    int(positions.numel()) for positions in prepared.positions_by_item
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
        item_lengths: Sequence[int],
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
        for start in range(0, int(rows.numel()), self.head_chunk_tokens):
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
                    _try_triton_local_logsumexp_stats(local_logits)
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

            logit_chunks = [
                chunk_offsets
                for item, match in zip(items, row_matches, strict=True)
                if item.request.logits
                for _, chunk_offsets in (
                    _match_chunk_offsets(
                        match,
                        start=start,
                        end=start + int(chunk_rows.numel()),
                    ),
                )
                if int(chunk_offsets.numel())
            ]
            logit_chunk_offsets = (
                torch.cat(logit_chunks).unique(sorted=True)
                if logit_chunks
                else torch.empty(0, dtype=torch.long, device=rows.device)
            )
            chunk_logits: torch.Tensor | None = None
            if int(logit_chunk_offsets.numel()):
                chunk_logits = _batch_seq_logits(
                    self._gather_tensor_parallel_logits(
                        local_logits.index_select(0, logit_chunk_offsets).unsqueeze(1)
                    ),
                    seq_len=int(logit_chunk_offsets.numel()),
                ).squeeze(0)

            for index, item in enumerate(items):
                offsets, chunk_offsets = _match_chunk_offsets(
                    row_matches[index],
                    start=start,
                    end=start + int(chunk_rows.numel()),
                )
                if int(offsets.numel()) == 0:
                    continue
                item_logits = logits[index]
                if item_logits is not None:
                    if chunk_logits is None:
                        raise RuntimeError("logits output requires gathered logits")
                    source_offsets, gathered_offsets = _matching_offsets(
                        chunk_offsets,
                        logit_chunk_offsets,
                    )
                    item_logits[offsets.index_select(0, source_offsets)] = (
                        chunk_logits.index_select(0, gathered_offsets)
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
                        current = TopK(
                            logprobs=torch.empty(
                                (item_lengths[index], int(values.logprobs.shape[1])),
                                device=values.logprobs.device,
                                dtype=values.logprobs.dtype,
                            ),
                            tokens=torch.empty(
                                (item_lengths[index], int(values.tokens.shape[1])),
                                device=values.tokens.device,
                                dtype=values.tokens.dtype,
                            ),
                        )
                        top_k[index] = current
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
            tensor_parallel_output_grad=True,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        )
        return cast(torch.Tensor, gathered).squeeze(1)

    def _prepare_packed_forward(
        self,
        batch: SharedPrefixPack,
    ) -> _PreparedPackedForward:
        topology = self._topology()
        batch = _pad_packed_batch(batch, multiple=int(topology.tp))
        if int(topology.cp) > 1:
            return self._prepare_context_parallel_forward(batch, topology=topology)
        from art.megatron.shared_prefix_state import create_shared_prefix_state

        handler = self.runtime.model_support_handler
        provider = self.runtime.provider
        return _PreparedPackedForward(
            tokens=batch.tokens.to(self.device),
            position_ids=batch.position_ids.to(self.device),
            attention_state=create_shared_prefix_state(
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
        batch: SharedPrefixPack,
        *,
        topology: "ParallelTopology",
    ) -> _PreparedPackedForward:
        from megatron.core import parallel_state as ps

        from art.megatron.context_parallel.runtime import (
            _dispatch_tensor,
            prepare_cp_micro,
        )
        from art.megatron.training.microbatches import (
            _context_parallel_config_for_provider,
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
        prepared = prepare_cp_micro(
            micro=sparse_micro,
            topology=topology,
            config=_context_parallel_config_for_provider(
                self.runtime.provider, self.device
            ),
            cp_group=ps.get_context_parallel_group(check_initialized=False),
            cp_rank=ps.get_context_parallel_rank(),
            build_gdn_execution_spec=handler.build_gdn_execution_spec,
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

    def _configure_optimizer(self, params: AdamParams) -> None:
        optimizer = self._optimizer()
        config = cast("OptimizerConfig | None", optimizer.config)
        if config is not None:
            config.lr = params.learning_rate
            config.adam_beta1 = params.beta1
            config.adam_beta2 = params.beta2
            config.weight_decay = params.weight_decay
            config.clip_grad = params.grad_clip_norm
        for group in optimizer.param_groups:
            param_group = cast(MutableMapping[str, object], group)
            param_group["lr"] = params.learning_rate
            param_group["weight_decay"] = params.weight_decay
            if "betas" in param_group:
                param_group["betas"] = (params.beta1, params.beta2)

    def _scale_main_grads(self, scale: float) -> None:
        if scale == 1.0:
            return
        for chunk in self.runtime.model:
            for param in chunk.parameters():
                grad = getattr(param, "main_grad", None)
                if isinstance(grad, torch.Tensor):
                    grad.mul_(scale)
                elif param.grad is not None:
                    param.grad.mul_(scale)


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
    batch: SharedPrefixPack,
    *,
    multiple: int,
) -> SharedPrefixPack:
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
    return SharedPrefixPack(
        tokens=torch.cat(
            (
                batch.tokens,
                torch.zeros((1, pad), dtype=batch.tokens.dtype, device=device),
            ),
            dim=1,
        ),
        group_ids=torch.cat((batch.group_ids, pad_group_ids), dim=1),
        parent_ids=torch.cat((batch.parent_ids, pad_group_ids), dim=1),
        position_ids=torch.cat(
            (
                batch.position_ids,
                torch.zeros((1, pad), dtype=batch.position_ids.dtype, device=device),
            ),
            dim=1,
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


def _vocab_parallel_target_logprobs(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
    *,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    start, _ = _vocab_range(local_logits)
    target_logits = _call_compiled(
        _owned_target_logits_for_rows,
        local_logits,
        labels,
        start,
        row_offsets,
    )
    target_logits = _all_reduce_tensor_parallel_sum(target_logits)
    return _call_compiled(_finish_target_logprobs, target_logits, labels, log_z)


def _owned_target_logits_for_rows(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    flat_labels = labels.reshape(int(labels.shape[0]), -1)
    local_labels = flat_labels - vocab_start
    owns_label = (
        (flat_labels != -100)
        & (local_labels >= 0)
        & (local_labels < int(local_logits.shape[1]))
    )
    rows = row_offsets.reshape(int(row_offsets.shape[0]), 1).expand_as(flat_labels)
    selected = local_logits[
        rows,
        local_labels.clamp(0, int(local_logits.shape[1]) - 1),
    ].float()
    return selected.masked_fill(~owns_label, 0.0).reshape(labels.shape)


def _finish_target_logprobs(
    target_logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
) -> torch.Tensor:
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

    return (
        [
            None if item_logprobs is None else anchor_tensor(item_logprobs)
            for item_logprobs in target_logprobs
        ],
        [
            None
            if item_top_k is None
            else TopK(
                logprobs=anchor_tensor(item_top_k.logprobs),
                tokens=item_top_k.tokens,
            )
            for item_top_k in top_k
        ],
    )


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


def _try_triton_local_logsumexp_stats(
    local_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return cast(
        tuple[torch.Tensor, torch.Tensor] | None,
        _try_triton_stats("local_logsumexp_stats", local_logits),
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
    local_values = local_values[:, :local_k] - log_z.unsqueeze(1)
    local_tokens = local_tokens[:, :local_k] + vocab_start

    from megatron.core import parallel_state as ps

    tp_size = int(ps.get_tensor_model_parallel_world_size())
    if tp_size <= 1:
        return TopK(logprobs=local_values, tokens=local_tokens)

    from torch.distributed.nn.functional import all_gather

    group = ps.get_tensor_model_parallel_group(check_initialized=False)
    gathered_values = cast(tuple[torch.Tensor, ...], all_gather(local_values, group))
    gathered_tokens = [torch.empty_like(local_tokens) for _ in range(tp_size)]
    dist.all_gather(gathered_tokens, local_tokens, group=group)
    values = torch.cat(gathered_values, dim=1)
    tokens = torch.cat(gathered_tokens, dim=1)
    top_values, top_offsets = torch.topk(values, k=k, dim=-1)
    return TopK(logprobs=top_values, tokens=tokens.gather(1, top_offsets))


def _vocab_parallel_log_z(local_logits: torch.Tensor) -> torch.Tensor:
    local_logits = local_logits.float()
    local_max = local_logits.max(dim=-1).values.detach()
    global_max = _all_reduce_tensor_parallel_max(local_max)
    local_sum = _call_compiled(_local_vocab_exp_sum, local_logits, global_max)
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
    from torch.distributed.nn.functional import all_reduce

    return cast(
        torch.Tensor,
        all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
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


def _call_compiled(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    if os.environ.get("ART_TRAINER_RANK_COMPILE", "0").lower() in {"0", "false"}:
        return fn(*args, **kwargs)
    compiled = _COMPILED_FUNCTIONS.get(fn)
    if compiled is None:
        compiled = cast(Callable[..., object], torch.compile(fn, dynamic=True))
        _COMPILED_FUNCTIONS[fn] = compiled
    try:
        return cast(Callable[P, R], compiled)(*args, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def _matching_offsets(
    positions: torch.Tensor,
    chunk_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(positions.numel()) == 0 or int(chunk_rows.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty
    sorted_rows, order = chunk_rows.sort()
    indices = torch.searchsorted(sorted_rows, positions)
    in_bounds = indices < int(sorted_rows.numel())
    source_offsets = torch.arange(
        int(positions.numel()),
        device=positions.device,
        dtype=torch.long,
    )[in_bounds]
    found = indices[in_bounds]
    keep = sorted_rows.index_select(0, found) == positions.index_select(
        0,
        source_offsets,
    )
    return source_offsets[keep], order.index_select(0, found[keep])


def _row_match(positions: torch.Tensor, rows: torch.Tensor) -> _RowMatch:
    source_offsets, row_offsets = _matching_offsets(positions, rows)
    if int(row_offsets.numel()) > 1:
        order = row_offsets.argsort()
        source_offsets = source_offsets.index_select(0, order)
        row_offsets = row_offsets.index_select(0, order)
    return _RowMatch(source_offsets=source_offsets, row_offsets=row_offsets)


def _match_chunk_offsets(
    match: _RowMatch,
    *,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = (match.row_offsets >= start) & (match.row_offsets < end)
    source_offsets = match.source_offsets[keep]
    return source_offsets, match.row_offsets[keep] - start


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


def _forward_output_grad_tensors(output: AnyForwardOutput) -> Iterator[torch.Tensor]:
    if output.target_logprobs is not None and output.target_logprobs.requires_grad:
        yield output.target_logprobs
    if output.top_k is not None and output.top_k.logprobs.requires_grad:
        yield output.top_k.logprobs
    if output.logits is not None and output.logits.requires_grad:
        yield output.logits
    if output.hidden_states is not None and output.hidden_states.requires_grad:
        yield output.hidden_states


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
