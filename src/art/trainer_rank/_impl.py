"""Private TrainerRank implementation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
import hashlib
import logging
import math
import os
from pathlib import Path
import struct
import threading
import time
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    NotRequired,
    Self,
    SupportsIndex,
    TypedDict,
    TypeVar,
    cast,
    overload,
)
import weakref

import torch
from torch.autograd.function import FunctionCtx
import torch.distributed as dist
from typing_extensions import TypeIs

from art.megatron.prefix_tree_packing import (
    PrefixTreePack,
    _local_position_pairs,
    estimate_prefix_tree_packed_tokens,
)
from art.trainer_rank._planner_cost import coefficient_version_for
from art.trainer_rank._prefix_tree_materializer import materialize_prefix_tree_layout
from art.trainer_rank._prefix_tree_planner import (
    CanonicalPrefixTree,
    PrefixTreeLayout,
    build_canonical_prefix_tree,
    prefix_tree_layout_candidates,
    select_prefix_tree_layout,
)
from art.trainer_rank._telemetry import phase as _telemetry_phase

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
    from art.trainer_rank._checkpoint import (
        CustomOptimizerState,
        LocalOptimizerState,
        PreparedCheckpoint,
        PreparedCustomPayload,
        _FinalizedSave,
        _PreparedSave,
    )
    from art.trainer_rank._lora_export import _PreparedLoraExport


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
ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)

_MEMORY_PROFILE_TRUST_GROWTH = 8

# Internal calibrated planner policy. These are deliberately not user knobs:
# prefix sharing, head chunking, and memory margins are planner decisions.
_MEMORY_SAFETY_FACTOR = 1.10
_MEMORY_RESERVE_FRACTION = 0.03
_HEAD_CHUNK_TOKENS = 512
_PLANNER_REFINEMENT_BUDGET = 2_000
_LAYOUT_SELECTION_CACHE_LIMIT = 64

# Test-only layout anchor forcing for paired acceptance measurement. Both
# variables must be set; the hook is inert in production.
logger = logging.getLogger(__name__)

_TEST_HOOKS_ENV = "ART_TRAINER_RANK_TEST_HOOKS"
_TEST_ANCHOR_ENV = "ART_TRAINER_RANK_TEST_ANCHOR"
# Test-only usable-memory cap (bytes) for split/decline acceptance cells.
_TEST_MEMORY_LIMIT_ENV = "ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES"

_U64_STRUCT = struct.Struct("<Q")

# Layout selected when the cost-optimal layout cannot be admitted: full sharing
# minimizes packed tokens, and its packed count is monotone in wave width, so
# it is the feasibility predicate the width search relies on.
_MEMORY_MINIMAL_ANCHOR = "full_sharing"
_CHECKPOINT_PREFETCH_EXECUTOR: tuple[int, ThreadPoolExecutor] | None = None
_CHECKPOINT_PREFETCH_EXECUTOR_LOCK = threading.Lock()


def _checkpoint_prefetch_executor() -> ThreadPoolExecutor:
    global _CHECKPOINT_PREFETCH_EXECUTOR
    pid = os.getpid()
    with _CHECKPOINT_PREFETCH_EXECUTOR_LOCK:
        if (
            _CHECKPOINT_PREFETCH_EXECUTOR is None
            or _CHECKPOINT_PREFETCH_EXECUTOR[0] != pid
        ):
            _CHECKPOINT_PREFETCH_EXECUTOR = (
                pid,
                ThreadPoolExecutor(thread_name_prefix="art-checkpoint-prefetch"),
            )
        return _CHECKPOINT_PREFETCH_EXECUTOR[1]


class _AdapterConfig(TypedDict):
    base_model_name_or_path: str
    revision: NotRequired[str | None]
    r: int
    lora_alpha: float
    target_modules: str | list[str]
    num_attention_heads: NotRequired[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    hidden_size: NotRequired[int]


class _Unset:
    pass


Unset = _Unset()
type AdapterSelection = str | None | _Unset


@dataclass(frozen=True)
class _LocalLoRASlotRef:
    name: str | None


@dataclass(frozen=True)
class ForwardOutput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    target_logprobs: LogprobsT
    top_k: TopKT
    logits: LogitsT
    hidden_states: HiddenStatesT
    checkpoint: str | None = None
    no_grad: bool = False


@dataclass(slots=True)
class ForwardInput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    input_tokens: torch.Tensor
    target_tokens: torch.Tensor | None = None
    top_k: int | None = None
    logits: bool = False
    hidden_states: bool = False
    no_grad: bool | None = None
    checkpoint: AdapterSelection = Unset

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
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
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
    ) -> "ForwardInput[torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None]": ...

    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
    ) -> Self:
        return object.__new__(cls)

    def __init__(
        self,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
    ) -> None:
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.logits = logits
        self.hidden_states = hidden_states
        self.no_grad = no_grad
        self.checkpoint = checkpoint
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")


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
    subforward_count: int = 1


class TrainerRankMemoryError(RuntimeError):
    """Bounded conservative planning could not safely admit this call.

    This is not an infeasibility proof: it means the planner's bounded search
    and conservative admission margin found no plan predicted to fit. The
    error reports only what the caller can act on.
    """

    predicted_peak_bytes: int
    usable_limit_bytes: int
    suggestion: str

    def __init__(
        self,
        message: str,
        *,
        predicted_peak_bytes: int = 0,
        usable_limit_bytes: int = 0,
        suggestion: str = "",
    ) -> None:
        super().__init__(message)
        self.predicted_peak_bytes = predicted_peak_bytes
        self.usable_limit_bytes = usable_limit_bytes
        self.suggestion = suggestion

    def __reduce__(self) -> tuple[object, ...]:
        # Keyword-only fields do not survive the default Exception reduce;
        # carry them as state so pickling across process boundaries keeps the
        # actionable numbers.
        return (
            type(self),
            (self.args[0] if self.args else "",),
            {
                "predicted_peak_bytes": self.predicted_peak_bytes,
                "usable_limit_bytes": self.usable_limit_bytes,
                "suggestion": self.suggestion,
            },
        )


class TrainerRankRuntimeSupportError(RuntimeError):
    """The current topology or runtime capability is not yet supported."""


class TrainerRankPartialExecutionError(TrainerRankMemoryError):
    """A subforward of an admitted split failed while executing.

    Distinct from an up-front refusal: model execution already began, so the
    call cannot transparently choose another split. Any earlier subforwards'
    graphs were released; runtime state (e.g. RNG) may have advanced.
    """


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
    logical_per_packed: float = 1.0
    # Fraction of a forward's observed peak still allocated after it returns
    # (activations retained for backward): a physical ratio, so it needs no
    # trusted denominator (the cold call's static estimate is far below the
    # real peak). ``None`` until observed, max-merged afterwards. Observed at
    # forward return only — it says nothing about the backward's own
    # workspace. Trusted only near the ``packed_tokens`` /
    # ``logical_per_packed`` scale it was observed at.
    retained_fraction: float | None = None


@dataclass(frozen=True)
class _CandidateMicroBatch(Generic[ForwardInputsT]):
    inputs: Sequence[ForwardInputsT]
    indices: tuple[int, ...]
    plan: "_AnyForwardPlan"
    check: _MemoryCheck
    stats_global_count: int
    rejected_candidates: int
    cold_start: bool


class _SlotGraphSentinel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        tensor: torch.Tensor,
        marker: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(marker)
        return tensor

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return grad_outputs[0], None


class _CustomSlotGraphSentinel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        tensor: torch.Tensor,
        marker: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(marker)
        return tensor

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        saved_tensors = cast(tuple[torch.Tensor, ...], getattr(ctx, "saved_tensors"))
        (marker,) = saved_tensors

        def finish() -> None:
            try:
                getattr(ctx, "saved_tensors")
            except RuntimeError:
                marker.fill_(True)

        torch.autograd.Variable._execution_engine.queue_callback(finish)
        return grad_outputs[0], None


@dataclass(eq=False)
class _CustomTensorTracker:
    trainer: weakref.ReferenceType[TrainerRank]
    ref: LoRASlotRef
    name: str
    generation: object
    active: bool = False

    def validate(self) -> TrainerRank:
        trainer = self.trainer()
        slot = (
            None
            if trainer is None or self.ref.name is None
            else trainer._checkpoint_slots.get(self.ref.name)
        )
        current = None if slot is None else slot.custom.get(self.name)
        if self.active and (
            current is None or current.generation is not self.generation
        ):
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {self.name!r} is stale because checkpoint "
                f"{self.ref.name!r} was replaced. Register it again from the current "
                "checkpoint before use."
            )
        if trainer is None:
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {self.name!r} no longer has a TrainerRank"
            )
        return trainer

    def record(self, marker: torch.Tensor) -> None:
        trainer = self.validate()
        trainer._slot_graphs().setdefault(self.ref, []).append(weakref.ref(marker))


class _TrackedTensor(torch.Tensor):
    _art_tracker: _CustomTensorTracker

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        tracker: _CustomTensorTracker,
    ) -> Self:
        value = torch.Tensor._make_subclass(cls, data, require_grad=False)
        value._art_tracker = tracker
        return value

    def __deepcopy__(self, memo: dict[int, object]) -> torch.Tensor:
        existing = memo.get(id(self))
        if isinstance(existing, torch.Tensor):
            return existing
        with torch._C.DisableTorchFunctionSubclass():
            result = self.as_subclass(torch.Tensor).detach().clone()
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> str | tuple[Any, ...]:
        with torch._C.DisableTorchFunctionSubclass():
            plain = self.as_subclass(torch.Tensor).detach().clone()
        return cast(str | tuple[Any, ...], plain.__reduce_ex__(proto))

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        return _tracked_tensor_function(func, types, args, kwargs or {})


class _TrackedParameter(torch.nn.Parameter):
    _art_tracker: _CustomTensorTracker

    def __new__(
        cls,
        data: torch.Tensor,
        tracker: _CustomTensorTracker,
        requires_grad: bool = True,
    ) -> Self:
        value = super().__new__(cls, data, requires_grad=requires_grad)
        value._art_tracker = tracker
        return value

    def __init__(
        self,
        data: torch.Tensor,
        tracker: _CustomTensorTracker,
        requires_grad: bool = True,
    ) -> None:
        del data, tracker, requires_grad

    def __setattr__(self, name: str, value: object) -> None:
        if name == "grad":
            with torch._C.DisableTorchFunctionSubclass():
                super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __deepcopy__(self, memo: dict[int, object]) -> torch.nn.Parameter:
        existing = memo.get(id(self))
        if isinstance(existing, torch.nn.Parameter):
            return existing
        with torch._C.DisableTorchFunctionSubclass():
            data = self.as_subclass(torch.Tensor).detach().clone()
            requires_grad = self.requires_grad
        result = torch.nn.Parameter(data, requires_grad=requires_grad)
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> str | tuple[Any, ...]:
        with torch._C.DisableTorchFunctionSubclass():
            data = self.as_subclass(torch.Tensor).detach().clone()
            requires_grad = self.requires_grad
        return cast(
            str | tuple[Any, ...],
            torch.nn.Parameter(data, requires_grad=requires_grad).__reduce_ex__(proto),
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        return _tracked_tensor_function(func, types, args, kwargs or {})


@dataclass(frozen=True)
class _DynamicOptimizer:
    optimizer: torch.optim.Optimizer
    master_params: tuple[torch.nn.Parameter, ...]


@dataclass(frozen=True)
class _CustomObject:
    kind: Literal["module", "parameter", "buffer"]
    value: torch.nn.Module | torch.nn.Parameter | torch.Tensor
    generation: object


@dataclass
class _CheckpointSlot:
    params: tuple[torch.nn.Parameter, ...] = ()
    config: _AdapterConfig | None = None
    optimizer: _DynamicOptimizer | None = None
    revision: int = 0
    custom: dict[str, _CustomObject] = dataclass_field(default_factory=dict)
    custom_payload: "PreparedCustomPayload | None" = None
    snapshot: bool = False


@dataclass(frozen=True)
class MaterializedCheckpoint:
    """A logical checkpoint and its rank-local materialized directory."""

    path: str
    directory: str


@dataclass
class PushedCheckpoint:
    _trainer: "TrainerRank"
    _path: str | None
    _directory: str | None
    _entered: bool = False
    _closed: bool = False

    def __enter__(self) -> "PushedCheckpoint":
        if self._entered or self._closed:
            raise RuntimeError("Pushed checkpoint context cannot be entered twice")
        self._trainer._push_checkpoint_sync(self._path, self._directory)
        self._entered = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._exit(exception)
        return False

    def _exit(self, body_error: BaseException | None) -> None:
        try:
            self._pop()
        except BaseException as pop_error:
            if body_error is not None:
                raise BaseExceptionGroup(
                    "checkpoint context body and cleanup both failed",
                    [body_error, pop_error],
                ) from None
            raise

    def _pop(self) -> None:
        if not self._entered:
            return
        ref = self._trainer._slot_ref(self._path)
        if not self._trainer._slot_stack or self._trainer._slot_stack[-1] != ref:
            raise RuntimeError("Pushed checkpoint stack changed before context exit")
        self._trainer.pop_checkpoint()
        self._entered = False
        self._closed = True


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
    planner_coefficients: int
    slot_group_count: int
    request_mix: tuple[str, ...]
    grad_enabled: bool
    grad_modes: tuple[bool, ...]


@dataclass(frozen=True)
class _ForwardGroupPlan:
    slot_ref: "LoRASlotRef | None"
    grad_enabled: bool
    request_indices: tuple[int, ...]
    items: tuple[_ForwardItem, ...]
    packed: PrefixTreePack


@dataclass(frozen=True)
class _FlatForwardPlan:
    request_count: int
    output_metadata: tuple[tuple[str | None, bool], ...]
    groups: tuple[_ForwardGroupPlan, ...]
    # Physical token count: every group's packed length rounded up to the TP
    # multiple that execution pads it to (see ``_physical_tokens``).
    packed_tokens: int
    logical_tokens: int
    output_bytes: int
    signature: _MemorySignature
    selected_max_depth: int = 0

    @property
    def subforward_count(self) -> int:
        return 1


@dataclass(frozen=True)
class _SplitForwardPlan:
    """One public forward executed as sequential subforwards.

    All returned graphs stay live together, so each subforward was admitted
    against the retained memory of the ones before it. ``request_indices``
    maps every subforward's outputs back to the caller's flat request order.
    """

    subforwards: tuple[_FlatForwardPlan, ...]
    request_indices: tuple[tuple[int, ...], ...]
    request_count: int

    @property
    def subforward_count(self) -> int:
        return len(self.subforwards)

    @property
    def groups(self) -> tuple[_ForwardGroupPlan, ...]:
        return tuple(group for plan in self.subforwards for group in plan.groups)

    @property
    def packed_tokens(self) -> int:
        return sum(plan.packed_tokens for plan in self.subforwards)

    @property
    def logical_tokens(self) -> int:
        return sum(plan.logical_tokens for plan in self.subforwards)

    @property
    def output_bytes(self) -> int:
        return sum(plan.output_bytes for plan in self.subforwards)

    @property
    def selected_max_depth(self) -> int:
        return max(plan.selected_max_depth for plan in self.subforwards)

    @property
    def signature(self) -> _MemorySignature:
        return self.subforwards[0].signature


_AnyForwardPlan = _FlatForwardPlan | _SplitForwardPlan


@dataclass(frozen=True)
class _SubforwardCost:
    """Memory terms of one candidate subforward while all graphs stay live.

    ``required`` is its transient peak while running (the planner estimate);
    ``retained`` is what stays allocated after it returns because its graph
    is kept for backward; ``ephemeral`` is the difference.
    """

    required: int
    retained: int

    @property
    def ephemeral(self) -> int:
        return self.required - self.retained


_MEMORY_ERROR_SUGGESTION = (
    "Use smaller top-level items, reduce output requests, or call "
    "dp_rank_forward with already-DP-local smaller inputs."
)


def _memory_error(
    *,
    context: str,
    message: str,
    packed_tokens: int,
    logical_tokens: int,
    check: _MemoryCheck,
) -> TrainerRankMemoryError:
    return TrainerRankMemoryError(
        f"{context}: {message}. "
        f"packed_tokens={packed_tokens} "
        f"logical_tokens={logical_tokens} "
        f"predicted_peak_gb={check.estimated_required_bytes / 1024**3:.3f} "
        f"usable_limit_gb={check.available_bytes / 1024**3:.3f}. "
        f"{_MEMORY_ERROR_SUGGESTION}",
        predicted_peak_bytes=check.estimated_required_bytes,
        usable_limit_bytes=check.available_bytes,
        suggestion=_MEMORY_ERROR_SUGGESTION,
    )


@dataclass(frozen=True)
class _ForwardRefusal:
    """Why no admissible plan was found. ``plan`` is the unsplit call."""

    plan: _FlatForwardPlan
    check: _MemoryCheck
    message: str

    def error(self, context: str) -> TrainerRankMemoryError:
        return _memory_error(
            context=context,
            message=self.message,
            packed_tokens=self.plan.packed_tokens,
            logical_tokens=self.plan.logical_tokens,
            check=self.check,
        )


def _wave_geometry(item_count: int, start: int, dp_size: int) -> tuple[int, int, int]:
    """Return (remaining, min_width, granularity) for a wave starting at start."""

    remaining = item_count - start
    min_width = min(dp_size, remaining)
    base_granularity = 1 if remaining < 64 else 8 if remaining < 256 else 32
    granularity = max(1, ((base_granularity + dp_size - 1) // dp_size) * dp_size)
    return remaining, min_width, granularity


def _normalize_wave_width(
    width: int, min_width: int, remaining: int, granularity: int
) -> int:
    width = max(min_width, min(width, remaining))
    if width in (min_width, remaining) or granularity <= 1:
        return width
    if width < granularity:
        return width
    return max(min_width, (width // granularity) * granularity)


def _local_wave_indices(
    start: int, width: int, dp_rank: int, dp_size: int
) -> tuple[int, ...]:
    """This DP rank's strided share of the global wave [start, start + width)."""

    return tuple(range(start + dp_rank, start + width, dp_size))


class _PlannerFacts(NamedTuple):
    """Topology and model facts the layout scorer prices against."""

    cp_size: int
    tp_size: int
    layers: int
    gdn_layers: int
    uses_gdn: bool
    # Score version for this runtime: the fitted table inside its calibrated
    # capability profile, the fallback outside it (see coefficient_version_for).
    coefficient_version: int


# Content hash, planner facts (including the coefficient version), anchor.
_LayoutKey = tuple[str, _PlannerFacts, str | None]


def _gdn_layer_count(model: torch.nn.Module) -> int:
    """Number of gated-delta-net layers in the model (0 when unavailable)."""

    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return 0
    return sum(isinstance(module, GatedDeltaNet) for module in model.modules())


def _split_chunks(
    order: Sequence[int],
    requests: Sequence[AnyForwardInput],
    count: int,
) -> tuple[tuple[int, ...], ...]:
    """Cut an ordered request list into ``count`` contiguous, token-balanced chunks."""

    count = max(1, min(count, len(order)))
    masses = [int(requests[index].input_tokens.numel()) for index in order]
    total = sum(masses)
    chunks: list[list[int]] = []
    current: list[int] = []
    cumulative = 0
    for position, index in enumerate(order):
        current.append(index)
        cumulative += masses[position]
        remaining_chunks = count - len(chunks) - 1
        remaining_items = len(order) - position - 1
        boundary = cumulative * count >= total * (len(chunks) + 1)
        if remaining_chunks > 0 and (boundary or remaining_items <= remaining_chunks):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return tuple(tuple(chunk) for chunk in chunks)


class TrainerRank:
    def __init__(self, runtime: TrainingRuntime) -> None:
        pp_size = int(getattr(runtime.provider, "pipeline_model_parallel_size", 1) or 1)
        if pp_size > 1 or len(runtime.model) > 1:
            raise TrainerRankRuntimeSupportError(
                "TrainerRank does not use the MCore forward/backward schedule and "
                "therefore requires PP=1 with exactly one local model chunk; "
                f"got pp={pp_size}, chunks={len(runtime.model)}"
            )
        # Tensor parallelism is admitted: the vocab-parallel head, sequence-
        # parallel gather, TP padding of packed batches and sharded LoRA
        # gradient reduction pre-date the planner, memory checks all-reduce
        # within the TP x CP group, the memory profile is keyed by topology so
        # TP calibrates itself online, and the fitted layout cost model prices
        # TP explicitly. Known limitation: the cold static memory estimate
        # ignores sharding (conservative).
        self.runtime: TrainingRuntime = runtime
        self.device: torch.device = next(runtime.model[0].parameters()).device
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
        # Layers that run the gated-delta-net path (Qwen3.5-4B: 24 of 32); the
        # cost model prices GDN state hand-offs per GDN layer, not per layer.
        self._gdn_layers = _gdn_layer_count(runtime.model[0])
        if self._gdn_layers == 0 and bool(
            getattr(runtime.model_support_handler, "build_gdn_execution_spec", False)
        ):
            self._gdn_layers = self._num_layers
        # The fitted layout cost model applies only inside the capability
        # profile it was calibrated on; other runtimes keep the previous score.
        parameter = next(runtime.model[0].parameters())
        capability: tuple[int, int] | None = None
        device_memory: int | None = None
        if parameter.device.type == "cuda":
            capability = torch.cuda.get_device_capability(parameter.device)
            device_memory = int(
                torch.cuda.get_device_properties(parameter.device).total_memory
            )
        spec = getattr(runtime, "model_support_spec", None)
        is_moe = bool(
            getattr(spec, "is_moe", False)
            or getattr(runtime.model_support_handler, "is_moe", False)
        )
        self._coefficient_version = coefficient_version_for(
            device_capability=capability,
            device_memory_bytes=device_memory,
            param_dtype=str(parameter.dtype),
            hidden_size=int(self._hidden_size),
            is_moe=is_moe,
        )
        if self._coefficient_version != 2:
            logger.warning(
                "TrainerRank layout cost model: runtime (capability=%s, device "
                "memory=%s, dtype=%s, hidden=%s, moe=%s) is outside the calibrated "
                "profile; using the version-%d score",
                capability,
                device_memory,
                parameter.dtype,
                self._hidden_size,
                is_moe,
                self._coefficient_version,
            )
        self._default_slot_ref: LoRASlotRef | None = None
        self._slot_stack: list[LoRASlotRef] = []
        self._checkpoint_slots: dict[str, _CheckpointSlot] = {}
        self._prepared_lora_exports: dict[str, tuple[str, _PreparedLoraExport]] = {}
        self._checkpoint_prefetches: dict[str, Future[PreparedCheckpoint]] = {}
        self._checkpoint_prefetch_sources: dict[str, str] = {}
        self._checkpoint_prefetch_lock = threading.Lock()
        self._checkpoint_mutation_lock = threading.RLock()
        self._checkpoint_process_group: dist.ProcessGroup | None = None
        self._checkpoint_finalize_process_group: dist.ProcessGroup | None = None
        self._checkpoint_group_lock = threading.Lock()
        self._checkpoint_prepare_lock = threading.Lock()
        self._checkpoint_finalize_lock = threading.Lock()
        self._checkpoint_save_condition = threading.Condition()
        self._checkpoint_save_sequence = 0
        self._checkpoint_save_next = 0
        self._checkpoint_save_skipped: set[int] = set()
        self._checkpoint_preparing_saves: set[str] = set()
        self._checkpoint_finalizing_saves: dict[str, Literal["finish", "abort"]] = {}
        self._checkpoint_save_outcomes: dict[str, Literal["finish", "abort"]] = {}
        self._prepared_checkpoint_saves: dict[str, _PreparedSave] = {}
        self._finalized_checkpoint_saves: dict[str, _FinalizedSave] = {}
        self._pending_slot_graphs: dict[
            LoRASlotRef, list[weakref.ReferenceType[torch.Tensor]]
        ] = {}
        self._pending_hybridep_graphs: list[weakref.ReferenceType[torch.Tensor]] = []
        self._hybridep_graph_tracking = False
        self._hybridep_buffer_id: int | None = None
        self._hybridep_rows_high_water = 0
        self._memory_profiles: dict[_MemorySignature, _MemoryProfile] = {}
        self._last_global_micro_batch_size: int | None = None
        # Bounded LRU: steady-state hits are temporally local (identical
        # content on consecutive calls); fresh-token training steps must not
        # accumulate entries for the lifetime of the rank.
        self._layout_selection_cache: OrderedDict[
            _LayoutKey,
            tuple[CanonicalPrefixTree, PrefixTreeLayout],
        ] = OrderedDict()
        self._tree_cache: OrderedDict[str, CanonicalPrefixTree] = OrderedDict()
        self._layout_cache_lock = threading.Lock()
        self._speculative_planner: ThreadPoolExecutor | None = None
        self._speculative_planning_future: Future[None] | None = None
        self._planning_seconds_accum = 0.0
        self._speculative_planning_seconds = 0.0
        self._last_forward_telemetry_snapshot: dict[str, Any] | None = None
        self.zero_grad()

    def zero_grad(self) -> None:
        for chunk in self.runtime.model:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()
        optimizer = self.runtime.optimizer
        if optimizer is not None:
            optimizer.zero_grad()
        for slot in self._checkpoint_slots.values():
            for param in slot.params:
                param.grad = None
        self._prune_slot_graphs()

    def module(
        self,
        name: str,
        factory: Callable[[], ModuleT],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> ModuleT:
        """Return a checkpoint-owned module, registering it on first access.

        Registration is collective across TrainerRank processes. The returned module is
        bound to the resolved checkpoint and is not selected by later push/pop calls.
        """
        value = self._custom_object(name, "module", factory, checkpoint=checkpoint)
        return cast(ModuleT, value)

    def parameter(
        self,
        name: str,
        factory: Callable[[], torch.Tensor | torch.nn.Parameter],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> torch.nn.Parameter:
        """Return a replicated checkpoint-owned trainable parameter."""
        value = self._custom_object(name, "parameter", factory, checkpoint=checkpoint)
        return cast(torch.nn.Parameter, value)

    def buffer(
        self,
        name: str,
        factory: Callable[[], torch.Tensor],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> torch.Tensor:
        """Return a replicated checkpoint-owned persistent tensor."""
        value = self._custom_object(name, "buffer", factory, checkpoint=checkpoint)
        return cast(torch.Tensor, value)

    def _custom_object(
        self,
        name: str,
        kind: Literal["module", "parameter", "buffer"],
        factory: Callable[[], object],
        *,
        checkpoint: AdapterSelection,
    ) -> object:
        from . import _checkpoint

        group = self._checkpoint_group()
        error: BaseException | None = None
        checkpoint_name: str | None = None
        try:
            if not isinstance(name, str) or not name or "." in name or "/" in name:
                raise ValueError(
                    "Custom checkpoint object name must be non-empty and contain "
                    "neither '.' nor '/'"
                )
            if not callable(factory):
                raise TypeError("Custom checkpoint object factory must be callable")
            checkpoint_name = self._resolve_custom_checkpoint(checkpoint)
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(
            error, "validate custom object registration", group
        )
        assert checkpoint_name is not None
        identity = (checkpoint_name, name, kind)
        if any(value != identity for value in _checkpoint._gather(identity, group)):
            raise TrainerRankSlotStateError(
                "Custom checkpoint object registration differs across ranks"
            )
        slot = self._checkpoint_slots[checkpoint_name]
        existing = slot.custom.get(name)
        registered = None if existing is None else existing.kind
        if any(value != registered for value in _checkpoint._gather(registered, group)):
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {name!r} is not registered consistently "
                "across ranks"
            )
        if existing is not None:
            if existing.kind != kind:
                raise TrainerRankSlotStateError(
                    f"Checkpoint {checkpoint_name!r} already registers {name!r} "
                    f"as a {existing.kind}, not a {kind}"
                )
            return existing.value
        custom: _CustomObject | None = None
        try:
            value = factory()
            generation = object()
            if kind == "module":
                if not isinstance(value, torch.nn.Module):
                    raise TypeError("module() factory must return torch.nn.Module")
                value = value.to(device=self.device)
                if slot.snapshot:
                    value.requires_grad_(False)
            elif kind == "parameter":
                if not isinstance(value, torch.Tensor):
                    raise TypeError("parameter() factory must return torch.Tensor")
                value = torch.nn.Parameter(
                    value.detach().to(device=self.device).clone(),
                    requires_grad=not slot.snapshot,
                )
            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError("buffer() factory must return torch.Tensor")
                value = value.detach().to(device=self.device).clone()
            custom = _CustomObject(kind, value, generation)
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(error, f"construct custom object {name!r}", group)
        assert custom is not None
        self._initialize_custom_object(checkpoint_name, name, custom)
        tracker: _CustomTensorTracker | None = None
        named_params: tuple[tuple[str, torch.nn.Parameter], ...] = ()
        new_params: tuple[torch.nn.Parameter, ...] = ()
        extended_optimizer: _DynamicOptimizer | None = None
        error = None
        try:
            tracker = _CustomTensorTracker(
                weakref.ref(self),
                self._slot_ref(checkpoint_name),
                name,
                custom.generation,
            )
            custom = _track_custom_object(custom, tracker)
            named_params = tuple(
                (key, parameter)
                for key, parameter in _custom_named_parameters(name, custom)
                if parameter.requires_grad
            )
            new_params = tuple(parameter for _key, parameter in named_params)
            self._tag_custom_parameters(new_params)
            if slot.optimizer is not None and new_params:
                extended_optimizer = self._extend_dynamic_optimizer(
                    checkpoint_name, named_params
                )
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(
            error, f"stage custom checkpoint object {name!r}", group
        )
        assert tracker is not None
        if extended_optimizer is not None:
            slot.optimizer = extended_optimizer
        slot.custom[name] = custom
        slot.params += new_params
        tracker.active = True
        return custom.value

    def _resolve_custom_checkpoint(self, checkpoint: AdapterSelection) -> str:
        if checkpoint is Unset:
            ref = self._slot_stack[-1] if self._slot_stack else self._default_slot_ref
            name = None if ref is None else ref.name
        else:
            name = cast(str | None, checkpoint)
        if name is None:
            raise TrainerRankSlotStateError(
                "Custom checkpoint objects require a loaded named checkpoint"
            )
        self._ensure_checkpoint_slots((name,))
        if name not in self._checkpoint_slots:
            raise TrainerRankSlotStateError(f"Unknown checkpoint: {name!r}")
        return name

    def _initialize_custom_object(
        self,
        checkpoint: str,
        name: str,
        custom: _CustomObject,
    ) -> None:
        from . import _checkpoint

        slot = self._checkpoint_slots[checkpoint]
        group = self._checkpoint_group()
        error: BaseException | None = None
        values: dict[str, torch.Tensor] = {}
        try:
            loaded = _checkpoint.load_custom_tensors(
                slot.custom_payload, name, custom.kind
            )
            if loaded is not None:
                record, tensors = loaded
                _validate_custom_schema(name, custom, record)
                _load_custom_state(custom, tensors)
            values = _custom_state(custom)
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(
            error, f"initialize custom object {name!r}", group
        )
        signature = _custom_signature(name, custom, values)
        signatures = _checkpoint._gather(signature, group)
        if any(value != signatures[0] for value in signatures):
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {name!r} differs across ranks"
            )
        payloads = _checkpoint._gather(
            {key: value.detach().cpu() for key, value in values.items()}, group
        )
        _load_custom_state(custom, payloads[0])

    @staticmethod
    def _tag_custom_parameters(params: Sequence[torch.nn.Parameter]) -> None:
        for param in params:
            setattr(param, "_art_custom_checkpoint_param", True)
            setattr(param, "allreduce", True)
            setattr(param, "lora_tp_sharded", False)
            setattr(param, "lora_shard_domain", "tp")
            setattr(param, "grad_sync_domain", "tp_default")
            setattr(param, "grad_sync_op", "avg")

    def _extend_dynamic_optimizer(
        self,
        name: str,
        params: Sequence[tuple[str, torch.nn.Parameter]],
    ) -> _DynamicOptimizer:
        from . import _checkpoint

        slot = self._checkpoint_slots[name]
        dynamic = slot.optimizer
        assert dynamic is not None
        restored = _checkpoint.load_custom_optimizer(
            slot.custom_payload, tuple(key for key, _param in params)
        )
        masters = []
        for key, param in params:
            state = restored.get(key)
            if state is not None:
                _validate_custom_optimizer_state(name, key, param, state)
            source = (
                param.detach().float()
                if state is None
                else state.master.to(param.device)
            )
            masters.append(torch.nn.Parameter(source.clone()))
        master_params = tuple(masters)
        all_masters = dynamic.master_params + master_params
        optimizer = torch.optim.AdamW(
            all_masters,
            **{
                key: dynamic.optimizer.defaults[key]
                for key in (
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "amsgrad",
                    "maximize",
                    "foreach",
                    "capturable",
                    "differentiable",
                    "fused",
                )
            },
        )
        optimizer.param_groups[0].update(
            {
                key: value
                for key, value in dynamic.optimizer.param_groups[0].items()
                if key != "params"
            }
        )
        optimizer.state.update(dynamic.optimizer.state)
        for (key, _param), master in zip(params, master_params, strict=True):
            state = restored.get(key)
            if state is not None:
                optimizer.state[master] = {
                    "step": torch.tensor(state.step, device=master.device),
                    "exp_avg": state.exp_avg.to(master.device).clone(),
                    "exp_avg_sq": state.exp_avg_sq.to(master.device).clone(),
                }
        return _DynamicOptimizer(optimizer, all_masters)

    def prefetch_checkpoints(
        self, *checkpoints: str | MaterializedCheckpoint
    ) -> asyncio.Task[None]:
        futures = []
        for checkpoint in checkpoints:
            logical, source = self._checkpoint_source(checkpoint)
            assert logical is not None and source is not None
            futures.append(self._register_checkpoint_prefetch(logical, source))

        async def prefetch() -> None:
            await asyncio.gather(
                *(self._await_checkpoint_prefetch(future) for future in futures)
            )

        return asyncio.create_task(prefetch())

    @staticmethod
    async def _await_checkpoint_prefetch(
        future: Future[PreparedCheckpoint],
    ) -> PreparedCheckpoint:
        return await asyncio.shield(asyncio.wrap_future(future))

    def _register_checkpoint_prefetch(
        self,
        checkpoint: str,
        source: str,
        prepare: Callable[[], PreparedCheckpoint] | None = None,
    ) -> Future[PreparedCheckpoint]:
        key = self._checkpoint_source_key(source)
        with self._checkpoint_prefetch_lock:
            previous = self._checkpoint_prefetch_sources.get(checkpoint)
            self._checkpoint_prefetch_sources[checkpoint] = key
            if (
                previous is not None
                and previous != key
                and previous not in self._checkpoint_prefetch_sources.values()
            ):
                self._checkpoint_prefetches.pop(previous, None)
            future = self._checkpoint_prefetches.get(key)
            if future is None or (
                future.done() and (future.cancelled() or future.exception() is not None)
            ):
                if prepare is None:
                    from ._checkpoint import prepare_checkpoint

                    prepare = lambda: prepare_checkpoint(key)
                future = _checkpoint_prefetch_executor().submit(prepare)
                self._checkpoint_prefetches[key] = future
            return future

    def _checkpoint_prefetch_waiter(self, *checkpoints: str) -> asyncio.Task[None]:
        with self._checkpoint_prefetch_lock:
            futures = [
                self._checkpoint_prefetches[self._checkpoint_prefetch_sources[name]]
                for name in checkpoints
                if name not in self._checkpoint_slots
            ]

        async def wait() -> None:
            await asyncio.gather(
                *(self._await_checkpoint_prefetch(future) for future in futures)
            )

        return asyncio.create_task(wait())

    def _prefetched_checkpoint(self, checkpoint: str) -> PreparedCheckpoint:
        with self._checkpoint_prefetch_lock:
            key = self._checkpoint_prefetch_sources.get(checkpoint)
            future = None if key is None else self._checkpoint_prefetches.get(key)
        if future is None:
            raise TrainerRankSlotStateError(
                f"Checkpoint {checkpoint!r} has not been prefetched"
            )
        return future.result()

    def _load_registered_checkpoint(self, checkpoint: str) -> None:
        from . import _checkpoint

        source: PreparedCheckpoint | None = None
        error: BaseException | None = None
        try:
            source = self._prefetched_checkpoint(checkpoint)
        except BaseException as exc:
            error = exc
        group = _checkpoint._ensure_group(self)
        _checkpoint.raise_distributed(error, "prepare checkpoint", group)
        assert source is not None
        _checkpoint.load_checkpoint(self, source, checkpoint)

    def _ensure_checkpoint_slots(self, checkpoints: Iterable[str]) -> None:
        from . import _checkpoint

        requested = tuple(dict.fromkeys(checkpoints))
        with self._checkpoint_mutation_lock:
            group = _checkpoint._ensure_group(self)
            names = sorted(
                {
                    name
                    for rank_names in _checkpoint._gather(requested, group)
                    for name in rank_names
                }
            )
            for name in names:
                with self._checkpoint_prefetch_lock:
                    state = (
                        name in self._checkpoint_slots,
                        name in self._checkpoint_prefetch_sources,
                    )
                states = _checkpoint._gather(state, group)
                if all(loaded for loaded, _prefetched in states):
                    continue
                if any(loaded for loaded, _prefetched in states):
                    raise TrainerRankSlotStateError(
                        f"Checkpoint {name!r} is not loaded consistently across ranks"
                    )
                if not all(prefetched for _loaded, prefetched in states):
                    raise TrainerRankSlotStateError(
                        f"Explicit selection references unloaded checkpoint {name!r}; "
                        "it has not been prefetched on every rank"
                    )
                self._load_registered_checkpoint(name)

    def load_checkpoint(self, checkpoint: str | MaterializedCheckpoint | None) -> None:
        logical, source = self._checkpoint_source(checkpoint)
        with self._checkpoint_mutation_lock:
            if self._slot_stack:
                raise RuntimeError("Cannot load a checkpoint while one is pushed")
            if logical is None:
                self._set_default_slot(self._slot_ref(None))
                return
            assert source is not None
            if (
                isinstance(checkpoint, MaterializedCheckpoint)
                or logical not in self._checkpoint_prefetch_sources
            ):
                self._register_checkpoint_prefetch(logical, source)
            self._load_registered_checkpoint(logical)
            self._set_default_slot(self._slot_ref(logical))

    def snapshot_checkpoint(self, source: str, destination: str) -> bool:
        """Clone a loaded checkpoint into a forward-only resident snapshot."""
        from . import _checkpoint

        self._ensure_checkpoint_slots((source,))
        return _checkpoint.snapshot_checkpoint(self, source, destination)

    def _discard_snapshot_checkpoint(self, checkpoint: str) -> None:
        """Discard a forward-only resident snapshot."""
        from . import _checkpoint

        _checkpoint.discard_snapshot_checkpoint(self, checkpoint)

    def push_checkpoint(
        self, checkpoint: str | MaterializedCheckpoint | None
    ) -> PushedCheckpoint:
        logical, directory = self._checkpoint_source(checkpoint)
        return PushedCheckpoint(self, logical, directory)

    def _push_checkpoint(self, checkpoint: str | MaterializedCheckpoint | None) -> None:
        logical, source = self._checkpoint_source(checkpoint)
        self._push_checkpoint_sync(logical, source)

    def _push_checkpoint_sync(
        self, logical_path: str | None, source_path: str | None
    ) -> None:
        with self._checkpoint_mutation_lock:
            if source_path is not None:
                assert logical_path is not None
                if (
                    logical_path not in self._checkpoint_slots
                    and logical_path not in self._checkpoint_prefetch_sources
                ):
                    self._register_checkpoint_prefetch(logical_path, source_path)
                self._ensure_checkpoint_slots((logical_path,))
            self._slot_stack.append(self._slot_ref(logical_path))

    def pop_checkpoint(self) -> None:
        with self._checkpoint_mutation_lock:
            if not self._slot_stack:
                raise RuntimeError("No pushed checkpoint to pop")
            self._slot_stack.pop()

    def save_checkpoint(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        self.prepare_checkpoint_save(output_dir, checkpoint_path)
        self.finish_checkpoint_save(output_dir)

    def prepare_checkpoint_save(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        from . import _checkpoint

        _checkpoint.prepare_checkpoint_save(
            self, output_dir, self._resolve_checkpoint_name(checkpoint_path)
        )

    def finish_checkpoint_save(self, output_dir: str) -> None:
        from . import _checkpoint

        _checkpoint.finish_checkpoint_save(self, output_dir)

    def abort_checkpoint_save(self, output_dir: str) -> None:
        from . import _checkpoint

        _checkpoint.abort_checkpoint_save(self, output_dir)

    def export_lora(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> int:
        from . import _lora_export

        return _lora_export.export_lora(
            self, output_dir, self._resolve_checkpoint_name(checkpoint_path)
        )

    def _prepare_lora_export(
        self,
        export_id: str,
        checkpoint_path: str | Literal["active"] = "active",
        *,
        owner_id: str,
    ) -> tuple[int, dict[str, float]]:
        from . import _lora_export

        return _lora_export.prepare_lora_export(
            self,
            export_id,
            self._resolve_checkpoint_name(checkpoint_path),
            owner_id=owner_id,
        )

    def _finish_lora_export(
        self, export_id: str, output_dir: str, *, owner_id: str
    ) -> dict[str, float]:
        from . import _lora_export

        return _lora_export.finish_lora_export(
            self, export_id, output_dir, owner_id=owner_id
        )

    def _abort_lora_export(self, export_id: str, *, owner_id: str) -> None:
        from . import _lora_export

        _lora_export.abort_lora_export(self, export_id, owner_id=owner_id)

    @staticmethod
    def _checkpoint_source_key(path: str) -> str:
        return str(Path(path).resolve())

    @staticmethod
    def _checkpoint_source(
        checkpoint: str | MaterializedCheckpoint | None,
    ) -> tuple[str | None, str | None]:
        if isinstance(checkpoint, MaterializedCheckpoint):
            return checkpoint.path, checkpoint.directory
        return checkpoint, checkpoint

    def _resolve_checkpoint_name(self, checkpoint_path: str | Literal["active"]) -> str:
        if checkpoint_path != "active":
            self._ensure_checkpoint_slots((checkpoint_path,))
            return checkpoint_path
        ref = self._slot_stack[-1] if self._slot_stack else self._default_slot_ref
        if ref is None or ref.name is None:
            raise TrainerRankSlotStateError("No active trainable checkpoint")
        return ref.name

    @staticmethod
    def _slot_state_error(message: str) -> TrainerRankSlotStateError:
        return TrainerRankSlotStateError(message)

    def _checkpoint_group(self) -> dist.ProcessGroup | None:
        from ._checkpoint import _ensure_group

        return _ensure_group(self)

    def _validate_checkpoint_adapter_config(
        self,
        name: str,
        adapter_config: Mapping[str, object] | None,
        *,
        alpha: float | None,
    ) -> _AdapterConfig | None:
        config = None if adapter_config is None else deepcopy(dict(adapter_config))
        if dist.is_available() and dist.is_initialized():
            gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, config, group=self._checkpoint_group())
            if any(value != config for value in gathered):
                raise ValueError(
                    f"Adapter config for checkpoint slot {name!r} differs across ranks"
                )
        if config is None:
            return None
        required = {"base_model_name_or_path", "r", "lora_alpha", "target_modules"}
        if missing := sorted(required - config.keys()):
            raise ValueError(
                f"Adapter config for checkpoint slot {name!r} is missing {missing}"
            )
        base_model = config["base_model_name_or_path"]
        rank = config["r"]
        config_alpha_value = config["lora_alpha"]
        target_modules = config["target_modules"]
        if not isinstance(base_model, str):
            raise TypeError(
                "adapter_config['base_model_name_or_path'] must be a string"
            )
        if base_model.startswith(("Qwen/Qwen3.5-", "Qwen/Qwen3.6-", "Qwen/Qwen3.8-")):
            dimensions = {
                "num_attention_heads": getattr(
                    self.runtime.provider, "num_attention_heads", None
                ),
                "num_key_value_heads": getattr(
                    self.runtime.provider, "num_query_groups", None
                ),
                "head_dim": getattr(self.runtime.provider, "kv_channels", None),
                "hidden_size": getattr(self.runtime.provider, "hidden_size", None),
            }
            for key, value in dimensions.items():
                if value is not None:
                    config[key] = int(value)
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("adapter_config['r'] must be an integer")
        if not isinstance(config_alpha_value, int | float) or isinstance(
            config_alpha_value, bool
        ):
            raise TypeError("adapter_config['lora_alpha'] must be numeric")
        if not isinstance(target_modules, str | list) or (
            isinstance(target_modules, list)
            and not all(isinstance(module, str) for module in target_modules)
        ):
            raise TypeError(
                "adapter_config['target_modules'] must be a string or list of strings"
            )
        if rank < 1:
            raise ValueError("adapter_config['r'] must be >= 1")
        config_alpha = float(config_alpha_value)
        if alpha is not None and float(alpha) != config_alpha:
            raise ValueError(
                f"alpha={alpha} conflicts with adapter_config lora_alpha={config_alpha}"
            )
        return cast(_AdapterConfig, config)

    def _validate_loaded_checkpoint_config(
        self, name: str, config: _AdapterConfig
    ) -> None:
        from art.megatron.lora import LoRA

        ref = self._slot_ref(name)
        slots = [
            slot
            for chunk in self.runtime.model
            for module in chunk.modules()
            if isinstance(module, LoRA)
            if (slot := module._slot(ref)) is not None
        ]
        expected = (int(config["r"]), float(config["lora_alpha"]))
        actual = {(slot.rank, slot.alpha) for slot in slots}
        if actual != {expected}:
            raise ValueError(
                f"Adapter config for checkpoint slot {name!r} declares "
                f"rank/alpha={expected}, loaded weights use {sorted(actual)}"
            )

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
        enabled = torch.is_grad_enabled() if no_grad is None else not no_grad
        batches = self._forward_micro_batches(inputs, checkpoint=checkpoint)
        while True:
            with torch.set_grad_enabled(enabled):
                try:
                    batch = next(batches)
                except StopIteration:
                    return
            yield batch

    def _forward_micro_batches(
        self,
        inputs: Iterable[ForwardInputs],
        *,
        checkpoint: AdapterSelection,
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        items = [_materialize(item) for item in inputs]
        requests = list(_flatten(items))
        self._validate_replicated_top_level_count(len(items))
        for _, indices in self._group_active_request_indices(
            requests, checkpoint=checkpoint
        ):
            for index in indices:
                self._forward_item(requests[index])
        start = 0
        self._reset_planning_telemetry()
        while start < len(items):
            with _telemetry_phase(
                "plan",
                {"global_start": start, "global_remaining": len(items) - start},
            ):
                candidate = self._select_next_micro_batch(
                    items, start, checkpoint=checkpoint
                )
            if isinstance(candidate.plan, _FlatForwardPlan):
                tracked_outputs, memory_baseline = (
                    self._run_flat_plan_with_memory_tracking(
                        candidate.plan,
                        context="forward_micro_batches",
                    )
                )
            else:
                tracked_outputs = self._execute_admitted_plan(
                    candidate.plan, context="forward_micro_batches"
                )
                memory_baseline = None
            flat_outputs = iter(tracked_outputs)
            outputs = [_unflatten(item, flat_outputs) for item in candidate.inputs]
            stop = start + candidate.stats_global_count
            if stop < len(items):
                self._last_global_micro_batch_size = max(
                    self._last_global_micro_batch_size or 0,
                    candidate.stats_global_count,
                )
                # Overlap next-wave planning with the caller's GPU time: the
                # width search seeds from the last wave's width, so pre-plan
                # that slice while this generator is suspended at the yield.
                self._submit_speculative_wave_planning(
                    items, stop, checkpoint=checkpoint
                )
            self._snapshot_planning_telemetry(candidate.plan, candidate.check)
            with _telemetry_phase(
                # This interval is controlled by the caller and normally contains
                # loss construction and backward for the yielded microbatch.
                "caller",
                self._telemetry_signature(candidate.plan),
                dedup_signature=self._telemetry_plan_signature(candidate.plan),
            ):
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
                        subforward_count=candidate.plan.subforward_count,
                    ),
                )
            # The caller normally runs backward while the micro-batch is yielded.
            # Include that peak in future planning; forward-only profiling can
            # otherwise admit a later micro-batch that leaves no collective or
            # optimizer headroom. Peak only: the retained observation belongs
            # to the forward's return, already recorded for this same plan.
            if isinstance(candidate.plan, _FlatForwardPlan):
                self._update_peak_memory_profile(candidate.plan, memory_baseline)
            start = stop

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
        enabled = torch.is_grad_enabled() if no_grad is None else not no_grad
        with torch.set_grad_enabled(enabled):
            self._reset_planning_telemetry()
            materialized = _materialize(inputs)
            requests = list(_flatten(materialized))
            plan = self._plan_admissible_forward(
                requests, checkpoint=checkpoint, context="dp_rank_forward"
            )
            tracked_outputs = self._execute_admitted_plan(
                plan, context="dp_rank_forward"
            )
            return _unflatten(materialized, iter(tracked_outputs))

    def _execute_admitted_plan(
        self, plan: _AnyForwardPlan, *, context: str
    ) -> list[AnyForwardOutput]:
        if isinstance(plan, _FlatForwardPlan):
            outputs, _baseline = self._run_flat_plan_with_memory_tracking(
                plan, context=context
            )
            return outputs
        merged: list[AnyForwardOutput | None] = [None] * plan.request_count
        for ordinal, (subforward, indices) in enumerate(
            zip(plan.subforwards, plan.request_indices, strict=True)
        ):
            try:
                outputs, _baseline = self._run_flat_plan_with_memory_tracking(
                    subforward, context=context
                )
            except TrainerRankMemoryError as error:
                # Model execution already began, so no replanning is possible
                # and the caller must not mistake this for an up-front refusal.
                raise TrainerRankPartialExecutionError(
                    f"{context}: subforward {ordinal + 1} of "
                    f"{plan.subforward_count} failed during execution "
                    f"({ordinal} of {plan.subforward_count} completed). {error}",
                    predicted_peak_bytes=error.predicted_peak_bytes,
                    usable_limit_bytes=error.usable_limit_bytes,
                    suggestion=error.suggestion,
                ) from error
            for index, output in zip(indices, outputs, strict=True):
                merged[index] = output
        if any(output is None for output in merged):
            raise AssertionError("split execution did not cover every request")
        return cast(list[AnyForwardOutput], merged)

    def _plan_admissible_forward(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection,
        context: str,
    ) -> _AnyForwardPlan:
        """Plan one forward (splitting if needed), recording telemetry, or raise."""

        found = self._find_admissible_forward(
            requests,
            checkpoint=checkpoint,
            refusal_prefix="forward is predicted to exceed available memory",
        )
        if isinstance(found, _ForwardRefusal):
            self._snapshot_planning_telemetry(found.plan, found.check)
            raise found.error(context)
        plan, check = found
        self._snapshot_planning_telemetry(plan, check)
        return plan

    def _find_admissible_forward(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection,
        refusal_prefix: str,
    ) -> tuple[_AnyForwardPlan, _MemoryCheck] | _ForwardRefusal:
        """Find an admissible plan: unsplit first, then the bounded split ladder.

        The unsplit plan is tried cost-optimal, then memory-minimal. If neither
        is admitted, the ladder tries 2, 4, ... subforwards (at most one
        request each), cutting the requests in prefix-local depth-first order
        into token-balanced chunks, and stops at the fewest subforwards whose
        rung check passes (``_admit_split_rung``). Exhausting the ladder is a
        refusal worded as "unable to find a feasible split": the search is
        bounded, so this is not a claim that none exists.

        Checkpoint slots are ensured exactly once, up front; everything after
        plans with ``ensure_slots=False`` so the number of collectives this
        rank performs does not depend on its (DP-local) inputs.
        """

        self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
        plan = self._plan_flat_forward(
            requests, checkpoint=checkpoint, ensure_slots=False
        )
        check = self._memory_check(plan)
        if check.fits:
            return plan, check
        # Best effort before splitting: the memory-minimal (full sharing)
        # layouts may fit where the cost-optimal ones do not.
        plan = self._plan_flat_forward(
            requests, checkpoint=checkpoint, memory_minimal=True, ensure_slots=False
        )
        check = self._memory_check(plan)
        if check.fits:
            return plan, check
        request_count = len(requests)
        if request_count == 1:
            return _ForwardRefusal(
                plan, check, f"{refusal_prefix}; a single request cannot be split"
            )
        if self._expert_parallel_active():
            return _ForwardRefusal(
                plan,
                check,
                f"{refusal_prefix}; unable to find a feasible split: internal "
                "splitting is disabled under expert parallelism in this release",
            )
        rows = tuple(
            request.input_tokens.detach().reshape(-1).to("cpu", torch.long)
            for request in requests
        )
        order = self._split_request_order(requests, rows, checkpoint=checkpoint)
        subforward_count = 2
        while True:
            chunks = _split_chunks(order, requests, subforward_count)
            split, check = self._admit_split_rung(
                chunks, requests, rows, checkpoint=checkpoint
            )
            if split is not None:
                return split, check
            if subforward_count >= request_count:
                break
            subforward_count = min(request_count, subforward_count * 2)
        return _ForwardRefusal(
            plan,
            check,
            f"{refusal_prefix}; unable to find a feasible split: every rung of "
            "the bounded ladder (2, 4, ..., one request per subforward) is "
            "predicted to exceed available memory once all returned graphs are "
            "live together",
        )

    def _expert_parallel_active(self) -> bool:
        try:
            from megatron.core import parallel_state as ps

            return int(ps.get_expert_model_parallel_world_size()) > 1
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return False

    def _admit_split_rung(
        self,
        chunks: Sequence[tuple[int, ...]],
        requests: Sequence[AnyForwardInput],
        rows: Sequence[torch.Tensor],
        *,
        checkpoint: AdapterSelection,
    ) -> tuple[_SplitForwardPlan | None, _MemoryCheck]:
        """Admit one rung of the ladder, or return its binding memory check.

        Every returned graph stays live, so subforward ``j`` needs its own
        transient peak plus the memory retained by the subforwards before it.
        Each of those sums is bounded by the rung check — all retained memory
        plus the largest ephemeral share — which therefore decides the rung by
        itself, in any execution order. The same quantity is the headroom the
        caller's backward can count on: every graph live and one subforward's
        forward-ephemeral memory free again. That is a heuristic for backward
        workspace, not a bound (a backward may need more than its forward's
        ephemeral memory); the ballast arm of the split-conversion gate
        measures it on a real cell.

        Cost: the cheap full-sharing lower bound (one O(tokens) CPU scan per
        chunk) rejects a rung without planning anything. A rung that survives
        is priced exactly with cost-optimal layouts and, failing that, with
        memory-minimal layouts, whose packed tokens equal the lower bound — so
        the planner runs for at most one rung, the one that executes.
        """

        lower = [
            self._split_chunk_lower_cost(
                [requests[index] for index in chunk],
                [rows[index] for index in chunk],
                checkpoint=checkpoint,
            )
            for chunk in chunks
        ]
        check = self._split_rung_check(lower)
        if not check.fits:
            return None, check
        for memory_minimal in (False, True):
            plans = [
                self._plan_flat_forward(
                    [requests[index] for index in chunk],
                    checkpoint=checkpoint,
                    memory_minimal=memory_minimal,
                    ensure_slots=False,
                )
                for chunk in chunks
            ]
            costs = [self._plan_cost(plan) for plan in plans]
            check = self._split_rung_check(costs)
            if check.fits:
                # Larger ephemeral first minimizes the running forward peak;
                # ties keep chunk order, so the partition is deterministic.
                order = sorted(
                    range(len(plans)), key=lambda i: (-costs[i].ephemeral, i)
                )
                return (
                    _SplitForwardPlan(
                        subforwards=tuple(plans[i] for i in order),
                        request_indices=tuple(tuple(chunks[i]) for i in order),
                        request_count=len(requests),
                    ),
                    check,
                )
        return None, check

    def _split_rung_check(self, costs: Sequence[_SubforwardCost]) -> _MemoryCheck:
        return self._memory_check_required(
            sum(cost.retained for cost in costs) + max(cost.ephemeral for cost in costs)
        )

    def _split_chunk_lower_cost(
        self,
        requests: Sequence[AnyForwardInput],
        rows: Sequence[torch.Tensor],
        *,
        checkpoint: AdapterSelection,
    ) -> _SubforwardCost:
        """Cheapest possible cost of a chunk: its full-sharing packed tokens."""

        groups = self._group_active_request_indices(
            requests, checkpoint=checkpoint, ensure_slots=False
        )
        packed_tokens = 0
        for _, group_indices in groups:
            estimated = estimate_prefix_tree_packed_tokens(
                (rows[index] for index in group_indices),
                max_depth=len(group_indices),
            )
            assert estimated is not None  # rows are CPU copies
            packed_tokens += self._physical_tokens(estimated)
        return self._subforward_cost(
            packed_tokens=packed_tokens,
            output_bytes=self._estimate_group_request_output_bytes(requests),
            signature=self._memory_signature_from_requests(
                requests,
                slot_group_count=len(groups),
                grad_modes=tuple(mode for (_, mode), _ in groups),
            ),
            logical_tokens=sum(int(row.numel()) for row in rows),
        )

    def _plan_cost(self, plan: _FlatForwardPlan) -> _SubforwardCost:
        return self._subforward_cost(
            packed_tokens=plan.packed_tokens,
            output_bytes=plan.output_bytes,
            signature=plan.signature,
            logical_tokens=plan.logical_tokens,
        )

    def _subforward_cost(
        self,
        *,
        packed_tokens: int,
        output_bytes: int,
        signature: _MemorySignature,
        logical_tokens: int,
    ) -> _SubforwardCost:
        required = self._estimate_required_memory_bytes_from_values(
            packed_tokens=packed_tokens,
            output_bytes=output_bytes,
            signature=signature,
            logical_tokens=logical_tokens,
        )
        retained = int(
            required
            * self._retained_fraction(
                signature, packed_tokens=packed_tokens, logical_tokens=logical_tokens
            )
        )
        return _SubforwardCost(required=required, retained=retained)

    def _retained_fraction(
        self,
        signature: _MemorySignature,
        *,
        packed_tokens: int,
        logical_tokens: int,
    ) -> float:
        """Share of a subforward's peak still allocated after it returns.

        Applied to the estimated peak, which is at least the real one whenever
        the estimate is trusted. 1.0 (everything retained) until observed. An
        observation is trusted only near the scale and sharing ratio it was
        made at — the growth range that already gates ``bytes_per_token`` —
        so a small profiled forward cannot authorize a much larger split.
        """

        profile = self._memory_profiles.get(signature)
        if profile is None or profile.retained_fraction is None:
            return 1.0
        if packed_tokens > profile.packed_tokens * _MEMORY_PROFILE_TRUST_GROWTH:
            return 1.0
        ratio = logical_tokens / max(1, packed_tokens)
        if ratio > profile.logical_per_packed * _MEMORY_PROFILE_TRUST_GROWTH:
            return 1.0
        return profile.retained_fraction

    def _split_request_order(
        self,
        requests: Sequence[AnyForwardInput],
        rows: Sequence[torch.Tensor],
        *,
        checkpoint: AdapterSelection,
    ) -> tuple[int, ...]:
        """Order requests in prefix-local depth-first order.

        Within each checkpoint group, requests follow the canonical tree's
        depth-first leaf order, so prefix-sharing requests are adjacent and
        contiguous cuts keep most sharing inside one chunk; requests that
        produce no outputs (and so join no group) follow in input order.
        """

        ordered: list[int] = []
        seen: set[int] = set()
        for _, group_indices in self._group_active_request_indices(
            requests, checkpoint=checkpoint, ensure_slots=False
        ):
            tree, _layout = self._select_group_layout(
                tuple(rows[index] for index in group_indices)
            )
            for sequence_indices in tree.sequence_indices_by_terminal:
                for position in sequence_indices:
                    ordered.append(group_indices[position])
                    seen.add(group_indices[position])
        ordered.extend(index for index in range(len(requests)) if index not in seen)
        return tuple(ordered)

    def _reset_planning_telemetry(self) -> None:
        self._planning_seconds_accum = 0.0
        with self._layout_cache_lock:
            self._speculative_planning_seconds = 0.0

    def _snapshot_planning_telemetry(
        self, plan: _AnyForwardPlan, check: _MemoryCheck
    ) -> None:
        with self._layout_cache_lock:
            speculative_seconds = self._speculative_planning_seconds
        partition = (
            plan.request_indices
            if isinstance(plan, _SplitForwardPlan)
            else (tuple(range(plan.request_count)),)
        )
        self._last_forward_telemetry_snapshot = {
            "planning_ms": self._planning_seconds_accum * 1_000.0,
            "speculative_planning_ms": speculative_seconds * 1_000.0,
            "selected_max_depth": plan.selected_max_depth,
            "subforward_count": plan.subforward_count,
            # Flat request indices executed by each subforward, in execution
            # order; lets callers backward per subforward if they wish.
            "subforward_request_indices": partition,
            "predicted_peak_bytes": check.estimated_required_bytes,
            "usable_limit_bytes": check.available_bytes,
        }

    def last_forward_telemetry(self) -> dict[str, Any]:
        """Concise planner telemetry for the most recent planned forward.

        ``planning_ms`` is critical-path planning accumulated across the whole
        public call (all waves of ``forward_micro_batches``, including the
        synchronous cost of submitting speculative work);
        ``speculative_planning_ms`` is worker CPU time hidden under the
        caller's GPU work; ``selected_max_depth`` describes the most recently
        materialized plan; ``predicted_peak_bytes`` / ``usable_limit_bytes``
        are the admitted plan's memory check (for a split: every retained
        graph plus the largest subforward's ephemeral share). A call refused
        with ``TrainerRankMemoryError`` is still reflected, with the binding
        check that refused it.
        """

        if self._last_forward_telemetry_snapshot is None:
            raise RuntimeError("no forward has completed planning yet")
        return dict(self._last_forward_telemetry_snapshot)

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
        params: AdamParams | Mapping[str, AdamParams],
        scale_grads: float | Mapping[str, float] = 1.0,
        checkpoints: Sequence[str] | None = None,
        on_live_graphs: Literal["allow", "error"] = "allow",
    ) -> dict[str, float]:
        if on_live_graphs not in ("allow", "error"):
            raise ValueError(
                "on_live_graphs must be either 'allow' or 'error', got "
                f"{on_live_graphs!r}"
            )
        params_by_checkpoint = dict(params) if isinstance(params, Mapping) else None
        if params_by_checkpoint is not None:
            if not params_by_checkpoint:
                raise ValueError("params mapping must select at least one checkpoint")
            if any(not isinstance(name, str) for name in params_by_checkpoint):
                raise TypeError("params keys must be checkpoint names")
            if any(
                not isinstance(value, AdamParams)
                for value in params_by_checkpoint.values()
            ):
                raise TypeError("params values must be AdamParams")
        elif not isinstance(params, AdamParams):
            raise TypeError(
                "params must be AdamParams or a mapping of checkpoint names"
            )
        if isinstance(scale_grads, Mapping):
            raw_scales = cast(Mapping[object, object], scale_grads)
            if not raw_scales:
                raise ValueError(
                    "scale_grads mapping must select at least one checkpoint"
                )
            if any(not isinstance(name, str) for name in raw_scales):
                raise TypeError("scale_grads keys must be checkpoint names")
            try:
                scales_by_checkpoint = {
                    cast(str, name): float(cast(Any, value))
                    for name, value in raw_scales.items()
                }
            except (TypeError, ValueError) as error:
                raise TypeError("scale_grads values must be floats") from error
            scale_grads_value = None
        else:
            scales_by_checkpoint = None
            scale_grads_value = float(scale_grads)
        configured = [
            tuple(value)
            for value in (params_by_checkpoint, scales_by_checkpoint)
            if value is not None
        ]
        if checkpoints is not None:
            configured.append(tuple(dict.fromkeys(checkpoints)))
        if configured and any(set(names) != set(configured[0]) for names in configured):
            raise ValueError(
                "params, scale_grads, and checkpoints must select the same "
                "checkpoint names"
            )
        checkpoint_selection = (
            checkpoints
            if checkpoints is not None
            else sorted(configured[0])
            if configured
            else None
        )
        self._guard_optim_step_configuration(
            checkpoint_selection, params, on_live_graphs
        )
        selected_checkpoints = self._selected_dynamic_checkpoints(checkpoint_selection)
        params_by_checkpoint = (
            params_by_checkpoint
            if params_by_checkpoint is not None
            else dict.fromkeys(selected_checkpoints, cast(AdamParams, params))
        )
        scales_by_checkpoint = (
            scales_by_checkpoint
            if scales_by_checkpoint is not None
            else dict.fromkeys(selected_checkpoints, cast(float, scale_grads_value))
        )
        if on_live_graphs == "error":
            self._guard_checkpoints_can_step(selected_checkpoints)
        with _telemetry_phase(
            "optim",
            {"checkpoint_count": len(selected_checkpoints)},
        ):
            return self._dynamic_optim_step(
                selected_checkpoints,
                params=params_by_checkpoint,
                scale_grads=scales_by_checkpoint,
            )

    def _guard_optim_step_configuration(
        self,
        checkpoints: Sequence[str] | None,
        params: AdamParams | Mapping[str, AdamParams],
        on_live_graphs: Literal["allow", "error"],
    ) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return

        def adam_values(
            value: AdamParams,
        ) -> tuple[float, float, float, float, float]:
            return (
                value.learning_rate,
                value.beta1,
                value.beta2,
                value.weight_decay,
                value.grad_clip_norm,
            )

        config_values = (
            tuple(
                (name, *adam_values(value))
                for name, value in sorted(
                    cast(Mapping[str, AdamParams], params).items()
                )
            )
            if isinstance(params, Mapping)
            else adam_values(params)
        )
        digest = hashlib.sha256(
            repr(
                (
                    None if checkpoints is None else tuple(checkpoints),
                    config_values,
                    on_live_graphs,
                )
            ).encode()
        ).digest()
        local = torch.tensor(tuple(digest), device=self.device, dtype=torch.uint8)
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        if any(not torch.equal(value, local) for value in gathered):
            raise TrainerRankSlotStateError(
                "Optimizer checkpoint selection or AdamParams differ across ranks"
            )

    def _load_checkpoint_slot(
        self,
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
        *,
        alpha: float,
        _prepared: bool = False,
    ) -> int:
        if self._slot_stack:
            raise RuntimeError("Cannot load a checkpoint while one is pushed")
        adapter_model = self._prepare_adapter_model(
            name, adapter_model, canonicalized=_prepared
        )
        from art.megatron.lora import load_lora_slot_into_model

        ref = self._slot_ref(name)
        self._guard_slot_can_load(ref)
        self._compact_lora_slot_keys()
        return load_lora_slot_into_model(
            self.runtime.model,
            ref,
            adapter_model,
            alpha=alpha,
            requires_grad=True,
        )

    def _compact_lora_slot_keys(self) -> None:
        from art.megatron.lora import LoRA

        for chunk in self.runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA):
                    continue
                slots = [
                    (ref, module._slot_modules[key])
                    for ref, key in module._slot_keys.items()
                ]
                module._slot_keys = {
                    ref: f"slot_{index}" for index, (ref, _slot) in enumerate(slots)
                }
                module._slot_modules = torch.nn.ModuleDict(
                    {f"slot_{index}": slot for index, (_ref, slot) in enumerate(slots)}
                )

    def _prepare_adapter_model(
        self,
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
        *,
        canonicalized: bool = False,
    ) -> dict[str, torch.Tensor]:
        templates = self._local_lora_adapter_templates()
        keys = set(adapter_model)
        expected = set(templates)
        if dist.is_available() and dist.is_initialized():
            gathered: list[set[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, expected, group=self._checkpoint_group())
            expected = set().union(*(value for value in gathered if value is not None))
        if unknown := sorted(keys - expected):
            preview = ", ".join(repr(key) for key in unknown[:8])
            more = "" if len(unknown) <= 8 else f", ... +{len(unknown) - 8} more"
            raise ValueError(
                f"Checkpoint {name!r} contains keys that do not match installed "
                f"LoRA wrapper sites: {preview}{more}. Configure the Megatron "
                "runtime with matching LoRA target modules before loading."
            )
        local_state = {
            key: tensor for key, tensor in adapter_model.items() if key in templates
        }
        adapter_model = (
            local_state
            if canonicalized
            else self.runtime.model_support_handler.canonicalize_loaded_lora_state(
                local_state, self.runtime.model
            )
        )
        if set(adapter_model) != set(local_state):
            raise TrainerRankSlotStateError(
                "Model-specific LoRA canonicalization changed the adapter key set "
                f"for checkpoint {name!r}."
            )
        return {
            key: tensor.to(
                device=templates[key].device,
                dtype=templates[key].dtype,
                non_blocking=True,
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

    def _iter_slot_parameters(self, ref: "LoRASlotRef") -> Iterator[torch.nn.Parameter]:
        from art.megatron.lora import iter_lora_slot_parameters

        return iter_lora_slot_parameters(self.runtime.model, ref)

    def _local_parameter_key_groups(self, name: str) -> tuple[tuple[str, ...], ...]:
        ref = self._slot_ref(name)
        return tuple(
            tuple(str(key) for key in expected(str(suffix).removesuffix(".weight")))
            for chunk in self.runtime.model
            for module in chunk.modules()
            if (lora_params := getattr(module, "_lora_params", None)) is not None
            if (expected := getattr(module, "_expected_weight_keys", None)) is not None
            for suffix, _param in lora_params(ref)
        )

    def _validate_checkpoint_consistency(
        self, name: str, loaded_sites: int, expected_keys: set[str]
    ) -> tuple[torch.nn.Parameter, ...]:
        params = tuple(self._iter_slot_parameters(self._slot_ref(name)))
        local_keys = {
            key for group in self._local_parameter_key_groups(name) for key in group
        }
        gathered = (
            [local_keys]
            if not (dist.is_available() and dist.is_initialized())
            else [None] * dist.get_world_size()
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered, local_keys, group=self._checkpoint_group())
        covered = set().union(*(keys for keys in gathered if keys is not None))
        if loaded_sites < 1 or covered != expected_keys:
            raise TrainerRankSlotStateError(
                f"Checkpoint {name!r} has inconsistent distributed coverage"
            )
        return params

    def _set_default_slot(self, ref: "LoRASlotRef") -> None:
        if self._slot_stack:
            raise RuntimeError("Cannot select a checkpoint while one is pushed")
        self._default_slot_ref = ref

    @staticmethod
    def _slot_ref(name: str | None) -> "LoRASlotRef":
        try:
            from art.megatron.lora import LoRASlotRef
        except ModuleNotFoundError as exc:
            if exc.name is None or not exc.name.startswith("megatron"):
                raise

            return cast("LoRASlotRef", _LocalLoRASlotRef(name=name))

        return LoRASlotRef(kind="checkpoint", name=name)

    def _resolve_slot_ref(
        self,
        request: AnyForwardInput,
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> "LoRASlotRef | None":
        selection = (
            request.checkpoint if request.checkpoint is not Unset else checkpoint
        )
        if selection is not Unset:
            name = cast(str | None, selection)
            if name is not None and name not in self._checkpoint_slots:
                raise TrainerRankSlotStateError(
                    f"Forward selects unloaded checkpoint {name!r}"
                )
            return self._slot_ref(name)
        if self._slot_stack:
            return self._slot_stack[-1]
        if self._default_slot_ref is not None:
            return self._default_slot_ref
        return self._slot_ref(None)

    def _selected_dynamic_checkpoints(
        self,
        checkpoints: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if checkpoints is not None:
            self._ensure_checkpoint_slots(checkpoints)
        loaded = set(self._checkpoint_slots)
        if not loaded:
            raise TrainerRankSlotStateError(
                "TrainerRank.optim_step requires a loaded checkpoint slot. Call "
                "load_checkpoint(...) and run backward on outputs produced by "
                "that slot before stepping."
            )
        requested = (
            tuple(
                sorted(
                    name for name in loaded if not self._checkpoint_slots[name].snapshot
                )
            )
            if checkpoints is None
            else tuple(dict.fromkeys(checkpoints))
        )
        if not requested:
            if checkpoints is None:
                raise TrainerRankSlotStateError(
                    "TrainerRank.optim_step requires a loaded trainable checkpoint "
                    "slot. Call load_checkpoint(...) and run backward on outputs "
                    "produced by that slot before stepping."
                )
            raise TrainerRankSlotStateError(
                "TrainerRank.optim_step(checkpoints=...) received no checkpoint "
                "names. Pass at least one loaded checkpoint slot."
            )
        if unknown := set(requested) - loaded:
            raise ValueError(f"Unknown checkpoint slots: {sorted(unknown)}")
        if snapshots := [
            name for name in requested if self._checkpoint_slots[name].snapshot
        ]:
            raise TrainerRankSlotStateError(
                "Snapshot checkpoints are forward-only and cannot be stepped: "
                f"{snapshots}"
            )
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
                    for param in self._checkpoint_slots[name].params
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
        params: Mapping[str, AdamParams],
        scale_grads: Mapping[str, float],
    ) -> dict[str, float]:
        self.runtime.model_support_handler.zero_internal_padding_grads(
            self.runtime.model
        )
        selected = []
        for name in checkpoint_names:
            slot_params = self._checkpoint_slots[name].params
            step_flags = self._dynamic_param_step_flags(slot_params)
            slot_grads = self._reduce_dynamic_grads(
                slot_params, scale_grads=scale_grads[name]
            )
            selected.append((name, slot_params, slot_grads, step_flags))

        grad_norms = dict(
            zip(
                checkpoint_names,
                _distributed_grad_norms(
                    [(model_params, grads) for _, model_params, grads, _ in selected]
                ),
                strict=True,
            )
        )
        grad_norm = math.sqrt(sum(value**2 for value in grad_norms.values()))
        metrics = {
            "grad_norm": float(grad_norm),
            "update_successful": float(math.isfinite(grad_norm)),
            "num_zeros_in_grad": 0.0,
        }
        for name in checkpoint_names:
            metrics[f"learning_rate/{name}"] = float(params[name].learning_rate)
            metrics[f"grad_norm/{name}"] = float(grad_norms[name])
        learning_rates = {params[name].learning_rate for name in checkpoint_names}
        if len(learning_rates) == 1:
            metrics["learning_rate"] = float(params[checkpoint_names[0]].learning_rate)
        if not math.isfinite(grad_norm):
            for name in checkpoint_names:
                for param in self._checkpoint_slots[name].params:
                    param.grad = None
                self._prune_slot_graphs(self._slot_ref(name))
            return metrics
        previous = {
            name: (
                slot.optimizer,
                None
                if slot.optimizer is None
                else [
                    {key: group[key] for key in ("lr", "betas", "weight_decay")}
                    for group in slot.optimizer.optimizer.param_groups
                ],
            )
            for name in checkpoint_names
            for slot in (self._checkpoint_slots[name],)
        }
        try:
            dynamics = {
                name: self._dynamic_optimizer(name, params[name])
                for name in checkpoint_names
            }
        except BaseException:
            for name, (optimizer, groups) in previous.items():
                self._checkpoint_slots[name].optimizer = optimizer
                if optimizer is not None and groups is not None:
                    for group, values in zip(
                        optimizer.optimizer.param_groups, groups, strict=True
                    ):
                        group.update(values)
            raise
        for name, model_params, grads, step_flags in selected:
            checkpoint_params = params[name]
            checkpoint_grad_norm = grad_norms[name]
            clip = (
                min(
                    1.0,
                    checkpoint_params.grad_clip_norm / (checkpoint_grad_norm + 1.0e-6),
                )
                if checkpoint_params.grad_clip_norm > 0.0
                else 1.0
            )
            dynamic = dynamics[name]
            for master, grad, should_step in zip(
                dynamic.master_params, grads, step_flags, strict=True
            ):
                master.grad = grad.mul(clip) if should_step else None
            dynamic.optimizer.step()
            dynamic.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for model, master in zip(
                    model_params, dynamic.master_params, strict=True
                ):
                    model.copy_(master)
                    model.grad = None
            self._prune_slot_graphs(self._slot_ref(name))
            self._checkpoint_slots[name].revision += 1
        return metrics

    def _dynamic_param_step_flags(
        self, params: Sequence[torch.nn.Parameter]
    ) -> tuple[bool, ...]:
        custom = [
            (index, param)
            for index, param in enumerate(params)
            if bool(getattr(param, "_art_custom_checkpoint_param", False))
        ]
        if not custom:
            return (True,) * len(params)
        flags = torch.tensor(
            [param.grad is not None for _, param in custom],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        result = [True] * len(params)
        for (index, _param), flag in zip(custom, flags.tolist(), strict=True):
            result[index] = bool(flag)
        return tuple(result)

    def _dynamic_optimizer(
        self,
        name: str,
        params: AdamParams,
    ) -> _DynamicOptimizer:
        slot = self._checkpoint_slots[name]
        dynamic = slot.optimizer
        if dynamic is None:
            dynamic = self._new_dynamic_optimizer(name, params)
            slot.optimizer = dynamic
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
        model_params = self._checkpoint_slots[name].params
        sources = model_params if master_params is None else tuple(master_params)
        if len(sources) != len(model_params) or any(
            not isinstance(source, torch.Tensor) for source in sources
        ):
            raise TrainerRankSlotStateError(
                f"Optimizer state for checkpoint slot {name!r} has "
                f"{len(sources)} master parameters; expected {len(model_params)}."
            )
        if any(
            tuple(source.shape) != tuple(model.shape)
            for source, model in zip(sources, model_params, strict=True)
        ):
            raise TrainerRankSlotStateError(
                f"Optimizer master parameter shape does not match checkpoint {name!r}"
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
        dynamic = _DynamicOptimizer(optimizer, masters)
        slot = self._checkpoint_slots[name]
        if slot.custom:
            from ._checkpoint import load_custom_optimizer

            named = tuple(
                pair
                for custom_name, custom in slot.custom.items()
                for pair in _custom_named_parameters(custom_name, custom)
                if pair[1].requires_grad
            )
            lora_count = len(model_params) - len(named)
            restored = load_custom_optimizer(
                slot.custom_payload, tuple(key for key, _param in named)
            )
            for (key, _param), master in zip(named, masters[lora_count:], strict=True):
                state = restored.get(key)
                if state is not None:
                    _validate_custom_optimizer_state(name, key, _param, state)
                    with torch.no_grad():
                        master.copy_(state.master.to(master.device))
                    optimizer.state[master] = {
                        "step": torch.tensor(state.step, device=master.device),
                        "exp_avg": state.exp_avg.to(master.device).clone(),
                        "exp_avg_sq": state.exp_avg_sq.to(master.device).clone(),
                    }
        return dynamic

    def _restore_canonical_optimizer(
        self,
        name: str,
        state: "LocalOptimizerState",
    ) -> _DynamicOptimizer:
        dynamic = self._new_dynamic_optimizer(
            name,
            AdamParams(
                learning_rate=state.config["learning_rate"],
                beta1=state.config["beta1"],
                beta2=state.config["beta2"],
                weight_decay=state.config["weight_decay"],
            ),
            master_params=state.masters,
        )
        dynamic.optimizer.param_groups[0]["eps"] = state.config["eps"]
        for master, exp_avg, exp_avg_sq, step in zip(
            dynamic.master_params,
            state.exp_avgs,
            state.exp_avg_sqs,
            state.steps,
            strict=True,
        ):
            if tuple(exp_avg.shape) != tuple(master.shape) or tuple(
                exp_avg_sq.shape
            ) != tuple(master.shape):
                raise TrainerRankSlotStateError(
                    f"Canonical optimizer moment shape does not match {name!r}"
                )
            dynamic.optimizer.state[master] = {
                "step": torch.tensor(step, dtype=torch.float32),
                "exp_avg": exp_avg.to(master.device, torch.float32).clone(),
                "exp_avg_sq": exp_avg_sq.to(master.device, torch.float32).clone(),
            }
        self._zero_dynamic_optimizer_padding(name, dynamic)
        return dynamic

    def _zero_dynamic_optimizer_padding(
        self,
        name: str,
        dynamic: _DynamicOptimizer,
    ) -> None:
        masks = self._dynamic_optimizer_padding_masks(name)
        with torch.no_grad():
            for param, mask in zip(dynamic.master_params, masks, strict=True):
                param.masked_fill_(mask, 0)
                for value in dynamic.optimizer.state.get(param, {}).values():
                    if isinstance(value, torch.Tensor) and value.shape == param.shape:
                        value.masked_fill_(mask, 0)

    def _dynamic_optimizer_padding_masks(self, name: str) -> tuple[torch.Tensor, ...]:
        params = self._checkpoint_slots[name].params
        masks = tuple(torch.zeros_like(param, dtype=torch.bool) for param in params)
        param_indices = {id(param): index for index, param in enumerate(params)}
        exported: dict[str, torch.Tensor] = {}
        owners: dict[str, tuple[int, int | None]] = {}
        mapped_indices: set[int] = set()
        ref = self._slot_ref(name)

        for chunk in self.runtime.model:
            for module in chunk.modules():
                lora_params = getattr(module, "_lora_params", None)
                expected_keys = getattr(module, "_expected_weight_keys", None)
                if not callable(lora_params) or not callable(expected_keys):
                    continue
                for suffix, param in lora_params(ref):
                    index = param_indices.get(id(param))
                    if index is None:
                        continue
                    mapped_indices.add(index)
                    keys = expected_keys(str(suffix).removesuffix(".weight"))
                    if int(param.ndim) == 3:
                        if len(keys) != int(param.shape[0]):
                            raise TrainerRankSlotStateError(
                                f"Cannot map optimizer padding for checkpoint "
                                f"{name!r}: {len(keys)} adapter keys describe "
                                f"{int(param.shape[0])} local experts."
                            )
                        for expert, key in enumerate(keys):
                            exported[str(key)] = torch.ones_like(param[expert].T)
                            owners[str(key)] = (index, expert)
                    elif len(keys) == 1:
                        key = str(keys[0])
                        exported[key] = torch.ones_like(param.T)
                        owners[key] = (index, None)
                    else:
                        raise TrainerRankSlotStateError(
                            f"Cannot map optimizer padding for checkpoint {name!r}: "
                            f"expected one adapter key, got {len(keys)}."
                        )

        if mapped_indices and (
            missing := sorted(
                index
                for index, param in enumerate(params)
                if index not in mapped_indices
                and not bool(getattr(param, "_art_custom_checkpoint_param", False))
            )
        ):
            raise TrainerRankSlotStateError(
                f"Cannot map optimizer padding for checkpoint {name!r}: parameter "
                f"indices {missing} do not belong to installed LoRA sites."
            )

        canonical = self.runtime.model_support_handler.canonicalize_loaded_lora_state(
            exported, self.runtime.model
        )
        for key, value in canonical.items():
            owner = owners.get(key)
            if owner is None or not isinstance(value, torch.Tensor):
                continue
            index, expert = owner
            mask = value.T == 0
            if expert is None:
                masks[index].copy_(mask)
            else:
                masks[index][expert].copy_(mask)
        return masks

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
            tuple[dist.ProcessGroup, dist.ReduceOp.RedOpType, list[torch.Tensor]],
        ] = {}

        def add(
            group: dist.ProcessGroup,
            op: dist.ReduceOp.RedOpType,
            grad: torch.Tensor,
        ) -> None:
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

    def _select_next_micro_batch(
        self,
        items: Sequence[ForwardInputsT],
        start: int,
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> _CandidateMicroBatch[ForwardInputsT]:
        dp_rank, dp_size = self._dp_rank_and_size()
        remaining, min_width, granularity = _wave_geometry(len(items), start, dp_size)
        if min_width <= 0:
            raise RuntimeError("cannot select an empty microbatch window")

        def normalize(width: int) -> int:
            return _normalize_wave_width(width, min_width, remaining, granularity)

        def local_slice(width: int) -> tuple[tuple[int, ...], list[ForwardInputsT]]:
            indices = _local_wave_indices(start, width, dp_rank, dp_size)
            return indices, [items[index] for index in indices]

        estimates: dict[int, tuple[_MemoryCheck, bool, bool] | None] = {}
        plans: dict[int, _FlatForwardPlan] = {}
        # Per-width layout mode chosen by admission: False = cost-optimal,
        # True = memory-minimal (full sharing). Materialization must build the
        # same layouts the admitted estimate priced.
        layout_modes: dict[int, bool] = {}
        exact_failed_width: int | None = None

        def estimate(width: int) -> tuple[_MemoryCheck, bool, bool] | None:
            nonlocal exact_failed_width
            width = normalize(width)
            if width in estimates:
                return estimates[width]
            indices, local_inputs = local_slice(width)
            local_requests = list(_flatten(local_inputs))
            values = self._estimate_flat_forward(local_requests, checkpoint=checkpoint)
            if not self._all_ranks_true(values is not None):
                estimates[width] = None
                return None
            assert values is not None
            logical_tokens = sum(
                int(request.input_tokens.numel()) for request in local_requests
            )

            def priced(
                packed_tokens: int,
                output_bytes: int,
                signature: _MemorySignature,
            ) -> tuple[_MemoryCheck, int, int, _MemorySignature]:
                return (
                    self._memory_check_required(
                        self._estimate_required_memory_bytes_from_values(
                            packed_tokens=packed_tokens,
                            output_bytes=output_bytes,
                            signature=signature,
                            logical_tokens=logical_tokens,
                        ),
                        sync_across_dp=True,
                    ),
                    packed_tokens,
                    output_bytes,
                    signature,
                )

            def priced_estimate(
                *, exact: bool, memory_minimal: bool
            ) -> tuple[_MemoryCheck, int, int, _MemorySignature] | None:
                estimated = self._estimate_flat_forward(
                    local_requests,
                    checkpoint=checkpoint,
                    exact=exact,
                    memory_minimal=memory_minimal,
                )
                return None if estimated is None else priced(*estimated)

            def trusted(packed_tokens: int, signature: _MemorySignature) -> bool:
                return self._all_ranks_have_memory_profile(
                    packed_tokens=packed_tokens, signature=signature
                )

            # The cheap no-sharing count is an upper bound on any planner
            # layout: valid for accepting a width (memory and profile trust),
            # never for rejecting one. Exact pricing runs when the bound would
            # reject on memory, or when it would reject on profile trust while
            # a profile exists — the selected layout may be far smaller than
            # the bound and squarely inside the profiled regime.
            selected = priced(*values)
            profiled = self._all_ranks_true(selected[3] in self._memory_profiles)
            needs_exact = not selected[0].fits or (
                profiled and not trusted(selected[1], selected[3])
            )
            if needs_exact and (
                exact_failed_width is None or width < exact_failed_width
            ):
                # Feasibility must be monotone in width for the outer search:
                # the cost-optimal layout can decline sharing at one width and
                # accept it at a wider one, but the memory-minimal (full
                # sharing) layout's packed count is monotone by construction.
                # Its cheap bound decides feasibility; planner pricing then
                # picks cost-optimal when that fits and memory-minimal
                # otherwise, recording the mode so materialization executes
                # exactly the layouts that were priced.
                minimal_bound = priced_estimate(exact=False, memory_minimal=True)
                if minimal_bound is not None and minimal_bound[0].fits:
                    for memory_minimal in (False, True):
                        exact = priced_estimate(
                            exact=True, memory_minimal=memory_minimal
                        )
                        assert exact is not None
                        selected = exact
                        if selected[0].fits and (
                            not profiled or trusted(selected[1], selected[3])
                        ):
                            layout_modes[width] = memory_minimal
                            break
                    else:
                        # Nothing fit and trusted; keep the memory-minimal
                        # pricing so the recorded failure is the monotone one.
                        if not selected[0].fits:
                            layout_modes.pop(width, None)
                elif minimal_bound is not None:
                    selected = minimal_bound
                if not selected[0].fits:
                    exact_failed_width = (
                        width
                        if exact_failed_width is None
                        else min(exact_failed_width, width)
                    )
            check, packed_tokens, _output_bytes, signature = selected
            result = (
                check,
                trusted(packed_tokens, signature),
                self._all_ranks_true(signature in self._memory_profiles),
            )
            estimates[width] = result
            return result

        rejected_widths: set[int] = set()

        def fits(width: int) -> tuple[bool, bool]:
            width = normalize(width)
            result = estimate(width)
            if result is None:
                # Estimator unavailable (device inputs): admit on the
                # materialized plan, trying the cost-optimal layouts first and
                # the memory-minimal layouts if those do not fit.
                plan = materialize(width)
                check = self._memory_check(plan, sync_across_dp=True)
                if not check.fits and not layout_modes.get(width, False):
                    layout_modes[width] = True
                    plans.pop(width, None)
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
                plan = self._plan_flat_forward(
                    list(_flatten(local_inputs)),
                    checkpoint=checkpoint,
                    memory_minimal=layout_modes.get(width, False),
                )
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
                # The smallest wave cannot run unsplit: best effort is the
                # bounded split ladder. Each DP rank runs it on its own share,
                # then all ranks agree on the outcome (one collective, always)
                # so a refusal is raised everywhere or nowhere.
                indices, local_inputs = local_slice(min_width)
                refusal_prefix = (
                    "smallest DP microbatch is predicted to exceed available memory"
                )
                found = self._find_admissible_forward(
                    list(_flatten(local_inputs)),
                    checkpoint=checkpoint,
                    refusal_prefix=refusal_prefix,
                )
                agreed = self._all_ranks_true(not isinstance(found, _ForwardRefusal))
                if isinstance(found, _ForwardRefusal):
                    raise found.error("forward_micro_batches")
                if not agreed:
                    raise _memory_error(
                        context="forward_micro_batches",
                        message=(
                            f"{refusal_prefix} on another DP rank, which was "
                            "unable to find a feasible split for its share"
                        ),
                        packed_tokens=first.plan.packed_tokens,
                        logical_tokens=first.plan.logical_tokens,
                        check=first.check,
                    )
                split_plan, split_check = found
                return _CandidateMicroBatch(
                    inputs=local_inputs,
                    indices=indices,
                    plan=split_plan,
                    check=split_check,
                    stats_global_count=min_width,
                    rejected_candidates=len(rejected_widths),
                    cold_start=True,
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

    def _forced_test_anchor(self) -> str | None:
        if os.environ.get(_TEST_HOOKS_ENV) != "1":
            return None
        return os.environ.get(_TEST_ANCHOR_ENV) or None

    def _planner_topology_facts(self) -> "_PlannerFacts":
        _dp, tp_size, cp_size, _pp = self._topology_key()
        uses_gdn = bool(
            getattr(
                self.runtime.model_support_handler, "build_gdn_execution_spec", False
            )
        )
        return _PlannerFacts(
            cp_size=cp_size,
            tp_size=tp_size,
            layers=self._num_layers,
            gdn_layers=self._gdn_layers if uses_gdn else 0,
            uses_gdn=uses_gdn,
            coefficient_version=self._coefficient_version,
        )

    def _layout_anchor(self, *, memory_minimal: bool) -> str | None:
        """Resolve the layout anchor: test forcing wins, else memory policy."""

        forced = self._forced_test_anchor()
        if forced is not None:
            return forced
        return _MEMORY_MINIMAL_ANCHOR if memory_minimal else None

    def _layout_cache_key(
        self,
        input_ids: Sequence[torch.Tensor],
        *,
        memory_minimal: bool = False,
    ) -> "_LayoutKey":
        facts = self._planner_topology_facts()
        hasher = hashlib.sha256()
        for tensor in input_ids:
            row = tensor.detach().reshape(-1).cpu().contiguous()
            hasher.update(str(row.dtype).encode("ascii"))
            hasher.update(_U64_STRUCT.pack(int(row.numel())))
            hasher.update(row.numpy())
        return (
            hasher.hexdigest(),
            facts,
            self._layout_anchor(memory_minimal=memory_minimal),
        )

    def _cached_group_layout(
        self,
        key: "_LayoutKey",
    ) -> tuple[CanonicalPrefixTree, PrefixTreeLayout] | None:
        with self._layout_cache_lock:
            cached = self._layout_selection_cache.get(key)
            if cached is not None:
                self._layout_selection_cache.move_to_end(key)
            return cached

    def _compute_group_layout(
        self,
        input_ids: Sequence[torch.Tensor],
        key: "_LayoutKey",
    ) -> tuple[CanonicalPrefixTree, PrefixTreeLayout]:
        """Plan one group and memoize the result.

        Pure with respect to TrainerRank state apart from the caches, so it is
        safe to run on the speculative planning thread; a concurrent duplicate
        computation of the same key is deterministic and harmless. The
        canonical tree is cached by content alone so the cost-optimal and
        memory-minimal layouts of one group share a single construction.
        """

        content_key, facts, anchor = key
        with self._layout_cache_lock:
            tree = self._tree_cache.get(content_key)
            if tree is not None:
                self._tree_cache.move_to_end(content_key)
        if tree is None:
            tree = build_canonical_prefix_tree(input_ids)
            with self._layout_cache_lock:
                self._tree_cache[content_key] = tree
                while len(self._tree_cache) > _LAYOUT_SELECTION_CACHE_LIMIT:
                    self._tree_cache.popitem(last=False)
        if anchor is not None:
            candidates = prefix_tree_layout_candidates(tree)
            matching = [
                candidate for candidate in candidates if anchor in candidate.labels
            ]
            if len(matching) != 1:
                raise ValueError(f"unknown forced layout anchor {anchor!r}")
            layout = matching[0].layout
        else:
            layout = select_prefix_tree_layout(
                tree,
                cp_size=facts.cp_size,
                layers=facts.layers,
                uses_gdn=facts.uses_gdn,
                tp_size=facts.tp_size,
                gdn_layers=facts.gdn_layers,
                coefficient_version=facts.coefficient_version,
                refinement_work_budget=_PLANNER_REFINEMENT_BUDGET,
            ).layout
        cached = (tree, layout)
        with self._layout_cache_lock:
            self._layout_selection_cache[key] = cached
            self._layout_selection_cache.move_to_end(key)
            while len(self._layout_selection_cache) > _LAYOUT_SELECTION_CACHE_LIMIT:
                self._layout_selection_cache.popitem(last=False)
        return cached

    def _select_group_layout(
        self,
        input_ids: Sequence[torch.Tensor],
        *,
        memory_minimal: bool = False,
    ) -> tuple[CanonicalPrefixTree, PrefixTreeLayout]:
        """Select one group's prefix-sharing layout, cached by content identity.

        The cache key is a raw-bytes content hash plus the topology, cost
        coefficients, and layout anchor, so identical steady-state groups (or
        groups pre-planned speculatively during the caller's GPU work) skip
        canonicalization and search entirely. ``memory_minimal`` selects the
        full-sharing layout instead of the cost-optimal one; the width search
        uses it when the cost-optimal layout cannot be admitted.
        """

        started = time.perf_counter()
        key = self._layout_cache_key(input_ids, memory_minimal=memory_minimal)
        cached = self._cached_group_layout(key)
        if cached is None:
            cached = self._compute_group_layout(input_ids, key)
        self._planning_seconds_accum += time.perf_counter() - started
        return cached

    def _speculative_planning_executor(self) -> ThreadPoolExecutor:
        if self._speculative_planner is None:
            self._speculative_planner = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="trainer-rank-speculative-planner",
            )
        return self._speculative_planner

    def _submit_speculative_wave_planning(
        self,
        items: Sequence[ForwardInputs],
        start: int,
        *,
        checkpoint: AdapterSelection,
    ) -> None:
        """Pre-plan the predicted next wave while the caller uses the GPU.

        Runs while this generator is suspended at a yield (the caller's
        forward/backward time). The prediction mirrors the width search
        exactly: the next wave seeds from the largest width so far and plans
        this DP rank's strided local slice. Grouping, immutable CPU token
        snapshots, and cache keys are produced on the calling thread; the
        worker only runs the pure, memoized planner over those snapshots, so
        speculation can never change a selected plan and cannot be poisoned by
        the caller mutating its tensors afterwards. A wrong prediction merely
        leaves an unused LRU entry. Speculation is skipped for CUDA inputs so
        the worker never touches the device.

        The synchronous submission cost here is on the critical path (it
        delays the yield) and is charged to planning telemetry; the worker's
        hidden CPU time is reported separately as ``speculative_planning_ms``.
        """

        started = time.perf_counter()
        try:
            dp_rank, dp_size = self._dp_rank_and_size()
            remaining, min_width, granularity = _wave_geometry(
                len(items), start, dp_size
            )
            if min_width <= 0:
                return
            width = _normalize_wave_width(
                self._last_global_micro_batch_size or min_width,
                min_width,
                remaining,
                granularity,
            )
            indices = _local_wave_indices(start, width, dp_rank, dp_size)
            requests = list(_flatten([items[index] for index in indices]))
            if not requests:
                return
            if any(request.input_tokens.device.type != "cpu" for request in requests):
                return
            # Slots were ensured for every input when the call began; skip the
            # ensure-collective so speculation adds no communication.
            groups = self._group_active_request_indices(
                requests, checkpoint=checkpoint, ensure_slots=False
            )
            pending: list[
                tuple[
                    tuple[torch.Tensor, ...],
                    _LayoutKey,
                ]
            ] = []
            for _, group_indices in groups:
                snapshots = tuple(
                    requests[index]
                    .input_tokens.detach()
                    .reshape(-1)
                    .to(dtype=torch.long)
                    .clone()
                    for index in group_indices
                )
                key = self._layout_cache_key(snapshots)
                if self._cached_group_layout(key) is None:
                    pending.append((snapshots, key))
        except Exception:
            # Prediction is best-effort; the real wave surfaces any genuine
            # input problem on the main thread.
            return
        finally:
            self._planning_seconds_accum += time.perf_counter() - started
        if not pending:
            return

        def warm() -> None:
            worker_started = time.perf_counter()
            for snapshots, key in pending:
                if self._cached_group_layout(key) is None:
                    self._compute_group_layout(snapshots, key)
            with self._layout_cache_lock:
                self._speculative_planning_seconds += (
                    time.perf_counter() - worker_started
                )

        self._speculative_planning_future = (
            self._speculative_planning_executor().submit(warm)
        )

    def _plan_flat_forward(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection = Unset,
        memory_minimal: bool = False,
        ensure_slots: bool = True,
    ) -> _FlatForwardPlan:
        plans: list[_ForwardGroupPlan] = []
        output_bytes = self._estimate_group_request_output_bytes(requests)
        logical_tokens = sum(int(request.input_tokens.numel()) for request in requests)
        groups = self._group_active_request_indices(
            requests, checkpoint=checkpoint, ensure_slots=ensure_slots
        )
        selected_max_depth = 0
        for (slot_ref, grad_enabled), group_indices in groups:
            items = tuple(
                self._forward_item(requests[index]) for index in group_indices
            )
            group_input_ids = tuple(item.input_ids for item in items)
            tree, layout = self._select_group_layout(
                group_input_ids, memory_minimal=memory_minimal
            )
            selected_max_depth = max(selected_max_depth, layout.maximum_depth)
            started = time.perf_counter()
            packed = materialize_prefix_tree_layout(
                group_input_ids, tree, layout, verify_shared_tokens=False
            )
            self._planning_seconds_accum += time.perf_counter() - started
            plans.append(
                _ForwardGroupPlan(
                    slot_ref=slot_ref,
                    grad_enabled=grad_enabled,
                    request_indices=tuple(group_indices),
                    items=items,
                    packed=packed,
                )
            )

        return _FlatForwardPlan(
            request_count=len(requests),
            output_metadata=tuple(
                self._forward_output_metadata(request, checkpoint=checkpoint)
                for request in requests
            ),
            groups=tuple(plans),
            packed_tokens=sum(
                self._physical_tokens(int(plan.packed.tokens.numel())) for plan in plans
            ),
            logical_tokens=logical_tokens,
            output_bytes=output_bytes,
            signature=self._memory_signature_from_requests(
                requests,
                slot_group_count=len(plans),
                grad_modes=tuple(mode for (_, mode), _ in groups),
            ),
            selected_max_depth=selected_max_depth,
        )

    def _estimate_flat_forward(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection = Unset,
        exact: bool = False,
        memory_minimal: bool = False,
    ) -> tuple[int, int, _MemorySignature] | None:
        """Estimate packed tokens for width probing.

        Cheap mode (``exact=False``) is one O(tokens) CPU walk of the packing
        primitive and preserves its CUDA None-contract: with
        ``memory_minimal=False`` it is the no-sharing count, an upper bound on
        any planner-selected layout (safe for accepting a width); with
        ``memory_minimal=True`` it is the full-sharing count, the lower bound
        whose feasibility is monotone in width (valid for rejecting one).
        ``exact=True`` prices the planner's actual layouts (memoized by
        content) and is used only inside the band where those bounds disagree.
        """

        groups = self._group_active_request_indices(requests, checkpoint=checkpoint)
        packed_tokens = 0
        for _, group_indices in groups:
            if exact:
                _, layout = self._select_group_layout(
                    tuple(
                        requests[index].input_tokens.reshape(-1).to(dtype=torch.long)
                        for index in group_indices
                    ),
                    memory_minimal=memory_minimal,
                )
                packed_tokens += self._physical_tokens(layout.packed_tokens)
                continue
            # Radix depth is bounded by the number of rows, so ``len(group)``
            # is an unlimited-sharing depth for this group; it is a bound for
            # estimation, not a sharing policy.
            group_packed_tokens = estimate_prefix_tree_packed_tokens(
                (requests[index].input_tokens.reshape(-1) for index in group_indices),
                max_depth=len(group_indices) if memory_minimal else 0,
            )
            if group_packed_tokens is None:
                return None
            packed_tokens += self._physical_tokens(group_packed_tokens)

        return (
            packed_tokens,
            self._estimate_group_request_output_bytes(requests),
            self._memory_signature_from_requests(
                requests,
                slot_group_count=len(groups),
                grad_modes=tuple(mode for (_, mode), _ in groups),
            ),
        )

    def _ensure_checkpoint_slots_for(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection,
    ) -> None:
        self._ensure_checkpoint_slots(
            cast(str, selection)
            for request in requests
            if (
                request.target_tokens is not None
                or request.logits
                or request.top_k is not None
                or request.hidden_states
            )
            if (
                selection := (
                    request.checkpoint
                    if request.checkpoint is not Unset
                    else checkpoint
                )
            )
            is not Unset
            and selection is not None
        )

    def _group_active_request_indices(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection = Unset,
        ensure_slots: bool = True,
    ) -> tuple[tuple[tuple["LoRASlotRef | None", bool], tuple[int, ...]], ...]:
        if ensure_slots:
            self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
        groups: dict[tuple[LoRASlotRef | None, bool], list[int]] = {}
        for index, request in enumerate(requests):
            if (
                request.target_tokens is not None
                or request.logits
                or request.top_k is not None
                or request.hidden_states
            ):
                groups.setdefault(
                    (
                        self._resolve_slot_ref(request, checkpoint=checkpoint),
                        (
                            torch.is_grad_enabled()
                            if request.no_grad is None
                            else not request.no_grad
                        ),
                    ),
                    [],
                ).append(index)
        return tuple((slot_ref, tuple(indices)) for slot_ref, indices in groups.items())

    def _run_flat_plan_with_memory_tracking(
        self,
        plan: _FlatForwardPlan,
        *,
        context: str,
    ) -> tuple[list[AnyForwardOutput], int | None]:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            baseline = int(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            baseline = None
        try:
            with _telemetry_phase(
                "forward",
                self._telemetry_signature(plan),
                dedup_signature=self._telemetry_plan_signature(plan),
                synchronized=torch.cuda.is_available() and self.device.type == "cuda",
            ):
                outputs = self._execute_flat_plan(plan)
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError as exc:
            raise _memory_error(
                context=context,
                message="CUDA OOM occurred despite the planner estimate",
                packed_tokens=plan.packed_tokens,
                logical_tokens=plan.logical_tokens,
                check=self._memory_check(plan),
            ) from exc
        if baseline is not None:
            self._update_peak_memory_profile(
                plan, baseline, int(torch.cuda.memory_allocated(self.device))
            )
        return outputs, baseline

    def _update_peak_memory_profile(
        self,
        plan: _FlatForwardPlan,
        baseline: int | None,
        retained_after: int | None = None,
    ) -> None:
        if baseline is None:
            return
        peak = int(torch.cuda.max_memory_allocated(self.device))
        self._update_memory_profile(
            plan,
            max(0, peak - baseline),
            retained_bytes=(
                None if retained_after is None else max(0, retained_after - baseline)
            ),
        )

    @staticmethod
    def _telemetry_plan_signature(plan: _AnyForwardPlan) -> dict[str, object]:
        return {
            "topology": plan.signature.topology,
            "planner_coefficients": plan.signature.planner_coefficients,
            "slot_group_count": plan.signature.slot_group_count,
            "request_mix": plan.signature.request_mix,
            "grad_enabled": plan.signature.grad_enabled,
            "grad_modes": plan.signature.grad_modes,
        }

    @classmethod
    def _telemetry_signature(cls, plan: _AnyForwardPlan) -> dict[str, object]:
        return {
            **cls._telemetry_plan_signature(plan),
            "request_count": plan.request_count,
            "packed_tokens": plan.packed_tokens,
            "logical_tokens": plan.logical_tokens,
            "group_packed_tokens": tuple(
                int(group.packed.tokens.numel()) for group in plan.groups
            ),
            "group_segment_counts": tuple(
                len(group.packed.segments) for group in plan.groups
            ),
        }

    def _execute_flat_plan(self, plan: _FlatForwardPlan) -> list[AnyForwardOutput]:
        outputs = [
            ForwardOutput(None, None, None, None, checkpoint, no_grad)
            for checkpoint, no_grad in plan.output_metadata
        ]
        self._validate_hybridep_topology()
        hybridep = (
            self._configure_hybridep(
                tuple(group.packed for group in plan.groups), topology=self._topology()
            )
            if plan.groups
            else None
        )
        try:
            for group_index, group in enumerate(plan.groups):
                from art.megatron.lora import use_lora_slot

                if hybridep is not None:
                    self._set_hybridep_rows(hybridep[0][group_index])
                with torch.set_grad_enabled(group.grad_enabled):
                    with use_lora_slot(group.slot_ref):
                        prepared = self._prepare_packed_forward(group.packed)
                        item_outputs = self._forward_packed(group.items, prepared)
                    item_outputs = [
                        replace(
                            output,
                            checkpoint=(
                                None if group.slot_ref is None else group.slot_ref.name
                            ),
                            no_grad=not group.grad_enabled,
                        )
                        for output in item_outputs
                    ]
                    item_outputs = self._track_slot_graph_outputs(
                        group.slot_ref, item_outputs
                    )
                for index, output in zip(
                    group.request_indices, item_outputs, strict=True
                ):
                    outputs[index] = output
        finally:
            if hybridep is not None:
                self._set_hybridep_rows(hybridep[1])
        return outputs

    def _track_slot_graph_outputs(
        self,
        ref: "LoRASlotRef | None",
        outputs: Sequence[AnyForwardOutput],
    ) -> list[AnyForwardOutput]:
        track_slot = ref is not None and ref.name is not None
        track_hybridep = bool(getattr(self, "_hybridep_graph_tracking", False))
        if not track_slot and not track_hybridep:
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
                checkpoint=output.checkpoint,
                no_grad=output.no_grad,
            )
            for output in outputs
        ]
        if marker is not None:
            marker_ref = weakref.ref(marker)
            if track_slot:
                self._slot_graphs().setdefault(ref, []).append(marker_ref)
            if track_hybridep:
                self._hybridep_graphs().append(marker_ref)
        return tracked_outputs

    def _forward_output_metadata(
        self,
        request: AnyForwardInput,
        *,
        checkpoint: AdapterSelection,
    ) -> tuple[str | None, bool]:
        selection = (
            request.checkpoint if request.checkpoint is not Unset else checkpoint
        )
        if selection is Unset:
            ref = self._slot_stack[-1] if self._slot_stack else self._default_slot_ref
            name = None if ref is None else ref.name
        else:
            name = cast(str | None, selection)
        enabled = (
            torch.is_grad_enabled() if request.no_grad is None else not request.no_grad
        )
        return name, not enabled

    def _hybridep_graphs(self) -> list[weakref.ReferenceType[torch.Tensor]]:
        graphs = getattr(self, "_pending_hybridep_graphs", None)
        if graphs is None:
            graphs = []
            self._pending_hybridep_graphs = graphs
        return graphs

    def _has_live_hybridep_graphs(self) -> bool:
        graphs = self._hybridep_graphs()
        graphs[:] = [marker for marker in graphs if marker() is not None]
        return bool(graphs)

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
                marker
                for marker in graphs.get(current, ())
                if _graph_marker_is_live(marker)
            ]
            if live:
                graphs[current] = live
            else:
                graphs.pop(current, None)

    def _has_live_slot_graph(self, ref: "LoRASlotRef") -> bool:
        self._prune_slot_graphs(ref)
        return bool(self._slot_graphs().get(ref))

    def _guard_slot_can_load(self, ref: "LoRASlotRef") -> None:
        slot = None if ref.name is None else self._checkpoint_slots.get(ref.name)
        if slot is not None and slot.snapshot:
            raise TrainerRankSlotStateError(
                f"Cannot load over forward-only snapshot checkpoint {ref.name!r}"
            )
        if slot is not None and any(param.grad is not None for param in slot.params):
            raise TrainerRankSlotStateError(
                f"Cannot load checkpoint {ref.name!r} while it has accumulated "
                "gradients. Call optim_step() or zero_grad() before replacing it."
            )
        if not self._has_live_slot_graph(ref):
            return
        raise TrainerRankSlotStateError(
            f"Cannot load checkpoint {ref.name!r} while outputs from an "
            "earlier forward using that slot still have a live backward graph. "
            "Activation checkpoint recompute resolves slots by name, so replacing "
            "the slot before backward can compute gradients with different LoRA "
            "weights than the original forward. Finish backward first; if the "
            "forward was abandoned, release all references to its outputs; or load "
            "the new weights under a different slot name."
        )

    def _guard_checkpoint_can_step(self, name: str) -> None:
        if not self._has_live_slot_graph(self._slot_ref(name)):
            return
        raise TrainerRankSlotStateError(
            f"Cannot optim_step checkpoint slot {name!r} while outputs from an "
            "earlier forward using that slot have not been backpropagated. Call "
            "loss.backward() without retaining the graph before optim_step(); if "
            "the forward was abandoned, release all references to its outputs."
        )

    def _guard_checkpoints_can_step(self, names: Sequence[str]) -> None:
        local_live = [self._has_live_slot_graph(self._slot_ref(name)) for name in names]
        if dist.is_available() and dist.is_initialized():
            live = torch.tensor(
                local_live,
                device=self.device,
                dtype=torch.int32,
            )
            dist.all_reduce(live, op=dist.ReduceOp.MAX)
            live_flags = live.tolist()
        else:
            live_flags = local_live
        blocked = [
            name for name, is_live in zip(names, live_flags, strict=True) if is_live
        ]
        if not blocked:
            return
        raise TrainerRankSlotStateError(
            f"Cannot optim_step checkpoint slots {blocked!r} while outputs from an "
            "earlier forward using those slots have a live backward graph on at "
            "least one rank. Call loss.backward() without retaining the graph "
            "before optim_step(); if the forward was abandoned, release all "
            "references to its outputs; or pass on_live_graphs='allow' to accept "
            "responsibility for any retained graphs."
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
        grad_modes: Iterable[bool],
    ) -> _MemorySignature:
        modes = tuple(sorted(grad_modes))
        return _MemorySignature(
            topology=self._topology_key(),
            planner_coefficients=self._coefficient_version,
            slot_group_count=slot_group_count,
            request_mix=tuple(
                sorted({_request_mix_key(request) for request in requests})
            ),
            grad_enabled=any(modes),
            grad_modes=modes,
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

    def _physical_tokens(self, packed_tokens: int) -> int:
        """Physical length of one packed group: padded to a multiple of TP.

        Execution pads every group independently (``_pad_packed_batch``), so
        admission, the cheap bounds and the memory profile all count tokens
        the same way; the omission would otherwise grow with the number of
        groups, not stay below TP.
        """

        multiple = max(1, self._topology_key()[1])
        return packed_tokens + (-packed_tokens % multiple)

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
                logical_tokens=forward.logical_tokens,
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
    def _forward_memory_group() -> dist.ProcessGroup | None:
        try:
            from megatron.core import parallel_state as ps

            return ps.get_tensor_and_context_parallel_group(check_initialized=False)
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return None

    def _estimate_required_memory_bytes_from_values(
        self,
        *,
        packed_tokens: int,
        output_bytes: int,
        signature: _MemorySignature,
        logical_tokens: int | None = None,
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
        # A profile learned under lighter sharing (lower logical/packed ratio)
        # underestimates the per-packed-token footprint of a deeper-shared
        # plan; scale the trusted estimate up by the ratio gap.
        ratio_scale = 1.0
        if profiled is not None and logical_tokens is not None:
            current_ratio = logical_tokens / max(1, packed_tokens)
            ratio_scale = max(1.0, current_ratio / profiled.logical_per_packed)
        if (
            profiled is None
            or profiled.packed_tokens * _MEMORY_PROFILE_TRUST_GROWTH < packed_tokens
        ):
            compute = static_compute
        else:
            compute = max(
                static_compute,
                int(profiled.bytes_per_token * packed_tokens * ratio_scale),
            )
        return int((output_bytes + compute) * _MEMORY_SAFETY_FACTOR)

    def _available_memory_bytes(self) -> int:
        if not (torch.cuda.is_available() and self.device.type == "cuda"):
            return 1 << 60
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        reusable_reserved = max(0, reserved - allocated)
        reserve = int(total * _MEMORY_RESERVE_FRACTION)
        available = max(0, int(free) + reusable_reserved - reserve)
        if os.environ.get(_TEST_HOOKS_ENV) == "1":
            limit = os.environ.get(_TEST_MEMORY_LIMIT_ENV)
            if limit:
                # Test-only: cap the usable budget relative to the current
                # allocation so acceptance cells can induce split/decline
                # behavior deterministically without ballast tensors.
                available = min(available, max(0, int(limit) - allocated))
        return available

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
        self,
        plan: _FlatForwardPlan,
        peak_delta_bytes: int,
        *,
        retained_bytes: int | None,
    ) -> None:
        if plan.packed_tokens <= 0:
            return
        compute_delta = max(0, peak_delta_bytes - plan.output_bytes)
        bytes_per_token = compute_delta / max(1, plan.packed_tokens)
        previous = self._memory_profiles.get(plan.signature)
        retained_fraction = None if previous is None else previous.retained_fraction
        if retained_bytes is not None:
            observed = min(1.0, retained_bytes / max(1, peak_delta_bytes))
            # Max-merge once observed. ``None`` (never observed) is distinct
            # from an observed 1.0, so a later, lower observation cannot
            # replace it.
            retained_fraction = (
                observed
                if retained_fraction is None
                else max(retained_fraction, observed)
            )
        self._memory_profiles[plan.signature] = _MemoryProfile(
            bytes_per_token=max(
                bytes_per_token,
                0.0 if previous is None else previous.bytes_per_token,
            ),
            packed_tokens=max(
                plan.packed_tokens,
                0 if previous is None else previous.packed_tokens,
            ),
            logical_per_packed=max(
                plan.logical_tokens / max(1, plan.packed_tokens),
                1.0 if previous is None else previous.logical_per_packed,
            ),
            retained_fraction=retained_fraction,
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
                    chunk_tokens=_HEAD_CHUNK_TOKENS,
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
                    chunk_tokens=_HEAD_CHUNK_TOKENS,
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
            range(0, int(rows.numel()), _HEAD_CHUNK_TOKENS)
        ):
            chunk_rows = rows[start : start + _HEAD_CHUNK_TOKENS]
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
        from art.megatron.training.microbatches import (
            _art_flex_sliding_windows,
            _gdn_planner_config_for_provider,
        )

        handler = self.runtime.model_support_handler
        provider = self.runtime.provider
        return _PreparedPackedForward(
            tokens=batch.tokens.to(self.device),
            position_ids=batch.position_ids.to(self.device),
            attention_state=create_prefix_tree_state(
                group_ids=batch.group_ids,
                parent_ids=batch.parent_ids,
                target_device=self.device,
                input_pos=batch.position_ids,
                sliding_windows=_art_flex_sliding_windows(provider),
                build_gdn_execution_spec=handler.build_gdn_execution_spec,
                model_support_handler=handler,
                attention_head_dim=provider.kv_channels,
                attention_value_head_dim=provider.kv_channels,
                gdn_planner_config=_gdn_planner_config_for_provider(provider, handler),
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

    def _configure_hybridep(
        self,
        batches: Sequence[PrefixTreePack],
        *,
        topology: "ParallelTopology",
    ) -> tuple[tuple[int, ...], int] | None:
        from megatron.core import parallel_state as ps

        expert_parallel_size = int(ps.get_expert_model_parallel_world_size())
        if expert_parallel_size <= 1:
            self._hybridep_graph_tracking = False
            return None
        self._validate_hybridep_topology(topology)
        if not batches:
            return None
        from megatron.core.transformer.moe import fused_a2a

        from art.megatron.train import (
            _ensure_hybridep_capacity,
            _hybridep_token_capacity,
        )

        padded = tuple(
            _pad_packed_batch(batch, multiple=int(topology.tp)) for batch in batches
        )
        sequence_length = max(int(batch.tokens.shape[1]) for batch in padded)
        rows = tuple(self._hybridep_rows(batch, topology=topology) for batch in padded)
        current = fused_a2a._hybrid_ep_buffer
        live = self._has_live_hybridep_graphs()
        required_capacity = _hybridep_token_capacity(sequence_length, int(topology.cp))
        if live and (
            current is None
            or id(current) != getattr(self, "_hybridep_buffer_id", None)
            or int(current.configurer.buffer_config.max_num_of_tokens_per_rank)
            < required_capacity
        ):
            raise TrainerRankSlotStateError(
                "Cannot grow or replace the HybridEP buffer while an earlier "
                "TrainerRank forward still has a live backward graph. Finish "
                "backward or release those outputs before forwarding a larger batch."
            )
        _ensure_hybridep_capacity(
            self.runtime,
            packed_sequence_length=sequence_length,
            context_parallel_size=int(topology.cp),
        )
        current = fused_a2a._hybrid_ep_buffer
        if current is None:
            raise RuntimeError("HybridEP buffer was not initialized")
        if live:
            high_water = max(*rows, int(getattr(self, "_hybridep_rows_high_water", 0)))
        else:
            high_water = max(rows)
        self._hybridep_buffer_id = id(current)
        self._hybridep_rows_high_water = high_water
        self._hybridep_graph_tracking = True
        return rows, high_water

    def _validate_hybridep_topology(
        self,
        topology: "ParallelTopology | None" = None,
    ) -> None:
        if topology is None:
            configured_ep = int(
                getattr(self.runtime.provider, "expert_model_parallel_size", 1) or 1
            )
            if configured_ep <= 1:
                return
            topology = self._topology()
        if int(topology.dp) > 1:
            raise NotImplementedError(
                "TrainerRank does not support combining data parallelism with "
                "expert parallelism because uneven DP inputs can desynchronize "
                "HybridEP collectives. For MoE models, use DP=1 with CP and EP "
                "set to the world size."
            )

    @staticmethod
    def _set_hybridep_rows(rows: int) -> None:
        from art.megatron.train import _set_hybridep_token_count

        _set_hybridep_token_count(rows)

    def _hybridep_rows(
        self,
        batch: PrefixTreePack,
        *,
        topology: "ParallelTopology",
    ) -> int:
        sequence_length = int(batch.tokens.shape[1])
        if int(topology.cp) <= 1:
            return sequence_length
        if int(topology.cp) > 1:
            from art.megatron.context_parallel.runtime import (
                context_parallel_rank_model_token_counts,
            )
            from art.megatron.training.microbatches import (
                _context_parallel_config_for_provider,
                _gdn_planner_config_for_provider,
            )

            handler = self.runtime.model_support_handler
            return max(
                context_parallel_rank_model_token_counts(
                    group_ids=batch.group_ids,
                    parent_ids=batch.parent_ids,
                    topology=topology,
                    config=_context_parallel_config_for_provider(
                        self.runtime.provider,
                        self.device,
                        handler,
                    ),
                    original_seq_len=sequence_length,
                    build_gdn_execution_spec=handler.build_gdn_execution_spec,
                    gdn_planner_config=_gdn_planner_config_for_provider(
                        self.runtime.provider, handler
                    ),
                )
            )
        raise AssertionError("unreachable")

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
            config=_context_parallel_config_for_provider(
                provider,
                self.device,
                handler,
            ),
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


def _validate_top_k(top_k: int, model: object) -> None:
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
        segments=batch.segments,
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


def _padded_vocab_size(model: object) -> int:
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
    return _distributed_grad_norms([(params, grads)])[0]


def _distributed_grad_norms(
    groups: Sequence[tuple[Sequence[torch.nn.Parameter], Sequence[torch.Tensor]]],
) -> tuple[float, ...]:
    if any(len(params) != len(grads) for params, grads in groups):
        raise ValueError("params and grads must have matching lengths")
    device = next(
        (grad.device for _params, grads in groups for grad in grads),
        torch.device("cpu"),
    )
    squared = torch.zeros(len(groups), device=device, dtype=torch.float32)
    for index, (params, grads) in enumerate(groups):
        for param, grad in zip(params, grads, strict=True):
            if _include_in_distributed_grad_norm(param):
                squared[index].add_(grad.float().square().sum())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return tuple(torch.sqrt(squared).tolist())


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


def _custom_parameters(custom: _CustomObject) -> Iterator[torch.nn.Parameter]:
    if custom.kind == "module":
        yield from cast(torch.nn.Module, custom.value).parameters()
    elif custom.kind == "parameter":
        yield cast(torch.nn.Parameter, custom.value)


def _walk_objects(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_objects(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_objects(item)
    else:
        yield value


def _tracked_tensor_function(
    func: Callable[..., object],
    types: tuple[type, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    del types
    tracked = tuple(
        value
        for value in _walk_objects((args, kwargs))
        if isinstance(value, _TrackedParameter | _TrackedTensor)
    )
    trackers = {id(value._art_tracker): value._art_tracker for value in tracked}
    for tracker in trackers.values():
        tracker.validate()
    if getattr(func, "__name__", "") in {
        "__format__",
        "__get__",
        "__hash__",
        "__len__",
        "__repr__",
        "__str__",
    }:
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    markers: dict[int, torch.Tensor] = {}
    replacements: dict[int, torch.Tensor] = {}

    def replace(value: object) -> object:
        if isinstance(value, _TrackedParameter | _TrackedTensor):
            cached = replacements.get(id(value))
            if cached is not None:
                return cached
            tracker = value._art_tracker
            with torch._C.DisableTorchFunctionSubclass():
                if (
                    tracker.active
                    and isinstance(value, _TrackedParameter)
                    and value.requires_grad
                    and torch.is_grad_enabled()
                ):
                    marker = markers.get(id(tracker))
                    if marker is None:
                        marker = torch.zeros((), dtype=torch.bool)
                        markers[id(tracker)] = marker
                        tracker.record(marker)
                    result = _CustomSlotGraphSentinel.apply(
                        value.as_subclass(torch.Tensor), marker
                    )
                else:
                    result = value.as_subclass(torch.Tensor)
            replacements[id(value)] = result
            return result
        if isinstance(value, tuple):
            values = tuple(replace(item) for item in value)
            return type(value)(*values) if hasattr(value, "_fields") else values
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    result = func(
        *cast(tuple[object, ...], replace(args)),
        **cast(dict[str, object], replace(kwargs)),
    )
    if markers and not any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in _walk_objects(result)
    ):
        for marker in markers.values():
            marker.fill_(True)
    return result


def _graph_marker_is_live(
    marker_ref: weakref.ReferenceType[torch.Tensor],
) -> bool:
    marker = marker_ref()
    return marker is not None and (marker.numel() == 0 or not bool(marker.item()))


def _track_custom_object(
    custom: _CustomObject,
    tracker: _CustomTensorTracker,
) -> _CustomObject:
    if custom.kind == "parameter":
        source = cast(torch.nn.Parameter, custom.value)
        with torch.no_grad():
            value = _TrackedParameter(
                source.detach().clone(), tracker, source.requires_grad
            )
        value.__dict__.update(
            (key, item)
            for key, item in source.__dict__.items()
            if key != "_art_tracker"
        )
        value._art_tracker = tracker
        return _CustomObject(custom.kind, value, custom.generation)
    if custom.kind == "buffer":
        source = cast(torch.Tensor, custom.value)
        with torch.no_grad():
            value = _TrackedTensor(source.detach().clone(), tracker)
        return _CustomObject(custom.kind, value, custom.generation)

    module = cast(torch.nn.Module, custom.value)
    parameters: dict[int, torch.nn.Parameter] = {}
    buffers: dict[int, torch.Tensor] = {}
    for child in module.modules():
        for key, source in child._parameters.items():
            if source is None:
                continue
            value = parameters.get(id(source))
            if value is None:
                with torch.no_grad():
                    value = _TrackedParameter(
                        source.detach().clone(), tracker, source.requires_grad
                    )
                value.__dict__.update(
                    (attribute, item)
                    for attribute, item in source.__dict__.items()
                    if attribute != "_art_tracker"
                )
                value._art_tracker = tracker
                parameters[id(source)] = value
            child._parameters[key] = value
        for key, source in child._buffers.items():
            if source is None:
                continue
            value = buffers.get(id(source))
            if value is None:
                with torch.no_grad():
                    value = _TrackedTensor(source.detach().clone(), tracker)
                buffers[id(source)] = value
            child._buffers[key] = value
    return custom


def _custom_layout(
    name: str,
    custom: _CustomObject,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if custom.kind == "parameter":
        parameter = cast(torch.nn.Parameter, custom.value)
        return ((name,),), (), (), (name,) if parameter.requires_grad else ()
    if custom.kind == "buffer":
        return (), ((name,),), (name,), ()

    module = cast(torch.nn.Module, custom.value)
    parameter_groups: dict[int, list[str]] = {}
    trainable: list[str] = []
    for key, parameter in module.named_parameters(remove_duplicate=False):
        full_key = f"{name}.{key}"
        aliases = parameter_groups.setdefault(id(parameter), [])
        if not aliases and parameter.requires_grad:
            trainable.append(full_key)
        aliases.append(full_key)
    buffer_groups: dict[int, list[str]] = {}
    persistent: list[str] = []
    for prefix, child in module.named_modules():
        for key, buffer in child._buffers.items():
            if buffer is None:
                continue
            local_key = f"{prefix}.{key}" if prefix else key
            full_key = f"{name}.{local_key}"
            buffer_groups.setdefault(id(buffer), []).append(full_key)
            if key not in child._non_persistent_buffers_set:
                persistent.append(full_key)

    def normalize(groups: Mapping[int, Sequence[str]]) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(tuple(sorted(group)) for group in groups.values()))

    return (
        normalize(parameter_groups),
        normalize(buffer_groups),
        tuple(sorted(persistent)),
        tuple(sorted(trainable)),
    )


def _validate_custom_schema(
    name: str,
    custom: _CustomObject,
    record: Mapping[str, object],
) -> None:
    actual = (
        tuple(
            tuple(group) for group in cast(list[list[str]], record["parameter_aliases"])
        ),
        tuple(
            tuple(group) for group in cast(list[list[str]], record["buffer_aliases"])
        ),
        tuple(cast(list[str], record["persistent_buffer_keys"])),
        tuple(cast(list[str], record["trainable_keys"])),
    )
    if actual != _custom_layout(name, custom):
        raise TrainerRankSlotStateError(
            f"Custom checkpoint object {name!r} schema differs from its factory"
        )


def _custom_signature(
    name: str,
    custom: _CustomObject,
    values: Mapping[str, torch.Tensor],
) -> tuple[object, ...]:
    state = tuple(
        (key, tuple(value.shape), str(value.dtype)) for key, value in values.items()
    )
    return name, custom.kind, state, *_custom_layout(name, custom)


def _custom_named_parameters(
    name: str, custom: _CustomObject
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    if custom.kind == "module":
        for key, parameter in cast(torch.nn.Module, custom.value).named_parameters():
            yield f"{name}.{key}", parameter
    elif custom.kind == "parameter":
        yield name, cast(torch.nn.Parameter, custom.value)


def _custom_state(custom: _CustomObject) -> dict[str, torch.Tensor]:
    if custom.kind == "module":
        return dict(cast(torch.nn.Module, custom.value).state_dict())
    return {"": cast(torch.Tensor, custom.value)}


def _load_custom_state(
    custom: _CustomObject,
    values: Mapping[str, torch.Tensor],
) -> None:
    if custom.kind == "module":
        module = cast(torch.nn.Module, custom.value)
        expected = module.state_dict()
        if set(values) != set(expected):
            raise TrainerRankSlotStateError(
                "Custom module tensor keys differ from checkpoint: "
                f"missing={sorted(set(expected) - set(values))[:8]} "
                f"unexpected={sorted(set(values) - set(expected))[:8]}"
            )
        for key, tensor in values.items():
            target = expected[key]
            if (
                tuple(tensor.shape) != tuple(target.shape)
                or tensor.dtype != target.dtype
            ):
                raise TrainerRankSlotStateError(
                    f"Custom module tensor {key!r} has shape/dtype "
                    f"{tuple(tensor.shape)}/{tensor.dtype}; expected "
                    f"{tuple(target.shape)}/{target.dtype}"
                )
        module.load_state_dict(values, strict=True)
        return
    if set(values) != {""}:
        raise TrainerRankSlotStateError(
            f"Custom {custom.kind} checkpoint must contain exactly one tensor"
        )
    target = cast(torch.Tensor, custom.value)
    tensor = values[""]
    if tuple(tensor.shape) != tuple(target.shape) or tensor.dtype != target.dtype:
        raise TrainerRankSlotStateError(
            f"Custom {custom.kind} has shape/dtype {tuple(target.shape)}/{target.dtype}; "
            f"checkpoint contains {tuple(tensor.shape)}/{tensor.dtype}"
        )
    with torch.no_grad():
        target.copy_(tensor.to(device=target.device))


def _validate_custom_optimizer_state(
    checkpoint: str,
    key: str,
    parameter: torch.nn.Parameter,
    state: "CustomOptimizerState",
) -> None:
    expected_shape = tuple(parameter.shape)
    tensors = {
        "master": state.master,
        "exp_avg": state.exp_avg,
        "exp_avg_sq": state.exp_avg_sq,
    }
    invalid = [
        name
        for name, tensor in tensors.items()
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32
    ]
    if invalid or not math.isfinite(state.step) or state.step < 0:
        raise TrainerRankSlotStateError(
            f"Custom optimizer state for {checkpoint!r}/{key!r} is invalid; "
            f"expected FP32 tensors with shape {expected_shape} and a nonnegative "
            f"finite step (invalid={invalid}, step={state.step})."
        )


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


def _is_forward_input(inputs: ForwardInputs) -> TypeIs[AnyForwardInput]:
    return isinstance(inputs, ForwardInput)


def _flatten(inputs: ForwardInputs) -> Iterator[AnyForwardInput]:
    if _is_forward_input(inputs):
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
