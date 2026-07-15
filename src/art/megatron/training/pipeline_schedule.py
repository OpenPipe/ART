from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
import time
from typing import Any, Generic, TypeVar, cast

from megatron.core import parallel_state as ps
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config
import torch

from art.megatron.training.model_chunks import ModelChunks

_T = TypeVar("_T")


@dataclass(frozen=True)
class ScheduleMicrobatch(Generic[_T]):
    order: int
    sample_index: int | None
    payload: _T


@dataclass
class PipelineScheduleTelemetry:
    schedule: str
    pp_rank: int
    pp_size: int
    vp_size: int
    num_microbatches: int
    compute_s_by_chunk: dict[int, float] = field(default_factory=dict)
    forward_calls_by_chunk: dict[int, int] = field(default_factory=dict)
    p2p_s: float = 0.0
    p2p_calls: int = 0
    schedule_wall_s: float = 0.0
    peak_memory_bytes: int = 0

    def metrics(self) -> dict[str, float]:
        metrics = {
            "pipeline/pp_rank": float(self.pp_rank),
            "pipeline/pp_size": float(self.pp_size),
            "pipeline/vp_size": float(self.vp_size),
            "pipeline/num_microbatches": float(self.num_microbatches),
            "pipeline/schedule_wall_s": self.schedule_wall_s,
            "pipeline/p2p_s": self.p2p_s,
            "pipeline/p2p_calls": float(self.p2p_calls),
            "pipeline/peak_memory_bytes": float(self.peak_memory_bytes),
            "pipeline/bubble_fraction_estimate": pipeline_bubble_fraction(
                pp_size=self.pp_size,
                vp_size=self.vp_size,
                num_microbatches=self.num_microbatches,
            ),
        }
        for chunk, value in sorted(self.compute_s_by_chunk.items()):
            metrics[f"pipeline/chunk_{chunk}/compute_s"] = value
            metrics[f"pipeline/chunk_{chunk}/forward_calls"] = float(
                self.forward_calls_by_chunk[chunk]
            )
        return metrics


def pipeline_bubble_fraction(
    *, pp_size: int, vp_size: int, num_microbatches: int
) -> float:
    if pp_size <= 1:
        return 0.0
    useful = max(1, num_microbatches * max(1, vp_size))
    bubbles = max(0, pp_size - 1)
    return bubbles / (useful + bubbles)


def validate_pipeline_topology(
    *,
    world_size: int,
    tp: int,
    cp: int,
    pp: int,
    ep: int,
    etp: int,
    vp: int,
    num_layers: int | None = None,
) -> None:
    values = {
        "world_size": world_size,
        "tp": tp,
        "cp": cp,
        "pp": pp,
        "ep": ep,
        "etp": etp,
        "vp": vp,
    }
    invalid = {name: value for name, value in values.items() if value < 1}
    if invalid:
        raise ValueError(f"Megatron topology sizes must be positive: {invalid}")
    dense = tp * cp * pp
    expert = etp * ep * pp
    if world_size % dense:
        raise ValueError(
            f"world_size={world_size} must be divisible by TP*CP*PP={dense}"
        )
    if world_size % expert:
        raise ValueError(
            f"world_size={world_size} must be divisible by ETP*EP*PP={expert}"
        )
    if vp > 1 and pp <= 1:
        raise ValueError("VPP requires pipeline_model_parallel_size > 1")
    if num_layers is not None and num_layers % (pp * vp):
        raise ValueError(
            f"num_layers={num_layers} must be divisible by PP*VPP={pp * vp}"
        )


def validate_fixed_microbatch_shapes(
    shapes: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    if not shapes:
        raise ValueError("MCore schedule requires at least one microbatch")
    expected = shapes[0]
    mismatches = [
        (index, shape) for index, shape in enumerate(shapes) if shape != expected
    ]
    if mismatches:
        raise ValueError(
            "MCore pipeline schedules require fixed [batch, sequence] shapes; "
            f"expected={expected}, mismatches={mismatches}"
        )
    batch, sequence = expected
    if batch < 1 or sequence < 1:
        raise ValueError(f"Invalid fixed microbatch shape: {expected}")
    return batch, sequence


def chunk_pre_process(model: torch.nn.Module) -> bool:
    return bool(_chunk_attr(model, "pre_process"))


def chunk_post_process(model: torch.nn.Module) -> bool:
    return bool(_chunk_attr(model, "post_process"))


def _chunk_attr(model: torch.nn.Module, name: str) -> Any:
    current: Any = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        for wrapper_name in ("module", "_orig_mod", "language_model"):
            wrapped = getattr(current, wrapper_name, None)
            if isinstance(wrapped, torch.nn.Module):
                current = wrapped
                break
        else:
            return None
    return None


class _TimedP2PCommunicator:
    def __init__(
        self, communicator: P2PCommunicator, telemetry: PipelineScheduleTelemetry
    ):
        self._communicator = communicator
        self._telemetry = telemetry

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._communicator, name)
        if not callable(value) or not (
            name.startswith("send") or name.startswith("recv")
        ):
            return value

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return value(*args, **kwargs)
            finally:
                self._telemetry.p2p_s += time.perf_counter() - start
                self._telemetry.p2p_calls += 1

        return timed


class MCoreScheduleAdapter(Generic[_T]):
    """Small ART boundary around MCore's PP1, PP and VPP schedules."""

    def __init__(
        self,
        *,
        model_chunks: ModelChunks,
        microbatches: Sequence[ScheduleMicrobatch[_T]],
        microbatch_shapes: Sequence[tuple[int, int]],
        routing_replay_enabled: bool = False,
    ) -> None:
        if not model_chunks:
            raise ValueError("MCore schedule requires at least one model chunk")
        if len(microbatches) != len(microbatch_shapes):
            raise ValueError("microbatch payload/shape counts differ")
        self.model_chunks = model_chunks
        self.microbatches = tuple(microbatches)
        self.micro_batch_size, self.seq_length = validate_fixed_microbatch_shapes(
            microbatch_shapes
        )
        self.pp_size = int(ps.get_pipeline_model_parallel_world_size())
        self.pp_rank = int(ps.get_pipeline_model_parallel_rank())
        self.vp_size = int(ps.get_virtual_pipeline_model_parallel_world_size() or 1)
        if self.vp_size != len(model_chunks):
            raise ValueError(
                "Local model chunk count must equal VPP size: "
                f"chunks={len(model_chunks)}, vpp={self.vp_size}"
            )
        self._configure(routing_replay_enabled=routing_replay_enabled)
        schedule = (
            "pp1"
            if self.pp_size == 1
            else "interleaved"
            if self.vp_size > 1
            else "noninterleaved"
        )
        self.telemetry = PipelineScheduleTelemetry(
            schedule=schedule,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            vp_size=self.vp_size,
            num_microbatches=len(self.microbatches),
        )
        self._chunk_by_id = {
            id(chunk): index for index, chunk in enumerate(model_chunks)
        }

    def _configure(self, *, routing_replay_enabled: bool) -> None:
        seen: set[int] = set()
        for chunk in self.model_chunks:
            config = get_model_config(chunk)
            if id(config) in seen:
                continue
            seen.add(id(config))
            if (
                routing_replay_enabled
                and self.pp_size > 1
                and _recompute_enabled(config)
            ):
                raise RuntimeError(
                    "MoE routing replay with pipeline 1F1B and activation recomputation "
                    "is disabled: recomputation may revisit microbatches out of forward "
                    "order. Disable recompute or routing replay for PP until the identity "
                    "recovery path has a distributed correctness test."
                )
            if self.vp_size > 1:
                group = int(
                    getattr(config, "microbatch_group_size_per_vp_stage", 0)
                    or self.pp_size
                )
                validate_vpp_microbatch_group(
                    num_microbatches=len(self.microbatches),
                    pp_size=self.pp_size,
                    group_size=group,
                )
                config.microbatch_group_size_per_vp_stage = group
                config.overlap_p2p_comm = True
                config.batch_p2p_comm = False
            elif self.pp_size > 1:
                config.overlap_p2p_comm = False
                config.batch_p2p_comm = True

    def independent_iterators(self) -> list[Iterator[ScheduleMicrobatch[_T]]]:
        return [iter(self.microbatches) for _ in self.model_chunks]

    def run(
        self,
        forward_step_func: Callable[
            ..., tuple[torch.Tensor, Callable[..., Any] | None]
        ],
        *,
        forward_only: bool,
        collect_non_loss_data: bool = False,
    ) -> list[Any]:
        def timed_forward(data_iterator: Any, model: Any, *args: Any) -> Any:
            chunk = self._chunk_by_id.get(id(model))
            if chunk is None:
                raise RuntimeError("MCore schedule passed an unknown local model chunk")
            start = time.perf_counter()
            try:
                return forward_step_func(data_iterator, model, *args)
            finally:
                elapsed = time.perf_counter() - start
                self.telemetry.compute_s_by_chunk[chunk] = (
                    self.telemetry.compute_s_by_chunk.get(chunk, 0.0) + elapsed
                )
                self.telemetry.forward_calls_by_chunk[chunk] = (
                    self.telemetry.forward_calls_by_chunk.get(chunk, 0) + 1
                )

        config = get_model_config(self.model_chunks[0])
        communicator: Any | None = None
        pg_collection: ProcessGroupCollection | None = None
        if self.pp_size > 1:
            communicator = _TimedP2PCommunicator(
                P2PCommunicator(
                    pp_group=ps.get_pipeline_model_parallel_group(), config=config
                ),
                self.telemetry,
            )
            pg_collection = _process_group_collection()
        start = time.perf_counter()
        outputs = get_forward_backward_func(
            pp_size=self.pp_size,
            vp_size=None if self.vp_size == 1 else self.vp_size,
        )(
            forward_step_func=timed_forward,
            data_iterator=self.independent_iterators(),
            model=self.model_chunks,
            num_microbatches=len(self.microbatches),
            seq_length=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            p2p_communicator=cast(Any, communicator),
            pg_collection=pg_collection,
        )
        self.telemetry.schedule_wall_s = time.perf_counter() - start
        if torch.cuda.is_available():
            self.telemetry.peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        return cast(list[Any], outputs)


def validate_vpp_microbatch_group(
    *, num_microbatches: int, pp_size: int, group_size: int
) -> None:
    if not (pp_size <= group_size <= num_microbatches):
        raise ValueError(
            "VPP microbatch group must be in [PP, num_microbatches]: "
            f"pp={pp_size}, group={group_size}, num_microbatches={num_microbatches}"
        )
    remainder = num_microbatches % group_size
    if 0 < remainder < pp_size:
        raise ValueError(
            "VPP final microbatch group must be empty or contain at least PP "
            f"microbatches: remainder={remainder}, pp={pp_size}"
        )


def _recompute_enabled(config: Any) -> bool:
    return bool(
        getattr(config, "recompute_granularity", None)
        or getattr(config, "recompute_method", None)
        or getattr(config, "recompute_num_layers", None)
        or getattr(config, "recompute_modules", None)
    )


def _process_group_collection() -> ProcessGroupCollection:
    groups = ProcessGroupCollection()
    groups.tp = ps.get_tensor_model_parallel_group()
    groups.pp = ps.get_pipeline_model_parallel_group()
    groups.cp = ps.get_context_parallel_group()
    groups.embd = ps.get_embedding_group(check_initialized=False)
    groups.pos_embd = ps.get_position_embedding_group(check_initialized=False)
    groups.dp_cp = ps.get_data_parallel_group(
        with_context_parallel=True, partial_data_parallel=False
    )
    return groups
