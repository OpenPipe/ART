from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Generic, Protocol, TypeVar, cast

from megatron.core import parallel_state as ps
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    get_forward_backward_func,
    get_schedule_table,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.timers import DummyTimer
from megatron.core.utils import get_model_config
import torch

from art.megatron.routing_replay import MoeRoutingReplayController
from art.megatron.training.model_chunks import ModelChunks
from art.megatron.training.trace import prepare_replay_local_input_token_uids


class _PreparedMicrobatch(Protocol):
    attention_state: Any
    local_token_uids: torch.Tensor | None


_T = TypeVar("_T", bound=_PreparedMicrobatch)


@dataclass(frozen=True)
class ScheduleMicrobatch(Generic[_T]):
    order: int
    sample_index: int | None
    payload: _T
    recompute_state: object | None = None


def _set_hybridep_token_count(rows: int) -> None:
    from megatron.core.transformer.moe import fused_a2a

    buffer = fused_a2a._hybrid_ep_buffer
    if buffer is None:
        raise RuntimeError("HybridEP buffer is not initialized")
    buffer.set_num_tokens_per_rank(rows)


def _validate_hybridep_token_counts(
    values: Sequence[int] | None, microbatch_count: int
) -> bool:
    enabled = ps.get_expert_model_parallel_world_size() > 1
    if enabled and (values is None or len(values) != microbatch_count):
        raise RuntimeError(
            "HybridEP requires one planned communication extent per microbatch"
        )
    return enabled


class PipelineMicrobatchState(Generic[_T]):
    def __init__(
        self,
        *,
        controller: MoeRoutingReplayController | None,
        hybridep_token_counts: Sequence[int] | None,
        microbatch_count: int,
    ) -> None:
        hybridep_enabled = _validate_hybridep_token_counts(
            hybridep_token_counts, microbatch_count
        )
        self._controller = controller
        self._hybridep_token_counts = (
            tuple(cast(Sequence[int], hybridep_token_counts))
            if hybridep_enabled
            else None
        )

    @property
    def enabled(self) -> bool:
        return self._controller is not None or self._hybridep_token_counts is not None

    def activate(self, item: ScheduleMicrobatch[_T]) -> None:
        prepared = item.payload
        if self._controller is not None:
            self._controller.begin_micro(item.sample_index, item.order)
            prepare_replay_local_input_token_uids(
                self._controller,
                prepared.local_token_uids,
                prepared.attention_state,
            )
        if self._hybridep_token_counts is not None:
            _set_hybridep_token_count(self._hybridep_token_counts[item.order])


@dataclass
class PipelineScheduleTelemetry:
    pp_rank: int
    pp_size: int
    vp_size: int
    num_microbatches: int
    real_microbatches: int
    dummy_microbatches: int
    micro_batch_size: int
    seq_length: int
    microbatch_group_size: int
    forward_compute_s_by_chunk: dict[int, float] = field(default_factory=dict)
    backward_compute_s_by_chunk: dict[int, float] = field(default_factory=dict)
    forward_host_s_by_chunk: dict[int, float] = field(default_factory=dict)
    forward_calls_by_chunk: dict[int, int] = field(default_factory=dict)
    p2p_s: float = 0.0
    p2p_calls: int = 0
    p2p_wait_s: float = 0.0
    p2p_wait_calls: int = 0
    schedule_wall_s: float = 0.0
    schedule_gpu_s: float = 0.0
    memory_allocated_start_bytes: int = 0
    peak_memory_bytes: int = 0
    _cuda_timers: _DeferredCudaTimers | None = field(default=None, repr=False)
    _metrics_cache: dict[str, float] | None = field(default=None, repr=False)

    def metrics(self) -> dict[str, float]:
        if self._metrics_cache is not None:
            return dict(self._metrics_cache)
        self._resolve_cuda_timers()
        bubble_fraction = pipeline_bubble_fraction(
            pp_size=self.pp_size,
            vp_size=self.vp_size,
            num_microbatches=self.num_microbatches,
        )
        metrics = {
            "pipeline/pp_rank": float(self.pp_rank),
            "pipeline/pp_size": float(self.pp_size),
            "pipeline/vp_size": float(self.vp_size),
            "pipeline/microbatches_per_dp_rank": float(self.num_microbatches),
            "pipeline/real_microbatches_per_dp_rank": float(self.real_microbatches),
            "pipeline/dummy_microbatches_per_dp_rank": float(self.dummy_microbatches),
            "pipeline/micro_batch_size": float(self.micro_batch_size),
            "pipeline/packed_sequence_length": float(self.seq_length),
            "pipeline/microbatch_group_size_per_vp_stage": float(
                self.microbatch_group_size
            ),
            "pipeline/schedule_wall_s": self.schedule_wall_s,
            "pipeline/schedule_gpu_s": self.schedule_gpu_s,
            "pipeline/p2p_s": self.p2p_s,
            "pipeline/p2p_call_host_s": self.p2p_s,
            "pipeline/p2p_calls": float(self.p2p_calls),
            "pipeline/p2p_wait_host_s": self.p2p_wait_s,
            "pipeline/p2p_wait_calls": float(self.p2p_wait_calls),
            "pipeline/memory_allocated_start_bytes": float(
                self.memory_allocated_start_bytes
            ),
            "pipeline/peak_memory_bytes": float(self.peak_memory_bytes),
            "pipeline/ideal_bubble_fraction": bubble_fraction,
        }
        for chunk, forward_compute in sorted(self.forward_compute_s_by_chunk.items()):
            backward_compute = self.backward_compute_s_by_chunk.get(chunk, 0.0)
            metrics[f"pipeline/chunk_{chunk}/compute_s"] = (
                forward_compute + backward_compute
            )
            metrics[f"pipeline/chunk_{chunk}/forward_compute_s"] = forward_compute
            metrics[f"pipeline/chunk_{chunk}/backward_compute_s"] = backward_compute
            metrics[f"pipeline/chunk_{chunk}/forward_host_s"] = (
                self.forward_host_s_by_chunk.get(chunk, 0.0)
            )
            metrics[f"pipeline/chunk_{chunk}/forward_calls"] = float(
                self.forward_calls_by_chunk[chunk]
            )
        metrics.update(self._stage_metrics())
        self._metrics_cache = metrics
        return dict(metrics)

    def _resolve_cuda_timers(self) -> None:
        timers = self._cuda_timers
        if timers is None:
            return
        timers.synchronize()
        self.schedule_gpu_s = timers.total("forward-backward")
        self.forward_compute_s_by_chunk = timers.by_chunk("forward-compute")
        self.backward_compute_s_by_chunk = timers.by_chunk("backward-compute")

    def _stage_metrics(self) -> dict[str, float]:
        local = [
            self.schedule_gpu_s,
            sum(self.forward_compute_s_by_chunk.values()),
            sum(self.backward_compute_s_by_chunk.values()),
            self.p2p_s,
            self.p2p_wait_s,
            float(self.peak_memory_bytes),
            *(
                values.get(chunk, 0.0)
                for values in (
                    self.forward_compute_s_by_chunk,
                    self.backward_compute_s_by_chunk,
                )
                for chunk in range(self.vp_size)
            ),
        ]
        rows = [local]
        if self.pp_size > 1:
            value = torch.tensor(
                local, device=torch.cuda.current_device(), dtype=torch.float64
            )
            gathered = torch.empty(
                self.pp_size * value.numel(), device=value.device, dtype=value.dtype
            )
            torch.distributed.all_gather_into_tensor(  # ty: ignore[possibly-missing-attribute]
                gathered,
                value,
                group=ps.get_pipeline_model_parallel_group(),
            )
            rows = gathered.view(self.pp_size, -1).cpu().tolist()

        metrics: dict[str, float] = {}
        for stage, row in enumerate(rows):
            prefix = f"pipeline/stage_{stage}"
            metrics[f"{prefix}/schedule_gpu_s"] = row[0]
            metrics[f"{prefix}/forward_compute_s"] = row[1]
            metrics[f"{prefix}/backward_compute_s"] = row[2]
            metrics[f"{prefix}/p2p_call_host_s"] = row[3]
            metrics[f"{prefix}/p2p_wait_host_s"] = row[4]
            metrics[f"{prefix}/peak_memory_bytes"] = row[5]
            for chunk in range(self.vp_size):
                metrics[f"{prefix}/chunk_{chunk}/forward_compute_s"] = row[6 + chunk]
                metrics[f"{prefix}/chunk_{chunk}/backward_compute_s"] = row[
                    6 + self.vp_size + chunk
                ]
        stage_compute = [row[1] + row[2] for row in rows]
        max_compute = max(stage_compute, default=0.0)
        metrics["pipeline/stage_compute_imbalance_fraction"] = (
            (max_compute - min(stage_compute)) / max_compute if max_compute > 0 else 0.0
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


class _DeferredCudaTimer:
    def __init__(self, owner: _DeferredCudaTimers, name: str) -> None:
        self._owner = owner
        self._name = name
        self._start: torch.cuda.Event | None = None

    def start(self, barrier: bool = False) -> None:
        if self._start is not None:
            raise RuntimeError(f"CUDA timer {self._name!r} is already running")
        if barrier:
            torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def stop(self, barrier: bool = False) -> None:
        if self._start is None:
            raise RuntimeError(f"CUDA timer {self._name!r} is not running")
        if barrier:
            torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._owner._spans.setdefault(self._name, []).append((self._start, end))
        self._start = None


class _DeferredCudaTimers:
    _TIMED_NAMES = {"forward-backward", "forward-compute", "backward-compute"}

    def __init__(
        self, *, forward_chunks: Sequence[int], backward_chunks: Sequence[int]
    ):
        self._chunk_sequences = {
            "forward-compute": tuple(forward_chunks),
            "backward-compute": tuple(backward_chunks),
        }
        self._timers = {
            name: _DeferredCudaTimer(self, name) for name in self._TIMED_NAMES
        }
        self._null_timer = DummyTimer()
        self._spans: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def __call__(self, name: str, **_kwargs: Any) -> _DeferredCudaTimer | DummyTimer:
        return self._timers.get(name, self._null_timer)

    def validate_counts(self, *, forward_only: bool) -> None:
        names = ("forward-compute",) + (() if forward_only else ("backward-compute",))
        for name in names:
            actual = len(self._spans.get(name, ()))
            expected = len(self._chunk_sequences[name])
            if actual != expected:
                raise RuntimeError(
                    f"MCore {name} count differs from the schedule table: "
                    f"expected={expected}, got={actual}"
                )

    def synchronize(self) -> None:
        schedule = self._spans.get("forward-backward", ())
        if schedule:
            schedule[-1][1].synchronize()

    def total(self, name: str) -> float:
        return (
            sum(start.elapsed_time(end) for start, end in self._spans.get(name, ()))
            / 1e3
        )

    def by_chunk(self, name: str) -> dict[int, float]:
        spans = self._spans.get(name, ())
        if not spans:
            return {}
        chunks = self._chunk_sequences[name]
        if len(spans) != len(chunks):
            raise RuntimeError(
                f"Cannot resolve {name}: spans={len(spans)}, chunks={len(chunks)}"
            )
        values: dict[int, float] = {}
        for chunk, (start, end) in zip(chunks, spans, strict=True):
            values[chunk] = values.get(chunk, 0.0) + start.elapsed_time(end) / 1e3
        return values


class _TimedWork:
    def __init__(self, work: Any, telemetry: PipelineScheduleTelemetry) -> None:
        self._work = work
        self._telemetry = telemetry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return self._work.wait(*args, **kwargs)
        finally:
            self._telemetry.p2p_wait_s += time.perf_counter() - start
            self._telemetry.p2p_wait_calls += 1


def _wrap_p2p_work(value: Any, telemetry: PipelineScheduleTelemetry) -> Any:
    if isinstance(value, torch.distributed.Work):  # ty: ignore[possibly-missing-attribute]
        return _TimedWork(value, telemetry)
    if isinstance(value, dict):
        return {key: _wrap_p2p_work(item, telemetry) for key, item in value.items()}
    if isinstance(value, list):
        return [_wrap_p2p_work(item, telemetry) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap_p2p_work(item, telemetry) for item in value)
    return value


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
                result = value(*args, **kwargs)
            finally:
                self._telemetry.p2p_s += time.perf_counter() - start
                self._telemetry.p2p_calls += 1
            return _wrap_p2p_work(result, self._telemetry)

        return timed


class MCoreScheduleAdapter(Generic[_T]):
    """Small ART boundary around MCore's PP1, PP and VPP schedules."""

    def __init__(
        self,
        *,
        model_chunks: ModelChunks,
        prepared_microbatches: Sequence[_T],
        sample_indices: Sequence[int | None],
        model_inputs: Sequence[torch.Tensor],
        moe_routing_replay_controller: MoeRoutingReplayController | None = None,
        hybridep_token_counts: Sequence[int] | None = None,
    ) -> None:
        if not model_chunks:
            raise ValueError("MCore schedule requires at least one model chunk")
        if not (len(prepared_microbatches) == len(sample_indices) == len(model_inputs)):
            raise ValueError("microbatch payload/sample/input counts differ")
        self.model_chunks = model_chunks
        self.microbatches = tuple(
            ScheduleMicrobatch(order, sample_index, prepared, prepared.attention_state)
            for order, (sample_index, prepared) in enumerate(
                zip(sample_indices, prepared_microbatches, strict=True)
            )
        )
        self._microbatch_state = PipelineMicrobatchState(
            controller=moe_routing_replay_controller,
            hybridep_token_counts=hybridep_token_counts,
            microbatch_count=len(self.microbatches),
        )
        self._active_recompute_state_id: int | None = None
        self.micro_batch_size, self.seq_length = validate_fixed_microbatch_shapes(
            [(int(value.shape[0]), int(value.shape[1])) for value in model_inputs]
        )
        self.pp_size = int(ps.get_pipeline_model_parallel_world_size())
        self.pp_rank = int(ps.get_pipeline_model_parallel_rank())
        self.vp_size = int(ps.get_virtual_pipeline_model_parallel_world_size() or 1)
        self.microbatch_group_size = len(self.microbatches)
        if self.vp_size != len(model_chunks):
            raise ValueError(
                "Local model chunk count must equal VPP size: "
                f"chunks={len(model_chunks)}, vpp={self.vp_size}"
            )
        self._validate_stage_ownership()
        self._configure()
        self.telemetry = PipelineScheduleTelemetry(
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            vp_size=self.vp_size,
            num_microbatches=len(self.microbatches),
            real_microbatches=sum(
                microbatch.sample_index is not None for microbatch in self.microbatches
            ),
            dummy_microbatches=sum(
                microbatch.sample_index is None for microbatch in self.microbatches
            ),
            micro_batch_size=self.micro_batch_size,
            seq_length=self.seq_length,
            microbatch_group_size=self.microbatch_group_size,
        )
        forward_chunks = [0] * len(self.microbatches)
        if self.vp_size > 1:
            table = get_schedule_table(
                len(self.microbatches), self.vp_size, self.microbatch_group_size
            )
            forward_chunks = [int(chunk) for _, chunk in table]
        backward_chunks = [self.vp_size - chunk - 1 for chunk in forward_chunks]
        if torch.cuda.is_available():
            self.telemetry._cuda_timers = _DeferredCudaTimers(
                forward_chunks=forward_chunks,
                backward_chunks=backward_chunks,
            )
        self._chunk_by_id = {
            id(chunk): index for index, chunk in enumerate(model_chunks)
        }

    def _validate_stage_ownership(self) -> None:
        for chunk_index, chunk in enumerate(self.model_chunks):
            expected_pre = self.pp_rank == 0 and chunk_index == 0
            expected_post = (
                self.pp_rank == self.pp_size - 1
                and chunk_index == len(self.model_chunks) - 1
            )
            actual_pre = chunk_pre_process(chunk)
            actual_post = chunk_post_process(chunk)
            if (actual_pre, actual_post) != (expected_pre, expected_post):
                raise RuntimeError(
                    "Megatron model chunk pipeline ownership is inconsistent: "
                    f"pp_rank={self.pp_rank}, chunk={chunk_index}, "
                    f"pre_process={actual_pre} (expected {expected_pre}), "
                    f"post_process={actual_post} (expected {expected_post})"
                )

    def _configure(self) -> None:
        vpp_group: int | None = None
        for config in _model_configs(self.model_chunks):
            config.variable_seq_lengths = False
            if self.vp_size > 1:
                group = int(
                    getattr(config, "microbatch_group_size_per_vp_stage", 0)
                    or self.pp_size
                )
                if vpp_group is not None and group != vpp_group:
                    raise ValueError(
                        "All VPP model chunks must use one microbatch group size: "
                        f"expected={vpp_group}, got={group}"
                    )
                vpp_group = group
                validate_vpp_microbatch_group(
                    num_microbatches=len(self.microbatches),
                    pp_size=self.pp_size,
                    group_size=group,
                )
                config.microbatch_group_size_per_vp_stage = group
                self.microbatch_group_size = group
                config.overlap_p2p_comm = True
                config.batch_p2p_comm = False
            elif self.pp_size > 1:
                config.overlap_p2p_comm = False
                config.batch_p2p_comm = True
                # PyTorch 2.11 does not need MCore's legacy batch-P2P device sync.
                config.batch_p2p_sync = False

    def activate(self, microbatch: ScheduleMicrobatch[_T]) -> None:
        if not self._microbatch_state.enabled:
            return
        state = microbatch.recompute_state
        state_id = id(microbatch if state is None else state)
        if state_id == self._active_recompute_state_id:
            return
        self._microbatch_state.activate(microbatch)
        self._active_recompute_state_id = state_id

    @contextmanager
    def _recompute_activation_hooks(self, *, enabled: bool) -> Iterator[None]:
        config = get_model_config(self.model_chunks[0])
        if (
            not enabled
            or self.pp_size <= 1
            or not self._microbatch_state.enabled
            or not _stateful_recompute_enabled(config)
        ):
            yield
            return
        _validate_stateful_recompute_mode(config)

        by_state_id: dict[int, ScheduleMicrobatch[_T]] = {}
        for microbatch in self.microbatches:
            state = microbatch.recompute_state
            if state is None:
                raise RuntimeError(
                    "Stateful PP recomputation requires recompute_state on every microbatch"
                )
            previous = by_state_id.setdefault(id(state), microbatch)
            if previous is not microbatch:
                raise RuntimeError(
                    "recompute_state must identify one logical microbatch"
                )

        def restore(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            microbatch = _find_bound_microbatch(by_state_id, (*args, kwargs))
            self.activate(microbatch)

        restore = torch.compiler.disable(restore)
        handles = [
            layer.register_forward_pre_hook(restore, with_kwargs=True)
            for layer in _transformer_layer_callers(self.model_chunks)
        ]
        if not handles:
            raise RuntimeError(
                "Stateful PP recomputation could not find TransformerLayer call sites"
            )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def independent_iterators(self) -> list[Iterator[ScheduleMicrobatch[_T]]]:
        return [iter(self.microbatches) for _ in self.model_chunks]

    @contextmanager
    def _telemetry_timer_context(self) -> Iterator[None]:
        timers = self.telemetry._cuda_timers
        if timers is None:
            yield
            return
        configs = _model_configs(self.model_chunks)
        previous = [config.timers for config in configs]
        for config in configs:
            config.timers = timers
        try:
            yield
        finally:
            for config, prior in zip(configs, previous, strict=True):
                config.timers = prior

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
                self.telemetry.forward_host_s_by_chunk[chunk] = (
                    self.telemetry.forward_host_s_by_chunk.get(chunk, 0.0) + elapsed
                )
                self.telemetry.forward_calls_by_chunk[chunk] = (
                    self.telemetry.forward_calls_by_chunk.get(chunk, 0) + 1
                )

        config = get_model_config(self.model_chunks[0])
        if not forward_only and bool(config.overlap_moe_expert_parallel_comm):
            raise RuntimeError(
                "ART's forward-step contract does not support MCore's combined "
                "EP-overlap schedule; disable overlap_moe_expert_parallel_comm"
            )
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
        self._active_recompute_state_id = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.telemetry.memory_allocated_start_bytes = int(
                torch.cuda.memory_allocated()
            )
        with (
            self._telemetry_timer_context(),
            self._recompute_activation_hooks(enabled=not forward_only),
        ):
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
        expected_calls = len(self.microbatches)
        invalid_calls = {
            chunk: self.telemetry.forward_calls_by_chunk.get(chunk, 0)
            for chunk in range(self.vp_size)
            if self.telemetry.forward_calls_by_chunk.get(chunk, 0) != expected_calls
        }
        if invalid_calls:
            raise RuntimeError(
                "MCore schedule did not run every local chunk once per microbatch: "
                f"expected={expected_calls}, got={invalid_calls}"
            )
        if self.telemetry._cuda_timers is not None:
            self.telemetry._cuda_timers.validate_counts(forward_only=forward_only)
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


def _stateful_recompute_enabled(config: Any) -> bool:
    granularity = getattr(config, "recompute_granularity", None)
    if granularity == "full":
        return True
    modules = set(getattr(config, "recompute_modules", None) or ())
    return granularity == "selective" and bool(modules & {"mlp", "moe"})


def _validate_stateful_recompute_mode(config: Any) -> None:
    if getattr(config, "recompute_granularity", None) != "full":
        raise RuntimeError(
            "HybridEP/MoE replay requires full-layer activation recomputation under "
            "PP; selective MLP/MoE checkpoints do not retain ART's exact microbatch "
            "state"
        )


def _transformer_layer_callers(
    model_chunks: Sequence[torch.nn.Module],
) -> list[torch.nn.Module]:
    from megatron.core.transformer.transformer_layer import TransformerLayer

    callers: dict[int, torch.nn.Module] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            original = getattr(module, "_orig_mod", None)
            if isinstance(original, TransformerLayer):
                callers[id(original)] = module
            elif isinstance(module, TransformerLayer):
                callers.setdefault(id(module), module)
    return list(callers.values())


def _find_bound_microbatch(
    by_state_id: dict[int, ScheduleMicrobatch[_T]],
    values: Sequence[Any],
) -> ScheduleMicrobatch[_T]:
    pending = list(values)
    seen: set[int] = set()
    match: ScheduleMicrobatch[_T] | None = None
    while pending:
        value = pending.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        microbatch = by_state_id.get(value_id)
        if microbatch is not None and microbatch.recompute_state is value:
            if match is not None and match is not microbatch:
                raise RuntimeError(
                    "Stateful PP recomputation received multiple microbatch states: "
                    f"orders={[match.order, microbatch.order]}"
                )
            match = microbatch
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list | tuple):
            pending.extend(value)
    if match is None:
        raise RuntimeError(
            "Stateful PP recomputation did not receive its exact microbatch state"
        )
    return match


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


def _model_configs(model_chunks: Sequence[torch.nn.Module]) -> list[Any]:
    configs: dict[int, Any] = {}
    for chunk in model_chunks:
        config = get_model_config(chunk)
        configs.setdefault(id(config), config)
    return list(configs.values())
