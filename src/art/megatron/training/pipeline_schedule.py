from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Generic, Protocol, TypeVar, cast

from megatron.core import parallel_state as ps
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config
import torch

from art.megatron.context_parallel.types import (
    TrainingMicrobatchWorkload,
    TrainingStepWorkload,
)
from art.megatron.routing_replay import MoeRoutingReplayController
from art.megatron.training.model_chunks import ModelChunks
from art.megatron.training.trace import prepare_replay_local_input_token_uids


class _PreparedMicrobatch(Protocol):
    attention_state: Any
    local_token_uids: torch.Tensor | None
    workload: TrainingMicrobatchWorkload


_T = TypeVar("_T", bound=_PreparedMicrobatch)


@dataclass(frozen=True)
class ScheduleMicrobatch(Generic[_T]):
    order: int
    sample_index: int | None
    payload: _T
    recompute_state: object | None = None


def _local_training_workload_values(
    microbatches: Sequence[ScheduleMicrobatch[Any]], cp_rank: int
) -> list[int]:
    real = tuple(item for item in microbatches if item.sample_index is not None)
    dummy = tuple(item for item in microbatches if item.sample_index is None)
    return [
        sum(item.payload.workload.logical_nonpadding_tokens for item in real),
        sum(item.payload.workload.loss_bearing_tokens for item in real),
        sum(item.payload.workload.executed_token_equivalents for item in microbatches),
        (
            sum(
                item.payload.workload.nominal_schedule_capacity_tokens
                for item in microbatches
            )
            if cp_rank == 0
            else 0
        ),
        sum(item.payload.workload.executed_token_equivalents for item in dummy),
        (
            sum(
                item.payload.workload.nominal_schedule_capacity_tokens for item in dummy
            )
            if cp_rank == 0
            else 0
        ),
        len(real) if cp_rank == 0 else 0,
        len(microbatches) - len(real) if cp_rank == 0 else 0,
    ]


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
        model_activator: Callable[[_T, int], None] | None,
    ) -> None:
        hybridep_enabled = _validate_hybridep_token_counts(
            hybridep_token_counts, microbatch_count
        )
        self._controller = controller
        self._model_activator = model_activator
        self._hybridep_token_counts = (
            tuple(cast(Sequence[int], hybridep_token_counts))
            if hybridep_enabled
            else None
        )

    @property
    def enabled(self) -> bool:
        return (
            self._controller is not None
            or self._hybridep_token_counts is not None
            or self._model_activator is not None
        )

    def activate(self, item: ScheduleMicrobatch[_T], chunk_index: int) -> None:
        prepared = item.payload
        if self._controller is not None:
            self._controller.begin_micro(
                item.sample_index,
                item.order,
                chunk_index=chunk_index,
            )
            prepare_replay_local_input_token_uids(
                self._controller,
                prepared.local_token_uids,
                prepared.attention_state,
            )
        if self._hybridep_token_counts is not None:
            _set_hybridep_token_count(self._hybridep_token_counts[item.order])
        if self._model_activator is not None:
            self._model_activator(prepared, chunk_index)


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
    schedule_wall_s: float = 0.0

    def metrics(self) -> dict[str, float]:
        bubble_fraction = pipeline_bubble_fraction(
            pp_size=self.pp_size,
            vp_size=self.vp_size,
            num_microbatches=self.num_microbatches,
        )
        return {
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
            "pipeline/ideal_bubble_fraction": bubble_fraction,
        }


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


def validate_microbatch_shapes(
    shapes: Sequence[tuple[int, int]],
) -> tuple[int, int, bool]:
    if not shapes:
        raise ValueError("MCore schedule requires at least one microbatch")
    invalid = [
        (index, shape)
        for index, shape in enumerate(shapes)
        if shape[0] != 1 or shape[1] < 1
    ]
    if invalid:
        raise ValueError(
            "ART pipeline microbatches must have [batch=1, sequence>0] shapes; "
            f"invalid={invalid}"
        )
    sequence_lengths = {shape[1] for shape in shapes}
    return 1, max(sequence_lengths), len(sequence_lengths) > 1


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


class _ArtP2PCommunicator(P2PCommunicator):
    def _communicate(self, *, tensor_shape: Any, **kwargs: Any) -> Any:
        multiplier = getattr(self.config, "art_pipeline_activation_multiplier", None)
        if tensor_shape is not None and multiplier is not None:
            tensor_shape = torch.Size(
                (*tensor_shape[:-1], multiplier, tensor_shape[-1])
            )
        return super()._communicate(tensor_shape=tensor_shape, **kwargs)


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
        model_activator: Callable[[_T, int], None] | None = None,
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
            model_activator=model_activator,
        )
        self._active_activation_key: tuple[int, int] | None = None
        self.pp_size = int(ps.get_pipeline_model_parallel_world_size())
        (
            self.micro_batch_size,
            local_seq_length,
            self.variable_seq_lengths,
        ) = validate_microbatch_shapes(
            [(int(value.shape[0]), int(value.shape[1])) for value in model_inputs]
        )
        self.seq_length = local_seq_length * (
            int(ps.get_context_parallel_world_size()) if self.pp_size > 1 else 1
        )
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
            config.variable_seq_lengths = self.pp_size > 1 and self.variable_seq_lengths
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

    def activate(self, microbatch: ScheduleMicrobatch[_T], chunk_index: int) -> None:
        if not self._microbatch_state.enabled:
            return
        activation_key = (microbatch.order, chunk_index)
        if activation_key == self._active_activation_key:
            return
        self._microbatch_state.activate(microbatch, chunk_index)
        self._active_activation_key = activation_key

    def training_workload(self) -> TrainingStepWorkload:
        values = torch.tensor(
            _local_training_workload_values(
                self.microbatches, int(ps.get_context_parallel_rank())
            ),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                values,
                group=ps.get_data_parallel_group(with_context_parallel=True),
            )
        (
            logical,
            loss_bearing,
            executed,
            nominal,
            dummy_executed,
            dummy_nominal,
            real_microbatches,
            dummy_microbatches,
        ) = values.cpu().tolist()
        return TrainingStepWorkload(
            logical_nonpadding_tokens=logical,
            loss_bearing_tokens=loss_bearing,
            executed_token_equivalents=executed,
            nominal_schedule_capacity_tokens=nominal,
            dummy_executed_token_equivalents=dummy_executed,
            dummy_schedule_capacity_tokens=dummy_nominal,
            real_microbatches=real_microbatches,
            dummy_microbatches=dummy_microbatches,
        )

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
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            chunk_index: int,
        ) -> None:
            microbatch = _find_bound_microbatch(by_state_id, (*args, kwargs))
            self.activate(microbatch, chunk_index)

        handles = []
        for chunk_index, chunk in enumerate(self.model_chunks):
            for layer in _transformer_layer_callers([chunk]):

                def restore_chunk(
                    module: Any,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    _chunk_index: int = chunk_index,
                ) -> None:
                    restore(
                        module,
                        args,
                        kwargs,
                        chunk_index=_chunk_index,
                    )

                handles.append(
                    layer.register_forward_pre_hook(
                        torch.compiler.disable(restore_chunk),
                        with_kwargs=True,
                    )
                )
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
        def activate(
            chunk_index: int,
        ) -> Iterator[ScheduleMicrobatch[_T]]:
            for microbatch in self.microbatches:
                self.activate(microbatch, chunk_index)
                yield microbatch

        return [activate(index) for index in range(len(self.model_chunks))]

    def run(
        self,
        forward_step_func: Callable[
            ..., tuple[torch.Tensor, Callable[..., Any] | None]
        ],
        *,
        forward_only: bool,
        collect_non_loss_data: bool = False,
    ) -> list[Any]:
        def forward_with_contiguous_output(
            data_iterator: Any, model: Any, *args: Any
        ) -> Any:
            result = forward_step_func(data_iterator, model, *args)
            output = result[0]
            if (
                self.pp_size > 1
                and isinstance(output, torch.Tensor)
                and output._base is not None
            ):
                return (output.clone(), *result[1:])
            return result

        config = get_model_config(self.model_chunks[0])
        if not forward_only and bool(config.overlap_moe_expert_parallel_comm):
            raise RuntimeError(
                "ART's forward-step contract does not support MCore's combined "
                "EP-overlap schedule; disable overlap_moe_expert_parallel_comm"
            )
        communicator: Any | None = None
        pg_collection: ProcessGroupCollection | None = None
        if self.pp_size > 1:
            communicator = _ArtP2PCommunicator(
                pp_group=ps.get_pipeline_model_parallel_group(), config=config
            )
            pg_collection = _process_group_collection()
        start = time.perf_counter()
        self._active_activation_key = None
        with self._recompute_activation_hooks(enabled=not forward_only):
            outputs = get_forward_backward_func(
                pp_size=self.pp_size,
                vp_size=None if self.vp_size == 1 else self.vp_size,
            )(
                forward_step_func=forward_with_contiguous_output,
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
