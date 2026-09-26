"""TrainerRank checkpoint-slot bookkeeping: prefetch registry, slot
loading and validation, the slot stack, and slot-graph liveness guards.

These are ``TrainerRank`` method bodies moved out of ``_impl`` verbatim: each
function takes the owning rank as ``self`` and ``TrainerRank`` binds them as
methods, so ``self._x(...)`` dispatch and per-instance overrides keep working.

Module globals the bodies used to read from ``_impl`` (``torch``, ``dist``,
sibling helpers) are still resolved through ``_impl`` at call time, so tests
that patch ``_impl.torch`` and friends keep intercepting them; only pure
stdlib helpers are imported here directly. Referencing ``_impl`` as a module
also lets the circular import resolve lazily.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast
import weakref

from art.trainer_rank import _impl

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank._checkpoint import PreparedCheckpoint
    from art.trainer_rank._impl import (
        AdapterSelection,
        AnyForwardInput,
        AnyForwardOutput,
        MaterializedCheckpoint,
        TrainerRank,
        _AdapterConfig,
    )


def _resolve_custom_checkpoint(self: TrainerRank, checkpoint: AdapterSelection) -> str:
    if checkpoint is _impl.Unset:
        ref = self._slot_stack[-1] if self._slot_stack else self._default_slot_ref
        name = None if ref is None else ref.name
    else:
        name = cast(str | None, checkpoint)
    if name is None:
        raise _impl.TrainerRankSlotStateError(
            "Custom checkpoint objects require a loaded named checkpoint"
        )
    self._ensure_checkpoint_slots((name,))
    if name not in self._checkpoint_slots:
        raise _impl.TrainerRankSlotStateError(f"Unknown checkpoint: {name!r}")
    return name


def prefetch_checkpoints(
    self: TrainerRank, *checkpoints: str | MaterializedCheckpoint
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


def _register_checkpoint_prefetch(
    self: TrainerRank,
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
            future = _impl._checkpoint_prefetch_executor().submit(prepare)
            self._checkpoint_prefetches[key] = future
        return future


def _checkpoint_prefetch_waiter(
    self: TrainerRank, *checkpoints: str
) -> asyncio.Task[None]:
    with self._checkpoint_prefetch_lock:
        futures = [
            self._checkpoint_prefetches[self._checkpoint_prefetch_sources[name]]
            for name in checkpoints
        ]

    async def wait() -> None:
        await asyncio.gather(
            *(self._await_checkpoint_prefetch(future) for future in futures)
        )

    return asyncio.create_task(wait())


def _prefetched_checkpoint(self: TrainerRank, checkpoint: str) -> PreparedCheckpoint:
    with self._checkpoint_prefetch_lock:
        key = self._checkpoint_prefetch_sources.get(checkpoint)
        future = None if key is None else self._checkpoint_prefetches.get(key)
    if future is None:
        raise _impl.TrainerRankSlotStateError(
            f"Checkpoint {checkpoint!r} has not been prefetched"
        )
    return future.result()


def _load_registered_checkpoint(self: TrainerRank, checkpoint: str) -> None:
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
    if checkpoint in self._snapshot_checkpoint_names:
        _checkpoint.load_checkpoint(self, source, checkpoint, forward_only=True)
    else:
        _checkpoint.load_checkpoint(self, source, checkpoint)


def _ensure_checkpoint_slots(self: TrainerRank, checkpoints: Iterable[str]) -> None:
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
                raise _impl.TrainerRankSlotStateError(
                    f"Checkpoint {name!r} is not loaded consistently across ranks"
                )
            if not all(prefetched for _loaded, prefetched in states):
                raise _impl.TrainerRankSlotStateError(
                    f"Explicit selection references unloaded checkpoint {name!r}; "
                    "it has not been prefetched on every rank"
                )
            self._load_registered_checkpoint(name)


def load_checkpoint(
    self: TrainerRank, checkpoint: str | MaterializedCheckpoint | None
) -> None:
    self._guard_forward_collective("load_checkpoint")
    logical, source = self._checkpoint_source(checkpoint)
    with self._checkpoint_mutation_lock:
        if self._slot_stack:
            raise RuntimeError("Cannot load a checkpoint while one is pushed")
        if logical is None:
            self._set_default_slot(self._slot_ref(None))
            return
        assert source is not None
        if (
            isinstance(checkpoint, _impl.MaterializedCheckpoint)
            or logical not in self._checkpoint_prefetch_sources
        ):
            self._register_checkpoint_prefetch(logical, source)
        self._load_registered_checkpoint(logical)
        self._set_default_slot(self._slot_ref(logical))


def _discard_snapshot_checkpoint(self: TrainerRank, checkpoint: str) -> None:
    """Discard a forward-only resident snapshot."""
    from . import _checkpoint

    _checkpoint.discard_snapshot_checkpoint(self, checkpoint)


def _push_checkpoint(
    self: TrainerRank, checkpoint: str | MaterializedCheckpoint | None
) -> None:
    logical, source = self._checkpoint_source(checkpoint)
    self._push_checkpoint_sync(logical, source)


def _push_checkpoint_sync(
    self: TrainerRank, logical_path: str | None, source_path: str | None
) -> None:
    self._guard_forward_collective("push_checkpoint")
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


def pop_checkpoint(self: TrainerRank) -> None:
    with self._checkpoint_mutation_lock:
        if not self._slot_stack:
            raise RuntimeError("No pushed checkpoint to pop")
        self._slot_stack.pop()


def _resolve_checkpoint_name(
    self: TrainerRank, checkpoint_path: str | Literal["active"]
) -> str:
    if checkpoint_path != "active":
        self._ensure_checkpoint_slots((checkpoint_path,))
        return checkpoint_path
    ref = self._slot_stack[-1] if self._slot_stack else self._default_slot_ref
    if ref is None or ref.name is None:
        raise _impl.TrainerRankSlotStateError("No active trainable checkpoint")
    return ref.name


def _checkpoint_group(self: TrainerRank) -> dist.ProcessGroup | None:
    from ._checkpoint import _ensure_group

    return _ensure_group(self)


def _validate_checkpoint_adapter_config(
    self: TrainerRank,
    name: str,
    adapter_config: Mapping[str, object] | None,
    *,
    alpha: float | None,
) -> _AdapterConfig | None:
    config = None if adapter_config is None else deepcopy(dict(adapter_config))
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        gathered: list[dict[str, object] | None] = [None] * _impl.dist.get_world_size()
        _impl.dist.all_gather_object(gathered, config, group=self._checkpoint_group())
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
        raise TypeError("adapter_config['base_model_name_or_path'] must be a string")
    if base_model.startswith(("Qwen/Qwen3.5-", "Qwen/Qwen3.6-", "Qwen/Qwen3.8-")):
        from art.megatron.model_support.lora_disk import (
            model_attention_dimensions,
        )

        config.update(model_attention_dimensions(self.runtime.provider))
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
    return cast(_impl._AdapterConfig, config)


def _validate_loaded_checkpoint_config(
    self: TrainerRank, name: str, config: _AdapterConfig
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


def _load_checkpoint_slot(
    self: TrainerRank,
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


def _iter_slot_parameters(
    self: TrainerRank, ref: "LoRASlotRef"
) -> Iterator[torch.nn.Parameter]:
    from art.megatron.lora import iter_lora_slot_parameters

    return iter_lora_slot_parameters(self.runtime.model, ref)


def _validate_checkpoint_consistency(
    self: TrainerRank, name: str, loaded_sites: int, expected_keys: set[str]
) -> tuple[torch.nn.Parameter, ...]:
    params = tuple(self._iter_slot_parameters(self._slot_ref(name)))
    local_keys = {
        key for group in self._local_parameter_key_groups(name) for key in group
    }
    gathered = (
        [local_keys]
        if not (_impl.dist.is_available() and _impl.dist.is_initialized())
        else [None] * _impl.dist.get_world_size()
    )
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        _impl.dist.all_gather_object(
            gathered, local_keys, group=self._checkpoint_group()
        )
    covered = set().union(*(keys for keys in gathered if keys is not None))
    if loaded_sites < 1 or covered != expected_keys:
        raise _impl.TrainerRankSlotStateError(
            f"Checkpoint {name!r} has inconsistent distributed coverage"
        )
    return params


def _set_default_slot(self: TrainerRank, ref: "LoRASlotRef") -> None:
    if self._slot_stack:
        raise RuntimeError("Cannot select a checkpoint while one is pushed")
    self._default_slot_ref = ref


def _resolve_slot_ref(
    self: TrainerRank,
    request: AnyForwardInput,
    *,
    checkpoint: AdapterSelection = _impl.Unset,
) -> "LoRASlotRef | None":
    selection = (
        request.checkpoint if request.checkpoint is not _impl.Unset else checkpoint
    )
    if selection is not _impl.Unset:
        name = cast(str | None, selection)
        if name is not None and name not in self._checkpoint_slots:
            raise _impl.TrainerRankSlotStateError(
                f"Forward selects unloaded checkpoint {name!r}"
            )
        return self._slot_ref(name)
    if self._slot_stack:
        return self._slot_stack[-1]
    if self._default_slot_ref is not None:
        return self._default_slot_ref
    return self._slot_ref(None)


def _selected_dynamic_checkpoints(
    self: TrainerRank,
    checkpoints: Sequence[str] | None,
) -> tuple[str, ...]:
    if checkpoints is not None:
        self._ensure_checkpoint_slots(checkpoints)
    loaded = set(self._checkpoint_slots)
    if not loaded:
        raise _impl.TrainerRankSlotStateError(
            "TrainerRank.optim_step requires a loaded checkpoint slot. Call "
            "load_checkpoint(...) and run backward on outputs produced by "
            "that slot before stepping."
        )
    requested = (
        tuple(
            sorted(name for name in loaded if not self._checkpoint_slots[name].snapshot)
        )
        if checkpoints is None
        else tuple(dict.fromkeys(checkpoints))
    )
    if not requested:
        if checkpoints is None:
            raise _impl.TrainerRankSlotStateError(
                "TrainerRank.optim_step requires a loaded trainable checkpoint "
                "slot. Call load_checkpoint(...) and run backward on outputs "
                "produced by that slot before stepping."
            )
        raise _impl.TrainerRankSlotStateError(
            "TrainerRank.optim_step(checkpoints=...) received no checkpoint "
            "names. Pass at least one loaded checkpoint slot."
        )
    if unknown := set(requested) - loaded:
        raise ValueError(f"Unknown checkpoint slots: {sorted(unknown)}")
    if snapshots := [
        name for name in requested if self._checkpoint_slots[name].snapshot
    ]:
        raise _impl.TrainerRankSlotStateError(
            f"Snapshot checkpoints are forward-only and cannot be stepped: {snapshots}"
        )
    flags = self._checkpoint_grad_flags(requested)
    selected = tuple(
        name for name, has_grad in zip(requested, flags, strict=True) if has_grad
    )
    if checkpoints is None:
        if selected:
            return selected
        raise _impl.TrainerRankSlotStateError(
            "TrainerRank.optim_step found loaded checkpoint slots, but none "
            "have gradients on any rank. Call loss.backward() first."
        )
    if missing := [
        name for name, has_grad in zip(requested, flags, strict=True) if not has_grad
    ]:
        raise _impl.TrainerRankSlotStateError(
            "TrainerRank.optim_step was asked to step checkpoint slots with no "
            f"gradients on any rank: {missing}. Call loss.backward() for those "
            "slots first, or omit them from checkpoints=[...]."
        )
    return selected


def _checkpoint_grad_flags(self: TrainerRank, names: Sequence[str]) -> tuple[bool, ...]:
    flags = _impl.torch.tensor(
        [
            any(param.grad is not None for param in self._checkpoint_slots[name].params)
            for name in names
        ],
        device=self.device,
        dtype=_impl.torch.int32,
    )
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        _impl.dist.all_reduce(flags, op=_impl.dist.ReduceOp.MAX)
    return tuple(bool(flag) for flag in flags.tolist())


def _ensure_checkpoint_slots_for(
    self: TrainerRank,
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
                if request.checkpoint is not _impl.Unset
                else checkpoint
            )
        )
        is not _impl.Unset
        and selection is not None
    )


def _track_slot_graph_outputs(
    self: TrainerRank,
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
        return cast(_impl.torch.Tensor, _impl._SlotGraphSentinel.apply(tensor, marker))

    tracked_outputs = [
        _impl.ForwardOutput(
            target_logprobs=track(output.target_logprobs),
            top_k=(
                None
                if output.top_k is None
                else _impl.TopK(
                    logprobs=cast(_impl.torch.Tensor, track(output.top_k.logprobs)),
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


def _slot_graphs(
    self: TrainerRank,
) -> dict["LoRASlotRef", list[weakref.ReferenceType[torch.Tensor]]]:
    graphs = getattr(self, "_pending_slot_graphs", None)
    if graphs is None:
        graphs = {}
        self._pending_slot_graphs = graphs
    return graphs


def _prune_slot_graphs(self: TrainerRank, ref: "LoRASlotRef | None" = None) -> None:
    graphs = self._slot_graphs()
    refs = tuple(graphs) if ref is None else (ref,)
    for current in refs:
        live = [
            marker
            for marker in graphs.get(current, ())
            if _impl._graph_marker_is_live(marker)
        ]
        if live:
            graphs[current] = live
        else:
            graphs.pop(current, None)


def _has_live_slot_graph(self: TrainerRank, ref: "LoRASlotRef") -> bool:
    self._prune_slot_graphs(ref)
    return bool(self._slot_graphs().get(ref))


def _guard_slot_can_load(self: TrainerRank, ref: "LoRASlotRef") -> None:
    slot = None if ref.name is None else self._checkpoint_slots.get(ref.name)
    if slot is not None and slot.snapshot:
        raise _impl.TrainerRankSlotStateError(
            f"Cannot load over forward-only snapshot checkpoint {ref.name!r}"
        )
    if slot is not None and any(param.grad is not None for param in slot.params):
        raise _impl.TrainerRankSlotStateError(
            f"Cannot load checkpoint {ref.name!r} while it has accumulated "
            "gradients. Call optim_step() or zero_grad() before replacing it."
        )
    if not self._has_live_slot_graph(ref):
        return
    raise _impl.TrainerRankSlotStateError(
        f"Cannot load checkpoint {ref.name!r} while outputs from an "
        "earlier forward using that slot still have a live backward graph. "
        "Activation checkpoint recompute resolves slots by name, so replacing "
        "the slot before backward can compute gradients with different LoRA "
        "weights than the original forward. Finish backward first; if the "
        "forward was abandoned, release all references to its outputs; or load "
        "the new weights under a different slot name."
    )


def _guard_checkpoint_can_step(self: TrainerRank, name: str) -> None:
    if not self._has_live_slot_graph(self._slot_ref(name)):
        return
    raise _impl.TrainerRankSlotStateError(
        f"Cannot optim_step checkpoint slot {name!r} while outputs from an "
        "earlier forward using that slot have not been backpropagated. Call "
        "loss.backward() without retaining the graph before optim_step(); if "
        "the forward was abandoned, release all references to its outputs."
    )


def _guard_checkpoints_can_step(self: TrainerRank, names: Sequence[str]) -> None:
    local_live = [self._has_live_slot_graph(self._slot_ref(name)) for name in names]
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        live = _impl.torch.tensor(
            local_live,
            device=self.device,
            dtype=_impl.torch.int32,
        )
        _impl.dist.all_reduce(live, op=_impl.dist.ReduceOp.MAX)
        live_flags = live.tolist()
    else:
        live_flags = local_live
    blocked = [name for name, is_live in zip(names, live_flags, strict=True) if is_live]
    if not blocked:
        return
    raise _impl.TrainerRankSlotStateError(
        f"Cannot optim_step checkpoint slots {blocked!r} while outputs from an "
        "earlier forward using those slots have a live backward graph on at "
        "least one rank. Call loss.backward() without retaining the graph "
        "before optim_step(); if the forward was abandoned, release all "
        "references to its outputs; or pass on_live_graphs='allow' to accept "
        "responsibility for any retained graphs."
    )
