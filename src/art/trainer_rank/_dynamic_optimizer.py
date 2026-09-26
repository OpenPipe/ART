"""TrainerRank dynamic (per-checkpoint) optimizer management: optim_step,
its configuration guard, and dynamic optimizer creation, extension,
restore, padding masks and step flags.

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

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import TYPE_CHECKING, Any, Literal, cast

from art.trainer_rank import _impl

if TYPE_CHECKING:
    import torch

    from art.trainer_rank._checkpoint import LocalOptimizerState
    from art.trainer_rank._impl import AdamParams, TrainerRank, _DynamicOptimizer


def _extend_dynamic_optimizer(
    self: TrainerRank,
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
            _impl._validate_custom_optimizer_state(name, key, param, state)
        source = (
            param.detach().float() if state is None else state.master.to(param.device)
        )
        masters.append(_impl.torch.nn.Parameter(source.clone()))
    master_params = tuple(masters)
    all_masters = dynamic.master_params + master_params
    optimizer = _impl.torch.optim.AdamW(
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
                "step": _impl.torch.tensor(state.step, device=master.device),
                "exp_avg": state.exp_avg.to(master.device).clone(),
                "exp_avg_sq": state.exp_avg_sq.to(master.device).clone(),
            }
    return _impl._DynamicOptimizer(optimizer, all_masters)


def optim_step(
    self: TrainerRank,
    *,
    params: AdamParams | Mapping[str, AdamParams],
    scale_grads: float | Mapping[str, float] = 1.0,
    checkpoints: Sequence[str] | None = None,
    on_live_graphs: Literal["allow", "error"] = "allow",
) -> dict[str, float]:
    self._guard_forward_collective("optim_step")
    if on_live_graphs not in ("allow", "error"):
        raise ValueError(
            f"on_live_graphs must be either 'allow' or 'error', got {on_live_graphs!r}"
        )
    params_by_checkpoint = dict(params) if isinstance(params, Mapping) else None
    if params_by_checkpoint is not None:
        if not params_by_checkpoint:
            raise ValueError("params mapping must select at least one checkpoint")
        if any(not isinstance(name, str) for name in params_by_checkpoint):
            raise TypeError("params keys must be checkpoint names")
        if any(
            not isinstance(value, _impl.AdamParams)
            for value in params_by_checkpoint.values()
        ):
            raise TypeError("params values must be AdamParams")
    elif not isinstance(params, _impl.AdamParams):
        raise TypeError("params must be AdamParams or a mapping of checkpoint names")
    if isinstance(scale_grads, Mapping):
        raw_scales = cast(Mapping[object, object], scale_grads)
        if not raw_scales:
            raise ValueError("scale_grads mapping must select at least one checkpoint")
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
            "params, scale_grads, and checkpoints must select the same checkpoint names"
        )
    checkpoint_selection = (
        checkpoints
        if checkpoints is not None
        else sorted(configured[0])
        if configured
        else None
    )
    self._guard_optim_step_configuration(checkpoint_selection, params, on_live_graphs)
    selected_checkpoints = self._selected_dynamic_checkpoints(checkpoint_selection)
    params_by_checkpoint = (
        params_by_checkpoint
        if params_by_checkpoint is not None
        else dict.fromkeys(selected_checkpoints, cast(_impl.AdamParams, params))
    )
    scales_by_checkpoint = (
        scales_by_checkpoint
        if scales_by_checkpoint is not None
        else dict.fromkeys(selected_checkpoints, cast(float, scale_grads_value))
    )
    if on_live_graphs == "error":
        self._guard_checkpoints_can_step(selected_checkpoints)
    with _impl._telemetry_phase(
        "optim",
        {"checkpoint_count": len(selected_checkpoints)},
    ):
        return self._dynamic_optim_step(
            selected_checkpoints,
            params=params_by_checkpoint,
            scale_grads=scales_by_checkpoint,
        )


def _guard_optim_step_configuration(
    self: TrainerRank,
    checkpoints: Sequence[str] | None,
    params: AdamParams | Mapping[str, AdamParams],
    on_live_graphs: Literal["allow", "error"],
) -> None:
    if not (_impl.dist.is_available() and _impl.dist.is_initialized()):
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
                cast(Mapping[str, _impl.AdamParams], params).items()
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
    local = _impl.torch.tensor(
        tuple(digest), device=self.device, dtype=_impl.torch.uint8
    )
    gathered = [
        _impl.torch.empty_like(local) for _ in range(_impl.dist.get_world_size())
    ]
    _impl.dist.all_gather(gathered, local)
    if any(not _impl.torch.equal(value, local) for value in gathered):
        raise _impl.TrainerRankSlotStateError(
            "Optimizer checkpoint selection or AdamParams differ across ranks"
        )


def _dynamic_optim_step(
    self: TrainerRank,
    checkpoint_names: Sequence[str],
    *,
    params: Mapping[str, AdamParams],
    scale_grads: Mapping[str, float],
) -> dict[str, float]:
    self.runtime.model_support_handler.zero_internal_padding_grads(self.runtime.model)
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
            _impl._distributed_grad_norms(
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
        with _impl.torch.no_grad():
            for model, master in zip(model_params, dynamic.master_params, strict=True):
                model.copy_(master)
                model.grad = None
        self._prune_slot_graphs(self._slot_ref(name))
        self._checkpoint_slots[name].revision += 1
    return metrics


def _dynamic_param_step_flags(
    self: TrainerRank, params: Sequence[torch.nn.Parameter]
) -> tuple[bool, ...]:
    custom = [
        (index, param)
        for index, param in enumerate(params)
        if bool(getattr(param, "_art_custom_checkpoint_param", False))
    ]
    if not custom:
        return (True,) * len(params)
    flags = _impl.torch.tensor(
        [param.grad is not None for _, param in custom],
        device=self.device,
        dtype=_impl.torch.int32,
    )
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        _impl.dist.all_reduce(flags, op=_impl.dist.ReduceOp.MAX)
    result = [True] * len(params)
    for (index, _param), flag in zip(custom, flags.tolist(), strict=True):
        result[index] = bool(flag)
    return tuple(result)


def _dynamic_optimizer(
    self: TrainerRank,
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
    self: TrainerRank,
    name: str,
    params: AdamParams,
    *,
    master_params: Sequence[torch.Tensor] | None = None,
) -> _DynamicOptimizer:
    model_params = self._checkpoint_slots[name].params
    sources = model_params if master_params is None else tuple(master_params)
    if len(sources) != len(model_params) or any(
        not isinstance(source, _impl.torch.Tensor) for source in sources
    ):
        raise _impl.TrainerRankSlotStateError(
            f"Optimizer state for checkpoint slot {name!r} has "
            f"{len(sources)} master parameters; expected {len(model_params)}."
        )
    if any(
        tuple(source.shape) != tuple(model.shape)
        for source, model in zip(sources, model_params, strict=True)
    ):
        raise _impl.TrainerRankSlotStateError(
            f"Optimizer master parameter shape does not match checkpoint {name!r}"
        )
    masters = tuple(
        _impl.torch.nn.Parameter(
            source.detach().to(device=model.device, dtype=_impl.torch.float32).clone()
        )
        for model, source in zip(
            model_params,
            sources,
            strict=True,
        )
    )
    optimizer = _impl.torch.optim.AdamW(
        masters,
        lr=params.learning_rate,
        betas=(params.beta1, params.beta2),
        weight_decay=params.weight_decay,
    )
    dynamic = _impl._DynamicOptimizer(optimizer, masters)
    slot = self._checkpoint_slots[name]
    if slot.custom:
        from ._checkpoint import load_custom_optimizer

        named = tuple(
            pair
            for custom_name, custom in slot.custom.items()
            for pair in _impl._custom_named_parameters(custom_name, custom)
            if pair[1].requires_grad
        )
        lora_count = len(model_params) - len(named)
        restored = load_custom_optimizer(
            slot.custom_payload, tuple(key for key, _param in named)
        )
        for (key, _param), master in zip(named, masters[lora_count:], strict=True):
            state = restored.get(key)
            if state is not None:
                _impl._validate_custom_optimizer_state(name, key, _param, state)
                with _impl.torch.no_grad():
                    master.copy_(state.master.to(master.device))
                optimizer.state[master] = {
                    "step": _impl.torch.tensor(state.step, device=master.device),
                    "exp_avg": state.exp_avg.to(master.device).clone(),
                    "exp_avg_sq": state.exp_avg_sq.to(master.device).clone(),
                }
    return dynamic


def _restore_canonical_optimizer(
    self: TrainerRank,
    name: str,
    state: "LocalOptimizerState",
) -> _DynamicOptimizer:
    dynamic = self._new_dynamic_optimizer(
        name,
        _impl.AdamParams(
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
            raise _impl.TrainerRankSlotStateError(
                f"Canonical optimizer moment shape does not match {name!r}"
            )
        dynamic.optimizer.state[master] = {
            "step": _impl.torch.tensor(step, dtype=_impl.torch.float32),
            "exp_avg": exp_avg.to(master.device, _impl.torch.float32).clone(),
            "exp_avg_sq": exp_avg_sq.to(master.device, _impl.torch.float32).clone(),
        }
    self._zero_dynamic_optimizer_padding(name, dynamic)
    return dynamic


def _zero_dynamic_optimizer_padding(
    self: TrainerRank,
    name: str,
    dynamic: _DynamicOptimizer,
) -> None:
    masks = self._dynamic_optimizer_padding_masks(name)
    with _impl.torch.no_grad():
        for param, mask in zip(dynamic.master_params, masks, strict=True):
            param.masked_fill_(mask, 0)
            for value in dynamic.optimizer.state.get(param, {}).values():
                if isinstance(value, _impl.torch.Tensor) and value.shape == param.shape:
                    value.masked_fill_(mask, 0)


def _dynamic_optimizer_padding_masks(
    self: TrainerRank, name: str
) -> tuple[torch.Tensor, ...]:
    params = self._checkpoint_slots[name].params
    masks = tuple(
        _impl.torch.zeros_like(param, dtype=_impl.torch.bool) for param in params
    )
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
                        raise _impl.TrainerRankSlotStateError(
                            f"Cannot map optimizer padding for checkpoint "
                            f"{name!r}: {len(keys)} adapter keys describe "
                            f"{int(param.shape[0])} local experts."
                        )
                    for expert, key in enumerate(keys):
                        exported[str(key)] = _impl.torch.ones_like(param[expert].T)
                        owners[str(key)] = (index, expert)
                elif len(keys) == 1:
                    key = str(keys[0])
                    exported[key] = _impl.torch.ones_like(param.T)
                    owners[key] = (index, None)
                else:
                    raise _impl.TrainerRankSlotStateError(
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
        raise _impl.TrainerRankSlotStateError(
            f"Cannot map optimizer padding for checkpoint {name!r}: parameter "
            f"indices {missing} do not belong to installed LoRA sites."
        )

    canonical = self.runtime.model_support_handler.canonicalize_loaded_lora_state(
        exported, self.runtime.model
    )
    for key, value in canonical.items():
        owner = owners.get(key)
        if owner is None or not isinstance(value, _impl.torch.Tensor):
            continue
        index, expert = owner
        mask = value.T == 0
        if expert is None:
            masks[index].copy_(mask)
        else:
            masks[index][expert].copy_(mask)
    return masks
