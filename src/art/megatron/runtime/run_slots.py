"""Megatron-owned state for production dynamic LoRA run slots."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
import threading
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

import torch
import torch.distributed as dist

from .. import checkpoint as _checkpoint

if TYPE_CHECKING:
    from art.megatron.lora import LoRASlotRef


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-13
    weight_decay: float = 0.1
    grad_clip_norm: float = 0.1


class MegatronRunSlotStateError(RuntimeError):
    pass


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


class MegatronRunSlots:
    """Own exact-shape LoRA slots and their independent optimizer state."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.device = next(runtime.model[0].parameters()).device
        self._checkpoint_slots: dict[str, _checkpoint.CheckpointSlot] = {}
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
        self._prepared_checkpoint_saves: dict[str, Any] = {}
        self._finalized_checkpoint_saves: OrderedDict[str, Any] = OrderedDict()

    def load_checkpoint_for_residency(
        self, name: str, directory: str
    ) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.Tensor, ...]]:
        """Prepare one complete initial run working set in CPU residency."""

        source = self._prepare_checkpoint(directory)
        return self.install_prepared_checkpoint_for_residency(
            name,
            source,
            restore_optimizer=True,
            require_optimizer=False,
        )

    def _prepare_checkpoint(self, directory: str) -> _checkpoint.PreparedCheckpoint:
        """Read and validate one checkpoint consistently across trainer ranks."""

        source: _checkpoint.PreparedCheckpoint | None = None
        error: BaseException | None = None
        try:
            source = _checkpoint.prepare_checkpoint(directory)
            if source.manifest is not None and source.manifest.get("custom_tensors"):
                raise MegatronRunSlotStateError(
                    "Megatron run slots do not support custom checkpoint tensors"
                )
        except BaseException as exc:
            error = exc
        group = _checkpoint._ensure_group(self)
        _checkpoint.raise_distributed(error, "prepare checkpoint", group)
        assert source is not None
        return source

    def install_prepared_checkpoint_for_residency(
        self,
        name: str,
        source: _checkpoint.PreparedCheckpoint,
        *,
        restore_optimizer: bool,
        require_optimizer: bool,
        materialize: Callable[[str], AbstractContextManager[Any]] | None = None,
    ) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.Tensor, ...]]:
        """Install CPU weights and optimizer objects for atomic L2 admission."""

        manifest = source.manifest
        weights_source = source
        if manifest is not None and manifest["optimizer"] is not None:
            weights_source = replace(
                source,
                manifest=cast(
                    Any,
                    {
                        **manifest,
                        "optimizer": None,
                        "parameters": {},
                        "steps": {},
                    },
                ),
            )
        adapter = (
            nullcontext()
            if materialize is None
            else materialize("adapter_model.safetensors")
        )
        with adapter:
            _checkpoint.load_checkpoint(self, weights_source, name)
        try:
            weights = self.checkpoint_slot_parameters(name)
            move_error: BaseException | None = None
            try:
                with torch.no_grad():
                    for tensor in weights:
                        tensor.data = tensor.detach().to(device="cpu")
            except BaseException as error:
                move_error = error
            _checkpoint.raise_distributed(
                move_error,
                "prepare checkpoint weights for CPU residency",
                _checkpoint._ensure_group(self),
            )

            has_optimizer = manifest is not None and manifest["optimizer"] is not None
            if restore_optimizer and has_optimizer:
                optimizer = self.prepare_checkpoint_slot_optimizer_for_residency(
                    name, source, materialize=materialize
                )
            elif restore_optimizer and require_optimizer:
                raise MegatronRunSlotStateError(
                    "Checkpoint does not contain the required optimizer state"
                )
            else:
                optimizer = self.prepare_fresh_checkpoint_slot_optimizer_for_residency(
                    name, OptimizerConfig(learning_rate=0.0)
                )
            tensors = (*weights, *optimizer)
            if (
                not weights
                or not optimizer
                or any(tensor.device.type != "cpu" for tensor in tensors)
            ):
                raise MegatronRunSlotStateError(
                    "Prepared run working set is not entirely CPU resident"
                )
            return weights, optimizer
        except BaseException:
            self.release_checkpoint_slot(name)
            raise

    def save_checkpoint(self, output_dir: str, checkpoint_path: str) -> None:
        self.prepare_checkpoint_save(output_dir, checkpoint_path)
        self.finish_checkpoint_save(output_dir)

    def prepare_checkpoint_save(self, output_dir: str, checkpoint_path: str) -> None:
        _checkpoint.prepare_checkpoint_save(self, output_dir, checkpoint_path)

    def finish_checkpoint_save(self, output_dir: str) -> None:
        _checkpoint.finish_checkpoint_save(self, output_dir)

    def abort_checkpoint_save(self, output_dir: str) -> None:
        _checkpoint.abort_checkpoint_save(self, output_dir)

    def checkpoint_slot_parameters(self, name: str) -> tuple[torch.nn.Parameter, ...]:
        try:
            return self._checkpoint_slots[name].params
        except KeyError as error:
            raise ValueError(f"Unknown checkpoint slot: {name!r}") from error

    def checkpoint_slot_optimizer_tensors(self, name: str) -> tuple[torch.Tensor, ...]:
        try:
            dynamic = self._checkpoint_slots[name].optimizer
        except KeyError as error:
            raise ValueError(f"Unknown checkpoint slot: {name!r}") from error
        if dynamic is None:
            return ()

        tensors: list[torch.Tensor] = []
        seen: set[int] = set()
        for master in dynamic.master_params:
            if id(master) not in seen:
                tensors.append(master)
                seen.add(id(master))
            state = dynamic.optimizer.state.get(master, {})
            for key in sorted(state):
                value = state[key]
                if (
                    isinstance(value, torch.Tensor)
                    and (key != "step" or value.device.type == "cuda")
                    and id(value) not in seen
                ):
                    tensors.append(value)
                    seen.add(id(value))
        return tuple(tensors)

    def prepare_fresh_checkpoint_slot_optimizer_for_residency(
        self, name: str, params: OptimizerConfig
    ) -> tuple[torch.Tensor, ...]:
        """Build a complete fresh optimizer while the run weights remain on CPU."""

        weights = self.checkpoint_slot_parameters(name)
        slot = self._checkpoint_slots[name]
        if slot.optimizer is not None:
            raise MegatronRunSlotStateError(
                f"Checkpoint slot {name!r} already has optimizer state"
            )
        if not weights or any(weight.device.type != "cpu" for weight in weights):
            raise MegatronRunSlotStateError(
                "Fresh optimizer preparation requires non-empty CPU weights"
            )
        dynamic = _checkpoint.build_adam_optimizer(
            weights,
            self._optimizer_config(params),
            padding_masks=self._optimizer_padding_masks(name),
        )
        slot.optimizer = dynamic
        tensors = self.checkpoint_slot_optimizer_tensors(name)
        if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
            slot.optimizer = None
            raise MegatronRunSlotStateError(
                "Fresh optimizer residency is not entirely CPU resident"
            )
        return tensors

    def prepare_checkpoint_slot_optimizer_for_residency(
        self,
        name: str,
        source: _checkpoint.PreparedCheckpoint,
        *,
        materialize: Callable[[str], AbstractContextManager[Any]] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Restore exact optimizer state against an installed CPU adapter."""

        slot = self._checkpoint_slots.get(name)
        if slot is None:
            raise ValueError(f"Unknown checkpoint slot: {name!r}")
        if slot.optimizer is not None:
            raise MegatronRunSlotStateError(
                f"Checkpoint slot {name!r} already has optimizer state"
            )
        if source.manifest is None or source.manifest["optimizer"] is None:
            raise MegatronRunSlotStateError(
                "Checkpoint does not contain optimizer state"
            )
        optimizer_state = _checkpoint._phase(
            lambda: (
                _checkpoint._optimizer_state(self, source, name)
                if materialize is None
                else _checkpoint._bounded_optimizer_state(
                    self, source, name, materialize
                )
            ),
            "prepare checkpoint optimizer state",
            _checkpoint._ensure_group(self),
        )
        dynamic = _checkpoint._phase(
            lambda: _checkpoint.build_adam_optimizer(
                slot.params,
                optimizer_state.config,
                restored=optimizer_state,
                padding_masks=self._optimizer_padding_masks(name),
            ),
            "restore checkpoint optimizer for CPU residency",
            _checkpoint._ensure_group(self),
        )
        slot.optimizer = dynamic
        tensors = self.checkpoint_slot_optimizer_tensors(name)
        if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
            slot.optimizer = None
            raise MegatronRunSlotStateError(
                "Restored optimizer residency is not entirely CPU resident"
            )
        return tensors

    def clear_checkpoint_slot_grads(self, name: str) -> None:
        for parameter in self.checkpoint_slot_parameters(name):
            parameter.grad = None

    def release_checkpoint_slot(self, name: str) -> None:
        ref = self._slot_ref(name)
        self._guard_slot_can_load(ref)
        from art.megatron.lora import LoRA

        for chunk in self.runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA):
                    continue
                key = module._slot_keys.pop(ref, None)
                if key is not None:
                    del module._slot_modules[key]
        self._checkpoint_slots.pop(name, None)

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

    def reduce_checkpoint_slot_grads(
        self,
        name: str,
        gradients: Sequence[torch.Tensor],
        *,
        scale_grads: float,
    ) -> tuple[torch.Tensor, ...]:
        params = self.checkpoint_slot_parameters(name)
        gradients = tuple(gradients)
        if len(gradients) != len(params) or any(
            tuple(gradient.shape) != tuple(parameter.shape)
            for gradient, parameter in zip(gradients, params, strict=True)
        ):
            raise ValueError("gradient layout does not match checkpoint slot")
        return self._reduce_gradient_tensors(params, gradients, scale_grads=scale_grads)

    def optim_step_reduced(
        self,
        name: str,
        *,
        params: OptimizerConfig,
        gradients: Sequence[torch.Tensor],
        step_flags: Sequence[bool],
    ) -> dict[str, float]:
        model_params = self.checkpoint_slot_parameters(name)
        gradients = tuple(gradients)
        step_flags = tuple(step_flags)
        if len(gradients) != len(model_params) or any(
            tuple(gradient.shape) != tuple(parameter.shape)
            for gradient, parameter in zip(gradients, model_params, strict=True)
        ):
            raise ValueError("reduced gradient layout does not match checkpoint slot")
        if len(step_flags) != len(model_params):
            raise ValueError("gradient step flags do not match checkpoint slot")

        for gradient, mask in zip(
            gradients, self._optimizer_padding_masks(name), strict=True
        ):
            gradient.masked_fill_(mask, 0)
        grad_norm = _distributed_grad_norm(model_params, gradients)
        if not torch.isfinite(torch.tensor(grad_norm)):
            for slot in self._checkpoint_slots.values():
                for parameter in slot.params:
                    parameter.grad = None
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
        dynamic = self._dynamic_optimizer(name, params)
        for master, gradient, should_step in zip(
            dynamic.master_params, gradients, step_flags, strict=True
        ):
            master.grad = gradient.mul(clip) if should_step else None
        dynamic.optimizer.step()
        dynamic.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for model, master in zip(model_params, dynamic.master_params, strict=True):
                model.copy_(master)
                model.grad = None
        self._checkpoint_slots[name].revision += 1
        return {
            "learning_rate": float(params.learning_rate),
            "grad_norm": float(grad_norm),
            "update_successful": 1.0,
            "num_zeros_in_grad": 0.0,
        }

    @staticmethod
    def _slot_state_error(message: str) -> MegatronRunSlotStateError:
        return MegatronRunSlotStateError(message)

    def _checkpoint_group(self) -> dist.ProcessGroup | None:
        return _checkpoint._ensure_group(self)

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

    @staticmethod
    def _slot_ref(name: str | None) -> LoRASlotRef:
        from art.megatron.lora import LoRASlotRef

        return LoRASlotRef(kind="checkpoint", name=name)

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
        slot_state = self._checkpoint_slots[name]
        if slot_state.custom or slot_state.custom_payload is not None:
            raise MegatronRunSlotStateError(
                "Megatron run slots do not support custom checkpoint tensors"
            )
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
        self,
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
        *,
        alpha: float,
        _prepared: bool = False,
    ) -> int:
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
        prepared = (
            local_state
            if canonicalized
            else self.runtime.model_support_handler.canonicalize_loaded_lora_state(
                local_state, self.runtime.model
            )
        )
        if set(prepared) != set(local_state):
            raise MegatronRunSlotStateError(
                "Model-specific LoRA canonicalization changed the adapter key set "
                f"for checkpoint {name!r}."
            )
        return {
            key: tensor.to(
                device=templates[key].device,
                dtype=templates[key].dtype,
                non_blocking=True,
            )
            for key, tensor in prepared.items()
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

    def _iter_slot_parameters(self, ref: LoRASlotRef) -> Iterator[torch.nn.Parameter]:
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
        gathered: list[set[str] | None] = (
            [local_keys]
            if not (dist.is_available() and dist.is_initialized())
            else [None] * dist.get_world_size()
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered, local_keys, group=self._checkpoint_group())
        covered = set().union(*(keys for keys in gathered if keys is not None))
        if loaded_sites < 1 or covered != expected_keys:
            raise MegatronRunSlotStateError(
                f"Checkpoint {name!r} has inconsistent distributed coverage"
            )
        return params

    def _guard_slot_can_load(self, ref: LoRASlotRef) -> None:
        slot = None if ref.name is None else self._checkpoint_slots.get(ref.name)
        if slot is not None and any(param.grad is not None for param in slot.params):
            raise MegatronRunSlotStateError(
                f"Cannot load checkpoint {ref.name!r} while it has accumulated "
                "gradients. Finish or discard the current command before replacing it."
            )

    def _dynamic_optimizer(
        self, name: str, params: OptimizerConfig
    ) -> _checkpoint.DynamicOptimizer:
        try:
            slot = self._checkpoint_slots[name]
        except KeyError as error:
            raise ValueError(f"Unknown checkpoint slot: {name!r}") from error
        dynamic = slot.optimizer
        if dynamic is None:
            dynamic = _checkpoint.build_adam_optimizer(
                slot.params,
                self._optimizer_config(params),
                padding_masks=self._optimizer_padding_masks(name),
            )
            slot.optimizer = dynamic
            return dynamic
        for group in dynamic.optimizer.param_groups:
            group["lr"] = params.learning_rate
            group["betas"] = (params.beta1, params.beta2)
            group["eps"] = params.eps
            group["weight_decay"] = params.weight_decay
        _checkpoint.zero_optimizer_padding(dynamic, self._optimizer_padding_masks(name))
        return dynamic

    @staticmethod
    def _optimizer_config(params: OptimizerConfig) -> _checkpoint.OptimizerConfig:
        return {
            "learning_rate": params.learning_rate,
            "beta1": params.beta1,
            "beta2": params.beta2,
            "eps": params.eps,
            "weight_decay": params.weight_decay,
        }

    def _optimizer_padding_masks(self, name: str) -> tuple[torch.Tensor, ...]:
        slot = self._checkpoint_slots[name]
        if slot.custom or slot.custom_payload is not None:
            raise MegatronRunSlotStateError(
                "Megatron run slots do not support custom checkpoint tensors"
            )
        params = self._checkpoint_slots[name].params
        return _checkpoint.lora_optimizer_padding_masks(
            params,
            model_chunks=self.runtime.model,
            slot_ref=self._slot_ref(name),
            canonicalize_loaded_state=(
                self.runtime.model_support_handler.canonicalize_loaded_lora_state
            ),
        )

    def _reduce_gradient_tensors(
        self,
        params: Sequence[torch.nn.Parameter],
        gradients: Sequence[torch.Tensor],
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
            gradient: torch.Tensor,
        ) -> None:
            key = (id(group), str(op), gradient.dtype, gradient.device)
            buckets.setdefault(key, (group, op, []))[2].append(gradient)

        reduced = tuple(
            gradient.detach().float().mul(scale_grads) for gradient in gradients
        )
        for param, gradient in zip(params, reduced, strict=True):
            if bool(getattr(param, "allreduce", True)):
                group = ps.get_data_parallel_group(with_context_parallel=True)
            else:
                group = ps.get_expert_data_parallel_group()
            if group is not None and group.size() > 1:
                add(group, dist.ReduceOp.SUM, gradient)

            sync = tensor_parallel_grad_sync(param, name="dynamic LoRA")
            if sync is not None:
                group, reduce_op = sync
                add(group, reduce_op, gradient)

        for group, op, gradients_in_bucket in buckets.values():
            coalesced_all_reduce(gradients_in_bucket, group=group, op=op)
        return reduced


def _distributed_grad_norm(
    params: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor]
) -> float:
    if len(params) != len(gradients):
        raise ValueError("params and gradients must have matching lengths")
    included = [
        gradient
        for param, gradient in zip(params, gradients, strict=True)
        if _include_in_distributed_grad_norm(param)
    ]
    device = gradients[0].device if gradients else torch.device("cpu")
    squared = torch.zeros((), device=device, dtype=torch.float32)
    for gradient in included:
        squared.add_(gradient.float().square().sum())
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


__all__ = [
    "MegatronRunSlotStateError",
    "MegatronRunSlots",
    "OptimizerConfig",
]
