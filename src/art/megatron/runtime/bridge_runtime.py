from __future__ import annotations

from collections.abc import Iterable, Mapping
import contextlib
import fnmatch
from typing import Any, cast

from megatron.bridge.models.common.unimodal import to_empty_if_meta_device
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    get_module_and_param_from_name,
)
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.utils import get_model_config
import torch

from art.megatron.model_support.spec import HfWeightSource


class ExpertTensorSlice:
    __slots__ = ("global_start", "global_stop", "tensor")

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        global_start: int,
        global_stop: int,
    ) -> None:
        self.tensor = tensor
        self.global_start = int(global_start)
        self.global_stop = int(global_stop)

    def get(self, global_expert: int) -> torch.Tensor:
        global_expert = int(global_expert)
        if not self.global_start <= global_expert < self.global_stop:
            raise RuntimeError(
                "expert slice cache miss for global expert "
                f"{global_expert}; cached range is "
                f"[{self.global_start}, {self.global_stop})"
            )
        return self.tensor[global_expert - self.global_start]

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def __getitem__(self, index: Any) -> torch.Tensor:
        if isinstance(index, int):
            return self.get(index)
        if isinstance(index, tuple) and index and isinstance(index[0], int):
            return self.get(index[0])[index[1:]]
        return self.tensor[index]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tensor, name)


def _pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or not torch.cuda.is_available():
        return tensor
    try:
        return tensor if tensor.is_pinned() else tensor.pin_memory()
    except RuntimeError:
        return tensor


def _iter_hf_param_names(hf_param: Any) -> Iterable[str]:
    if isinstance(hf_param, str):
        yield hf_param
        return
    if isinstance(hf_param, Mapping):
        for value in hf_param.values():
            yield from _iter_hf_param_names(value)


def _needs_local_hf_prefetch(task: Any) -> bool:
    if task is None or task.megatron_module is None:
        return False
    if _needs_expert_slice_prefetch(task):
        return False
    mapping = task.mapping
    # ART Qwen3.5 expert mappings slice the full HF expert tensor before
    # delegating to the inner TP mapping, so every ETP rank needs the source.
    if type(mapping).__name__ in {
        "_ArtExpertMLPGateUpProjMapping",
        "_ArtExpertMLPDownProjMapping",
    }:
        return True
    tp_size = int(getattr(mapping, "tp_size", 1))
    if tp_size <= 1:
        return True
    if type(mapping).__name__ == "DirectMapping":
        return True
    return int(getattr(mapping, "tp_rank", 0)) == 0


def _needs_expert_slice_prefetch(task: Any) -> bool:
    mapping = task.mapping
    return (
        int(getattr(mapping, "ep_size", 1)) > 1
        and bool(getattr(mapping, "is_expert", False))
        and bool(getattr(mapping, "is_grouped_export", False))
        and isinstance(getattr(mapping, "hf_param", None), str)
    )


def _expert_slice_range(task: Any) -> tuple[int, int]:
    mapping = task.mapping
    config = getattr(task.megatron_module, "config", None)
    num_experts = int(getattr(config, "num_moe_experts", 0) or 0)
    ep_size = int(getattr(mapping, "ep_size", 1))
    ep_rank = int(getattr(mapping, "ep_rank", 0))
    if num_experts <= 0 or ep_size <= 1 or num_experts % ep_size != 0:
        raise RuntimeError(
            "cannot slice fused expert HF weights with "
            f"num_experts={num_experts}, ep_size={ep_size}"
        )
    experts_per_rank = num_experts // ep_size
    start = ep_rank * experts_per_rank
    return start, start + experts_per_rank


def _load_hf_tensor_slice(
    hf_state_dict: Mapping[str, torch.Tensor],
    key: str,
    *,
    start: int,
    stop: int,
) -> torch.Tensor:
    source = getattr(hf_state_dict, "source", None)
    if source is None or not hasattr(source, "key_to_filename_map"):
        raise RuntimeError(
            "fused expert EP loading requires a safetensors-backed HF state "
            f"dict for key {key!r}"
        )
    key_to_filename = source.key_to_filename_map
    if key not in key_to_filename:
        raise KeyError(f"HF tensor key {key!r} not found in safetensors index")
    from safetensors import safe_open

    file_path = source.path / key_to_filename[key]
    with safe_open(file_path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        shape = tuple(int(dim) for dim in tensor_slice.get_shape())
        if not shape or start < 0 or stop > shape[0] or start >= stop:
            raise RuntimeError(
                f"invalid expert slice [{start}, {stop}) for {key!r} with shape {shape}"
            )
        index = (slice(start, stop),) + (slice(None),) * (len(shape) - 1)
        return tensor_slice[index]


def _direct_hf_weight_source(key: str) -> HfWeightSource:
    return HfWeightSource(logical_key=key, physical_key_options=((key,),))


def _planned_hf_weight_source(
    bridge: MegatronModelBridge | None,
    key: str,
    *,
    task: Any | None,
) -> HfWeightSource:
    source_fn = (
        None if bridge is None else getattr(bridge, "_art_hf_weight_source", None)
    )
    source = (
        None
        if source_fn is None
        else cast(HfWeightSource | None, source_fn(key, task=task))
    )
    if source is None:
        return _direct_hf_weight_source(key)
    if source.logical_key != key:
        raise RuntimeError(
            f"handler returned HF source for {source.logical_key!r} while loading {key!r}"
        )
    if not source.physical_key_options or any(
        not option for option in source.physical_key_options
    ):
        raise RuntimeError(f"handler returned empty HF source options for {key!r}")
    return source


def _source_options_message(source: HfWeightSource) -> str:
    return ", ".join(str(option) for option in source.physical_key_options)


def _select_physical_key_option(
    source: HfWeightSource,
    hf_state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    for option in source.physical_key_options:
        if all(key in hf_state_dict for key in option):
            return option
    raise KeyError(
        f"HF tensor source for {source.logical_key!r} not found; "
        f"tried {_source_options_message(source)}"
    )


def _materialize_hf_weight_source(
    bridge: MegatronModelBridge | None,
    source: HfWeightSource,
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    selected_option: tuple[str, ...],
) -> torch.Tensor:
    if source.kind == "direct":
        if len(selected_option) != 1:
            raise RuntimeError(
                "direct HF source must select exactly one physical key for "
                f"{source.logical_key!r}; got {selected_option!r}"
            )
        return hf_state_dict[selected_option[0]]
    if source.kind == "bridge_materialized":
        if bridge is None:
            raise RuntimeError(
                f"HF source for {source.logical_key!r} requires Megatron Bridge"
            )
        return bridge.maybe_modify_loaded_hf_weight(source.logical_key, hf_state_dict)
    raise RuntimeError(
        f"unknown HF source kind {source.kind!r} for {source.logical_key!r}"
    )


def load_unique_hf_keys_once(
    tasks: Iterable[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    bridge: MegatronModelBridge | None = None,
) -> dict[str, torch.Tensor | ExpertTensorSlice]:
    task_list = list(tasks)
    prefetch_task_by_key: dict[str, Any] = {}
    for task in task_list:
        if not _needs_local_hf_prefetch(task):
            continue
        for key in _iter_hf_param_names(task.mapping.hf_param):
            prefetch_task_by_key.setdefault(key, task)
    keys = sorted(prefetch_task_by_key)
    expert_slice_ranges: dict[str, tuple[int, int]] = {}
    expert_slice_task_by_key: dict[str, Any] = {}
    for task in task_list:
        if task is None or task.megatron_module is None:
            continue
        if not _needs_expert_slice_prefetch(task):
            continue
        start, stop = _expert_slice_range(task)
        key = cast(str, task.mapping.hf_param)
        previous = expert_slice_ranges.get(key)
        expert_slice_ranges[key] = (
            (start, stop)
            if previous is None
            else (min(previous[0], start), max(previous[1], stop))
        )
        expert_slice_task_by_key.setdefault(key, task)
    cache: dict[str, torch.Tensor | ExpertTensorSlice] = {}
    direct_physical_by_logical: dict[str, str] = {}
    materialized_source_by_key: dict[str, tuple[HfWeightSource, tuple[str, ...]]] = {}
    for key in keys:
        source = _planned_hf_weight_source(
            bridge,
            key,
            task=prefetch_task_by_key.get(key),
        )
        selected_option = _select_physical_key_option(source, hf_state_dict)
        if source.kind == "direct":
            if len(selected_option) != 1:
                raise RuntimeError(
                    "direct HF source must select exactly one physical key for "
                    f"{source.logical_key!r}; got {selected_option!r}"
                )
            direct_physical_by_logical[key] = selected_option[0]
        else:
            materialized_source_by_key[key] = (source, selected_option)

    physical_direct_keys = sorted(set(direct_physical_by_logical.values()))
    if physical_direct_keys and hasattr(hf_state_dict, "__getitem__"):
        hf_state_dict_getter = cast(Any, hf_state_dict)
        loaded = (
            hf_state_dict_getter[physical_direct_keys]
            if not isinstance(hf_state_dict, dict)
            else {key: hf_state_dict[key] for key in physical_direct_keys}
        )
    else:
        loaded = {key: hf_state_dict[key] for key in physical_direct_keys}
    loaded_direct = cast(Mapping[str, torch.Tensor], loaded)
    cache.update(
        {
            logical_key: _pin_cpu_tensor(loaded_direct[physical_key])
            for logical_key, physical_key in direct_physical_by_logical.items()
        }
    )
    for key, (source, selected_option) in materialized_source_by_key.items():
        cache[key] = _pin_cpu_tensor(
            _materialize_hf_weight_source(
                bridge,
                source,
                hf_state_dict,
                selected_option=selected_option,
            )
        )
    for key, (start, stop) in expert_slice_ranges.items():
        source = _planned_hf_weight_source(
            bridge,
            key,
            task=expert_slice_task_by_key.get(key),
        )
        selected_option = _select_physical_key_option(source, hf_state_dict)
        if source.kind != "direct":
            tensor = _materialize_hf_weight_source(
                bridge,
                source,
                hf_state_dict,
                selected_option=selected_option,
            )
            if not tensor.ndim or start < 0 or stop > tensor.shape[0] or start >= stop:
                raise RuntimeError(
                    f"invalid expert slice [{start}, {stop}) for {key!r} "
                    f"with shape {tuple(tensor.shape)}"
                )
            cache[key] = ExpertTensorSlice(
                _pin_cpu_tensor(tensor[start:stop]),
                global_start=start,
                global_stop=stop,
            )
            continue
        if len(selected_option) != 1:
            raise RuntimeError(
                "direct HF source must select exactly one physical key for "
                f"{source.logical_key!r}; got {selected_option!r}"
            )
        cache[key] = ExpertTensorSlice(
            _pin_cpu_tensor(
                _load_hf_tensor_slice(
                    hf_state_dict,
                    selected_option[0],
                    start=start,
                    stop=stop,
                )
            ),
            global_start=start,
            global_stop=stop,
        )
    return cache


class _CachedStateLookup(Mapping[str, torch.Tensor | ExpertTensorSlice]):
    def __init__(
        self,
        *,
        cache: Mapping[str, torch.Tensor | ExpertTensorSlice],
        source: Mapping[str, torch.Tensor],
    ) -> None:
        self._cache = cache
        self._source = source

    def __getitem__(self, key: str) -> torch.Tensor | ExpertTensorSlice:
        if key in self._cache:
            return self._cache[key]
        return _pin_cpu_tensor(self._source[key])

    def __iter__(self):
        seen = set(self._cache)
        yield from self._cache
        for key in self._source:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(set(self._cache).union(self._source))


def _materialization_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _apply_pre_wrap_hook(
    model: list[MegatronModule],
    pre_wrap_hook: Any,
) -> list[MegatronModule]:
    if pre_wrap_hook is None:
        return model
    if not callable(pre_wrap_hook):
        raise RuntimeError("pre_wrap_hook must be callable")
    updated = pre_wrap_hook(model)
    return model if updated is None else updated


def _set_tp_attrs(model: list[MegatronModule]) -> None:
    from megatron.core import tensor_parallel

    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )


def _wrap_with_mp_wrapper(
    model: list[MegatronModule],
    model_config: Any,
    mixed_precision_wrapper: Any,
) -> list[MegatronModule]:
    if not (model_config.fp16 or model_config.bf16) or mixed_precision_wrapper is None:
        return model
    keep_in_fp32: list[tuple[Any, torch.Tensor]] = []
    for model_module in model:
        for submodule in model_module.modules():
            if hasattr(submodule, "_maintain_float32_expert_bias"):
                expert_bias = getattr(submodule, "expert_bias", None)
                if expert_bias is not None:
                    keep_in_fp32.append((submodule, expert_bias.data.clone()))
    wrapped = [
        mixed_precision_wrapper(model_config, model_module) for model_module in model
    ]
    for submodule, fp32_data in keep_in_fp32:
        submodule.expert_bias.data = fp32_data
    return wrapped


def _art_get_model(
    model_provider: ModelProviderMixin,
    ddp_config: DistributedDataParallelConfig,
    model_type=ModelType.encoder_or_decoder,
    overlap_param_gather_with_optimizer_step: bool = False,
    fp16: bool | None = None,
    bf16: bool | None = None,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    use_cpu_initialization: None | bool = False,
    init_model_with_meta_device: bool | None = None,
    pre_wrap_hook: Any = None,
    mixed_precision_wrapper: Any = Float16Module,
    *,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    from megatron.bridge.models import model_provider as model_provider_module

    if fp16:
        setattr(model_provider, "fp16", fp16)
    if bf16:
        setattr(model_provider, "bf16", bf16)

    setattr(model_provider, "use_cpu_initialization", bool(use_cpu_initialization))
    if init_model_with_meta_device:
        setattr(model_provider, "init_model_with_meta_device", True)
        with torch.device("meta"):
            model = model_provider_module._create_model(
                model_provider,
                model_type,
                pg_collection=pg_collection,
            )
    else:
        model = model_provider_module._create_model(
            model_provider,
            model_type,
            pg_collection=pg_collection,
        )

    if init_model_with_meta_device and not use_torch_fsdp2 and not use_megatron_fsdp:
        device = _materialization_device()
        model = [
            to_empty_if_meta_device(model_module, device=device)
            for model_module in model
        ]

    model = _apply_pre_wrap_hook(model, pre_wrap_hook)
    _set_tp_attrs(model)
    model_provider_module._print_num_params(model, pg_collection=pg_collection)
    model_config = get_model_config(model[0])

    if (
        not use_torch_fsdp2
        and not model_config.use_cpu_initialization
        and not model_config.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    model = _wrap_with_mp_wrapper(model, model_config, mixed_precision_wrapper)
    if model_provider_module.correct_amax_history_if_needed is not None:
        model_provider_module.correct_amax_history_if_needed(cast(Any, model))
    if wrap_with_ddp:
        model = model_provider_module._ddp_wrap(
            model,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            pg_collection=pg_collection,
        )
    return model


def _column_parallel_hf_to_megatron(
    self: ColumnParallelMapping,
    hf_weights: torch.Tensor,
    megatron_module: torch.nn.Module,
) -> torch.Tensor:
    if self.tp_size == 1:
        return hf_weights
    normalized_param = self._normalize_expert_param_name(self.megatron_param)
    target_param = get_module_and_param_from_name(
        cast(Any, megatron_module), normalized_param
    )[1]
    if self.tp_rank == 0:
        full_size = hf_weights.shape[0]
        if full_size % self.tp_size != 0:
            raise ValueError(
                f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks"
            )
        splits = list(torch.chunk(hf_weights, self.tp_size, dim=0))
    else:
        splits = None
    return self.scatter_to_tp_ranks(
        splits,
        target_param.shape,
        target_param.dtype,
        target_param.device,
    )


def _scatter_to_tp_ranks(
    self: MegatronParamMapping,
    splits: list[torch.Tensor] | None,
    output_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    src_rank: int = 0,
) -> torch.Tensor:
    if self.tp_size == 1:
        return cast(list[torch.Tensor], splits)[0].to(
            device=device, dtype=dtype, non_blocking=True
        )
    output = torch.empty(output_shape, dtype=dtype, device=device)
    dist = cast(Any, torch.distributed)
    global_src = dist.get_global_rank(group=self.tp_group, group_rank=src_rank)
    scatter_list = None
    if self.tp_rank == src_rank and splits:
        scatter_list = [
            shard.to(device=device, dtype=dtype, non_blocking=True) for shard in splits
        ]
    dist.scatter(output, scatter_list, src=global_src, group=self.tp_group)
    return output


def _replicated_hf_to_megatron(
    self: ReplicatedMapping,
    hf_weights: torch.Tensor,
    megatron_module: torch.nn.Module,
) -> torch.Tensor:
    if hasattr(megatron_module, "weight"):
        target_device = cast(Any, megatron_module).weight.device
    else:
        target_device = next(megatron_module.parameters()).device
    if self.tp_size == 1:
        return hf_weights.to(device=target_device, non_blocking=True)
    broadcast_device = target_device
    if (
        broadcast_device.type != "cuda"
        or broadcast_device.index != torch.cuda.current_device()
    ):
        broadcast_device = _materialization_device()
    if self.tp_rank == 0:
        tensor = hf_weights.to(device=cast(Any, broadcast_device), non_blocking=True)
    else:
        tensor = torch.empty_like(hf_weights, device=cast(Any, broadcast_device))
    return self.broadcast_tensor_to_tp_ranks(tensor, src_rank=0)


def _optimized_load_weights_hf_to_megatron(
    self: MegatronModelBridge,
    hf_pretrained: Any,
    megatron_model: Any,
    allowed_mismatched_params: list[str] | None = None,
) -> list[Any]:
    if not isinstance(megatron_model, list):
        megatron_model = [megatron_model]
    with contextlib.ExitStack() as stack:
        if hasattr(megatron_model[0], "hide_teacher_model"):
            stack.enter_context(megatron_model[0].hide_teacher_model())
        if hasattr(megatron_model[0], "hide_loss_modules"):
            stack.enter_context(megatron_model[0].hide_loss_modules())
        tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
    hf_state_dict = hf_pretrained.state
    raw_cache = load_unique_hf_keys_once(tasks, hf_state_dict, bridge=self)
    cached_state = _CachedStateLookup(cache=raw_cache, source=hf_state_dict)
    description = f"Loading from {hf_pretrained.model_name_or_path}"
    pending_device_copy = False
    for task in self._with_progress_tracking(tasks, description):
        if task is None or task.megatron_module is None:
            continue
        hf_param = task.mapping.hf_param
        if (
            isinstance(hf_param, str)
            and hf_param in raw_cache
            and hf_param not in hf_state_dict
        ):
            hf_weights = raw_cache[hf_param]
        else:
            hf_weights = self.maybe_modify_loaded_hf_weight(
                hf_param, cast(Mapping[str, torch.Tensor], cached_state)
            )
        converted_weights = task.mapping.hf_to_megatron(
            hf_weights, task.megatron_module
        )
        if converted_weights is None:
            continue
        assert task.param_weight is not None, (
            "param_weight is required for HF->Megatron conversion"
        )
        if converted_weights.shape != task.param_weight.shape:
            is_whitelisted = False
            if allowed_mismatched_params:
                for pattern in allowed_mismatched_params:
                    if fnmatch.fnmatch(
                        task.mapping.megatron_param, pattern
                    ) or fnmatch.fnmatch(task.param_name, pattern):
                        is_whitelisted = True
                        break
            if is_whitelisted:
                continue
            raise ValueError(
                f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                f"  Expected shape: {task.param_weight.shape}\n"
                f"  Got shape: {converted_weights.shape}\n"
                f"  Bridge type: {type(task.mapping).__name__}\n"
                f"  HF mapping: {task.mapping.hf_param}"
            )
        task.param_weight.data.copy_(converted_weights, non_blocking=True)
        if task.param_weight.device.type == "cuda":
            pending_device_copy = True
    if pending_device_copy and torch.cuda.is_available():
        torch.cuda.synchronize()
    self._broadcast_shared_embeddings(megatron_model)
    return megatron_model


def install_art_bridge_runtime_patches() -> None:
    from megatron.bridge.models import model_provider as model_provider_module

    _patch_router_gating_linear_empty_input()
    _patch_bias_swiglu_empty_input()
    _patch_moe_unpermute_empty_input()
    if not getattr(
        model_provider_module.get_model, "__art_meta_materialization__", False
    ):
        setattr(_art_get_model, "__art_meta_materialization__", True)
        setattr(model_provider_module, "get_model", _art_get_model)
    if not getattr(
        MegatronParamMapping.scatter_to_tp_ranks, "__art_non_blocking__", False
    ):
        setattr(_scatter_to_tp_ranks, "__art_non_blocking__", True)
        setattr(MegatronParamMapping, "scatter_to_tp_ranks", _scatter_to_tp_ranks)
    if not getattr(ColumnParallelMapping.hf_to_megatron, "__art_cast_last__", False):
        setattr(_column_parallel_hf_to_megatron, "__art_cast_last__", True)
        setattr(
            ColumnParallelMapping, "hf_to_megatron", _column_parallel_hf_to_megatron
        )
    if not getattr(ReplicatedMapping.hf_to_megatron, "__art_cast_last__", False):
        setattr(_replicated_hf_to_megatron, "__art_cast_last__", True)
        setattr(ReplicatedMapping, "hf_to_megatron", _replicated_hf_to_megatron)
    if not getattr(
        MegatronModelBridge.load_weights_hf_to_megatron, "__art_cached_load__", False
    ):
        setattr(_optimized_load_weights_hf_to_megatron, "__art_cached_load__", True)
        setattr(
            MegatronModelBridge,
            "load_weights_hf_to_megatron",
            _optimized_load_weights_hf_to_megatron,
        )


def _patch_router_gating_linear_empty_input() -> None:
    from megatron.core.transformer.moe import moe_utils, router

    if getattr(moe_utils.router_gating_linear, "__art_empty_safe__", False):
        return

    original_router_gating_linear = moe_utils.router_gating_linear

    def _router_gating_linear_empty_safe(
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        router_dtype: torch.dtype,
    ) -> torch.Tensor:
        if int(inp.numel()) != 0:
            return original_router_gating_linear(inp, weight, bias, router_dtype)
        zero = inp.to(router_dtype).sum() * 0.0 + weight.to(router_dtype).sum() * 0.0
        if bias is not None:
            zero = zero + bias.to(router_dtype).sum() * 0.0
        return zero.expand(*inp.shape[:-1], int(weight.shape[0]))

    setattr(_router_gating_linear_empty_safe, "__art_empty_safe__", True)
    setattr(moe_utils, "router_gating_linear", _router_gating_linear_empty_safe)
    setattr(router, "router_gating_linear", _router_gating_linear_empty_safe)


def _patch_bias_swiglu_empty_input() -> None:
    from megatron.core.fusions import fused_bias_swiglu
    from megatron.core.transformer import mlp
    from megatron.core.transformer.moe import experts, shared_experts

    if getattr(fused_bias_swiglu.bias_swiglu_impl, "__art_empty_safe__", False):
        return

    original_bias_swiglu_impl = fused_bias_swiglu.bias_swiglu_impl
    original_weighted_bias_swiglu_impl = fused_bias_swiglu.weighted_bias_swiglu_impl

    def _empty_swiglu_output(
        input: torch.Tensor,
        bias: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_shape = (*input.shape[:-1], int(input.shape[-1]) // 2)
        zero = input.sum() * 0.0
        if bias is not None:
            zero = zero + bias.to(dtype=input.dtype).sum() * 0.0
        if weights is not None:
            zero = zero + weights.to(dtype=input.dtype).sum() * 0.0
        return zero.expand(output_shape).clone()

    def _bias_swiglu_empty_safe(
        input: torch.Tensor,
        bias: torch.Tensor | None,
        fp8_input_store: bool = False,
        cpu_offload_input: bool = False,
    ) -> torch.Tensor:
        if int(input.numel()) != 0:
            return original_bias_swiglu_impl(
                input, bias, fp8_input_store, cpu_offload_input
            )
        return _empty_swiglu_output(input, bias=bias)

    def _weighted_bias_swiglu_empty_safe(
        input: torch.Tensor,
        bias: torch.Tensor | None,
        weights: torch.Tensor,
        fp8_input_store: bool = False,
    ) -> torch.Tensor:
        if int(input.numel()) != 0:
            return original_weighted_bias_swiglu_impl(
                input, bias, weights, fp8_input_store
            )
        return _empty_swiglu_output(input, bias=bias, weights=weights)

    setattr(_bias_swiglu_empty_safe, "__art_empty_safe__", True)
    setattr(_weighted_bias_swiglu_empty_safe, "__art_empty_safe__", True)
    setattr(fused_bias_swiglu, "bias_swiglu_impl", _bias_swiglu_empty_safe)
    setattr(
        fused_bias_swiglu,
        "weighted_bias_swiglu_impl",
        _weighted_bias_swiglu_empty_safe,
    )
    setattr(mlp, "bias_swiglu_impl", _bias_swiglu_empty_safe)
    setattr(mlp, "weighted_bias_swiglu_impl", _weighted_bias_swiglu_empty_safe)
    setattr(experts, "weighted_bias_swiglu_impl", _weighted_bias_swiglu_empty_safe)
    setattr(shared_experts, "bias_swiglu_impl", _bias_swiglu_empty_safe)


def _patch_moe_unpermute_empty_input() -> None:
    from megatron.core.transformer.moe import moe_utils, token_dispatcher

    if getattr(moe_utils.unpermute, "__art_empty_safe__", False):
        return

    original_unpermute = moe_utils.unpermute

    def _unpermute_empty_safe(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
        routing_map: torch.Tensor | None = None,
        fused: bool = False,
        drop_and_pad: bool = False,
        pad_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(permuted_tokens.numel()) != 0:
            return original_unpermute(
                permuted_tokens,
                sorted_indices,
                restore_shape,
                probs=probs,
                routing_map=routing_map,
                fused=fused,
                drop_and_pad=drop_and_pad,
                pad_offsets=pad_offsets,
            )
        zero = (
            permuted_tokens.sum() * 0.0 + sorted_indices.sum().to(permuted_tokens) * 0.0
        )
        if probs is not None:
            zero = zero + probs.to(dtype=permuted_tokens.dtype).sum() * 0.0
        return zero.expand(tuple(int(dim) for dim in restore_shape)).clone()

    setattr(_unpermute_empty_safe, "__art_empty_safe__", True)
    setattr(moe_utils, "unpermute", _unpermute_empty_safe)
    setattr(token_dispatcher, "unpermute", _unpermute_empty_safe)
