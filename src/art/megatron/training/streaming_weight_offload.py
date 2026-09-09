from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import suppress
import logging
import os
import threading
from typing import Any, Literal

from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.random import is_checkpointing
from pydantic import BaseModel, ConfigDict, Field
import torch

from .model_chunks import ModelChunks

logger = logging.getLogger(__name__)

LayerOffloadStatus = Literal["cpu", "gpu", "loading"]
LAYER_STATUS_CPU: LayerOffloadStatus = "cpu"
LAYER_STATUS_GPU: LayerOffloadStatus = "gpu"
LAYER_STATUS_LOADING: LayerOffloadStatus = "loading"
STREAMING_INSTALLED_MESSAGE = (
    "Installed streaming frozen weight offload for %d layers (%d rank-local params)"
)
STREAMING_COMPILED_LAYERS_MESSAGE = (
    "Streaming weight offload managing compiled transformer layers"
)
# Quantized/custom kernels may vector-load weights and scales from streamed views.
STREAMED_PARAM_ALIGNMENT_BYTES = 256


def _rank0_info(rank: int, message: str, *args: object) -> None:
    if rank == 0:
        logger.info(message, *args)


class StreamingWeightOffloadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    num_layers: int = Field(default=0, ge=0)
    num_slots: int = Field(default=4, ge=2)
    resident_layers: int = Field(default=2, ge=1)


class _ParamSpec:
    def __init__(
        self,
        *,
        name: str,
        param: torch.nn.Parameter,
        offset: int,
        numel: int,
        shape: torch.Size,
    ) -> None:
        self.name = name
        self.param = param
        self.offset = offset
        self.numel = numel
        self.shape = shape


class _TensorGroup:
    def __init__(
        self, *, dtype: torch.dtype, cpu_flat: torch.Tensor, specs: list[_ParamSpec]
    ):
        self.dtype = dtype
        self.cpu_flat = cpu_flat
        self.specs = specs


class _LoadSlot:
    def __init__(self, index: int):
        self.index = index
        self.owner: _LayerState | None = None
        self.release_stream: torch.cuda.Stream | None = None
        self.pinned: dict[torch.dtype, torch.Tensor] = {}
        self.gpu: dict[torch.dtype, torch.Tensor] = {}

    def ensure_capacity(self, dtype: torch.dtype, numel: int) -> None:
        pinned = self.pinned.get(dtype)
        if pinned is None or pinned.numel() < numel:
            self.pinned[dtype] = torch.empty(
                numel, dtype=dtype, device="cpu", pin_memory=True
            )
        gpu = self.gpu.get(dtype)
        if gpu is None or gpu.numel() < numel:
            self.gpu[dtype] = torch.empty(
                numel, dtype=dtype, device=torch.cuda.current_device()
            )


class _LayerState:
    def __init__(self, index: int, layer: torch.nn.Module, groups: list[_TensorGroup]):
        self.index = index
        self.layer = layer
        self.groups = groups
        self.status: LayerOffloadStatus = LAYER_STATUS_GPU
        self.slot: _LoadSlot | None = None
        self.load_event: torch.cuda.Event | None = None
        self.load_ready = False
        self.load_error: BaseException | None = None


class StreamingWeightOffloader:
    def __init__(
        self,
        *,
        layers: list[torch.nn.Module],
        rank: int,
        config: StreamingWeightOffloadConfig,
    ) -> None:
        self.rank = rank
        self.config = config
        selected_layers = layers[: config.num_layers or len(layers)]
        self.layers = [
            _LayerState(i, layer, _build_tensor_groups(_frozen_cuda_parameters(layer)))
            for i, layer in enumerate(selected_layers)
        ]
        self.device = torch.cuda.current_device()
        self.h2d_stream = torch.cuda.Stream()
        self.slots = [_LoadSlot(i) for i in range(config.num_slots)]
        self._condition = threading.Condition()
        self._queue: deque[tuple[_LayerState, _LoadSlot]] = deque()
        self._worker_error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._load_worker,
            name=f"streaming_weight_offload_rank{rank}",
            daemon=True,
        )
        self._hooks: list[Any] = []

    def install(self) -> None:
        if not self.layers:
            raise RuntimeError(
                "Streaming weight offload found no transformer layers to manage"
            )
        param_count = sum(
            spec.numel
            for layer in self.layers
            for group in layer.groups
            for spec in group.specs
        )
        if param_count == 0:
            raise RuntimeError(
                "Streaming weight offload found no frozen CUDA parameters to manage"
            )
        self._worker.start()
        for layer_state in self.layers:
            self._hooks.append(
                layer_state.layer.register_forward_pre_hook(
                    lambda module, inputs, state=layer_state: self._pre_forward(state)
                )
            )
            self._hooks.append(
                layer_state.layer.register_forward_hook(
                    lambda module, inputs, output, state=layer_state: (
                        self._post_forward(state)
                    )
                )
            )
        self.offload_all()
        _rank0_info(
            self.rank, STREAMING_INSTALLED_MESSAGE, len(self.layers), param_count
        )

    def begin_job(self) -> None:
        self._prefetch_window(0, 1, self.config.resident_layers)

    def finish_job(self) -> None:
        self.offload_all()

    def remove(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def offload_all(self) -> None:
        for layer_state in self.layers:
            self._ensure_offloaded(layer_state)

    def _pre_forward(self, layer_state: _LayerState) -> None:
        recompute_forward = _is_recompute_forward()
        if recompute_forward:
            self._offload_recomputed_successors(layer_state.index)
        self._finish_load(layer_state)
        if recompute_forward:
            self._prefetch_window(
                layer_state.index - 1, -1, self.config.resident_layers - 1
            )
        else:
            self._prefetch_window(
                layer_state.index + 1, 1, self.config.resident_layers - 1
            )

    def _post_forward(self, layer_state: _LayerState) -> None:
        if not torch.is_grad_enabled():
            self._start_offload(layer_state)
            self._prefetch_window(layer_state.index + self.config.resident_layers, 1, 1)

    def _offload_recomputed_successors(self, index: int) -> None:
        for layer_state in self.layers[index + 1 :]:
            if layer_state.status in {LAYER_STATUS_GPU, LAYER_STATUS_LOADING}:
                self._ensure_offloaded(layer_state)

    def _start_load(self, index: int) -> None:
        self._check_worker_error()
        if index < 0 or index >= len(self.layers):
            return
        layer_state = self.layers[index]
        if layer_state.status in {LAYER_STATUS_GPU, LAYER_STATUS_LOADING}:
            return
        if layer_state.status != LAYER_STATUS_CPU:
            raise RuntimeError(f"Unexpected layer offload state {layer_state.status!r}")
        slot = self._acquire_slot()
        layer_state.slot = slot
        layer_state.load_event = None
        layer_state.load_ready = False
        layer_state.load_error = None
        layer_state.status = LAYER_STATUS_LOADING
        slot.owner = layer_state
        with self._condition:
            self._queue.append((layer_state, slot))
            self._condition.notify()

    def _prefetch_window(self, start_index: int, step: int, count: int) -> None:
        if step not in {-1, 1}:
            raise RuntimeError(f"Unexpected streaming prefetch step {step}")
        self._check_worker_error()
        if count <= 0:
            return
        end_index = start_index + step * count
        for index in range(start_index, end_index, step):
            self._start_load(index)

    def _finish_load(self, layer_state: _LayerState) -> None:
        self._check_worker_error()
        if layer_state.status == LAYER_STATUS_GPU:
            return
        if layer_state.status == LAYER_STATUS_CPU:
            self._start_load(layer_state.index)
        if layer_state.status != LAYER_STATUS_LOADING:
            raise RuntimeError(f"Unexpected layer load state {layer_state.status!r}")
        self._wait_for_load_launch(layer_state)
        if layer_state.load_error is not None:
            raise RuntimeError(
                f"Streaming weight offload failed while loading layer {layer_state.index}"
            ) from layer_state.load_error
        if layer_state.load_event is None or layer_state.slot is None:
            raise RuntimeError(f"Unexpected layer load state {layer_state.status!r}")
        # Transformer Engine can launch work on internal streams. Complete the H2D
        # copy before installing the parameter pointer so every downstream stream
        # observes initialized weights.
        layer_state.load_event.synchronize()
        self._install_gpu_views(layer_state)
        layer_state.load_event = None
        layer_state.status = LAYER_STATUS_GPU

    def _ensure_offloaded(self, layer_state: _LayerState) -> None:
        if layer_state.status == LAYER_STATUS_CPU:
            return
        if layer_state.status == LAYER_STATUS_LOADING:
            self._finish_load(layer_state)
        if layer_state.status == LAYER_STATUS_GPU:
            self._start_offload(layer_state)

    def _start_offload(self, layer_state: _LayerState) -> None:
        if layer_state.status == LAYER_STATUS_CPU:
            return
        if layer_state.status == LAYER_STATUS_LOADING:
            self._finish_load(layer_state)
        if layer_state.status != LAYER_STATUS_GPU:
            raise RuntimeError(f"Unexpected layer offload state {layer_state.status!r}")
        current_stream = torch.cuda.current_stream()
        slot = layer_state.slot
        if slot is not None:
            for tensor in slot.gpu.values():
                tensor.record_stream(current_stream)
            slot.owner = None
            slot.release_stream = current_stream
        self._install_cpu_views(layer_state)
        layer_state.slot = None
        layer_state.status = LAYER_STATUS_CPU

    def _acquire_slot(self) -> _LoadSlot:
        free_slots = [slot for slot in self.slots if slot.owner is None]
        if not free_slots:
            raise RuntimeError(
                "Streaming weight offload has no free load slots; increase "
                "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_SLOTS"
            )
        return next(
            (slot for slot in free_slots if slot.release_stream is None),
            free_slots[0],
        )

    def _load_worker(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                layer_state, slot = self._queue.popleft()
            try:
                self._run_load(layer_state, slot)
            except BaseException as exc:  # noqa: BLE001 - propagated to training thread.
                with self._condition:
                    layer_state.load_error = exc
                    layer_state.load_ready = True
                    self._worker_error = exc
                    self._condition.notify_all()

    def _run_load(self, layer_state: _LayerState, slot: _LoadSlot) -> None:
        for group in layer_state.groups:
            slot.ensure_capacity(group.dtype, group.cpu_flat.numel())
            slot.pinned[group.dtype][: group.cpu_flat.numel()].copy_(
                group.cpu_flat,
                non_blocking=False,
            )
        release_stream = slot.release_stream
        if release_stream is not None:
            self.h2d_stream.wait_stream(release_stream)
            slot.release_stream = None
        with torch.cuda.stream(self.h2d_stream):
            for group in layer_state.groups:
                n = group.cpu_flat.numel()
                gpu_tensor = slot.gpu[group.dtype][:n]
                gpu_tensor.copy_(slot.pinned[group.dtype][:n], non_blocking=True)
                gpu_tensor.record_stream(self.h2d_stream)
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
        with self._condition:
            layer_state.load_event = event
            layer_state.load_ready = True
            self._condition.notify_all()

    def _wait_for_load_launch(self, layer_state: _LayerState) -> None:
        with self._condition:
            while not layer_state.load_ready and self._worker_error is None:
                self._condition.wait()
        self._check_worker_error()

    def _check_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(
                "Streaming weight offload worker failed"
            ) from self._worker_error

    def _install_cpu_views(self, layer_state: _LayerState) -> None:
        for group in layer_state.groups:
            for spec in group.specs:
                _validate_streamed_param(spec)
                spec.param.data = group.cpu_flat[
                    spec.offset : spec.offset + spec.numel
                ].view(spec.shape)

    def _install_gpu_views(self, layer_state: _LayerState) -> None:
        if layer_state.slot is None:
            raise RuntimeError(
                "Cannot install GPU views before a layer has a load slot"
            )
        for group in layer_state.groups:
            gpu_flat = layer_state.slot.gpu[group.dtype]
            for spec in group.specs:
                _validate_streamed_param(spec)
                spec.param.data = gpu_flat[spec.offset : spec.offset + spec.numel].view(
                    spec.shape
                )


def streaming_weight_offload_config_from_env() -> StreamingWeightOffloadConfig:
    config = StreamingWeightOffloadConfig(
        enabled=_env_flag("ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD"),
        num_layers=_env_int("ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_LAYERS", 0),
        num_slots=_env_int("ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_SLOTS", 4),
        resident_layers=_env_int(
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_RESIDENT_LAYERS", 2
        ),
    )
    if config.resident_layers > config.num_slots:
        raise RuntimeError(
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_RESIDENT_LAYERS must be <= "
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_SLOTS"
        )
    return config


def install_streaming_weight_offload(
    *,
    model: ModelChunks,
    rank: int,
    compile_enabled: bool,
    config: StreamingWeightOffloadConfig,
) -> StreamingWeightOffloader | None:
    if not config.enabled:
        return None
    layers = _transformer_layers(model)
    if not layers:
        raise RuntimeError("Streaming weight offload could not find transformer layers")
    _validate_checkpoint_shape(layers[0])
    if compile_enabled:
        _rank0_info(rank, STREAMING_COMPILED_LAYERS_MESSAGE)
    offloader = StreamingWeightOffloader(layers=layers, rank=rank, config=config)
    offloader.install()
    return offloader


def maybe_install_streaming_weight_offload(
    *,
    model: ModelChunks,
    rank: int,
    compile_enabled: bool,
) -> StreamingWeightOffloader | None:
    return install_streaming_weight_offload(
        model=model,
        rank=rank,
        compile_enabled=compile_enabled,
        config=streaming_weight_offload_config_from_env(),
    )


def _validate_checkpoint_shape(layer: torch.nn.Module) -> None:
    config = getattr(layer, "config", None)
    if (
        getattr(config, "recompute_granularity", None) != "full"
        or getattr(config, "recompute_method", None) != "uniform"
        or int(getattr(config, "recompute_num_layers", 0) or 0) != 1
    ):
        raise RuntimeError(
            "Streaming weight offload requires full uniform activation recompute with "
            "recompute_num_layers=1"
        )


def _transformer_layers(model: Sequence[torch.nn.Module]) -> list[torch.nn.Module]:
    layers: list[torch.nn.Module] = []
    for chunk in model:
        module = _unwrap_module(chunk)
        gpt_module = (
            module
            if isinstance(module, GPTModel)
            else getattr(module, "language_model", None)
        )
        decoder = getattr(gpt_module, "decoder", None)
        chunk_layers = getattr(decoder, "layers", None)
        if chunk_layers is not None:
            layers.extend(list(chunk_layers))
    return layers


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr_name in ("_orig_mod", "module"):
            child = getattr(current, attr_name, None)
            if isinstance(child, torch.nn.Module):
                current = child
                break
        else:
            return current
    return current


def _frozen_cuda_parameters(
    module: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in module.named_parameters()
        if isinstance(param, torch.nn.Parameter)
        and not param.requires_grad
        and param.device.type == "cuda"
    ]


def _build_tensor_groups(
    params: list[tuple[str, torch.nn.Parameter]],
) -> list[_TensorGroup]:
    grouped: dict[torch.dtype, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, param in params:
        grouped.setdefault(param.dtype, []).append((name, param))
    groups: list[_TensorGroup] = []
    for dtype, dtype_params in grouped.items():
        element_size = dtype_params[0][1].element_size()
        alignment_numel = max(
            1,
            (STREAMED_PARAM_ALIGNMENT_BYTES + element_size - 1) // element_size,
        )
        specs: list[_ParamSpec] = []
        offset = 0
        for name, param in dtype_params:
            offset = _align_numel(offset, alignment_numel)
            numel = param.numel()
            specs.append(
                _ParamSpec(
                    name=name,
                    param=param,
                    offset=offset,
                    numel=numel,
                    shape=param.shape,
                )
            )
            offset += numel
        cpu_flat = torch.empty(offset, dtype=dtype, device="cpu")
        for spec in specs:
            cpu_flat[spec.offset : spec.offset + spec.numel].copy_(
                spec.param.detach().view(-1).cpu()
            )
        groups.append(_TensorGroup(dtype=dtype, cpu_flat=cpu_flat, specs=specs))
    return groups


def _align_numel(offset: int, alignment_numel: int) -> int:
    return ((offset + alignment_numel - 1) // alignment_numel) * alignment_numel


def _validate_streamed_param(spec: _ParamSpec) -> None:
    if spec.param.requires_grad:
        raise RuntimeError(
            "Streaming weight offload cannot manage trainable parameter "
            f"{spec.name}; trainable parameters must remain owned by Megatron buffers"
        )


def _is_recompute_forward() -> bool:
    return is_checkpointing() and torch.is_grad_enabled()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    with suppress(ValueError):
        return int(raw)
    raise RuntimeError(f"{name} must be an integer, got {raw!r}")
