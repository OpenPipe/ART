from dataclasses import dataclass, field
import gc
from typing import Any, Sequence

import torch


@dataclass
class OffloadState:
    pinned_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    is_offloaded: bool = False


def offload_to_cpu(
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    rank: int,
    offload_state: OffloadState,
) -> None:
    """Offload model params and optimizer state to CPU pinned memory."""
    if offload_state.is_offloaded:
        return
    pinned_buffers = offload_state.pinned_buffers

    for chunk in model:
        for module in chunk.modules():
            for attr in ["A_T", "B_T"]:
                if not hasattr(module, attr):
                    continue
                param = getattr(module, attr)
                if (
                    not isinstance(param, torch.nn.Parameter)
                    or param.device.type != "cuda"
                ):
                    continue
                key = f"{id(module)}_{attr}"
                if (
                    key not in pinned_buffers
                    or pinned_buffers[key].shape != param.shape
                    or pinned_buffers[key].dtype != param.dtype
                ):
                    pinned_buffers[key] = torch.empty(
                        param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                    )
                pinned_buffers[key].copy_(param.data, non_blocking=True)
                param.data = pinned_buffers[key]

    # Offload remaining model parameters (including base weights).
    for chunk in model:
        for param in chunk.parameters():
            if not isinstance(param, torch.nn.Parameter) or param.device.type != "cuda":
                continue
            key = f"param_{id(param)}"
            if (
                key not in pinned_buffers
                or pinned_buffers[key].shape != param.shape
                or pinned_buffers[key].dtype != param.dtype
            ):
                pinned_buffers[key] = torch.empty(
                    param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                )
            pinned_buffers[key].copy_(param.data, non_blocking=True)
            param.data = pinned_buffers[key]

    for param_id, opt_state in optimizer.optimizer.state.items():
        for k, v in opt_state.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                key = f"opt_{id(param_id)}_{k}"
                if (
                    key not in pinned_buffers
                    or pinned_buffers[key].shape != v.shape
                    or pinned_buffers[key].dtype != v.dtype
                ):
                    pinned_buffers[key] = torch.empty(
                        v.shape, dtype=v.dtype, device="cpu", pin_memory=True
                    )
                pinned_buffers[key].copy_(v, non_blocking=True)
                opt_state[k] = pinned_buffers[key]

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    offload_state.is_offloaded = True
    if rank == 0:
        print("Offloaded model params and optimizer to CPU")


def reload_to_gpu(
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    rank: int,
    offload_state: OffloadState,
    device: torch.device | str | None = None,
) -> None:
    """Reload model params and optimizer state to GPU."""
    if not offload_state.is_offloaded:
        return

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    for chunk in model:
        for module in chunk.modules():
            for attr in ["A_T", "B_T"]:
                if not hasattr(module, attr):
                    continue
                param = getattr(module, attr)
                if (
                    not isinstance(param, torch.nn.Parameter)
                    or param.device.type != "cpu"
                ):
                    continue
                gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
                gpu_tensor.copy_(param.data, non_blocking=True)
                param.data = gpu_tensor

    # Reload remaining model parameters (including base weights).
    for chunk in model:
        for param in chunk.parameters():
            if not isinstance(param, torch.nn.Parameter) or param.device.type != "cpu":
                continue
            gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
            gpu_tensor.copy_(param.data, non_blocking=True)
            param.data = gpu_tensor

    for opt_state in optimizer.optimizer.state.values():
        for k, v in opt_state.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                gpu_tensor = torch.empty(v.shape, dtype=v.dtype, device=device)
                gpu_tensor.copy_(v, non_blocking=True)
                opt_state[k] = gpu_tensor

    torch.cuda.synchronize()
    offload_state.is_offloaded = False
    if rank == 0:
        print("Reloaded LoRA params and optimizer to GPU")
