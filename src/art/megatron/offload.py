from collections.abc import Iterator
from dataclasses import dataclass, field
import gc
from typing import Any, Sequence, cast

from megatron.core.distributed import DistributedDataParallel
import torch

from art.megatron.model_chunks import unwrap_megatron_chunk


@dataclass
class OffloadState:
    pinned_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    is_offloaded: bool = False


def _iter_megatron_param_buffers(model: Sequence[torch.nn.Module]) -> Iterator[Any]:
    for chunk in model:
        ddp_chunk = unwrap_megatron_chunk(chunk)
        if not isinstance(ddp_chunk, DistributedDataParallel):
            raise RuntimeError(
                "Expected Megatron chunk wrapped by DistributedDataParallel, got "
                f"{type(ddp_chunk).__name__}"
            )
        ddp_buffers = cast(Sequence[Any] | None, ddp_chunk.__dict__.get("buffers"))
        expert_buffers = cast(
            Sequence[Any] | None, ddp_chunk.__dict__.get("expert_parallel_buffers")
        )
        if ddp_buffers is None or expert_buffers is None:
            raise RuntimeError(
                "Megatron DistributedDataParallel chunk is missing expected "
                "param buffer attributes"
            )
        yield from ddp_buffers
        yield from expert_buffers


def offload_to_cpu(
    model: Sequence[torch.nn.Module],
    rank: int,
    offload_state: OffloadState,
) -> None:
    """Offload model params to CPU pinned memory."""
    if offload_state.is_offloaded:
        return
    pinned_buffers = offload_state.pinned_buffers

    for param_buffer in _iter_megatron_param_buffers(model):
        param_buffer.offload_to_cpu(move_params=True, move_grads=True)

    # Megatron remaps trainable params into contiguous DDP buffers. Offload those via the
    # native buffer APIs above, and only manually offload frozen params here.
    for chunk in model:
        for param in chunk.parameters():
            if (
                not isinstance(param, torch.nn.Parameter)
                or param.requires_grad
                or param.device.type != "cuda"
            ):
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

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    offload_state.is_offloaded = True
    if rank == 0:
        print("Offloaded model params to CPU")


def reload_to_gpu(
    model: Sequence[torch.nn.Module],
    rank: int,
    offload_state: OffloadState,
    device: torch.device | str | None = None,
) -> None:
    """Reload model params to GPU."""
    if not offload_state.is_offloaded:
        return

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    for param_buffer in _iter_megatron_param_buffers(model):
        param_buffer.reload_from_cpu(move_params=True, move_grads=True)

    # Reload frozen params that were manually offloaded.
    for chunk in model:
        for param in chunk.parameters():
            if (
                not isinstance(param, torch.nn.Parameter)
                or param.requires_grad
                or param.device.type != "cpu"
            ):
                continue
            gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
            gpu_tensor.copy_(param.data, non_blocking=True)
            param.data = gpu_tensor

    torch.cuda.synchronize()
    offload_state.is_offloaded = False
    if rank == 0:
        print("Reloaded LoRA params to GPU")
