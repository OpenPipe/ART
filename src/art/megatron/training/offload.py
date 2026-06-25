from collections.abc import Iterator
from dataclasses import dataclass, field
import logging
from typing import Any, Sequence, cast

from megatron.core.distributed import DistributedDataParallel
import torch

from .model_chunks import unwrap_megatron_chunk

logger = logging.getLogger(__name__)

OFFLOADED_TRAINABLE_BUFFERS_MESSAGE = (
    "Offloaded Megatron trainable param buffers to CPU"
)
RELOADED_TRAINABLE_BUFFERS_MESSAGE = "Reloaded Megatron trainable param buffers to GPU"
OFFLOADED_FROZEN_PARAMS_MESSAGE = "Offloaded frozen model params to CPU"
RELOADED_FROZEN_PARAMS_MESSAGE = "Reloaded frozen model params to GPU"


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


def _rank0_info(rank: int, message: str) -> None:
    if rank == 0:
        logger.info(message)


def offload_trainable_buffers_to_cpu(
    model: Sequence[torch.nn.Module],
    rank: int,
) -> None:
    for param_buffer in _iter_megatron_param_buffers(model):
        param_buffer.offload_to_cpu(move_params=True, move_grads=True)
    _rank0_info(rank, OFFLOADED_TRAINABLE_BUFFERS_MESSAGE)


def reload_trainable_buffers_to_gpu(
    model: Sequence[torch.nn.Module],
    rank: int,
) -> None:
    for param_buffer in _iter_megatron_param_buffers(model):
        param_buffer.reload_from_cpu(move_params=True, move_grads=True)
    _rank0_info(rank, RELOADED_TRAINABLE_BUFFERS_MESSAGE)


def offload_to_cpu(
    model: Sequence[torch.nn.Module],
    rank: int,
    offload_state: OffloadState,
) -> None:
    """Offload model params to CPU pinned memory."""
    if offload_state.is_offloaded:
        return
    pinned_buffers = offload_state.pinned_buffers

    offload_trainable_buffers_to_cpu(model, rank)

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
    offload_state.is_offloaded = True
    _rank0_info(rank, OFFLOADED_FROZEN_PARAMS_MESSAGE)


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

    reload_trainable_buffers_to_gpu(model, rank)

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
    _rank0_info(rank, RELOADED_FROZEN_PARAMS_MESSAGE)
