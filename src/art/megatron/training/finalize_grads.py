from collections.abc import Iterable
from typing import Any, Literal, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.transformer.module import MegatronModule
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

GradSyncDomain = Literal["tp_default", "expert_tp"]
GradSyncOp = Literal["none", "sum", "avg"]

TP_DEFAULT_GRAD_SYNC_DOMAIN: GradSyncDomain = "tp_default"
EXPERT_TP_GRAD_SYNC_DOMAIN: GradSyncDomain = "expert_tp"
GRAD_SYNC_OP_NONE: GradSyncOp = "none"
GRAD_SYNC_OP_SUM: GradSyncOp = "sum"
GRAD_SYNC_OP_AVG: GradSyncOp = "avg"
VALID_SYNC_OPS = (GRAD_SYNC_OP_NONE, GRAD_SYNC_OP_SUM, GRAD_SYNC_OP_AVG)


def _iter_named_trainable_parameters(
    model: list[MegatronModule],
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    seen: set[int] = set()
    for chunk_index, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if getattr(param, "_art_dynamic_lora_slot", False):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield f"chunk{chunk_index}.{name}", param


def _resolve_domain_group(
    domain: GradSyncDomain,
) -> Any | None:
    if domain == TP_DEFAULT_GRAD_SYNC_DOMAIN:
        group = ps.get_tensor_model_parallel_group(check_initialized=False)
        if group is None or group.size() <= 1:
            return None
        return group
    if domain != EXPERT_TP_GRAD_SYNC_DOMAIN:
        raise RuntimeError(f"Unknown grad sync domain: {domain}")

    group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    if group is None or group.size() <= 1:
        return None
    return group


def _resolve_reduce_op(op: GradSyncOp) -> Any:
    if op == GRAD_SYNC_OP_SUM:
        return torch.distributed.ReduceOp.SUM  # ty: ignore[possibly-missing-attribute]
    if op == GRAD_SYNC_OP_AVG:
        return torch.distributed.ReduceOp.AVG  # ty: ignore[possibly-missing-attribute]
    raise RuntimeError(f"Unknown grad sync op: {op}")


def tensor_parallel_grad_sync(
    param: torch.nn.Parameter,
    *,
    name: str,
) -> tuple[Any, Any] | None:
    domain: GradSyncDomain = getattr(
        param, "grad_sync_domain", TP_DEFAULT_GRAD_SYNC_DOMAIN
    )
    group = _resolve_domain_group(domain)
    if group is None:
        return None
    op: GradSyncOp = getattr(param, "grad_sync_op", GRAD_SYNC_OP_NONE)
    if op not in VALID_SYNC_OPS:
        raise RuntimeError(f"{name}: unsupported grad_sync_op={op}")
    if op == GRAD_SYNC_OP_NONE:
        return None
    return group, _resolve_reduce_op(op)


def coalesced_all_reduce(
    grads: list[torch.Tensor],
    *,
    group: Any,
    op: Any,
) -> None:
    coalesced = _flatten_dense_tensors(grads)
    reduced = (
        coalesced.float()
        if torch.is_floating_point(coalesced) and coalesced.dtype != torch.float32
        else coalesced
    )
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        reduced,
        op=op,
        group=group,
    )
    if reduced is not coalesced:
        reduced = reduced.to(dtype=coalesced.dtype)
    for grad, synced in zip(grads, _unflatten_dense_tensors(reduced, grads)):
        grad.copy_(synced)


def flush_param_grads_to_main_grads(model_chunks: Iterable[torch.nn.Module]) -> None:
    """Fallback for direct jobs when DDP post-hooks leave grads in param.grad.

    Megatron's distributed optimizer reads gradients from `main_grad`, which is
    normally populated by DDP backward post-hooks. Some direct ART runtimes can
    reach finalize/step with gradients still in `param.grad`, so copy them over
    using the same guard Megatron uses in its hook implementation.
    """
    for chunk in model_chunks:
        for param in chunk.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if not hasattr(param, "main_grad"):
                continue
            main_grad = cast(torch.Tensor, param.main_grad)
            if not getattr(param, "grad_added_to_main_grad", False) or getattr(
                param, "zero_out_wgrad", False
            ):
                main_grad.add_(param.grad.to(dtype=main_grad.dtype))
            param.grad = None


def finalize_model_grads_extended(
    model: list[MegatronModule],
    num_tokens: torch.Tensor | None = None,
) -> None:
    """Run Megatron finalize, then apply extra LoRA grad-sync reductions.

    Megatron finalize handles DP/CP(via `param.allreduce=True`)(and expert-DP via `param.allreduce=False`) internally.
    This extension handles extra TP/expert-TP reductions for params annotated
    with grad_sync_* metadata.
    """
    # All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    # embedding grads across first and last pipeline stages (if not tied)
    finalize_model_grads(
        cast(list[torch.nn.Module], model),
        num_tokens=num_tokens,
    )

    buckets: dict[
        tuple[int, str, torch.dtype, torch.device],
        tuple[Any, Any, list[torch.Tensor]],
    ] = {}

    for name, param in _iter_named_trainable_parameters(model):
        sync = tensor_parallel_grad_sync(param, name=name)
        if sync is None:
            continue

        if not hasattr(param, "main_grad"):
            raise RuntimeError(
                f"{name}: expected main_grad for tensor-parallel grad sync, but attribute is missing"
            )
        grad = param.main_grad
        if grad is None:
            raise RuntimeError(
                f"{name}: expected non-None main_grad for tensor-parallel grad sync"
            )
        local_grad = cast(  # local part of dtensor
            torch.Tensor, grad._local_tensor if hasattr(grad, "_local_tensor") else grad
        )
        group, reduce_op = sync
        key = (id(group), str(reduce_op), local_grad.dtype, local_grad.device)
        buckets.setdefault(key, (group, reduce_op, []))[2].append(local_grad)

    for group, op, grads in buckets.values():
        coalesced_all_reduce(grads, group=group, op=op)
