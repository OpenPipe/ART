from __future__ import annotations

from getpass import getuser
import hashlib
import importlib
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, cast

from quack.gemm import gemm as quack_gemm
import torch

_PADDED_LOW_RANK_TARGET = 8


def _validate_rank(rank: int) -> None:
    if rank <= 0:
        raise ValueError(f"Grouped LoRA QuACK backend requires rank > 0, got {rank}")
    if rank >= _PADDED_LOW_RANK_TARGET and rank % _PADDED_LOW_RANK_TARGET != 0:
        raise ValueError(
            "Grouped LoRA QuACK backend requires rank < 8 or a multiple of 8, "
            f"got {rank}"
        )


def _env_positive_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _quack_compile_lock_dir() -> Path:
    path = (
        Path(os.environ["ART_QUACK_COMPILE_LOCK_DIR"])
        if "ART_QUACK_COMPILE_LOCK_DIR" in os.environ
        else Path(tempfile.gettempdir()) / getuser() / "art_quack_compile_locks"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    return args + tuple(sorted(kwargs.items())) if kwargs else args


def _key_hash(key: tuple[Any, ...]) -> str:
    try:
        data = pickle.dumps(key)
    except Exception:
        data = repr(key).encode()
    return hashlib.sha256(data).hexdigest()


def _install_quack_compile_lock() -> None:
    """Serialize QuACK CuTe compilation misses across distributed ranks.

    QuACK protects disk-cache load/store with file locks, but it compiles cache
    misses outside that lock. ETP runs can make every rank compile the same
    grouped-LoRA backward GEMM at once, which has crashed inside CuTe DSL codegen.
    Hot in-process cache hits bypass this guard, so kernel launches stay unlocked.
    """
    import quack.cache_utils as quack_cache_utils

    gemm_module = importlib.import_module("quack.gemm")
    compile_gemm = getattr(gemm_module, "_compile_gemm")
    if bool(getattr(compile_gemm, "_art_compile_lock_wrapped", False)):
        return
    cache = getattr(compile_gemm, "cache", None)

    def locked_compile_gemm(*args: Any, **kwargs: Any) -> Any:
        key = _cache_key(args, kwargs)
        if isinstance(cache, dict) and key in cache:
            return compile_gemm(*args, **kwargs)
        lock_path = _quack_compile_lock_dir() / f"{_key_hash(key)}.lock"
        with quack_cache_utils.FileLock(lock_path, exclusive=True, timeout=600):
            return compile_gemm(*args, **kwargs)

    locked_compile_gemm_any = cast(Any, locked_compile_gemm)
    locked_compile_gemm_any.cache = getattr(compile_gemm, "cache", None)
    locked_compile_gemm_any.cache_clear = getattr(compile_gemm, "cache_clear", None)
    locked_compile_gemm_any.cache_info = getattr(compile_gemm, "cache_info", None)
    locked_compile_gemm_any._art_compile_lock_wrapped = True
    setattr(gemm_module, "_compile_gemm", locked_compile_gemm)


def _quack_gemm(*args: Any, **kwargs: Any) -> Any:
    _install_quack_compile_lock()
    return quack_gemm(*args, **kwargs)


def _tokens_per_expert_to_tensor(
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(tokens_per_expert, list):
        return torch.tensor(tokens_per_expert, dtype=torch.int64, device="cpu")
    if tokens_per_expert.ndim != 1:
        raise ValueError(
            f"tokens_per_expert must be 1D, got shape {tuple(tokens_per_expert.shape)}"
        )
    return tokens_per_expert.detach().to(dtype=torch.int64).contiguous()


def _build_expert_offsets(
    tokens_per_expert: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.empty(
        tokens_per_expert.numel() + 1,
        dtype=torch.int64,
        device=tokens_per_expert.device,
    )
    offsets[0] = 0
    offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)
    return offsets.to(device=device, dtype=torch.int32)


def _validate_inputs(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    counts = _tokens_per_expert_to_tensor(tokens_per_expert)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if a_t.ndim != 3:
        raise ValueError(f"a_t must be 3D, got shape {tuple(a_t.shape)}")
    if b_t.ndim != 3:
        raise ValueError(f"b_t must be 3D, got shape {tuple(b_t.shape)}")
    rank = a_t.shape[-1]
    _validate_rank(rank)
    if b_t.shape[-2] != rank:
        raise ValueError(f"Expected b_t rank dim {rank}, got shape {tuple(b_t.shape)}")
    if a_t.shape[0] != b_t.shape[0]:
        raise ValueError(
            "a_t and b_t must have the same number of experts, "
            f"got {a_t.shape[0]} and {b_t.shape[0]}"
        )
    if a_t.shape[1] != x.shape[1]:
        raise ValueError(
            f"a_t input dim must match x.shape[1], got {a_t.shape[1]} and {x.shape[1]}"
        )
    if counts.numel() != a_t.shape[0]:
        raise ValueError(
            "tokens_per_expert length must match number of experts, "
            f"got {counts.numel()} and {a_t.shape[0]}"
        )
    if x.device.type != "cuda" or a_t.device != x.device or b_t.device != x.device:
        raise ValueError("x, a_t, and b_t must be CUDA tensors on the same device")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"Unsupported dtype {x.dtype}; expected fp16 or bf16")
    if a_t.dtype != x.dtype or b_t.dtype != x.dtype:
        raise ValueError(
            f"Dtype mismatch: x={x.dtype}, a_t={a_t.dtype}, b_t={b_t.dtype}"
        )
    return counts


def _validate_dual_inputs(
    x: torch.Tensor,
    gate_a_t: torch.Tensor,
    gate_b_t: torch.Tensor,
    up_a_t: torch.Tensor,
    up_b_t: torch.Tensor,
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    counts = _validate_inputs(x, gate_a_t, gate_b_t, tokens_per_expert)
    if up_a_t.ndim != 3:
        raise ValueError(f"up_a_t must be 3D, got shape {tuple(up_a_t.shape)}")
    if up_b_t.ndim != 3:
        raise ValueError(f"up_b_t must be 3D, got shape {tuple(up_b_t.shape)}")
    up_rank = up_a_t.shape[-1]
    _validate_rank(up_rank)
    if up_b_t.shape[-2] != up_rank:
        raise ValueError(
            f"Expected up_b_t rank dim {up_rank}, got shape {tuple(up_b_t.shape)}"
        )
    if up_a_t.shape[0] != gate_a_t.shape[0] or up_b_t.shape[0] != gate_b_t.shape[0]:
        raise ValueError(
            "Gate and up tensors must have the same number of experts, "
            f"got gate={gate_a_t.shape[0]} up={up_a_t.shape[0]}"
        )
    if up_a_t.shape[1] != x.shape[1]:
        raise ValueError(
            f"up_a_t input dim must match x.shape[1], got {up_a_t.shape[1]} and {x.shape[1]}"
        )
    if up_a_t.device != x.device or up_b_t.device != x.device:
        raise ValueError(
            "x, up_a_t, and up_b_t must be CUDA tensors on the same device"
        )
    if up_a_t.dtype != x.dtype or up_b_t.dtype != x.dtype:
        raise ValueError(
            f"Dtype mismatch: x={x.dtype}, up_a_t={up_a_t.dtype}, up_b_t={up_b_t.dtype}"
        )
    return counts


def _effective_rank(rank: int) -> int:
    if rank < _PADDED_LOW_RANK_TARGET:
        return _PADDED_LOW_RANK_TARGET
    return rank


def _pad_a_t(a_t: torch.Tensor, effective_rank: int) -> torch.Tensor:
    pad_rank = effective_rank - a_t.shape[-1]
    if pad_rank <= 0:
        return a_t.contiguous()
    pad = a_t.new_zeros(a_t.shape[0], a_t.shape[1], pad_rank)
    return torch.cat((a_t, pad), dim=-1).contiguous()


def _pad_b_t(b_t: torch.Tensor, effective_rank: int) -> torch.Tensor:
    pad_rank = effective_rank - b_t.shape[1]
    if pad_rank <= 0:
        return b_t.contiguous()
    pad = b_t.new_zeros(b_t.shape[0], pad_rank, b_t.shape[2])
    return torch.cat((b_t, pad), dim=1).contiguous()


def _proj_tile_n(rank: int) -> int:
    override = _env_positive_int("ART_QUACK_PROJ_TILE_N")
    if override is not None:
        return override
    if rank <= 32:
        return 32
    return 64 if rank <= 64 else 128


def _matmul_tile_n(out_features: int) -> int:
    override = _env_positive_int("ART_QUACK_MATMUL_TILE_N")
    if override is not None:
        return override
    return 128 if out_features >= 128 else 64


def _grad_a_tile_m(rank: int) -> int:
    override = _env_positive_int("ART_QUACK_GRAD_A_TILE_M")
    if override is not None:
        return override
    return 128


def _grad_b_tile_m(rank: int) -> int:
    override = _env_positive_int("ART_QUACK_GRAD_B_TILE_M")
    if override is not None:
        return override
    return 64 if rank <= 64 else 128


def _varlen_quack_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_features: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
    tile_n: int,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            a.shape[0],
            out_features,
            device=a.device,
            dtype=a.dtype,
        )
    else:
        if out.shape != (a.shape[0], out_features):
            raise ValueError(
                f"Expected output shape {(a.shape[0], out_features)}, got {tuple(out.shape)}"
            )
        if out.device != a.device or out.dtype != a.dtype:
            raise ValueError(
                f"Output tensor must match input device/dtype, got {out.device}/{out.dtype}"
            )
    _quack_gemm(
        a,
        b,
        out,
        None,
        None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        persistent=True,
        alpha=alpha,
        cu_seqlens_m=expert_offsets,
    )
    return out


def _varlen_quack_gemm_k(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    batch_count: int,
    out_shape_m: int,
    out_shape_n: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
    tile_n: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    out = torch.empty(
        batch_count,
        out_shape_m,
        out_shape_n,
        device=a.device,
        dtype=a.dtype,
    )
    _quack_gemm(
        a,
        b,
        out,
        None,
        None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        persistent=True,
        alpha=alpha,
        cu_seqlens_k=expert_offsets,
    )
    return out


class _QuackGroupedLoraFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        a_t: torch.Tensor,
        b_t: torch.Tensor,
        counts: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        expert_offsets = _build_expert_offsets(counts, device=x.device)
        actual_rank = a_t.shape[-1]
        effective_rank = _effective_rank(actual_rank)
        a_t_eff = _pad_a_t(a_t, effective_rank)
        b_t_eff = _pad_b_t(b_t, effective_rank)
        proj_weights = a_t_eff.permute(0, 2, 1).contiguous()
        apply_weights = b_t_eff.permute(0, 2, 1).contiguous()

        tmp = _varlen_quack_gemm(
            x.contiguous(),
            proj_weights,
            out_features=effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(effective_rank),
        )
        out = _varlen_quack_gemm(
            tmp,
            apply_weights,
            out_features=b_t.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(b_t.shape[-1]),
            alpha=scale,
        )

        ctx.save_for_backward(x, a_t_eff, b_t_eff, tmp, expert_offsets)
        ctx.actual_rank = actual_rank
        ctx.effective_rank = effective_rank
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        if len(grad_outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one gradient output, got {len(grad_outputs)}"
            )
        x, a_t_eff, b_t_eff, tmp, expert_offsets = ctx.saved_tensors
        effective_rank = ctx.effective_rank
        actual_rank = ctx.actual_rank
        scale = ctx.scale
        grad_out = cast(torch.Tensor, grad_outputs[0])
        assert grad_out.stride(-1) == 1, (
            "QuACK grouped LoRA backward requires grad_out stride(-1) == 1"
        )

        grad_tmp = _varlen_quack_gemm(
            grad_out,
            b_t_eff.contiguous(),
            out_features=effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(effective_rank),
            alpha=scale,
        )
        grad_x = _varlen_quack_gemm(
            grad_tmp,
            a_t_eff.contiguous(),
            out_features=x.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(x.shape[-1]),
        )
        grad_a_eff = _varlen_quack_gemm_k(
            x.transpose(0, 1),
            grad_tmp.transpose(0, 1),
            batch_count=a_t_eff.shape[0],
            out_shape_m=a_t_eff.shape[1],
            out_shape_n=effective_rank,
            expert_offsets=expert_offsets,
            tile_m=_grad_a_tile_m(effective_rank),
            tile_n=_proj_tile_n(effective_rank),
        )
        grad_b_eff = _varlen_quack_gemm_k(
            tmp.transpose(0, 1),
            grad_out.transpose(0, 1),
            batch_count=b_t_eff.shape[0],
            out_shape_m=effective_rank,
            out_shape_n=b_t_eff.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=_grad_b_tile_m(effective_rank),
            tile_n=_matmul_tile_n(b_t_eff.shape[-1]),
            alpha=scale,
        )
        return (
            grad_x,
            grad_a_eff[:, :, :actual_rank].contiguous(),
            grad_b_eff[:, :actual_rank, :].contiguous(),
            None,
            None,
        )


class _QuackGroupedLoraDualFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_a_t: torch.Tensor,
        gate_b_t: torch.Tensor,
        up_a_t: torch.Tensor,
        up_b_t: torch.Tensor,
        counts: torch.Tensor,
        scale_gate: float,
        scale_up: float,
    ) -> torch.Tensor:
        expert_offsets = _build_expert_offsets(counts, device=x.device)
        gate_actual_rank = gate_a_t.shape[-1]
        up_actual_rank = up_a_t.shape[-1]
        gate_effective_rank = _effective_rank(gate_actual_rank)
        up_effective_rank = _effective_rank(up_actual_rank)

        gate_a_t_eff = _pad_a_t(gate_a_t, gate_effective_rank)
        up_a_t_eff = _pad_a_t(up_a_t, up_effective_rank)
        gate_b_t_eff = _pad_b_t(gate_b_t, gate_effective_rank)
        up_b_t_eff = _pad_b_t(up_b_t, up_effective_rank)

        a_cat_eff = torch.cat((gate_a_t_eff, up_a_t_eff), dim=-1).contiguous()
        proj_weights = a_cat_eff.permute(0, 2, 1).contiguous()
        gate_apply_weights = gate_b_t_eff.permute(0, 2, 1).contiguous()
        up_apply_weights = up_b_t_eff.permute(0, 2, 1).contiguous()

        total_effective_rank = gate_effective_rank + up_effective_rank
        tmp_cat = _varlen_quack_gemm(
            x.contiguous(),
            proj_weights,
            out_features=total_effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(total_effective_rank),
        )
        tmp_gate, tmp_up = torch.split(
            tmp_cat, [gate_effective_rank, up_effective_rank], dim=1
        )

        gate_out_features = gate_b_t.shape[-1]
        up_out_features = up_b_t.shape[-1]
        out = torch.empty(
            x.shape[0],
            gate_out_features + up_out_features,
            device=x.device,
            dtype=x.dtype,
        )
        _varlen_quack_gemm(
            tmp_gate,
            gate_apply_weights,
            out_features=gate_out_features,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(gate_out_features),
            alpha=scale_gate,
            out=out[:, :gate_out_features],
        )
        _varlen_quack_gemm(
            tmp_up,
            up_apply_weights,
            out_features=up_out_features,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(up_out_features),
            alpha=scale_up,
            out=out[:, gate_out_features:],
        )

        ctx.save_for_backward(
            x,
            a_cat_eff,
            gate_b_t_eff,
            up_b_t_eff,
            tmp_cat,
            expert_offsets,
        )
        ctx.gate_actual_rank = gate_actual_rank
        ctx.up_actual_rank = up_actual_rank
        ctx.gate_effective_rank = gate_effective_rank
        ctx.up_effective_rank = up_effective_rank
        ctx.gate_out_features = gate_out_features
        ctx.scale_gate = scale_gate
        ctx.scale_up = scale_up
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        if len(grad_outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one gradient output, got {len(grad_outputs)}"
            )
        x, a_cat_eff, gate_b_t_eff, up_b_t_eff, tmp_cat, expert_offsets = (
            ctx.saved_tensors
        )
        gate_actual_rank = ctx.gate_actual_rank
        up_actual_rank = ctx.up_actual_rank
        gate_effective_rank = ctx.gate_effective_rank
        up_effective_rank = ctx.up_effective_rank
        gate_out_features = ctx.gate_out_features
        scale_gate = ctx.scale_gate
        scale_up = ctx.scale_up

        grad_out = cast(torch.Tensor, grad_outputs[0])
        assert grad_out.stride(-1) == 1, (
            "QuACK grouped FC1 dual LoRA backward requires grad_out stride(-1) == 1"
        )
        grad_gate = grad_out[:, :gate_out_features]
        grad_up = grad_out[:, gate_out_features:]
        tmp_gate, tmp_up = torch.split(
            tmp_cat, [gate_effective_rank, up_effective_rank], dim=1
        )

        grad_tmp_gate = _varlen_quack_gemm(
            grad_gate,
            gate_b_t_eff.contiguous(),
            out_features=gate_effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(gate_effective_rank),
            alpha=scale_gate,
        )
        grad_tmp_up = _varlen_quack_gemm(
            grad_up,
            up_b_t_eff.contiguous(),
            out_features=up_effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(up_effective_rank),
            alpha=scale_up,
        )
        grad_tmp_cat = torch.cat((grad_tmp_gate, grad_tmp_up), dim=1).contiguous()

        total_effective_rank = gate_effective_rank + up_effective_rank
        grad_x = _varlen_quack_gemm(
            grad_tmp_cat,
            a_cat_eff.contiguous(),
            out_features=x.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(x.shape[-1]),
        )
        grad_a_cat_eff = _varlen_quack_gemm_k(
            x.transpose(0, 1),
            grad_tmp_cat.transpose(0, 1),
            batch_count=a_cat_eff.shape[0],
            out_shape_m=a_cat_eff.shape[1],
            out_shape_n=total_effective_rank,
            expert_offsets=expert_offsets,
            tile_m=_grad_a_tile_m(total_effective_rank),
            tile_n=_proj_tile_n(total_effective_rank),
        )
        grad_b_gate_eff = _varlen_quack_gemm_k(
            tmp_gate.transpose(0, 1),
            grad_gate.transpose(0, 1),
            batch_count=gate_b_t_eff.shape[0],
            out_shape_m=gate_effective_rank,
            out_shape_n=gate_b_t_eff.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=_grad_b_tile_m(gate_effective_rank),
            tile_n=_matmul_tile_n(gate_b_t_eff.shape[-1]),
            alpha=scale_gate,
        )
        grad_b_up_eff = _varlen_quack_gemm_k(
            tmp_up.transpose(0, 1),
            grad_up.transpose(0, 1),
            batch_count=up_b_t_eff.shape[0],
            out_shape_m=up_effective_rank,
            out_shape_n=up_b_t_eff.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=_grad_b_tile_m(up_effective_rank),
            tile_n=_matmul_tile_n(up_b_t_eff.shape[-1]),
            alpha=scale_up,
        )
        grad_a_gate_eff, grad_a_up_eff = torch.split(
            grad_a_cat_eff, [gate_effective_rank, up_effective_rank], dim=2
        )
        return (
            grad_x,
            grad_a_gate_eff[:, :, :gate_actual_rank].contiguous(),
            grad_b_gate_eff[:, :gate_actual_rank, :].contiguous(),
            grad_a_up_eff[:, :, :up_actual_rank].contiguous(),
            grad_b_up_eff[:, :up_actual_rank, :].contiguous(),
            None,
            None,
            None,
        )


# Dynamo tracing through CuTe's DLPack interop fails on FakeTensor, so keep the
# QuACK grouped kernels eager while the surrounding layer stays compiled.
@torch.compiler.disable
def quack_grouped_lora(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: list[int] | torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Run grouped LoRA with the QuACK varlen GEMM backend.

    Assumptions required by the caller:
    - `counts` is ordered by local expert index and `sum(counts) == x.shape[0]`.
    - `counts` length matches `a_t.shape[0] == b_t.shape[0]`.
    - `x.shape[1] == a_t.shape[1]` and `a_t.shape[-1] == b_t.shape[-2]`.
    - `x`, `a_t`, and `b_t` are CUDA tensors on the same device with fp16 or bf16 dtype.

    The value-based `sum(counts)` check is intentionally omitted to avoid a host-device
    synchronization in the hot path.
    """
    counts_tensor = _validate_inputs(x, a_t, b_t, counts)
    return _QuackGroupedLoraFn.apply(x, a_t, b_t, counts_tensor, scale)


@torch.compiler.disable
def quack_grouped_lora_dual(
    x: torch.Tensor,
    gate_a_t: torch.Tensor,
    gate_b_t: torch.Tensor,
    up_a_t: torch.Tensor,
    up_b_t: torch.Tensor,
    counts: list[int] | torch.Tensor,
    *,
    scale_gate: float = 1.0,
    scale_up: float = 1.0,
) -> torch.Tensor:
    """Run grouped FC1 gate/up LoRA with a shared QuACK projection path."""
    counts_tensor = _validate_dual_inputs(x, gate_a_t, gate_b_t, up_a_t, up_b_t, counts)
    return _QuackGroupedLoraDualFn.apply(
        x,
        gate_a_t,
        gate_b_t,
        up_a_t,
        up_b_t,
        counts_tensor,
        scale_gate,
        scale_up,
    )
