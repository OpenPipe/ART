"""Canonicalize dense TileLang ``chunk_bwd_dqkwg`` calls onto ``B=1, IS_VARLEN=True``.

In ``fla-core==0.5.0``, ``fla.ops.common.backends.tilelang.chunk_bwd._build_kernel``
bakes ``B`` as a compile-time JIT cache key when ``IS_VARLEN=False``
(``cu_seqlens=None``), whereas ``T``, ``NT``, ``total_h``, and ``Ncu`` are all
``T.dynamic(...)`` when ``IS_VARLEN=True``. Because ``run_gdn_bucket`` dispatches
uniform-length local buckets with ``cu_seqlens=None`` and variable ``B``, every
unseen uniform batch size triggers a multi-second live TileLang compile stall.

Flattening dense ``(B, T, ...)`` inputs to ``(1, B * T, ...)`` with uniform
``cu_seqlens`` and ``chunk_indices`` routes every dense and varlen call through
the single ``B=1, IS_VARLEN=True`` kernel, preserving exact memory layout, grid
work, and bit-for-bit arithmetic with zero post-startup TileLang compiles for
any batch size ``B >= 1``.
"""

from __future__ import annotations

from collections import OrderedDict
import functools
import os
import threading
from typing import Any

_TILELANG_VARLEN_INDEX_CACHE_MAX = 64


def ensure_tilelang_varlen_bwd(*, torch_mod: Any | None = None) -> bool:
    """Install dense-to-varlen canonicalization on ``chunk_bwd_dqkwg_tilelang``.

    Idempotent; returns ``False`` when ``FLA_TILELANG`` is disabled or FLA's
    TileLang chunk-backward module is unavailable.
    """
    if os.environ.get("FLA_TILELANG", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    try:
        import fla.ops.common.backends.tilelang.chunk_bwd as tl_chunk_bwd
    except Exception:
        return False

    orig_fn = getattr(tl_chunk_bwd, "chunk_bwd_dqkwg_tilelang", None)
    if not callable(orig_fn):
        return False
    if getattr(orig_fn, "__art_tilelang_varlen_bwd__", False):
        return True

    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False

    index_cache_lock = threading.Lock()
    index_cache: OrderedDict[tuple[str, int, int, int], tuple[Any, Any]] = OrderedDict()

    def _get_uniform_varlen_indices(
        device: Any,
        batch_size: int,
        seq_len: int,
        chunk_size: int,
    ) -> tuple[Any, Any]:
        key = (str(device), batch_size, seq_len, chunk_size)
        with index_cache_lock:
            cached = index_cache.get(key)
            if cached is not None:
                index_cache.move_to_end(key)
                return cached
        nt_local = (seq_len + chunk_size - 1) // chunk_size
        cu_sl = torch_mod.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            device=device,
            dtype=torch_mod.int32,
        )
        seq_idx = torch_mod.arange(
            batch_size, device=device, dtype=torch_mod.int32
        ).repeat_interleave(nt_local)
        chunk_idx = torch_mod.arange(
            nt_local, device=device, dtype=torch_mod.int32
        ).repeat(batch_size)
        c_idx = torch_mod.stack((seq_idx, chunk_idx), dim=1)
        pair = (cu_sl, c_idx)
        with index_cache_lock:
            index_cache[key] = pair
            index_cache.move_to_end(key)
            while len(index_cache) > _TILELANG_VARLEN_INDEX_CACHE_MAX:
                index_cache.popitem(last=False)
        return pair

    @functools.wraps(orig_fn)
    def _canonical_chunk_bwd_dqkwg_tilelang(
        q: Any,
        k: Any,
        v: Any,
        do: Any,
        h: Any,
        dh: Any,
        w: Any = None,
        g: Any = None,
        g_gamma: Any = None,
        dv: Any = None,
        scale: Any = None,
        cu_seqlens: Any = None,
        chunk_size: int = 64,
        chunk_indices: Any = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        if (
            cu_seqlens is None
            and chunk_indices is None
            and g_gamma is None
            and getattr(k, "ndim", 0) == 4
            and getattr(v, "ndim", 0) == 4
            and q.shape == k.shape
            and v.shape[:3] == k.shape[:3]
            and do.shape == v.shape
            and chunk_size > 0
        ):
            batch_size, seq_len, num_heads, key_dim = (int(d) for d in k.shape)
            val_dim = int(v.shape[-1])
            nt_local = (seq_len + chunk_size - 1) // chunk_size
            expected_h_elems = (
                batch_size * nt_local * num_heads * int(h.shape[-2]) * int(h.shape[-1])
                if getattr(h, "ndim", 0) >= 3
                else -1
            )
            if (
                batch_size >= 1
                and seq_len > 0
                and getattr(h, "ndim", 0) >= 3
                and getattr(dh, "ndim", 0) >= 3
                and h.shape[-2:] == dh.shape[-2:]
                and int(h.numel()) == expected_h_elems
                and int(dh.numel()) == expected_h_elems
                and (w is None or w.shape == k.shape)
                and (dv is None or dv.shape == v.shape)
                and (g is None or g.shape == (batch_size, seq_len, num_heads))
            ):
                total_t = batch_size * seq_len

                def _flat_4d(tensor: Any, last_dim: int) -> Any:
                    contig = tensor if tensor.is_contiguous() else tensor.contiguous()
                    return contig.reshape(1, total_t, num_heads, last_dim)

                q_f = _flat_4d(q, key_dim)
                k_f = _flat_4d(k, key_dim)
                v_f = _flat_4d(v, val_dim)
                do_f = _flat_4d(do, val_dim)
                w_f = _flat_4d(w, key_dim) if w is not None else None
                dv_f = _flat_4d(dv, val_dim) if dv is not None else None
                g_f = (
                    (g if g.is_contiguous() else g.contiguous()).reshape(
                        1, total_t, num_heads
                    )
                    if g is not None
                    else None
                )
                h_f = (h if h.is_contiguous() else h.contiguous()).reshape(
                    1, batch_size * nt_local, num_heads, h.shape[-2], h.shape[-1]
                )
                dh_f = (dh if dh.is_contiguous() else dh.contiguous()).reshape(
                    1, batch_size * nt_local, num_heads, dh.shape[-2], dh.shape[-1]
                )
                cu_sl, c_idx = _get_uniform_varlen_indices(
                    q.device, batch_size, seq_len, int(chunk_size)
                )
                dq, dk, dw_out, dg = orig_fn(
                    q_f,
                    k_f,
                    v_f,
                    do_f,
                    h_f,
                    dh_f,
                    w=w_f,
                    g=g_f,
                    g_gamma=None,
                    dv=dv_f,
                    scale=scale,
                    cu_seqlens=cu_sl,
                    chunk_size=chunk_size,
                    chunk_indices=c_idx,
                    use_exp2=use_exp2,
                    transpose_state_layout=transpose_state_layout,
                )
                return (
                    dq.reshape(batch_size, seq_len, num_heads, key_dim),
                    dk.reshape(batch_size, seq_len, num_heads, key_dim),
                    (
                        dw_out.reshape(batch_size, seq_len, num_heads, key_dim)
                        if dw_out is not None
                        else None
                    ),
                    (
                        dg.reshape(batch_size, seq_len, num_heads)
                        if dg is not None
                        else None
                    ),
                )

        return orig_fn(
            q,
            k,
            v,
            do,
            h,
            dh,
            w=w,
            g=g,
            g_gamma=g_gamma,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )

    setattr(_canonical_chunk_bwd_dqkwg_tilelang, "__art_tilelang_varlen_bwd__", True)
    setattr(_canonical_chunk_bwd_dqkwg_tilelang, "_art_index_cache", index_cache)
    tl_chunk_bwd.chunk_bwd_dqkwg_tilelang = _canonical_chunk_bwd_dqkwg_tilelang
    return True
