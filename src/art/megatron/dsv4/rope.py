from functools import lru_cache
import math
from typing import Any, cast

from megatron.core.transformer import TransformerConfig
import torch
from torch import nn

_DEVICE_ROPE_CACHES: dict[tuple[object, ...], torch.Tensor] = {}


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """Precompute the complex rotary frequencies for RoPE, with optional YaRN smoothing.

    When ``original_seq_len > 0``, applies YaRN factor rescaling interpolated
    by a linear ramp between ``beta_fast`` and ``beta_slow``. Otherwise the
    base frequencies are used verbatim.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply RoPE in-place to the last dim of ``x``.

    ``x`` has shape ``[..., dim]`` where ``dim`` is even; the last-dim pairs are
    treated as complex numbers multiplied by ``freqs_cis``. When ``inverse=True``
    the conjugate rotation is applied (used for the indexer's inverse rope).
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        if freqs_cis.ndim == 2:
            freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
        else:
            freqs_cis = freqs_cis.view(x.size(0), x.size(1), x.size(-1))
    else:
        if freqs_cis.ndim == 2:
            freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        else:
            freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def wrapped_precompute_freqs_cis(
    config: TransformerConfig,
    rope_head_dim: int,
    base: float,
    yarn_disabled: bool = False,
    seqlen: int = 65536,
):
    cfg = cast(Any, config)

    # yarn_disabled=True makes precompute_freqs_cis skip YaRN interpolation.
    # DSV4 Flash keeps YaRN enabled for sliding and compressed attention.
    original_seq_len = 0 if yarn_disabled else cfg.original_max_position_embeddings

    inputs = dict(
        dim=rope_head_dim,
        seqlen=seqlen,
        original_seq_len=original_seq_len,
        base=base,
        factor=cfg.rotary_scaling_factor,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
    )

    assert cfg.rotary_scaling_factor in (4, 16), (
        f"Unexpected rotary_scaling_factor: {cfg.rotary_scaling_factor}"
    )
    expected_original = 0 if yarn_disabled else 65536
    assert inputs == dict(
        dim=rope_head_dim,
        seqlen=seqlen,
        original_seq_len=expected_original,
        base=base,
        factor=cfg.rotary_scaling_factor,
        beta_fast=32,
        beta_slow=1,
    )

    return precompute_freqs_cis(**inputs)


def _rope_cache_key(
    config: TransformerConfig,
    *,
    rope_head_dim: int,
    base: float,
    yarn_disabled: bool,
    device: torch.device,
) -> tuple[object, ...]:
    cfg = cast(Any, config)
    device_index = (
        torch.cuda.current_device()
        if device.type == "cuda" and device.index is None
        else device.index
    )
    original_seq_len = 0 if yarn_disabled else cfg.original_max_position_embeddings
    return (
        device.type,
        device_index,
        int(rope_head_dim),
        int(original_seq_len),
        float(base),
        float(cfg.rotary_scaling_factor),
        int(cfg.beta_fast),
        int(cfg.beta_slow),
    )


def _get_device_rope_cache(
    config: TransformerConfig,
    *,
    rope_head_dim: int,
    base: float,
    yarn_disabled: bool,
    device: torch.device,
    seqlen: int,
) -> torch.Tensor:
    seqlen = max(1, int(seqlen))
    key = _rope_cache_key(
        config,
        rope_head_dim=rope_head_dim,
        base=base,
        yarn_disabled=yarn_disabled,
        device=device,
    )
    cached = _DEVICE_ROPE_CACHES.get(key)
    if cached is None or cached.shape[0] < seqlen:
        cached = wrapped_precompute_freqs_cis(
            config,
            rope_head_dim=rope_head_dim,
            base=base,
            yarn_disabled=yarn_disabled,
            seqlen=seqlen,
        ).to(device)
        _DEVICE_ROPE_CACHES[key] = cached
    return cached


def configure_rope_cache(
    module: nn.Module,
    config: TransformerConfig,
    *,
    rope_head_dim: int,
    base: float,
    yarn_disabled: bool = False,
) -> None:
    setattr(module, "_dsv4_rope_config", config)
    setattr(module, "_dsv4_rope_head_dim", int(rope_head_dim))
    setattr(module, "_dsv4_rope_base", float(base))
    setattr(module, "_dsv4_rope_yarn_disabled", bool(yarn_disabled))
    module.register_buffer(
        "freqs_cis", torch.empty(0, dtype=torch.complex64), persistent=False
    )


def materialize_rope_cache(
    module: nn.Module, device: torch.device | None = None, seqlen: int = 65536
) -> bool:
    config = getattr(module, "_dsv4_rope_config", None)
    if config is None:
        return False
    if device is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if device.type == "meta":
        return False
    freqs_cis = _get_device_rope_cache(
        config,
        rope_head_dim=int(getattr(module, "_dsv4_rope_head_dim")),
        base=float(getattr(module, "_dsv4_rope_base")),
        yarn_disabled=bool(getattr(module, "_dsv4_rope_yarn_disabled")),
        device=device,
        seqlen=seqlen,
    )
    module.freqs_cis = freqs_cis
    return True


def get_rope_cache(
    module: nn.Module, *, seqlen: int, device: torch.device
) -> torch.Tensor:
    freqs_cis = cast(torch.Tensor, module.freqs_cis)
    if (
        freqs_cis.numel() == 0
        or freqs_cis.device != device
        or freqs_cis.shape[0] < seqlen
    ):
        materialize_rope_cache(module, device, seqlen=seqlen)
        freqs_cis = cast(torch.Tensor, module.freqs_cis)
    return freqs_cis[:seqlen]


def get_rope_cache_at_positions(
    module: nn.Module, *, position_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    freqs_cis = cast(torch.Tensor, module.freqs_cis)
    safe_positions = position_ids.to(device=device, dtype=torch.long).clamp_min(0)
    seqlen = max(1, safe_positions.numel())
    if (
        freqs_cis.numel() == 0
        or freqs_cis.device != device
        or freqs_cis.shape[0] < seqlen
    ):
        materialize_rope_cache(module, device, seqlen=seqlen)
        freqs_cis = cast(torch.Tensor, module.freqs_cis)
    return freqs_cis.index_select(0, safe_positions.reshape(-1)).view(
        *safe_positions.shape,
        freqs_cis.shape[-1],
    )
