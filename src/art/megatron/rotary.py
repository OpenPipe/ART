from typing import Any

import torch


def absolute_rotary_pos_emb(
    module: Any,
    *,
    max_position: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rotary = module.rotary_pos_emb
    cache = getattr(module, "_art_absolute_rotary_pos_emb_cache", None)
    if cache is None:
        cache = {}
        setattr(module, "_art_absolute_rotary_pos_emb_cache", cache)
    key = (str(device), str(dtype), max_position + 1)
    if (cached := cache.get(key)) is not None:
        return cached
    freqs = rotary.get_freqs_non_repeated(max_position + 1)
    if not rotary.rotary_interleaved:
        result = torch.cat((freqs, freqs), dim=-1)
    else:
        result = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
            freqs.shape[0], -1
        )
    result = result[:, None, None, :].to(device=device, dtype=dtype)
    cache[key] = result
    return result


def rotary_pos_emb_for_positions(
    module: Any,
    position_ids: torch.Tensor,
    *,
    max_sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if position_ids.ndim != 2:
        raise ValueError(
            f"Rotary position ids must be rank two, got {position_ids.ndim}"
        )
    table = absolute_rotary_pos_emb(
        module,
        max_position=max_sequence_length - 1,
        dtype=dtype,
        device=device,
    )
    batch, sequence = position_ids.shape
    embedding = int(table.shape[-1])
    return (
        table.view(max_sequence_length, embedding)
        .index_select(0, position_ids.reshape(-1))
        .view(batch, sequence, embedding)
        .permute(1, 0, 2)
        .contiguous()
        .unsqueeze(2)
    )
