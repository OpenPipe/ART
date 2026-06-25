from __future__ import annotations

from typing import Any, Sequence, cast


def _context_parallel_world_size(config: Any) -> int:
    from megatron.core import parallel_state as ps
    from torch.distributed import is_initialized

    if is_initialized() and ps.model_parallel_is_initialized():
        return int(ps.get_context_parallel_world_size())
    return int(getattr(config, "context_parallel_size", 1) or 1)


def _build_absolute_rotary_pos_emb(
    module: Any,
    *,
    max_position: int,
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    rotary_pos_emb = module.rotary_pos_emb
    cache = getattr(module, "_art_absolute_rotary_pos_emb_cache", None)
    if cache is None:
        cache = {}
        setattr(module, "_art_absolute_rotary_pos_emb_cache", cache)
    cache_key = (str(device), max_position + 1)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    freqs = rotary_pos_emb.get_freqs_non_repeated(max_position + 1)
    if not rotary_pos_emb.rotary_interleaved:
        absolute_rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)
    else:
        absolute_rotary_pos_emb = torch.stack(
            (freqs.view(-1, 1), freqs.view(-1, 1)),
            dim=-1,
        ).view(freqs.shape[0], -1)
    absolute_rotary_pos_emb = absolute_rotary_pos_emb[:, None, None, :].to(
        device=device,
        dtype=dtype,
    )
    cache[cache_key] = absolute_rotary_pos_emb
    return absolute_rotary_pos_emb


def qwen3_forward_kwargs(model: Any, **kwargs: Any) -> dict[str, Any]:
    attention_bias = kwargs.get("attention_bias")
    from art.megatron.context_parallel.types import ArtContextParallelState

    module = model
    while hasattr(module, "module"):
        module = module.module
    gpt_module = getattr(module, "language_model", module)
    if isinstance(attention_bias, ArtContextParallelState):
        setattr(
            gpt_module,
            "_art_qwen3_rotary_seq_len",
            int(attention_bias.rank_plan.original_seq_len),
        )
    else:
        setattr(gpt_module, "_art_qwen3_rotary_seq_len", None)
    return {"extra_block_kwargs": kwargs}


def install_qwen3_text_preprocess_patch(model_chunks: Sequence[Any]) -> None:
    from megatron.core.models.gpt.gpt_model import GPTModel
    import torch

    for chunk in list(model_chunks):
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        gpt_module = (
            module
            if isinstance(module, GPTModel)
            else cast(GPTModel, getattr(module, "language_model"))
        )
        preprocess = gpt_module._preprocess

        def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
            position_ids = kwargs.get("position_ids")
            rotary_pos_emb = getattr(gpt_module, "rotary_pos_emb", None)
            rotary_cp_group = getattr(rotary_pos_emb, "cp_group", None)
            config = getattr(gpt_module, "config", None)
            cp_world_size = _context_parallel_world_size(config)
            uses_dispatched_local_cp_positions = (
                isinstance(position_ids, torch.Tensor)
                and position_ids.ndim == 2
                and cp_world_size > 1
                and rotary_cp_group is not None
            )
            if uses_dispatched_local_cp_positions:
                setattr(rotary_pos_emb, "cp_group", None)
            try:
                preproc_output = list(_preprocess(*args, **kwargs))
            finally:
                if uses_dispatched_local_cp_positions:
                    setattr(rotary_pos_emb, "cp_group", rotary_cp_group)
            decoder_input = cast(torch.Tensor, preproc_output[0])
            if not decoder_input.requires_grad and decoder_input.is_leaf:
                decoder_input.requires_grad_(True)
            position_ids = cast(torch.Tensor, position_ids)
            table = cast(torch.Tensor, preproc_output[1])
            if table is None:
                return tuple(preproc_output)
            embedding_dim = int(table.shape[-1])
            if (
                rotary_pos_emb is not None
                and getattr(gpt_module, "position_embedding_type", None) == "rope"
                and cp_world_size > 1
            ):
                rotary_seq_len = cast(
                    int,
                    getattr(gpt_module, "_art_qwen3_rotary_seq_len", None),
                )
                table_source = _build_absolute_rotary_pos_emb(
                    gpt_module,
                    max_position=int(rotary_seq_len) - 1,
                    dtype=table.dtype,
                    device=table.device,
                )
            else:
                table_source = table
            batch_size, sequence_length = position_ids.shape
            gathered = table_source.view(
                table_source.shape[0], embedding_dim
            ).index_select(
                0,
                position_ids.reshape(-1),
            )
            preproc_output[1] = (
                gathered.view(batch_size, sequence_length, embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
                .unsqueeze(2)
            )
            return tuple(preproc_output)

        gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]
