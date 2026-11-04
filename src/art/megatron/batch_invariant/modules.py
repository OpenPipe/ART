"""Replace instance forwards before DDP without changing parameter/checkpoint keys.

TE owns storage and Megatron owns the parallel topology. ART owns arithmetic.
No TE private helpers, global monkeypatches or alternate opt-in policy are used.
"""

import types

import torch

from .ops import linear, rms_norm


def has_invariant_forward(module, function):
    """Recognize our exact bound arithmetic, never a user-set marker alone."""
    forward = vars(module).get("forward")
    return (
        type(forward) is types.MethodType
        and forward.__self__ is module
        and forward.__func__ is function
    )


def _norm(module, x, *, fused=False):
    return rms_norm(
        x,
        module.layer_norm_weight if fused else module.weight,
        module.eps if module.eps is not None else torch.finfo(x.dtype).eps,
        zero_centered=getattr(module, "zero_centered_gamma", False),
    )


def _input_parallel(module, x):
    if getattr(module, "parallel_mode", None) != "column" or module.tp_size <= 1:
        return x
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
    )

    if module.sequence_parallel:
        return gather_from_sequence_parallel_region(x, group=module.tp_group)
    return copy_to_tensor_model_parallel_region(x, group=module.tp_group)


def _output_parallel(module, x):
    if getattr(module, "parallel_mode", None) != "row" or module.tp_size <= 1:
        return x
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    if module.sequence_parallel:
        return reduce_scatter_to_sequence_parallel_region(x, group=module.tp_group)
    return reduce_from_tensor_model_parallel_region(x, group=module.tp_group)


def _te_linear(module, x, is_first_microbatch=None, fp8_output=False, fp8_grad=False):
    del is_first_microbatch
    if fp8_output or fp8_grad or getattr(module, "fp8", False):
        raise ValueError("batch-invariant trainer requires unquantized GEMMs")
    fused = hasattr(module, "layer_norm_weight")
    local = _norm(module, x, fused=True) if fused else x
    gathered = _input_parallel(module, local)
    weight = module.weight.to(x.dtype)
    out = _output_parallel(module, linear(gathered, weight))
    bias = getattr(module, "bias", None) if getattr(module, "use_bias", False) else None
    if bias is not None:
        bias = bias.to(x.dtype)
        if module.apply_bias:
            from .ops import _Bias

            out = _Bias.apply(out, bias)
    returns = [out]
    if module.return_bias:
        returns.append(bias)
    if fused and module.return_layernorm_output:
        returns.append(gathered if module.return_layernorm_output_gathered else local)
    result = tuple(returns) if len(returns) > 1 else out
    # Megatron TE adapters always return (output, bias), including no-bias.
    if hasattr(module, "te_return_bias") and not module.te_return_bias:
        result = (result, None)
    module.is_first_microbatch = False
    return result


def _te_grouped(module, x, m_splits, is_first_microbatch=None):
    del is_first_microbatch
    if sum(m_splits) != x.shape[0] or len(m_splits) != module.num_gemms:
        raise ValueError("grouped GEMM splits must cover every input row")
    outputs, biases = [], []
    for i, part in enumerate(x.split(m_splits)):
        weight = getattr(module, f"weight{i}").to(x.dtype)
        bias = getattr(module, f"bias{i}", None) if module.use_bias else None
        bias = None if bias is None else bias.to(x.dtype)
        outputs.append(linear(part, weight, bias if module.apply_bias else None))
        biases.append(bias)
    out = torch.cat(outputs)
    module.is_first_microbatch = False
    if module.return_bias:
        return out, biases
    return (out, None) if hasattr(module, "te_return_bias") else out


def _rms_forward(module, x):
    return _norm(module, x)


def _residual_rms_forward(module, x):
    return _norm(module, x), x


def _torch_linear(module, x):
    return linear(x, module.weight, module.bias)


def _head_forward(module, input_, weight=None, runtime_gather_output=None):
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
    )

    weight = module.weight if weight is None else weight
    if weight is None:
        raise ValueError("output head requires a weight")
    x = input_
    if module.sequence_parallel:
        x = gather_from_sequence_parallel_region(x, group=module.tp_group)
    elif not module.disable_grad_reduce and not module.allreduce_dgrad:
        x = copy_to_tensor_model_parallel_region(x, group=module.tp_group)
    elif module.allreduce_dgrad:
        x = copy_to_tensor_model_parallel_region(x, group=module.tp_group)
    out = linear(x, weight, None if module.skip_bias_add else module.bias)
    gather = (
        module.gather_output if runtime_gather_output is None else runtime_gather_output
    )
    if gather:
        out = gather_from_tensor_model_parallel_region(out, group=module.tp_group)
    return out, module.bias if module.skip_bias_add else None


def _router_gating(module, x):
    dtype = getattr(module.config, "moe_router_dtype", None)
    dtype = {"fp32": torch.float32, "fp64": torch.float64}.get(dtype, x.dtype)
    return linear(
        x.to(dtype),
        module.weight.to(dtype),
        None if module.bias is None else module.bias.to(dtype),
    )


def install(chunks):
    """Pre-wrap hook for LoRA and full training; install once and return chunks.

    Fail before mutation for arithmetic modes that would silently bypass the
    contract. Shared tensors, dynamic slots, optimizer metadata and hooks survive.
    """
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.moe.router import TopKRouter
    import transformer_engine.pytorch as te

    replacements = []
    for chunk in chunks:
        for module in chunk.modules():
            if getattr(module, "_art_batch_invariant", False):
                continue
            forward = None
            if isinstance(module, te.LayerNormLinear):
                if module.normalization != "RMSNorm":
                    raise ValueError("batch-invariant fused linear requires RMSNorm")
                forward = _te_linear
            elif isinstance(module, te.Linear):
                forward = _te_linear
            elif isinstance(module, te.GroupedLinear):
                forward = _te_grouped
            elif isinstance(module, (te.RMSNorm, torch.nn.RMSNorm)):
                forward = (
                    _residual_rms_forward
                    if type(module).__name__ == "TEFusedResidualRMSNorm"
                    else _rms_forward
                )
            elif isinstance(module, torch.nn.Linear):
                forward = _torch_linear
            elif isinstance(module, ColumnParallelLinear):
                forward = _head_forward
            if forward is not None:
                config = getattr(module, "config", None)
                if any(
                    getattr(config, key, False)
                    for key in (
                        "fp8",
                        "fp4",
                        "delay_wgrad_compute",
                        "defer_embedding_wgrad_compute",
                    )
                ):
                    raise ValueError(
                        "batch-invariant arithmetic does not support quantized/delayed GEMMs"
                    )
                if len(getattr(module, "weight_names", ["weight"])) != 1:
                    raise ValueError(
                        "batch-invariant GEMM requires unsplit parameter storage"
                    )
                replacements.append((module, "forward", forward))
            if isinstance(module, TopKRouter):
                replacements.append((module, "gating", _router_gating))
    for module, name, function in replacements:
        setattr(module, name, types.MethodType(function, module))
        module._art_batch_invariant = True
    return chunks
