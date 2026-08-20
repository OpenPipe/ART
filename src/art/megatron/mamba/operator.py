from __future__ import annotations

from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from types import MethodType
from typing import Any, NamedTuple, Protocol, Sequence, cast

import torch
import torch.distributed as dist

from art.megatron.gdn.conv_gelu import packed_varlen_causal_conv

MAMBA_SSM_VERSION = "2.3.2.post1"
MAMBA_SSM_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
MAMBA_FAMILY_KEY = "mamba_2"
MAMBA_KERNEL_ID = (
    "mamba_ssm_2_3_2_post1_e9594ce1.chunk128.conv4.fp32_state.canonical_replay.v2"
)
MAMBA_LAYOUT_KEY = "mamba2.z_x_b_c_dt.head_group.v1"


class MambaBucketPlan(Protocol):
    length: int
    row_indices: torch.Tensor
    position_indices: torch.Tensor
    dense_token_indices: torch.Tensor | None
    dense_real_mask: torch.Tensor | None
    flat_token_indices: torch.Tensor | None
    family_indices_cpu: torch.Tensor | None
    family_indices_cpu_tuple: tuple[int, ...]
    parent_indices_cpu: torch.Tensor | None
    parent_indices_cpu_tuple: tuple[int, ...] | None
    output_mask: torch.Tensor | None
    real_token_count_static: int
    needs_final_state: bool

    @property
    def segment_count(self) -> int: ...


class MambaLocalParameters(NamedTuple):
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    dt_bias: torch.Tensor
    a_log: torch.Tensor
    d: torch.Tensor
    head_dim: int
    state_dim: int
    num_groups: int
    chunk_size: int


class MambaStateBundle(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


class _MambaRuntime(NamedTuple):
    contract: Any
    execution_spec: Any
    execution_plan: Any
    exchange_plan: Any
    cp_group: Any | None


def run_mamba_tree(
    projected: torch.Tensor,
    buckets_by_depth: Sequence[Sequence[MambaBucketPlan]],
    params: MambaLocalParameters,
) -> torch.Tensor:
    """Convolve physical segments once, then execute canonical scan columns."""

    _validate_projected(projected, params)
    kinds = {
        getattr(bucket, "output_mask", None) is not None
        for buckets in buckets_by_depth
        for bucket in buckets
    }
    if kinds != {False, True}:
        raise ValueError(
            "Mamba tree plans must pair physical convolution and canonical scan buckets"
        )
    postconv = _convolve_mamba_tree(projected, buckets_by_depth, params)
    output = projected.new_zeros(
        (*projected.shape[:2], params.dt_bias.numel() * params.head_dim)
    )
    cache = _MambaSsmStateCache(params, projected)
    postconv_flat = postconv.flatten(0, 1)
    output_flat = output.flatten(0, 1)
    for buckets in buckets_by_depth:
        for bucket in buckets:
            output_mask = getattr(bucket, "output_mask", None)
            if output_mask is None:
                continue
            parent_states = cache.parent_states(bucket)
            flat_indices = _bucket_tensor(bucket, "flat_token_indices")
            compact = postconv_flat.index_select(0, flat_indices)
            compact_output, final_states = _run_mamba_scan_bucket(
                compact,
                bucket,
                parent_states,
                params,
                output_final_state=bucket.needs_final_state,
            )
            write = _bucket_tensor(bucket, "output_mask")
            output_flat = output_flat.index_copy(
                0, flat_indices[write], compact_output[write]
            )
            if bucket.needs_final_state:
                if final_states is None:
                    raise RuntimeError(
                        "Mamba tree bucket did not return required final state"
                    )
                cache.append(bucket, final_states)
    return output_flat.view_as(output)


def run_mamba_bucket(
    compact_projected: torch.Tensor,
    bucket: MambaBucketPlan,
    parent_states: MambaStateBundle,
    params: MambaLocalParameters,
    *,
    output_final_state: bool,
) -> tuple[torch.Tensor, MambaStateBundle | None]:
    """Run one physical projected segment bucket through convolution and SSD."""

    heads = int(params.dt_bias.numel())
    inner = heads * params.head_dim
    group_width = params.num_groups * params.state_dim
    z, conv_input, dt = torch.split(
        compact_projected,
        [inner, inner + 2 * group_width, heads],
        dim=-1,
    )
    _validate_conv_bucket(conv_input, bucket, parent_states.conv, params)
    convolved, conv_final = packed_varlen_causal_conv(
        conv_input,
        _bucket_tensor(bucket, "cu_seqlens"),
        parent_states.conv,
        params.conv_weight,
        params.conv_bias,
        activation="silu",
        output_final_state=output_final_state,
    )
    output, ssm_final = _run_mamba_scan_bucket(
        torch.cat((z, convolved, dt), dim=-1),
        bucket,
        parent_states.ssm,
        params,
        output_final_state=output_final_state,
        require_output_mask=False,
    )
    final = (
        MambaStateBundle(conv_final, ssm_final)
        if conv_final is not None and ssm_final is not None
        else None
    )
    if output_final_state and final is None:
        raise RuntimeError("Mamba physical bucket omitted required final state")
    if final is not None and (
        final.conv.dtype != compact_projected.dtype or final.ssm.dtype != torch.float32
    ):
        raise TypeError("Mamba physical final-state dtypes violate the contract")
    return output, final


def _run_mamba_scan_bucket(
    compact_postconv: torch.Tensor,
    bucket: MambaBucketPlan,
    parent_states: torch.Tensor,
    params: MambaLocalParameters,
    *,
    output_final_state: bool,
    require_output_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the official autograd SSD on one canonical scan-column bucket."""

    _validate_scan_bucket(
        compact_postconv,
        bucket,
        parent_states,
        params,
        require_output_mask=require_output_mask,
    )
    heads = int(params.dt_bias.numel())
    inner = heads * params.head_dim
    group_width = params.num_groups * params.state_dim
    z, convolved, dt = torch.split(
        compact_postconv,
        [inner, inner + 2 * group_width, heads],
        dim=-1,
    )
    x, b, c = torch.split(convolved, [inner, group_width, group_width], dim=-1)
    batch, length = bucket.segment_count, bucket.length
    dense_indices = _bucket_tensor(bucket, "dense_token_indices")
    x = _compact_to_dense(x, dense_indices, batch, length).view(
        batch, length, heads, params.head_dim
    )
    b = _compact_to_dense(b, dense_indices, batch, length).view(
        batch, length, params.num_groups, params.state_dim
    )
    c = _compact_to_dense(c, dense_indices, batch, length).view(
        batch, length, params.num_groups, params.state_dim
    )
    z = _compact_to_dense(z, dense_indices, batch, length).view(
        batch, length, heads, params.head_dim
    )
    dense_dt = dt.new_full((batch * length, heads), -torch.inf)
    dense_dt = dense_dt.index_copy(0, dense_indices, dt).view(batch, length, heads)
    scan = _mamba_chunk_scan_combined()
    result = scan(
        x,
        dense_dt,
        -torch.exp(params.a_log.float()),
        b,
        c,
        params.chunk_size,
        D=params.d.float(),
        z=z,
        dt_bias=params.dt_bias.float(),
        initial_states=parent_states,
        dt_softplus=True,
        return_final_states=output_final_state,
        state_dtype=torch.float32,
    )
    if output_final_state:
        dense_output, ssm_final = result
    else:
        dense_output, ssm_final = result, None
    compact_output = dense_output.reshape(batch * length, inner).index_select(
        0, dense_indices
    )
    if output_final_state and ssm_final is None:
        raise RuntimeError("Mamba bucket final-state contract was not satisfied")
    if ssm_final is not None and ssm_final.dtype != torch.float32:
        raise TypeError("Mamba final SSM state must remain FP32")
    return compact_output, ssm_final


def _convolve_mamba_tree(
    projected: torch.Tensor,
    buckets_by_depth: Sequence[Sequence[MambaBucketPlan]],
    params: MambaLocalParameters,
) -> torch.Tensor:
    heads = int(params.dt_bias.numel())
    inner = heads * params.head_dim
    group_width = params.num_groups * params.state_dim
    projected_flat = projected.flatten(0, 1)
    z, conv_input, dt = torch.split(
        projected_flat, [inner, inner + 2 * group_width, heads], dim=-1
    )
    convolved_flat = conv_input.new_zeros(conv_input.shape)
    cache = _MambaConvStateCache(params, projected)
    for buckets in buckets_by_depth:
        for bucket in buckets:
            if getattr(bucket, "output_mask", None) is not None:
                continue
            flat_indices = _bucket_tensor(bucket, "flat_token_indices")
            compact = conv_input.index_select(0, flat_indices)
            parent_states = cache.parent_states(bucket)
            _validate_conv_bucket(compact, bucket, parent_states, params)
            convolved, final_states = packed_varlen_causal_conv(
                compact,
                _bucket_tensor(bucket, "cu_seqlens"),
                parent_states,
                params.conv_weight,
                params.conv_bias,
                activation="silu",
                output_final_state=bucket.needs_final_state,
            )
            convolved_flat = convolved_flat.index_copy(0, flat_indices, convolved)
            if bucket.needs_final_state:
                if final_states is None:
                    raise RuntimeError("Mamba convolution omitted required final state")
                cache.append(bucket, final_states)
    return torch.cat((z, convolved_flat, dt), dim=-1).view_as(projected)


class _MambaConvStateCache:
    def __init__(self, params: MambaLocalParameters, reference: torch.Tensor) -> None:
        self.params = params
        self.reference = reference
        self.states_by_family: list[torch.Tensor | None] = []

    def append(self, bucket: MambaBucketPlan, states: torch.Tensor) -> None:
        families = bucket.family_indices_cpu_tuple
        if states.shape[0] != len(families):
            raise ValueError("Mamba convolution states must match bucket families")
        if families:
            self.states_by_family.extend(
                None for _ in range(max(families) + 1 - len(self.states_by_family))
            )
        for row, family in enumerate(families):
            self.states_by_family[family] = states[row]

    def parent_states(self, bucket: MambaBucketPlan) -> torch.Tensor:
        parents = bucket.parent_indices_cpu_tuple
        if parents is None:
            raise ValueError("Mamba bucket requires prebuilt CPU parent indices")
        zero_conv = self.reference.new_zeros(
            self.params.conv_weight.shape[0],
            self.params.conv_weight.shape[1] - 1,
        )
        missing = []
        selected = []
        for parent in parents:
            if parent < 0:
                selected.append(zero_conv)
            elif (
                parent >= len(self.states_by_family)
                or self.states_by_family[parent] is None
            ):
                missing.append(parent)
            else:
                selected.append(cast(torch.Tensor, self.states_by_family[parent]))
        if missing:
            raise RuntimeError(
                f"Mamba convolution is missing parent states for families {missing}"
            )
        return torch.stack(selected)


class _MambaSsmStateCache:
    def __init__(self, params: MambaLocalParameters, reference: torch.Tensor) -> None:
        self.params = params
        self.reference = reference
        self.states_by_family: list[torch.Tensor | None] = []

    def append(self, bucket: MambaBucketPlan, states: torch.Tensor) -> None:
        families = bucket.family_indices_cpu_tuple
        if states.shape[0] != len(families):
            raise ValueError("Mamba SSM states must match bucket boundary families")
        if families:
            self.states_by_family.extend(
                None for _ in range(max(families) + 1 - len(self.states_by_family))
            )
        for row, family in enumerate(families):
            self.states_by_family[family] = states[row]

    def parent_states(self, bucket: MambaBucketPlan) -> torch.Tensor:
        parents = bucket.parent_indices_cpu_tuple
        if parents is None:
            raise ValueError("Mamba scan bucket requires boundary parent indices")
        zero = torch.zeros(
            self.params.dt_bias.numel(),
            self.params.head_dim,
            self.params.state_dim,
            dtype=torch.float32,
            device=self.reference.device,
        )
        missing = []
        selected = []
        for parent in parents:
            if parent < 0:
                selected.append(zero)
            elif (
                parent >= len(self.states_by_family)
                or self.states_by_family[parent] is None
            ):
                missing.append(parent)
            else:
                selected.append(cast(torch.Tensor, self.states_by_family[parent]))
        if missing:
            raise RuntimeError(f"Mamba scan is missing boundary states {missing}")
        return torch.stack(selected)


def _compact_to_dense(
    compact: torch.Tensor,
    dense_indices: torch.Tensor,
    batch: int,
    length: int,
) -> torch.Tensor:
    dense = compact.new_zeros((batch * length, compact.shape[-1]))
    return dense.index_copy(0, dense_indices, compact)


def _validate_projected(projected: torch.Tensor, params: MambaLocalParameters) -> None:
    _validate_parameters(params)
    if projected.ndim != 3:
        raise ValueError(
            f"Mamba projected input must be [batch, sequence, width], got {projected.shape}"
        )
    expected = _projected_width(params)
    if projected.shape[-1] != expected:
        raise ValueError(
            f"Mamba projected width must be {expected}, got {projected.shape[-1]}"
        )


def _validate_scan_bucket(
    postconv: torch.Tensor,
    bucket: MambaBucketPlan,
    states: torch.Tensor,
    params: MambaLocalParameters,
    *,
    require_output_mask: bool,
) -> None:
    if postconv.ndim != 2 or tuple(postconv.shape) != (
        bucket.real_token_count_static,
        _projected_width(params),
    ):
        raise ValueError(
            "compact Mamba bucket does not match its planned token/feature shape"
        )
    _validate_parameters(params)
    heads = params.dt_bias.numel()
    expected_ssm = (bucket.segment_count, heads, params.head_dim, params.state_dim)
    if tuple(states.shape) != expected_ssm or states.dtype != torch.float32:
        raise TypeError("Mamba boundary SSM states must have the planned FP32 shape")
    dense_indices = _bucket_tensor(bucket, "dense_token_indices")
    dense_real_mask = _bucket_tensor(bucket, "dense_real_mask")
    if dense_indices.device != postconv.device:
        raise ValueError("Mamba scan indices were not materialized on the input device")
    if tuple(dense_real_mask.shape) != (bucket.segment_count, bucket.length):
        raise ValueError("Mamba dense real-token mask has the wrong shape")
    if require_output_mask and tuple(_bucket_tensor(bucket, "output_mask").shape) != (
        bucket.real_token_count_static,
    ):
        raise ValueError("Mamba scan output mask has the wrong shape")


def _validate_conv_bucket(
    compact: torch.Tensor,
    bucket: MambaBucketPlan,
    states: torch.Tensor,
    params: MambaLocalParameters,
) -> None:
    _validate_parameters(params)
    conv_channels = params.conv_weight.shape[0]
    expected = (
        bucket.segment_count,
        conv_channels,
        params.conv_weight.shape[1] - 1,
    )
    if compact.ndim != 2 or tuple(compact.shape) != (
        bucket.real_token_count_static,
        conv_channels,
    ):
        raise ValueError("compact Mamba convolution input violates its plan")
    if tuple(states.shape) != expected or states.dtype != compact.dtype:
        raise TypeError("Mamba convolution states violate their planned shape/dtype")


def _validate_parameters(params: MambaLocalParameters) -> None:
    heads, conv_channels = params.dt_bias.numel(), params.conv_weight.shape[0]
    if params.conv_weight.ndim != 2 or params.conv_weight.shape[1] < 1:
        raise ValueError("Mamba conv weight must be [channels, kernel_width]")
    if (
        conv_channels
        != heads * params.head_dim + 2 * params.num_groups * params.state_dim
    ):
        raise ValueError("Mamba conv channels do not match head/group dimensions")
    if params.conv_bias is not None and tuple(params.conv_bias.shape) != (
        conv_channels,
    ):
        raise ValueError("Mamba conv bias shape does not match conv channels")
    if tuple(params.a_log.shape) != (heads,) or tuple(params.dt_bias.shape) != (heads,):
        raise ValueError("Mamba A_log/dt_bias must contain one value per local head")
    if tuple(params.d.shape) not in ((heads,), (heads, params.head_dim)):
        raise ValueError("Mamba D must be per-head or per-head-dimension")


def _projected_width(params: MambaLocalParameters) -> int:
    heads = int(params.dt_bias.numel())
    return (
        2 * heads * params.head_dim + 2 * params.num_groups * params.state_dim + heads
    )


def _bucket_tensor(bucket: MambaBucketPlan, name: str) -> torch.Tensor:
    tensor = getattr(bucket, name, None)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Mamba bucket is missing materialized {name}")
    return tensor


@cache
def _mamba_chunk_scan_combined() -> Any:
    try:
        installed = version("mamba-ssm")
    except PackageNotFoundError as error:
        raise ImportError(
            f"ART Mamba-2 requires mamba-ssm=={MAMBA_SSM_VERSION}"
        ) from error
    if installed != MAMBA_SSM_VERSION:
        raise ImportError(
            f"ART Mamba-2 requires mamba-ssm=={MAMBA_SSM_VERSION}, found {installed}"
        )
    try:
        module = import_module("mamba_ssm.ops.triton.ssd_combined")
    except ImportError as error:
        raise ImportError(
            "mamba-ssm is installed without the official Triton SSD implementation"
        ) from error
    scan = getattr(module, "mamba_chunk_scan_combined", None)
    if not callable(scan):
        raise ImportError("mamba-ssm does not expose mamba_chunk_scan_combined")
    if "state_dtype" not in signature(scan).parameters:
        raise ImportError(
            "ART Mamba-2 requires official mamba-ssm revision "
            f"{MAMBA_SSM_REVISION} with FP32 state autograd support"
        )
    return scan


def install_prefix_tree_mamba_hooks(model_chunks: Sequence[Any]) -> None:
    """Install explicit ART state plumbing on a Megatron Mamba model."""

    if not model_chunks:
        raise ValueError("Mamba hook installation requires at least one model chunk")
    MambaModel, MambaStack, MambaLayer, MambaMixer, _ = _mcore_mamba_types()
    for chunk_index, chunk in enumerate(model_chunks):
        if not isinstance(chunk, torch.nn.Module):
            raise TypeError("Mamba model chunks must be torch modules")
        found = {"model": 0, "stack": 0, "layer": 0, "mixer": 0}
        visited: set[int] = set()
        for wrapped in chunk.modules():
            module = _physical_module(wrapped)
            if id(module) in visited:
                continue
            visited.add(id(module))
            if isinstance(module, MambaModel):
                found["model"] += 1
                _install_forward(
                    module, "_art_mamba_model_forward", _mamba_model_forward
                )
                if not hasattr(module, "_art_decoder_hidden"):
                    module._art_decoder_hidden = MethodType(  # type: ignore[attr-defined]
                        _art_decoder_hidden, module
                    )
            elif isinstance(module, MambaStack):
                found["stack"] += 1
                _install_forward(
                    module, "_art_mamba_stack_forward", _mamba_stack_forward
                )
            elif isinstance(module, MambaLayer):
                found["layer"] += 1
                _install_forward(
                    module, "_art_mamba_layer_forward", _mamba_layer_forward
                )
            elif isinstance(module, MambaMixer):
                found["mixer"] += 1
                _install_forward(
                    module, "_art_mamba_mixer_forward", _mamba_mixer_forward
                )
        if found["model"] != 1 or found["stack"] != 1:
            raise RuntimeError(
                "each Mamba model chunk must contain exactly one model and stack; "
                f"chunk={chunk_index} counts={found}"
            )
        if found["layer"] != found["mixer"]:
            raise RuntimeError(
                "Mamba hook installation found unpaired local layers and mixers; "
                f"chunk={chunk_index} counts={found}"
            )


def _install_forward(module: Any, name: str, replacement: Any) -> None:
    original_name = f"{name}_original"
    if hasattr(module, original_name):
        return
    setattr(module, original_name, module.forward)
    module.forward = MethodType(replacement, module)


def _mamba_model_forward(
    self: Any,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    inference_context: Any | None = None,
    runtime_gather_output: bool | None = None,
    *,
    inference_params: Any | None = None,
    loss_mask: torch.Tensor | None = None,
    packed_seq_params: Any | None = None,
    padding_mask: torch.Tensor | None = None,
    is_spec_decode: bool | None = None,
    extra_block_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    attention_bias = _extract_attention_bias(extra_block_kwargs, required=True)
    del loss_mask
    _validate_prefix_invocation(
        self,
        inference_context=inference_context,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
    )
    if is_spec_decode:
        raise RuntimeError("ART prefix-tree Mamba does not support speculative decode")
    if bool(getattr(self, "mtp_process", False)):
        raise RuntimeError("ART prefix-tree Mamba does not support MTP")
    _mamba_runtime(attention_bias)
    hidden_states = _mamba_decoder_hidden(
        self,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        decoder_input=decoder_input,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        attention_bias=attention_bias,
    )
    if not self.post_process:
        return hidden_states
    output_weight = (
        self.shared_embedding_or_output_weight()
        if self.share_embeddings_and_output_weights
        else None
    )
    logits, _ = self.output_layer(
        hidden_states,
        weight=output_weight,
        runtime_gather_output=runtime_gather_output,
    )
    logits = self._scale_logits(logits)
    if labels is None:
        return logits.transpose(0, 1).contiguous()
    return self.compute_language_model_loss(labels, logits)


def _art_decoder_hidden(
    self: Any,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params: Any | None,
    extra_block_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    """Return PP1 decoder hidden state for ART's selective LM head."""

    if not self.pre_process or not self.post_process:
        raise RuntimeError("_art_decoder_hidden requires a PP1 Mamba model chunk")
    pp_group = getattr(getattr(self, "pg_collection", None), "pp", None)
    if pp_group is not None and callable(getattr(pp_group, "size", None)):
        if int(pp_group.size()) != 1:
            raise RuntimeError(
                "_art_decoder_hidden requires pipeline parallel size one"
            )
    attention_bias = _extract_attention_bias(extra_block_kwargs, required=True)
    _validate_prefix_invocation(
        self,
        inference_context=None,
        inference_params=None,
        packed_seq_params=packed_seq_params,
    )
    _mamba_runtime(attention_bias)
    return _mamba_decoder_hidden(
        self,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        decoder_input=None,
        packed_seq_params=packed_seq_params,
        padding_mask=None,
        attention_bias=attention_bias,
    )


def _mamba_decoder_hidden(
    model: Any,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input: torch.Tensor | None,
    packed_seq_params: Any | None,
    padding_mask: torch.Tensor | None,
    attention_bias: Any,
) -> torch.Tensor:
    if decoder_input is None and model.pre_process:
        decoder_input = model.embedding(
            input_ids=input_ids,
            position_ids=position_ids,
        )
    rotary_pos_emb = None
    if model.position_embedding_type == "rope":
        rotary_seq_len = model.rotary_pos_emb.get_rotary_seq_len(
            None,
            model.decoder,
            decoder_input,
            model.config,
            packed_seq_params,
        )
        rotary_pos_emb = model.rotary_pos_emb(
            rotary_seq_len,
            packed_seq=bool(
                packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            ),
        )
    return model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=None,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        attention_bias=attention_bias,
    )


def _mamba_stack_forward(
    self: Any,
    hidden_states: Any,
    attention_mask: torch.Tensor,
    inference_context: Any | None = None,
    rotary_pos_emb: torch.Tensor | None = None,
    *,
    inference_params: Any | None = None,
    packed_seq_params: Any | None = None,
    padding_mask: torch.Tensor | None = None,
    attention_bias: Any | None = None,
) -> torch.Tensor:
    if attention_bias is None:
        raise ValueError("ART MambaStack requires explicit recurrent state")
    _validate_prefix_invocation(
        self,
        inference_context=inference_context,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
    )
    _mamba_runtime(attention_bias)
    if not self.pre_process:
        hidden_states = self.input_tensor
    _, _, _, _, WrappedTensor = _mcore_mamba_types()
    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("MambaStack requires a tensor hidden state")

    def run_range(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(start, end):
            value = _call_mamba_stack_layer(
                self.layers[index],
                hidden_states=value,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                padding_mask=padding_mask,
                attention_bias=attention_bias,
            )
        return value

    granularity = getattr(self.config, "recompute_granularity", None)
    if granularity not in (None, "full"):
        raise RuntimeError(
            "ART prefix-tree Mamba supports eager or full activation recomputation"
        )
    if granularity == "full" and self.training:
        from megatron.core import tensor_parallel

        method = getattr(self.config, "recompute_method", None)
        count = int(getattr(self.config, "recompute_num_layers", 0) or 0)
        if method not in ("uniform", "block") or count <= 0:
            raise RuntimeError(
                "ART prefix-tree Mamba full recompute requires uniform/block and "
                "a positive recompute_num_layers"
            )
        # Reentrant checkpoints attach autograd only through explicit inputs.
        if torch.is_grad_enabled() and not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        def checkpoint_range(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
            def forward(tensor: torch.Tensor) -> torch.Tensor:
                return run_range(tensor, start, end)

            return tensor_parallel.checkpoint(
                forward,
                bool(self.config.distribute_saved_activations),
                value,
            )

        if method == "uniform":
            layer_index = 0
            while layer_index < len(self.layers):
                end = min(layer_index + count, len(self.layers))
                hidden_states = checkpoint_range(hidden_states, layer_index, end)
                layer_index = end
        else:
            for layer_index in range(len(self.layers)):
                hidden_states = (
                    checkpoint_range(hidden_states, layer_index, layer_index + 1)
                    if layer_index < count
                    else run_range(hidden_states, layer_index, layer_index + 1)
                )
    else:
        hidden_states = run_range(hidden_states, 0, len(self.layers))
    if self.post_process and self.post_layer_norm:
        hidden_states = self.final_norm(hidden_states)
    from megatron.core.utils import make_viewless_tensor

    return make_viewless_tensor(
        inp=hidden_states,
        requires_grad=hidden_states.requires_grad,
        keep_graph=True,
    )


def _call_mamba_stack_layer(
    layer: Any,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    attention_bias: Any,
) -> torch.Tensor:
    _, _, MambaLayer, _, _ = _mcore_mamba_types()
    from megatron.core.transformer.transformer_layer import TransformerLayer

    physical = _physical_module(layer)
    if isinstance(physical, TransformerLayer):
        result = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=padding_mask,
        )
    elif isinstance(physical, MambaLayer):
        result = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=None,
            packed_seq_params=None,
            attention_bias=attention_bias,
        )
    else:
        result = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=None,
            packed_seq_params=None,
        )
    return result[0] if isinstance(result, tuple) else result


def _mamba_layer_forward(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    inference_context: Any | None = None,
    rotary_pos_emb: torch.Tensor | None = None,
    *,
    inference_params: Any | None = None,
    packed_seq_params: Any | None = None,
    attention_bias: Any | None = None,
) -> torch.Tensor:
    if attention_bias is None:
        raise ValueError("ART MambaLayer requires explicit recurrent state")
    _validate_prefix_invocation(
        self,
        inference_context=inference_context,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
    )
    del rotary_pos_emb
    from megatron.core.typed_torch import apply_module

    residual = (
        hidden_states.float() if self.config.fp32_residual_connection else hidden_states
    )
    hidden_states = apply_module(self.norm)(
        hidden_states.to(dtype=self.config.params_dtype)
    )
    mixer_out_with_bias = self.mixer(
        hidden_states,
        inference_context=None,
        packed_seq_params=None,
        attention_bias=attention_bias,
    )
    with self.bias_dropout_add_exec_handler():
        return self.mamba_bda(
            training=self.training,
            fused=self.config.bias_dropout_fusion,
        )(mixer_out_with_bias, residual, self.hidden_dropout)


def _mamba_mixer_forward(
    self: Any,
    hidden_states: torch.Tensor,
    inference_context: Any | None = None,
    *,
    inference_params: Any | None = None,
    packed_seq_params: Any | None = None,
    attention_bias: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if attention_bias is None:
        raise ValueError("ART MambaMixer requires explicit recurrent state")
    projected, projection_bias = self.in_proj(hidden_states)
    if projection_bias is not None:
        raise RuntimeError(
            "ART prefix-tree Mamba requires a bias-free input projection"
        )
    if projected.ndim != 3:
        raise ValueError("Mamba input projection must return [sequence, batch, width]")
    output = _run_mamba_prefix_tree_eager(
        self,
        projected,
        attention_bias,
        inference_context=inference_context,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
    )
    if self.rmsnorm:
        output = self.norm(output)
    return self.out_proj(output)


@torch.compiler.disable
def _run_mamba_prefix_tree_eager(
    mixer: Any,
    projected: torch.Tensor,
    attention_bias: Any,
    *,
    inference_context: Any | None,
    inference_params: Any | None,
    packed_seq_params: Any | None,
) -> torch.Tensor:
    """Keep Python tree orchestration and raw A2A outside compiled compute."""

    _validate_prefix_invocation(
        mixer,
        inference_context=inference_context,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
    )
    runtime = _mamba_runtime(attention_bias, mixer=mixer, device=projected.device)
    batch, sequence = int(projected.shape[1]), int(projected.shape[0])
    if (batch, sequence) != (
        int(runtime.execution_spec.batch_size),
        int(runtime.execution_spec.sequence_length),
    ) and runtime.execution_plan.cp_size == 1:
        raise ValueError("CP1 Mamba projection shape does not match its execution spec")
    projected_flat = projected.transpose(0, 1).reshape(-1, projected.shape[-1])
    rank = int(runtime.execution_plan.cp_rank)
    local_token_count = int(runtime.execution_plan.external_token_counts_by_rank[rank])
    if runtime.execution_plan.cp_size == 1:
        local_projected = projected_flat.index_select(
            0, runtime.exchange_plan.canonical_flat_token_positions
        )
    else:
        if batch != 1 or projected_flat.shape[0] < local_token_count:
            raise ValueError(
                "CP Mamba requires one packed row with trailing-only padding"
            )
        local_projected = projected_flat[:local_token_count]
    from .exchange import (
        exchange_mamba_head_shards_to_attention,
        exchange_mamba_projected_to_head_shards,
    )

    canonical_projected = exchange_mamba_projected_to_head_shards(
        local_projected,
        runtime.exchange_plan,
        group=runtime.cp_group,
    )
    dense_shape = (
        int(runtime.execution_spec.batch_size),
        int(runtime.execution_spec.sequence_length),
        canonical_projected.shape[-1],
    )
    dense_projected = canonical_projected.new_zeros(
        (dense_shape[0] * dense_shape[1], dense_shape[2])
    ).index_copy(
        0,
        runtime.exchange_plan.canonical_flat_token_positions,
        canonical_projected,
    )
    params = _local_mamba_parameters(mixer, runtime)
    dense_output = run_mamba_tree(
        dense_projected.view(dense_shape),
        runtime.execution_plan.tree_segment_buckets_by_depth,
        params,
    )
    canonical_output = dense_output.flatten(0, 1).index_select(
        0, runtime.exchange_plan.canonical_flat_token_positions
    )
    local_output = exchange_mamba_head_shards_to_attention(
        canonical_output,
        runtime.exchange_plan,
        group=runtime.cp_group,
    )
    if runtime.execution_plan.cp_size == 1:
        restored = local_output.new_zeros(
            (projected_flat.shape[0], local_output.shape[-1])
        ).index_copy(
            0,
            runtime.exchange_plan.canonical_flat_token_positions,
            local_output,
        )
    else:
        padding = projected_flat.shape[0] - local_token_count
        restored = torch.cat(
            (local_output, local_output.new_zeros((padding, local_output.shape[-1])))
        )
    return restored.view(batch, sequence, -1).transpose(0, 1).contiguous()


def _local_mamba_parameters(mixer: Any, runtime: _MambaRuntime) -> MambaLocalParameters:
    plan = runtime.exchange_plan
    rank = int(runtime.execution_plan.cp_rank)
    conv_positions = plan.conv_feature_positions_by_rank[rank]
    head_positions = plan.head_positions_by_rank[rank]
    group_positions = plan.group_positions_by_rank[rank]
    head_feature_positions = plan.head_feature_positions_by_rank[rank]
    d = mixer.D.index_select(
        0, head_feature_positions if mixer.D_has_hdim else head_positions
    )
    if mixer.D_has_hdim:
        d = d.view(head_positions.numel(), mixer.headdim)
    return MambaLocalParameters(
        conv_weight=mixer.conv1d.weight.squeeze(1).index_select(0, conv_positions),
        conv_bias=(
            None
            if mixer.conv1d.bias is None
            else mixer.conv1d.bias.index_select(0, conv_positions)
        ),
        dt_bias=mixer.dt_bias.index_select(0, head_positions),
        a_log=mixer.A_log.index_select(0, head_positions),
        d=d,
        head_dim=int(mixer.headdim),
        state_dim=int(mixer.d_state),
        num_groups=int(group_positions.numel()),
        chunk_size=int(mixer.chunk_size),
    )


def _extract_attention_bias(
    extra_block_kwargs: dict[str, Any] | None,
    *,
    required: bool = False,
) -> Any | None:
    if extra_block_kwargs is None:
        if required:
            raise ValueError("ART Mamba requires extra_block_kwargs['attention_bias']")
        return None
    if not isinstance(extra_block_kwargs, dict) or set(extra_block_kwargs) != {
        "attention_bias"
    }:
        raise ValueError(
            "ART Mamba extra_block_kwargs must contain only attention_bias"
        )
    attention_bias = extra_block_kwargs["attention_bias"]
    if attention_bias is None:
        raise ValueError("ART Mamba attention_bias must not be None")
    return attention_bias


def _validate_prefix_invocation(
    module: Any,
    *,
    inference_context: Any | None,
    inference_params: Any | None,
    packed_seq_params: Any | None,
) -> None:
    if inference_context is not None or inference_params is not None:
        raise RuntimeError("ART prefix-tree Mamba does not support inference state")
    if packed_seq_params is not None:
        raise RuntimeError("ART prefix-tree Mamba requires ART dense-row metadata")
    config = module.config
    if bool(getattr(config, "fp8", False)) or getattr(config, "fp4", None) is not None:
        raise RuntimeError("ART prefix-tree Mamba does not yet support FP8 or FP4")


def _mamba_runtime(
    attention_state: Any,
    mixer: Any | None = None,
    device: torch.device | None = None,
) -> _MambaRuntime:
    from art.megatron.context_parallel.types import (
        HeadShardedRecurrentRankExecutionPlan,
    )
    from art.megatron.recurrent import (
        LinearRecurrentContract,
        RecurrentPackedExecutionSpec,
    )

    from .exchange import MambaHeadShardDevicePlan

    contract = getattr(attention_state, "linear_recurrent_contract", None)
    execution_spec = getattr(attention_state, "recurrent_execution_spec", None)
    execution_plan = getattr(attention_state, "recurrent_execution_plan", None)
    if not isinstance(contract, LinearRecurrentContract):
        raise TypeError("ART Mamba state is missing its linear recurrent contract")
    if not isinstance(execution_spec, RecurrentPackedExecutionSpec):
        raise TypeError("ART Mamba state is missing its recurrent execution spec")
    if not isinstance(execution_plan, HeadShardedRecurrentRankExecutionPlan):
        raise TypeError("ART Mamba state is missing its head-sharded execution plan")
    exchange_plan = execution_plan.exchange_plan
    if not isinstance(exchange_plan, MambaHeadShardDevicePlan):
        raise TypeError("ART Mamba exchange plan was not materialized on device")
    if (
        contract.family_key != MAMBA_FAMILY_KEY
        or contract.contract_version != "1"
        or contract.partition_kind != "head_sharded_full_tree"
        or contract.local_kernel_implementation_id != MAMBA_KERNEL_ID
        or contract.layout_compatibility_key != MAMBA_LAYOUT_KEY
        or contract.activation != "silu"
    ):
        raise ValueError("ART Mamba recurrent contract identity is incompatible")
    if tuple(stream.name for stream in contract.projected_streams) != (
        "z",
        "x",
        "B",
        "C",
        "dt",
    ) or tuple(state.name for state in contract.states) != ("conv", "ssm"):
        raise ValueError("ART Mamba contract stream/state order is incompatible")
    cpu = exchange_plan.cpu
    if (
        execution_plan.cp_size != cpu.cp_size
        or execution_plan.cp_rank >= execution_plan.cp_size
        or len(execution_plan.external_token_counts_by_rank) != cpu.cp_size
        or sum(execution_plan.external_token_counts_by_rank) != cpu.total_token_count
        or execution_spec.real_token_count != cpu.total_token_count
        or len(cpu.canonical_flat_token_positions) != cpu.total_token_count
        or max(cpu.canonical_flat_token_positions, default=-1)
        >= execution_spec.batch_size * execution_spec.sequence_length
    ):
        raise ValueError("ART Mamba execution and exchange plans disagree")
    cp_group = getattr(attention_state, "cp_group", None)
    if execution_plan.cp_size > 1 and cp_group is None:
        raise ValueError("ART Mamba CP state is missing its process group")
    if execution_plan.cp_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for ART Mamba CP")
        if (
            dist.get_world_size(cp_group) != execution_plan.cp_size
            or dist.get_rank(cp_group) != execution_plan.cp_rank
        ):
            raise ValueError(
                "ART Mamba execution-plan rank does not match its CP group"
            )
    runtime = _MambaRuntime(
        contract=contract,
        execution_spec=execution_spec,
        execution_plan=execution_plan,
        exchange_plan=exchange_plan,
        cp_group=cp_group,
    )
    if mixer is not None:
        _validate_mixer_contract(mixer, runtime)
    if device is not None:
        _validate_device_plan(runtime, device)
    return runtime


def _validate_mixer_contract(mixer: Any, runtime: _MambaRuntime) -> None:
    contract, cpu = runtime.contract, runtime.exchange_plan.cpu
    rank = int(runtime.execution_plan.cp_rank)
    heads = len(cpu.head_positions_by_rank[rank])
    groups = len(cpu.group_positions_by_rank[rank])
    conv_channels = heads * cpu.head_dim + 2 * groups * cpu.state_dim
    conv_channels_local_tp = (
        cpu.inner_width_local_tp + 2 * cpu.groups_local_tp * cpu.state_dim
    )
    expected_stream_widths = (
        cpu.inner_width_local_tp,
        cpu.inner_width_local_tp,
        cpu.groups_local_tp * cpu.state_dim,
        cpu.groups_local_tp * cpu.state_dim,
        cpu.heads_local_tp,
    )
    if (
        tuple(stream.width for stream in contract.projected_streams)
        != expected_stream_widths
    ):
        raise ValueError("ART Mamba projected stream widths do not match the mixer")
    if tuple(contract.states[0].shape) != (conv_channels, int(mixer.d_conv) - 1):
        raise ValueError("ART Mamba convolution state shape does not match the mixer")
    if contract.states[0].dtype != str(mixer.config.params_dtype).removeprefix(
        "torch."
    ):
        raise ValueError("ART Mamba convolution state dtype does not match the mixer")
    if (
        tuple(contract.states[1].shape)
        != (
            heads,
            int(mixer.headdim),
            int(mixer.d_state),
        )
        or contract.states[1].dtype != "float32"
    ):
        raise ValueError("ART Mamba SSM state contract does not match the mixer")
    if (
        contract.convolution_width != int(mixer.d_conv)
        or contract.local_chunk_size != int(mixer.chunk_size)
        or contract.local_chunk_size != 128
        or not bool(mixer.rmsnorm)
        or bool(mixer.norm_before_gate)
        or mixer.activation != "silu"
        or tuple(mixer.norm.weight.shape) != (cpu.inner_width_local_tp,)
        or int(mixer.norm.group_size) != cpu.inner_width_local_tp // cpu.groups_local_tp
        or cpu.heads_local_tp != int(mixer.nheads_local_tp)
        or cpu.head_dim != int(mixer.headdim)
        or cpu.groups_local_tp != int(mixer.ngroups_local_tp)
        or cpu.state_dim != int(mixer.d_state)
        or tuple(mixer.conv1d.weight.shape)
        != (conv_channels_local_tp, 1, contract.convolution_width)
        or (
            mixer.conv1d.bias is not None
            and tuple(mixer.conv1d.bias.shape) != (conv_channels_local_tp,)
        )
        or tuple(mixer.dt_bias.shape) != (cpu.heads_local_tp,)
        or tuple(mixer.A_log.shape) != (cpu.heads_local_tp,)
        or tuple(mixer.D.shape)
        != (
            (cpu.inner_width_local_tp,)
            if bool(mixer.D_has_hdim)
            else (cpu.heads_local_tp,)
        )
        or int(getattr(mixer.cp, "cp_size", cpu.cp_size)) != cpu.cp_size
        or (cpu.cp_size > 1 and int(getattr(mixer.cp, "cp_rank", -1)) != rank)
    ):
        raise ValueError("ART Mamba mixer geometry does not match its execution plan")


def _validate_device_plan(runtime: _MambaRuntime, device: torch.device) -> None:
    plan = runtime.exchange_plan
    exchange_indices = (
        *plan.token_positions_by_rank,
        *plan.projected_feature_positions_by_rank,
        *plan.conv_feature_positions_by_rank,
        *plan.head_positions_by_rank,
        *plan.group_positions_by_rank,
        *plan.head_feature_positions_by_rank,
        plan.canonical_to_received_positions,
        plan.canonical_flat_token_positions,
    )
    if any(index.device != device for index in exchange_indices):
        raise ValueError("ART Mamba exchange plan is on the wrong device")
    for buckets in runtime.execution_plan.tree_segment_buckets_by_depth:
        for bucket in buckets:
            names = (
                "cu_seqlens",
                "dense_token_indices",
                "dense_real_mask",
                "flat_token_indices",
                "parent_indices",
            ) + (("output_mask",) if bucket.output_mask is not None else ())
            for name in names:
                if _bucket_tensor(bucket, name).device != device:
                    raise ValueError(f"ART Mamba bucket {name} is on the wrong device")


def _physical_module(module: Any) -> Any:
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        original = getattr(module, "_orig_mod", None)
        if not isinstance(original, torch.nn.Module):
            return module
        module = original
    raise RuntimeError("cyclic torch.compile module wrapper")


@cache
def _mcore_mamba_types() -> tuple[
    type[Any], type[Any], type[Any], type[Any], type[Any]
]:
    from megatron.core.models.mamba.mamba_model import MambaModel
    from megatron.core.ssm.mamba_block import MambaStack
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.utils import WrappedTensor

    return MambaModel, MambaStack, MambaLayer, MambaMixer, WrappedTensor
