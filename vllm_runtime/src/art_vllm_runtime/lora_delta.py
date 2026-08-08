from collections.abc import Iterable
from contextlib import contextmanager
import math
from typing import Any

import torch

ART_LORA_DELTA_UPDATE_KIND = "lora_delta"
_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_GATE_UP_A_SUFFIX = ".base_layer.lora_A.weight"
_GATE_UP_B_SUFFIX = ".base_layer.lora_B.weight"
_PEFT_PREFIX = "base_model.model."
_UNSUPPORTED_MERGED_DELTA_TARGETS_KEY = (
    "art_merged_lora_delta_unsupported_target_modules"
)
_BLOCK_FP8_SCALE_ATTR = "_art_block_fp8_scale"
_BLOCK_FP8_SIZE_ATTR = "_art_block_fp8_size"


def _lora_scaling(adapter_config: dict[str, Any]) -> float:
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    return alpha / math.sqrt(rank) if adapter_config.get("use_rslora") else alpha / rank


def _checkpoint_base(base: str) -> str:
    if base.startswith(_PEFT_PREFIX):
        base = base.removeprefix(_PEFT_PREFIX)
    return base.removesuffix(".base_layer")


def _lora_delta(
    *,
    a_key: str,
    b_key: str,
    lora_tensors: dict[str, torch.Tensor],
    previous_lora_tensors: dict[str, torch.Tensor] | None,
    scaling: float,
) -> torch.Tensor:
    delta = lora_tensors[b_key].float().matmul(lora_tensors[a_key].float())
    delta.mul_(scaling)
    if previous_lora_tensors is None:
        return delta
    previous_delta = (
        previous_lora_tensors[b_key]
        .float()
        .matmul(previous_lora_tensors[a_key].float())
    )
    return delta.sub_(previous_delta.mul_(scaling))


def _unpack_expert_lora_b(tensor: torch.Tensor, *, rank: int) -> torch.Tensor:
    num_experts = tensor.shape[1] // rank
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _merged_delta_skips_experts(adapter_config: dict[str, Any]) -> bool:
    targets = adapter_config.get(_UNSUPPORTED_MERGED_DELTA_TARGETS_KEY) or ()
    return "experts" in set(targets)


def _iter_lora_checkpoint_deltas(
    lora_tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
    previous_lora_tensors: dict[str, torch.Tensor] | None,
) -> Iterable[tuple[str, torch.Tensor]]:
    rank = int(adapter_config["r"])
    scaling = _lora_scaling(adapter_config)
    skip_expert_deltas = _merged_delta_skips_experts(adapter_config)
    consumed: set[str] = set()
    for a_key in sorted(lora_tensors):
        if a_key.endswith(_GATE_UP_A_SUFFIX):
            prefix = a_key.removesuffix(_GATE_UP_A_SUFFIX)
            b_key = prefix + _GATE_UP_B_SUFFIX
            consumed.update((a_key, b_key))
            if skip_expert_deltas:
                continue
            a_tensor = lora_tensors[a_key]
            b_tensor = _unpack_expert_lora_b(lora_tensors[b_key], rank=rank)
            previous_b = (
                _unpack_expert_lora_b(previous_lora_tensors[b_key], rank=rank)
                if previous_lora_tensors is not None
                else None
            )
            checkpoint_prefix = _checkpoint_base(prefix)
            for expert_id, b_expert in enumerate(b_tensor):
                expert_a = a_tensor[expert_id * rank : (expert_id + 1) * rank]
                delta = b_expert.float().matmul(expert_a.float()).mul_(scaling)
                if previous_b is not None:
                    assert previous_lora_tensors is not None
                    previous_a = previous_lora_tensors[a_key][
                        expert_id * rank : (expert_id + 1) * rank
                    ]
                    delta.sub_(
                        previous_b[expert_id]
                        .float()
                        .matmul(previous_a.float())
                        .mul_(scaling)
                    )
                gate_delta, up_delta = delta.chunk(2, dim=0)
                yield f"{checkpoint_prefix}.{expert_id}.gate_proj.weight", gate_delta
                yield f"{checkpoint_prefix}.{expert_id}.up_proj.weight", up_delta
            continue
        if not a_key.endswith(_LORA_A_SUFFIX):
            continue
        prefix = a_key.removesuffix(_LORA_A_SUFFIX)
        b_key = prefix + _LORA_B_SUFFIX
        consumed.update((a_key, b_key))
        if prefix.endswith(".experts"):
            if skip_expert_deltas:
                continue
            a_tensor = lora_tensors[a_key]
            b_tensor = _unpack_expert_lora_b(lora_tensors[b_key], rank=rank)
            previous_b = (
                _unpack_expert_lora_b(previous_lora_tensors[b_key], rank=rank)
                if previous_lora_tensors is not None
                else None
            )
            checkpoint_prefix = _checkpoint_base(prefix)
            for expert_id, b_expert in enumerate(b_tensor):
                expert_a = a_tensor[expert_id * rank : (expert_id + 1) * rank]
                delta = b_expert.float().matmul(expert_a.float()).mul_(scaling)
                if previous_b is not None:
                    assert previous_lora_tensors is not None
                    previous_a = previous_lora_tensors[a_key][
                        expert_id * rank : (expert_id + 1) * rank
                    ]
                    delta.sub_(
                        previous_b[expert_id]
                        .float()
                        .matmul(previous_a.float())
                        .mul_(scaling)
                    )
                yield f"{checkpoint_prefix}.{expert_id}.down_proj.weight", delta
            continue
        yield (
            f"{_checkpoint_base(prefix)}.weight",
            _lora_delta(
                a_key=a_key,
                b_key=b_key,
                lora_tensors=lora_tensors,
                previous_lora_tensors=previous_lora_tensors,
                scaling=scaling,
            ),
        )
    unexpected = sorted(set(lora_tensors) - consumed)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA tensor keys: {unexpected}")


def _default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.copy_(loaded_weight.view(param.shape))
        return
    assert param.size() == loaded_weight.size(), (
        f"Attempted to load weight ({loaded_weight.size()}) into parameter "
        f"({param.size()})"
    )
    param.data.copy_(loaded_weight)


def _call_weight_loader(
    loader: Any,
    loader_param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if not hasattr(loader_param, "load_merged_column_weight"):
        owner = getattr(loader, "__self__", None)
        legacy_loader = getattr(owner, "weight_loader", None)
        if (
            legacy_loader is not None
            and legacy_loader is not loader
            and getattr(loader, "__name__", "") == "weight_loader_v2"
        ):
            return legacy_loader(loader_param, loaded_weight, *args, **kwargs)
    return loader(loader_param, loaded_weight, *args, **kwargs)


def _e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    bits = scale.view(torch.uint8).to(torch.int32) << 23
    return bits.view(torch.float32)


def _requantize_block_fp8_delta(
    param: torch.Tensor,
    delta: torch.Tensor,
) -> None:
    scale = getattr(param, _BLOCK_FP8_SCALE_ATTR)
    block_m, block_k = getattr(param, _BLOCK_FP8_SIZE_ATTR)
    _requantize_block_fp8_tensors(param.data, scale.data, delta, block_m, block_k)


def _requantize_block_fp8_tensors(
    weight: torch.Tensor,
    scale_data: torch.Tensor,
    delta: torch.Tensor,
    block_m: int,
    block_k: int,
) -> None:
    if weight.ndim == 3 and scale_data.ndim == 2:
        weight = weight.flatten(0, 1)
        delta = delta.flatten(0, 1)
    scale_float = (
        _e8m0_to_float(scale_data)
        if scale_data.dtype in (torch.float8_e8m0fnu, torch.uint8)
        else scale_data.float()
    )
    expanded = scale_float.repeat_interleave(block_m, -2).repeat_interleave(block_k, -1)
    merged = weight.float().mul_(expanded).add_(delta)
    leading = merged.shape[:-2]
    blocks = merged.view(
        *leading,
        merged.shape[-2] // block_m,
        block_m,
        merged.shape[-1] // block_k,
        block_k,
    )
    block_amax = blocks.abs().amax(dim=(-3, -1))
    new_scale = torch.pow(
        2.0,
        torch.ceil(
            torch.log2((block_amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny))
        ),
    )
    new_scale.masked_fill_(block_amax == 0, 1.0)
    expanded = new_scale.repeat_interleave(block_m, -2).repeat_interleave(block_k, -1)
    weight.copy_((merged / expanded).clamp_(-448, 448))
    if scale_data.dtype == torch.float8_e8m0fnu:
        scale_data.copy_(new_scale.to(scale_data.dtype))
    elif scale_data.dtype == torch.uint8:
        scale_data.copy_(new_scale.to(torch.float8_e8m0fnu).view(torch.uint8))
    else:
        scale_data.copy_(new_scale)


def _load_block_fp8_expert_delta(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    original_loader: Any,
    kwargs: dict[str, Any],
) -> bool | None:
    owner = getattr(original_loader, "__self__", None)
    map_expert = getattr(owner, "_map_global_expert_id_to_local_expert_id", None)
    if map_expert is None or "expert_id" not in kwargs:
        return None
    local_expert = map_expert(kwargs["expert_id"])
    if local_expert == -1:
        return False
    block_m, block_k = getattr(param, _BLOCK_FP8_SIZE_ATTR)
    weight = param.data[local_expert]
    scale = getattr(param, _BLOCK_FP8_SCALE_ATTR).data[local_expert]
    shard_id = kwargs["shard_id"]
    if shard_id in ("w1", "w3"):
        rows = loaded_weight.shape[-2]
        offset = 0 if shard_id == "w1" else weight.shape[-2] - rows
        weight = weight.narrow(-2, offset, rows)
        scale = scale.narrow(-2, offset // block_m, rows // block_m)
    assert weight.shape == loaded_weight.shape
    _requantize_block_fp8_tensors(
        weight,
        scale,
        loaded_weight.float(),
        block_m,
        block_k,
    )
    return True


def _additive_weight_loader(
    original_loader: Any,
    block_fp8_deltas: dict[torch.Tensor, torch.Tensor],
) -> Any:
    def load_delta(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        real_data = param.data
        is_block_fp8 = hasattr(param, _BLOCK_FP8_SCALE_ATTR)
        if is_block_fp8:
            expert_result = _load_block_fp8_expert_delta(
                param, loaded_weight, original_loader, kwargs
            )
            if expert_result is not None:
                return expert_result
        scratch = block_fp8_deltas.get(param)
        if scratch is None:
            scratch = torch.zeros_like(
                real_data,
                dtype=torch.float32 if is_block_fp8 else None,
            )
        param.data = scratch
        try:
            result = _call_weight_loader(
                original_loader,
                param,
                loaded_weight,
                *args,
                **kwargs,
            )
        finally:
            param.data = real_data
        if result is not False:
            if is_block_fp8:
                block_fp8_deltas[param] = scratch
            else:
                real_data.add_(scratch)
        return result

    return load_delta


@contextmanager
def _additive_weight_loaders(model: Any) -> Any:
    originals: list[tuple[torch.Tensor, bool, Any]] = []
    block_fp8_deltas: dict[torch.Tensor, torch.Tensor] = {}
    for param in model.parameters():
        has_loader = hasattr(param, "weight_loader")
        original_loader = getattr(param, "weight_loader", _default_weight_loader)
        originals.append((param, has_loader, original_loader))
        setattr(
            param,
            "weight_loader",
            _additive_weight_loader(original_loader, block_fp8_deltas),
        )
    try:
        yield
    except BaseException:
        raise
    else:
        for param, delta in block_fp8_deltas.items():
            _requantize_block_fp8_delta(param, delta)
    finally:
        for param, has_loader, original_loader in originals:
            if has_loader:
                setattr(param, "weight_loader", original_loader)
            else:
                delattr(param, "weight_loader")


@contextmanager
def _normalized_quantization_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "quantization_config", None) is not None:
        yield
        return
    config.quantization_config = {"quant_method": None}
    try:
        yield
    finally:
        config.quantization_config = None


def apply_lora_delta_update(
    *,
    model: Any,
    lora_tensors: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    previous_lora_tensors: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if previous_lora_tensors is not None and set(lora_tensors) != set(
        previous_lora_tensors
    ):
        raise RuntimeError(
            "LoRA update key set changed: "
            f"current={sorted(lora_tensors)} previous={sorted(previous_lora_tensors)}"
        )
    with (
        torch.no_grad(),
        _additive_weight_loaders(model),
        _normalized_quantization_config(model),
    ):
        model.load_weights(
            _iter_lora_checkpoint_deltas(
                lora_tensors,
                adapter_config=adapter_config,
                previous_lora_tensors=previous_lora_tensors,
            )
        )
    return {
        name: tensor.detach().clone() for name, tensor in sorted(lora_tensors.items())
    }
