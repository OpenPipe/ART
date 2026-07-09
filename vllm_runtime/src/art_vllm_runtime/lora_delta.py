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


def _additive_weight_loader(param: torch.Tensor, original_loader: Any) -> Any:
    def load_delta(
        loader_param: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        real_data = loader_param.data
        scratch = torch.zeros_like(real_data)
        loader_param.data = scratch
        try:
            result = _call_weight_loader(
                original_loader,
                loader_param,
                loaded_weight,
                *args,
                **kwargs,
            )
        finally:
            loader_param.data = real_data
        if result is not False:
            real_data.add_(scratch)
        return result

    return load_delta


@contextmanager
def _additive_weight_loaders(model: Any) -> Any:
    originals: list[tuple[torch.Tensor, bool, Any]] = []
    for param in model.parameters():
        has_loader = hasattr(param, "weight_loader")
        original_loader = getattr(param, "weight_loader", _default_weight_loader)
        originals.append((param, has_loader, original_loader))
        param.weight_loader = _additive_weight_loader(param, original_loader)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for param, has_loader, original_loader in originals:
            if has_loader:
                param.weight_loader = original_loader  # type: ignore[attr-defined]
            else:
                delattr(param, "weight_loader")


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
    with torch.no_grad(), _additive_weight_loaders(model):
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
