from __future__ import annotations

import argparse
from contextlib import ExitStack
import faulthandler
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, cast

import torch
import torch.nn.functional as F

from art.megatron import train as megatron_train
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
)
from art.megatron.routing_replay import (
    ParallelTopology as ReplayParallelTopology,
)
from art.megatron.training.trace import prepare_replay_local_input_token_uids
from art.megatron.weights.merged_weight_export import build_art_conversion_tasks
from art.preprocessing.pack import packed_tensors_from_dir

from .gdn_fp32_reference import install_megatron_qwen35_gdn_fp32_reference
from .hf_parity import (
    HF_PARITY_REPORT_FILENAME,
    HfParityRunRequest,
    _hf_parity_phase_pass_fns,
    build_hf_parity_report,
    build_parity_sample_indices,
    build_tensor_map_metric_rows,
    set_hf_config_num_layers,
    summarize_tensor_pair,
    zero_hf_dropout_config,
)
from .hf_parity_canonicalization import hf_tensor_map_to_art_canonical
from .oracle_harness import (
    ORACLE_TOPOLOGY,
    TEST_DEFAULT_FLEX_BACKEND,
    _read_json,
    _write_json,
)
from .oracle_worker import (
    _apply_requested_flex_backend_patch,
    _apply_test_attention_full_fp32_patch,
    _apply_test_flex_inner_fp32_patch,
    _assert_runtime_configuration,
    _build_optimizer_config,
    _configure_cuda_precision,
    _configure_provider,
    _set_deterministic_seed,
)
from .test_inputs import build_sft_trajectory_tensors_from_packed_tensors

HF_PARITY_DEBUG_ENV = "ART_HF_PARITY_DEBUG"
_DEBUG_START_TIME = time.perf_counter()
_VISUAL_HF_PREFIXES = ("model.visual.", "visual.")
_HF_MOE_ROUTER_NAME_PATTERN = re.compile(
    r"^(?:"
    r"model\.layers\.(?P<gate_layer>\d+)\.mlp\.gate|"
    r"model(?:\.language_model)?\.layers\.(?P<mlp_router_layer>\d+)\.mlp\.router|"
    r"model(?:\.language_model)?\.layers\.(?P<router_layer>\d+)\.router"
    r")$"
)
_REPLAY_ROUTER_LAYER_PATTERN = re.compile(
    r"^chunk_\d+\.layer_(?P<layer>\d+)\.mlp\.router$"
)
_GATE_WEIGHT_PATTERN = re.compile(
    r"^model(?:\.language_model)?\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$"
)
_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^model(?:\.language_model)?\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?:down_proj|gate_proj|up_proj)\.weight$"
)
_GEMMA4_ROUTER_PROJ_WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>model(?:\.language_model)?\.layers\.\d+\.)"
    r"router\.proj\.weight$"
)
_GEMMA4_SHARED_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>model(?:\.language_model)?\.layers\.\d+\.)"
    r"mlp\.(?:gate_proj|up_proj)\.weight$"
)
_GEMMA4_ABSENT_V_PROJ_WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>model(?:\.language_model)?\.layers\.\d+\.self_attn\.)"
    r"v_proj\.weight$"
)
_GEMMA4_REPARAMETERIZED_NORM_GRAD_PATTERN = re.compile(
    r"^model(?:\.language_model)?\.layers\.\d+\.pre_feedforward_layernorm_2\.weight$"
)


def _hf_moe_router_key(module_name: str) -> str | None:
    match = _HF_MOE_ROUTER_NAME_PATTERN.match(module_name)
    if match is None:
        return None
    layer = (
        match.group("gate_layer")
        or match.group("mlp_router_layer")
        or match.group("router_layer")
    )
    return f"chunk_00.layer_{int(layer):04d}.mlp.router"


def _hf_router_num_experts(module: Any, router_scores: torch.Tensor) -> int:
    config = getattr(module, "config", None)
    return int(
        getattr(
            module,
            "num_experts",
            getattr(config, "num_experts", router_scores.shape[-1]),
        )
    )


class _HfMoeRoutingCapture:
    def __init__(self, model: Any) -> None:
        self._handles: list[Any] = []
        self._routes: dict[str, dict[int, RouterCallRoute]] = {}
        self._active_sample_index: int | None = None
        self._active_micro_slot = 0
        for module_name, module in model.named_modules():
            router_key = _hf_moe_router_key(module_name)
            if router_key is None:
                continue
            self._routes[router_key] = {}
            self._handles.append(
                module.register_forward_hook(self._make_hook(router_key, module))
            )

    @property
    def enabled(self) -> bool:
        return bool(self._handles)

    def set_active_micro(self, sample_index: int | None, micro_slot: int) -> None:
        self._active_sample_index = sample_index
        self._active_micro_slot = micro_slot

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def build_replay_bundle(
        self,
        *,
        topology: ReplayParallelTopology,
    ) -> MoeRoutingReplayBundle | None:
        if not self.enabled:
            return None
        routers: dict[str, StepRouterRoutes] = {}
        max_topk = 0
        num_global_tokens: int | None = None
        for router_key in sorted(self._routes):
            calls = self._routes[router_key]
            if not calls:
                raise RuntimeError(f"HF parity captured no routes for '{router_key}'")
            routers[router_key] = StepRouterRoutes(calls=calls)
            for route in calls.values():
                max_topk = max(max_topk, route.max_topk)
                if num_global_tokens is None:
                    num_global_tokens = route.num_global_tokens
                elif num_global_tokens != route.num_global_tokens:
                    raise RuntimeError(
                        "HF parity routing capture token count mismatch: "
                        f"expected={num_global_tokens}, got={route.num_global_tokens}, "
                        f"router='{router_key}'"
                    )
        if num_global_tokens is None:
            raise RuntimeError("HF parity routing capture produced no route tokens")
        return MoeRoutingReplayBundle(
            topology=topology,
            num_steps=1,
            max_topk=max_topk,
            router_keys=sorted(routers),
            steps={
                0: StepRoutes(
                    routers=routers,
                    global_token_uids=torch.arange(
                        num_global_tokens, dtype=torch.int64
                    ),
                )
            },
        )

    def _make_hook(self, router_key: str, module: Any) -> Any:
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 3:
                raise RuntimeError(
                    f"Expected HF router tuple output for '{router_key}', got {type(output)}"
                )
            router_scores = output[1]
            router_indices = output[2]
            if not isinstance(router_scores, torch.Tensor) or not isinstance(
                router_indices, torch.Tensor
            ):
                raise RuntimeError(
                    f"Expected tensor router outputs for '{router_key}', "
                    f"got scores={type(router_scores)} indices={type(router_indices)}"
                )
            route = RouterCallRoute(
                expert_indices=router_indices.detach().cpu().to(torch.int32),
                expert_probs=router_scores.detach().cpu().to(torch.float32),
                expert_mask=torch.ones_like(
                    router_indices.detach().cpu(), dtype=torch.bool
                ),
                num_experts=_hf_router_num_experts(module, router_scores),
                sample_index=self._active_sample_index,
                micro_slot=(
                    None
                    if self._active_sample_index is not None
                    else self._active_micro_slot
                ),
            )
            self._routes[router_key][len(self._routes[router_key])] = route

        return _hook


def _debug(message: str) -> None:
    if os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    elapsed = time.perf_counter() - _DEBUG_START_TIME
    print(f"[hf_parity +{elapsed:8.2f}s] {message}", flush=True)


def _enable_debug_traceback_dump() -> None:
    if os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)


def _debug_enabled() -> bool:
    return os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _install_bridge_timing_debug(provider_bundle: Any) -> None:
    if not _debug_enabled():
        return
    provider = provider_bundle.provider
    pre_wrap_hooks = list(getattr(provider, "_pre_wrap_hooks", []))
    _debug(
        "registered pre-wrap hooks: "
        + ", ".join(
            getattr(hook, "__qualname__", repr(hook)) for hook in pre_wrap_hooks
        )
    )
    timed_hooks = []
    for index, hook in enumerate(pre_wrap_hooks):
        label = f"pre_wrap_hook[{index}]"

        def _timed_hook(
            model: list[Any], _hook: Any = hook, _label: str = label
        ) -> list[Any]:
            start = time.perf_counter()
            _debug(f"{_label}: start")
            try:
                return _hook(model)
            finally:
                _debug(f"{_label}: done in {time.perf_counter() - start:.2f}s")

        timed_hooks.append(_timed_hook)
    if pre_wrap_hooks:
        provider._pre_wrap_hooks = timed_hooks

    model_bridge = getattr(provider_bundle.bridge, "_model_bridge", None)
    if model_bridge is None:
        return
    if getattr(model_bridge, "_art_hf_parity_timing_wrapped", False):
        return
    original = model_bridge.load_weights_hf_to_megatron

    def _timed_load_weights(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        _debug("bridge.load_weights_hf_to_megatron: start")
        try:
            return original(*args, **kwargs)
        finally:
            _debug(
                "bridge.load_weights_hf_to_megatron: done in "
                f"{time.perf_counter() - start:.2f}s"
            )

    model_bridge.load_weights_hf_to_megatron = _timed_load_weights
    model_bridge._art_hf_parity_timing_wrapped = True


def _load_hf_model(
    *,
    base_model: str,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    set_hf_config_num_layers(config, num_layers)
    zero_hf_dropout_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.train()
    return cast(Any, model).to(device)


def _collect_hf_grads(model: Any) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param)
        grads[name] = grad.detach().cpu().to(dtype=torch.float32)
    return grads


def _bridge_compatible_hf_key(key: str, expected_keys: set[str]) -> str:
    if key in expected_keys:
        return key
    if key.startswith("model."):
        prefixed = f"model.language_model.{key.removeprefix('model.')}"
        if prefixed in expected_keys:
            return prefixed
    if key.startswith("model.language_model."):
        stripped = f"model.{key.removeprefix('model.language_model.')}"
        if stripped in expected_keys:
            return stripped
    return key


def _normalize_hf_tensor_map_for_bridge(
    hf_map: dict[str, torch.Tensor],
    expected_keys: set[str],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in hf_map.items():
        normalized_key = _bridge_compatible_hf_key(key, expected_keys)
        if normalized_key in normalized:
            raise RuntimeError(
                f"Duplicate normalized HF key '{normalized_key}' from '{key}'"
            )
        normalized[normalized_key] = value
    return normalized


def _active_embedding_token_rows(
    micro_inputs: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    active_token_ids: list[torch.Tensor] = []
    for micro in micro_inputs:
        attention_mask = micro["attention_mask"].reshape(-1).to(dtype=torch.bool)
        if not bool(attention_mask.any()):
            continue
        active_token_ids.append(micro["input_ids"].reshape(-1)[attention_mask].cpu())
    if not active_token_ids:
        return torch.zeros((0,), dtype=torch.long)
    return torch.unique(torch.cat(active_token_ids, dim=0), sorted=True)


def _active_router_rows_by_layer(
    replay_bundle: MoeRoutingReplayBundle | None,
) -> dict[int, torch.Tensor]:
    if replay_bundle is None:
        return {}
    active_rows: dict[int, torch.Tensor] = {}
    step_routes = replay_bundle.steps.get(0)
    if step_routes is None:
        return {}
    for router_key, router_routes in step_routes.routers.items():
        match = _REPLAY_ROUTER_LAYER_PATTERN.match(router_key)
        if match is None:
            continue
        layer_index = int(match.group("layer"))
        layer_rows: list[torch.Tensor] = []
        for route in router_routes.calls.values():
            if route.expert_indices.numel() == 0:
                continue
            layer_rows.append(route.expert_indices[route.expert_mask].to(torch.long))
        if layer_rows:
            active_rows[layer_index] = torch.unique(
                torch.cat(layer_rows, dim=0),
                sorted=True,
            )
    return active_rows


def _loss_active_last_layer_experts(
    replay_bundle: MoeRoutingReplayBundle | None,
    micro_inputs: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    *,
    layer_index: int,
) -> set[int]:
    if replay_bundle is None:
        return set()
    experts: set[int] = set()
    step_routes = replay_bundle.steps.get(0)
    if step_routes is None:
        return experts
    for router_key, router_routes in step_routes.routers.items():
        match = _REPLAY_ROUTER_LAYER_PATTERN.match(router_key)
        if match is None or int(match.group("layer")) != layer_index:
            continue
        for route in router_routes.calls.values():
            micro_index = (
                sample_indices.index(route.sample_index)
                if route.sample_index is not None
                else route.micro_slot
            )
            if micro_index is None:
                continue
            micro = micro_inputs[micro_index]
            actual_len = max(int(micro["attention_mask"].reshape(-1).sum().item()), 1)
            shifted_labels = megatron_train.shift_tensor(
                micro["labels"].reshape(-1)[:actual_len].unsqueeze(0), -100
            ).reshape(-1)
            loss_mask = (shifted_labels != -100).cpu()
            selected = route.expert_indices[loss_mask][route.expert_mask[loss_mask]]
            experts.update(int(expert) for expert in selected.reshape(-1).tolist())
    return experts


def _focus_derivative_tensor_map(
    tensor_map: dict[str, torch.Tensor],
    *,
    active_embedding_rows: torch.Tensor,
    active_router_rows: dict[int, torch.Tensor],
    last_layer_index: int,
    loss_active_last_layer_experts: set[int],
) -> dict[str, torch.Tensor]:
    focused: dict[str, torch.Tensor] = {}
    for key, value in tensor_map.items():
        if match := _EXPERT_WEIGHT_PATTERN.match(key):
            if (
                int(match.group("layer")) == last_layer_index
                and int(match.group("expert")) not in loss_active_last_layer_experts
            ):
                continue
        focused_value = value
        if (
            key == "model.language_model.embed_tokens.weight"
            and active_embedding_rows.numel() > 0
        ):
            focused_value = value.index_select(0, active_embedding_rows)
        elif match := _GATE_WEIGHT_PATTERN.match(key):
            active_rows = active_router_rows.get(int(match.group("layer")))
            if active_rows is not None and active_rows.numel() > 0:
                focused_value = value.index_select(0, active_rows)
        focused[key] = focused_value
    return focused


def _run_hf_sft_step(
    *,
    base_model: str,
    num_layers: int,
    micro_inputs: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    topology: ReplayParallelTopology,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    MoeRoutingReplayBundle | None,
]:
    _debug("loading HF model")
    model = _load_hf_model(
        base_model=base_model,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )
    if dtype == torch.float32:
        _install_hf_qwen35_gdn_fp32_reference(model, base_model=base_model)
    route_capture = _HfMoeRoutingCapture(model)
    _debug("running HF forward/backward")
    model.zero_grad(set_to_none=True)
    loss_sum = torch.tensor(0.0, device=device)
    token_count = 0
    trainable_losses: list[torch.Tensor] = []
    total_token_count = max(
        sum(
            int(megatron_train._count_sft_trainable_tokens(micro))
            for micro in micro_inputs
        ),
        1,
    )
    for micro_slot, (micro, sample_index) in enumerate(
        zip(micro_inputs, sample_indices, strict=True)
    ):
        route_capture.set_active_micro(sample_index, micro_slot)
        attention_mask = micro["attention_mask"].reshape(-1)
        actual_len = max(int(attention_mask.sum().item()), 1)
        input_ids = micro["input_ids"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
        labels = micro["labels"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
        hf_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        logits = model(
            input_ids=input_ids,
            attention_mask=hf_attention_mask,
            use_cache=False,
        ).logits
        shifted_labels = megatron_train.shift_tensor(labels, -100)
        per_token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(shifted_labels.shape)
        mask = shifted_labels != -100
        masked_losses = per_token_loss[mask]
        trainable_losses.append(masked_losses.detach().cpu())
        loss_sum = loss_sum + masked_losses.sum()
        token_count += int(mask.sum().item())
        (masked_losses.sum() / total_token_count).backward()
    grads = _collect_hf_grads(model)
    routing_replay_bundle = route_capture.build_replay_bundle(topology=topology)
    scalar_loss = (loss_sum / max(token_count, 1)).detach().cpu().reshape(1)
    output_vector = torch.cat(trainable_losses, dim=0).to(dtype=torch.float32)
    route_capture.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _debug("finished HF step")
    return output_vector, scalar_loss, grads, routing_replay_bundle


def _install_hf_qwen35_gdn_fp32_reference(model: Any, *, base_model: str) -> None:
    model_key = base_model.lower()
    if "qwen3.5" not in model_key and "qwen3_5" not in model_key:
        return
    patched = 0
    for module in model.modules():
        module_impl = sys.modules.get(type(module).__module__)
        torch_impl = getattr(module_impl, "torch_chunk_gated_delta_rule", None)
        if torch_impl is None or not hasattr(module, "chunk_gated_delta_rule"):
            continue
        module.chunk_gated_delta_rule = torch_impl
        patched += 1
    if patched == 0:
        raise RuntimeError("Qwen3.5 HF parity found no GDN modules to patch")


def _build_megatron_runtime(
    request: HfParityRunRequest,
    *,
    moe_routing_replay_bundle: MoeRoutingReplayBundle | None = None,
) -> megatron_train.TrainingRuntime:
    return megatron_train.build_training_runtime(
        model_identifier=request.case_config.base_model,
        provider_torch_dtype=_dtype_for_precision(request.case_config.precision),
        provider_bundle_configure=_install_bridge_timing_debug,
        provider_configure=lambda provider: _configure_provider(
            provider, ORACLE_TOPOLOGY, request.case_config
        ),
        optimizer_config=_build_optimizer_config(request.case_config),
        moe_routing_replay_bundle=moe_routing_replay_bundle,
        moe_routing_replay_strict=True,
        print_env=False,
        trainable_parameter_mode="base_model",
        allow_unvalidated_arch=request.case_config.allow_unvalidated_arch,
    )


def _dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported HF parity precision: {precision}")


def _megatron_task_tensor(
    task: Any,
    *,
    mode: str,
) -> torch.Tensor:
    param = cast(torch.nn.Parameter, task.param_weight)
    if mode == "grad":
        grad = param.grad
        if grad is None:
            grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = torch.zeros_like(param)
        if hasattr(grad, "_local_tensor"):
            grad = cast(torch.Tensor, grad._local_tensor)
        return cast(torch.Tensor, grad)
    if mode == "param":
        return param.detach()
    raise ValueError(f"Unsupported task-tensor mode: {mode}")


def _mapping_supports_derivative_parity(mapping: Any) -> bool:
    from megatron.bridge.models.conversion.param_mapping import (
        RMSNorm2ZeroCenteredRMSNormMapping,
    )

    return not isinstance(mapping, RMSNorm2ZeroCenteredRMSNormMapping)


def _is_language_hf_param_name(name: str) -> bool:
    return not name.startswith(_VISUAL_HF_PREFIXES)


def _language_hf_param_names(mapping: Any) -> list[str]:
    hf_param = mapping.hf_param
    if isinstance(hf_param, str):
        return [hf_param]
    if isinstance(hf_param, dict):
        return [value for value in hf_param.values() if isinstance(value, str)]
    return []


def _mapping_targets_language_only(mapping: Any) -> bool:
    names = _language_hf_param_names(mapping)
    if not names:
        return True
    return all(_is_language_hf_param_name(name) for name in names)


def _filter_language_only_tensor_map(
    tensor_map: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in tensor_map.items()
        if _is_language_hf_param_name(key)
    }


def _is_gemma4_model_bridge(model_bridge: Any) -> bool:
    return "Gemma4" in type(model_bridge).__name__


def _add_converted_hf_grad(
    converted: dict[str, torch.Tensor],
    additive_keys: set[str],
    key: str,
    value: torch.Tensor,
    *,
    additive: bool = False,
) -> None:
    if key in converted:
        converted[key] = converted[key] + value
    else:
        converted[key] = value
    if additive:
        additive_keys.add(key)


def _maybe_modify_converted_hf_grad(
    model_bridge: Any,
    task: Any,
    converted_weights_dict: dict[str, torch.Tensor],
    hf_state_dict: Any,
    *,
    model_is_moe: bool,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    if not _is_gemma4_model_bridge(model_bridge):
        return (
            model_bridge.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            ),
            set(),
        )

    converted: dict[str, torch.Tensor] = {}
    additive_keys: set[str] = set()
    for hf_name, tensor in converted_weights_dict.items():
        if hf_name not in hf_state_dict:
            if match := _GEMMA4_ABSENT_V_PROJ_WEIGHT_PATTERN.match(hf_name):
                k_name = f"{match.group('prefix')}k_proj.weight"
                hf_state_dict[k_name]
                _add_converted_hf_grad(
                    converted,
                    additive_keys,
                    k_name,
                    tensor.float(),
                    additive=True,
                )
            continue
        grad = tensor.float()

        if model_is_moe and (
            match := _GEMMA4_ROUTER_PROJ_WEIGHT_PATTERN.match(hf_name)
        ):
            prefix = match.group("prefix")
            scale = hf_state_dict[f"{prefix}router.scale"].float().to(grad.device)
            ln2 = (
                hf_state_dict[f"{prefix}pre_feedforward_layernorm_2.weight"]
                .float()
                .to(grad.device)
            )
            hf_weight = hf_state_dict[hf_name].float().to(grad.device)
            root = grad.shape[-1] ** -0.5
            factor = scale * root / ln2
            # Gemma 4 imports fold HF preprocessing into MCore weights. Value
            # export divides by this factor, but derivative export must apply the
            # chain rule and accumulate the induced norm-weight gradient.
            _add_converted_hf_grad(converted, additive_keys, hf_name, grad * factor)
            _add_converted_hf_grad(
                converted,
                additive_keys,
                f"{prefix}pre_feedforward_layernorm_2.weight",
                (grad * hf_weight * (-scale * root / ln2.square()).unsqueeze(0)).sum(
                    dim=0
                ),
                additive=True,
            )
            continue

        if model_is_moe and (
            match := _GEMMA4_SHARED_EXPERT_WEIGHT_PATTERN.match(hf_name)
        ):
            prefix = match.group("prefix")
            pffl = (
                hf_state_dict[f"{prefix}pre_feedforward_layernorm.weight"]
                .float()
                .to(grad.device)
            )
            ln2 = (
                hf_state_dict[f"{prefix}pre_feedforward_layernorm_2.weight"]
                .float()
                .to(grad.device)
            )
            hf_weight = hf_state_dict[hf_name].float().to(grad.device)
            factor = pffl / ln2
            _add_converted_hf_grad(converted, additive_keys, hf_name, grad * factor)
            _add_converted_hf_grad(
                converted,
                additive_keys,
                f"{prefix}pre_feedforward_layernorm_2.weight",
                (grad * hf_weight * (-pffl / ln2.square()).unsqueeze(0)).sum(dim=0),
                additive=True,
            )
            continue

        _add_converted_hf_grad(converted, additive_keys, hf_name, tensor)
    return converted, additive_keys


def _convert_megatron_tasks_to_hf(
    runtime: megatron_train.TrainingRuntime,
    *,
    mode: str,
    tasks: list[Any] | None = None,
) -> dict[str, torch.Tensor]:
    if tasks is None:
        tasks = [
            task
            for task in build_art_conversion_tasks(
                bridge=runtime.bridge,
                model=runtime.model,
            )
            if isinstance(task.param_weight, torch.nn.Parameter)
        ]
    model_bridge = runtime.bridge._model_bridge
    hf_state_dict = runtime.bridge.hf_pretrained.state
    grouped_buffers: dict[str, dict[int, torch.Tensor]] = {}
    converted: dict[str, torch.Tensor] = {}
    additive_grad_keys: set[str] = set()
    for task in tasks:
        tensor = _megatron_task_tensor(task, mode=mode)
        converted_weights_dict = task.mapping.megatron_to_hf(
            tensor,
            task.megatron_module,
        )
        task_additive_grad_keys: set[str] = set()
        if getattr(task.mapping, "is_grouped_export", False):
            merged_result = model_bridge._accumulate_grouped_export(
                task,
                converted_weights_dict,
                runtime.model[0].config,
                grouped_buffers,
                hf_state_dict,
            )
            if merged_result is None:
                continue
            converted_weights_dict = merged_result
        else:
            if mode == "grad":
                converted_weights_dict, task_additive_grad_keys = (
                    _maybe_modify_converted_hf_grad(
                        model_bridge,
                        task,
                        converted_weights_dict,
                        hf_state_dict,
                        model_is_moe=runtime.model_support_handler.is_moe,
                    )
                )
            else:
                converted_weights_dict = model_bridge.maybe_modify_converted_hf_weight(
                    task,
                    converted_weights_dict,
                    hf_state_dict,
                )
        for hf_name, value in converted_weights_dict.items():
            if not _is_language_hf_param_name(hf_name):
                continue
            value = value.detach().cpu().to(dtype=torch.float32)
            if hf_name in converted:
                if mode == "grad" and (
                    hf_name in additive_grad_keys or hf_name in task_additive_grad_keys
                ):
                    converted[hf_name] = converted[hf_name] + value
                    additive_grad_keys.add(hf_name)
                    continue
                raise RuntimeError(f"Duplicate converted HF key '{hf_name}' in {mode}")
            converted[hf_name] = value
            if hf_name in task_additive_grad_keys:
                additive_grad_keys.add(hf_name)
    return converted


def _run_megatron_sft_step(
    *,
    request: HfParityRunRequest,
    micro_inputs: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    device: torch.device,
    moe_routing_replay_bundle: MoeRoutingReplayBundle | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    runtime = _build_megatron_runtime(
        request,
        moe_routing_replay_bundle=moe_routing_replay_bundle,
    )
    _assert_runtime_configuration(runtime.model, request.case_config, ORACLE_TOPOLOGY)
    assert runtime.optimizer is not None
    if moe_routing_replay_bundle is not None:
        controller = runtime.moe_routing_replay_controller
        if controller is None:
            raise RuntimeError(
                "Expected MoE routing replay controller to be configured"
            )
        controller.set_step(
            step_index=0,
            sample_index=sample_indices,
            global_grad_accumulation_sequences=request.case_config.grad_accumulation_sequences,
        )
    _debug("initializing Megatron optimizer state")
    megatron_train._eager_initialize_optimizer_state(runtime.optimizer)
    tasks = [
        task
        for task in build_art_conversion_tasks(
            bridge=runtime.bridge,
            model=runtime.model,
        )
        if isinstance(task.param_weight, torch.nn.Parameter)
    ]
    _debug(f"built {len(tasks)} Megatron conversion tasks")
    for chunk in runtime.model:
        if hasattr(chunk, "zero_grad_buffer"):
            chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]
        for param in chunk.parameters():
            param.grad = None
    loss_sum = torch.tensor(0.0, device=device)
    token_count = 0
    trainable_losses: list[torch.Tensor] = []
    for micro_order, micro in enumerate(micro_inputs):
        if runtime.moe_routing_replay_controller is not None:
            runtime.moe_routing_replay_controller.begin_micro(
                sample_indices[micro_order],
                micro_order,
            )
        prepared_micro = megatron_train._prepare_dense_sft_micro(
            micro,
            device=device,
            provider=runtime.provider,
            model_support_handler=runtime.model_support_handler,
        )
        prepare_replay_local_input_token_uids(
            runtime.moe_routing_replay_controller,
            prepared_micro.local_token_uids,
            prepared_micro.attention_state,
        )
        attention_mask = megatron_train._placeholder_attention_mask(device)
        forward_kwargs = runtime.model_support_handler.get_forward_kwargs(
            runtime.model[0],
            attention_bias=prepared_micro.attention_state,
        )
        per_token_loss = runtime.model[0](
            input_ids=prepared_micro.input_ids,
            position_ids=prepared_micro.position_ids,
            attention_mask=attention_mask,
            labels=prepared_micro.labels,
            **forward_kwargs,
        )
        masked_losses = per_token_loss[prepared_micro.loss_mask]
        trainable_losses.append(masked_losses.detach().cpu())
        loss_sum = loss_sum + masked_losses.sum()
        token_count += int(prepared_micro.loss_mask.sum().item())
        masked_losses.sum().backward()
    _debug("finished Megatron forward/backward")
    num_tokens = megatron_train._local_trainable_sft_token_count_tensor(
        micro_inputs,
        device=device,
    )
    megatron_train.flush_param_grads_to_main_grads(runtime.model)
    megatron_train.finalize_model_grads_extended(
        megatron_train.as_megatron_api_chunks(runtime.model),
        num_tokens=num_tokens,
    )
    _debug("finalized Megatron grads")
    derivative_tasks = [
        task
        for task in tasks
        if _mapping_supports_derivative_parity(task.mapping)
        and _mapping_targets_language_only(task.mapping)
    ]
    _debug(f"retained {len(derivative_tasks)} derivative-safe conversion tasks")
    grads = _convert_megatron_tasks_to_hf(
        runtime,
        mode="grad",
        tasks=derivative_tasks,
    )
    _debug("exported Megatron grads")
    if runtime.moe_routing_replay_controller is not None:
        runtime.moe_routing_replay_controller.finalize_step()
    scalar_loss = (loss_sum / max(token_count, 1)).detach().cpu().reshape(1)
    output_vector = torch.cat(trainable_losses, dim=0).to(dtype=torch.float32)
    _debug("finished Megatron step")
    return output_vector, scalar_loss, grads


def _normalize_hf_grads_for_bridge(
    hf_grads: dict[str, torch.Tensor],
    *,
    expected_grad_keys: set[str],
) -> dict[str, torch.Tensor]:
    hf_grads = _filter_language_only_tensor_map(hf_grads)
    hf_grads = hf_tensor_map_to_art_canonical(
        hf_grads,
        expected_keys=expected_grad_keys,
    )
    normalized_hf_grads = _normalize_hf_tensor_map_for_bridge(
        hf_grads,
        expected_grad_keys,
    )
    return {
        key: normalized_hf_grads[key]
        for key in sorted(expected_grad_keys)
        if key in normalized_hf_grads
    }


def _drop_gemma4_reparameterized_norm_grads(
    tensor_map: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in tensor_map.items()
        if _GEMMA4_REPARAMETERIZED_NORM_GRAD_PATTERN.match(key) is None
    }


def _worker_run(request: HfParityRunRequest) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HF parity requires at least one CUDA device")
    torch.cuda.set_device(0)
    _set_deterministic_seed(request.case_config.seed)
    _configure_cuda_precision(request.case_config)
    _enable_debug_traceback_dump()

    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    trajectory_tensors = build_sft_trajectory_tensors_from_packed_tensors(
        packed_tensors
    )
    zero_template = megatron_train._zero_contribution_sft_inputs(trajectory_tensors[0])
    sample_indices = build_parity_sample_indices(
        num_sequences=len(trajectory_tensors),
        global_grad_accumulation_sequences=request.case_config.grad_accumulation_sequences,
    )
    micro_inputs = megatron_train.select_sft_micro_inputs(
        trajectory_tensors,
        sample_indices,
        zero_template,
    )
    replay_topology = ReplayParallelTopology.model_validate(
        ORACLE_TOPOLOGY.model_dump(
            include={"tp", "ep", "etp", "dp", "sp", "cp", "pp", "vpp"},
            mode="python",
        )
    )
    device = torch.device("cuda", 0)
    flex_patch_stack = ExitStack()
    flex_patch_stack.enter_context(
        _apply_requested_flex_backend_patch(TEST_DEFAULT_FLEX_BACKEND)
    )
    dtype = _dtype_for_precision(request.case_config.precision)
    if dtype == torch.float32:
        flex_patch_stack.enter_context(
            _apply_test_flex_inner_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        flex_patch_stack.enter_context(
            _apply_test_attention_full_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        install_megatron_qwen35_gdn_fp32_reference(
            flex_patch_stack,
            base_model=request.case_config.base_model,
        )
    try:
        _debug("starting HF parity worker")
        hf_outputs, hf_loss, hf_grads, moe_routing_replay_bundle = _run_hf_sft_step(
            base_model=request.case_config.base_model,
            num_layers=request.case_config.num_layers,
            micro_inputs=micro_inputs,
            sample_indices=sample_indices,
            topology=replay_topology,
            device=device,
            dtype=dtype,
        )
        megatron_outputs, megatron_loss, megatron_grads = _run_megatron_sft_step(
            request=request,
            micro_inputs=micro_inputs,
            sample_indices=sample_indices,
            device=device,
            moe_routing_replay_bundle=moe_routing_replay_bundle,
        )
        _debug("finished HF and Megatron steps, building report")
        normalized_hf_grads = _normalize_hf_grads_for_bridge(
            hf_grads,
            expected_grad_keys=set(megatron_grads.keys()),
        )
        if "gemma-4" in request.case_config.base_model.lower():
            # Gemma 4 Bridge stores HF-only preprocessing parameters as buffers and
            # folds them into Megatron weights. The fused linear gradients are
            # compared after the chain-rule export above, but this norm's base
            # gradient is not an independent HF-coordinate gradient in the reduced
            # Megatron parameterization used by the shipped LoRA path.
            normalized_hf_grads = _drop_gemma4_reparameterized_norm_grads(
                normalized_hf_grads
            )
            megatron_grads = _drop_gemma4_reparameterized_norm_grads(megatron_grads)
        active_embedding_rows = _active_embedding_token_rows(micro_inputs)
        active_router_rows = _active_router_rows_by_layer(moe_routing_replay_bundle)
        last_layer_index = request.case_config.num_layers - 1
        loss_active_last_layer_experts = _loss_active_last_layer_experts(
            moe_routing_replay_bundle,
            micro_inputs,
            sample_indices,
            layer_index=last_layer_index,
        )
        normalized_hf_grads = _focus_derivative_tensor_map(
            normalized_hf_grads,
            active_embedding_rows=active_embedding_rows,
            active_router_rows=active_router_rows,
            last_layer_index=last_layer_index,
            loss_active_last_layer_experts=loss_active_last_layer_experts,
        )
        megatron_grads = _focus_derivative_tensor_map(
            megatron_grads,
            active_embedding_rows=active_embedding_rows,
            active_router_rows=active_router_rows,
            last_layer_index=last_layer_index,
            loss_active_last_layer_experts=loss_active_last_layer_experts,
        )
        outputs_summary = summarize_tensor_pair(hf_outputs, megatron_outputs)
        loss_summary = summarize_tensor_pair(hf_loss, megatron_loss)
        grads_rows = build_tensor_map_metric_rows(
            phase="grads",
            reference=normalized_hf_grads,
            candidate=megatron_grads,
            phase_pass_fns=_hf_parity_phase_pass_fns(),
        )
        report = build_hf_parity_report(
            request=request,
            outputs_summary=outputs_summary,
            loss_summary=loss_summary,
            grads_rows=grads_rows,
        )
        _write_json(
            Path(request.output_dir) / HF_PARITY_REPORT_FILENAME,
            report.model_dump(mode="json"),
        )
        _debug("wrote HF parity report")
    finally:
        flex_patch_stack.close()
        if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
            torch.distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def run_worker_cli(run_request_path: Path) -> None:
    request = HfParityRunRequest.model_validate(_read_json(run_request_path))
    _worker_run(request)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron HF parity worker")
    parser.add_argument("--run-request", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
