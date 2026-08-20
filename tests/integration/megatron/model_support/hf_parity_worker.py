from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
import faulthandler
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.nn.functional as F

from art.megatron import train as megatron_train
from art.megatron.context_parallel.block_mask import prepare_block_mask_context
from art.megatron.prefix_tree import parse_prefix_tree_row
from art.megatron.prefix_tree_state import create_prefix_tree_state
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
)
from art.megatron.routing_replay import (
    ParallelTopology as ReplayParallelTopology,
)
from art.megatron.training import microbatches as megatron_microbatches
from art.megatron.training.trace import prepare_replay_local_input_token_uids
from art.megatron.weights.conversion_tasks import build_art_conversion_tasks
from art.preprocessing.pack import packed_tensors_from_dir

from .base_megatron_session import (
    BaseMegatronSessionKey,
    active_base_megatron_session,
    initialize_single_rank_process_group,
)
from .fp32_grouped_gemm import (
    allow_fp32_grouped_gemm_fallback_for_model_support_tests,
)
from .gdn_fp32_reference import install_megatron_qwen35_gdn_fp32_reference
from .hf_parity import (
    HF_PARITY_REPORT_FILENAME,
    HfParityRunRequest,
    _hf_parity_phase_pass_fns_for_case,
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

allow_fp32_grouped_gemm_fallback_for_model_support_tests()

HF_PARITY_DEBUG_ENV = "ART_HF_PARITY_DEBUG"
_DEBUG_START_TIME = time.perf_counter()
_VISUAL_HF_PREFIXES = ("model.visual.", "visual.")
_HF_MOE_ROUTER_NAME_PATTERN = re.compile(
    r"^(?:"
    r"(?:model\.)?backbone\.layers\."
    r"(?P<nemotron_gate_layer>\d+)\.mixer\.gate|"
    r"model\.layers\.(?P<gate_layer>\d+)\.mlp\.gate|"
    r"model(?:\.language_model)?\.layers\.(?P<mlp_router_layer>\d+)\.mlp\.router|"
    r"model(?:\.language_model)?\.layers\.(?P<router_layer>\d+)\.router"
    r")$"
)
_REPLAY_ROUTER_LAYER_PATTERN = re.compile(
    r"^chunk_\d+\.layer_(?P<layer>\d+)\.mlp\.router$"
)
_DISTRIBUTED_PROCESS_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)
_GATE_WEIGHT_PATTERN = re.compile(
    r"^model(?:\.language_model)?\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$"
)
_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^model(?:\.language_model)?\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?:down_proj|gate_proj|up_proj)\.weight$"
)
_NEMOTRON_H_GATE_WEIGHT_PATTERN = re.compile(
    r"^(?:model\.)?backbone\.layers\."
    r"(?P<layer>\d+)\.mixer\.gate\.weight$"
)
_NEMOTRON_H_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^(?:model\.)?backbone\.layers\."
    r"(?P<layer>\d+)\.mixer\.experts\."
    r"(?P<expert>\d+)\.(?:down_proj|up_proj)\.weight$"
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
        match.group("nemotron_gate_layer")
        or match.group("gate_layer")
        or match.group("mlp_router_layer")
        or match.group("router_layer")
    )
    return f"chunk_00.layer_{int(layer):04d}.mlp.router"


def _hf_router_expert_counts(module: Any) -> list[int]:
    config = getattr(module, "config", None)
    return [
        int(value)
        for owner, field in (
            (module, "num_experts"),
            (config, "num_experts"),
            (module, "n_routed_experts"),
            (config, "n_routed_experts"),
        )
        if (value := getattr(owner, field, None)) is not None
    ]


def _hf_router_num_experts(module: Any, _router_scores: torch.Tensor) -> int:
    counts = _hf_router_expert_counts(module)
    if not counts:
        raise RuntimeError("HF parity router is missing expert count metadata")
    if any(count != counts[0] for count in counts[1:]):
        raise RuntimeError(f"HF parity router has conflicting expert counts: {counts}")
    return counts[0]


def _expected_hf_moe_router_names(
    model: Any, named_modules: list[tuple[str, Any]]
) -> set[str]:
    modules_by_name = dict(named_modules)
    expected = {
        name
        for name, module in named_modules
        if name.rsplit(".", 1)[-1] in {"gate", "router"}
        and _hf_router_expert_counts(module)
    }
    prefixes = [
        prefix for prefix in ("backbone", "model.backbone") if prefix in modules_by_name
    ]
    if len(prefixes) > 1:
        raise RuntimeError(f"HF parity found ambiguous backbone prefixes: {prefixes}")
    if prefixes:
        pattern = getattr(
            getattr(modules_by_name[prefixes[0]], "config", None),
            "hybrid_override_pattern",
            getattr(getattr(model, "config", None), "hybrid_override_pattern", None),
        )
    else:
        pattern = None
    if isinstance(pattern, str):
        expected.update(
            f"{prefixes[0]}.layers.{layer}.mixer.gate"
            for layer, layer_type in enumerate(pattern)
            if layer_type == "E"
        )
    return expected


def _nemotron_router_output(
    module: Any,
    router_key: str,
    output: tuple[Any, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    router_indices, router_scores = output
    if not isinstance(router_indices, torch.Tensor) or router_indices.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise RuntimeError(
            f"Nemotron router indices dtype must be int32 or int64 for '{router_key}'"
        )
    if (
        not isinstance(router_scores, torch.Tensor)
        or not router_scores.is_floating_point()
    ):
        raise RuntimeError(
            f"Nemotron router weights dtype must be floating for '{router_key}'"
        )
    if router_indices.ndim != 2 or router_scores.ndim != 2:
        raise RuntimeError(f"Nemotron router outputs must be rank-2 for '{router_key}'")
    if router_indices.shape != router_scores.shape:
        raise RuntimeError(
            f"Nemotron router output shape mismatch for '{router_key}': "
            f"{tuple(router_indices.shape)} vs {tuple(router_scores.shape)}"
        )
    top_k = getattr(module, "top_k", None)
    if top_k is None or int(top_k) <= 0:
        raise RuntimeError(f"Nemotron router is missing top_k for '{router_key}'")
    if int(router_indices.shape[1]) != int(top_k):
        raise RuntimeError(
            f"Nemotron router top-k width mismatch for '{router_key}': "
            f"{int(router_indices.shape[1])} != {int(top_k)}"
        )
    if router_indices.device != router_scores.device:
        raise RuntimeError(f"Nemotron router outputs changed device for '{router_key}'")
    if not bool(torch.isfinite(router_scores).all()) or bool((router_scores < 0).any()):
        raise RuntimeError(f"Nemotron router weights are invalid for '{router_key}'")
    return router_scores, router_indices


def _glm_router_output(
    module: Any, router_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = router_logits.sigmoid()
    choice = scores + module.e_score_correction_bias
    groups = int(module.n_group)
    group_scores = (
        choice.view(choice.shape[0], groups, -1).topk(2, dim=-1).values.sum(-1)
    )
    selected_groups = group_scores.topk(
        int(module.topk_group), dim=-1, sorted=False
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    choice = choice.masked_fill(
        ~group_mask.unsqueeze(-1)
        .expand_as(choice.view(choice.shape[0], groups, -1))
        .reshape_as(choice),
        float("-inf"),
    )
    indices = choice.topk(int(module.top_k), dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if bool(module.norm_topk_prob):
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * float(module.routed_scaling_factor), indices


class _HfMoeRoutingCapture:
    def __init__(self, model: Any, *, required: bool = False) -> None:
        self._handles: list[Any] = []
        self._routes: dict[str, dict[int, RouterCallRoute]] = {}
        self._active_sample_index: int | None = None
        self._active_micro_slot = 0
        self._active_token_uids: torch.Tensor | None = None
        self._active_token_span: int | None = None
        self._assembled_routes: dict[str, dict[int, RouterCallRoute]] = {}
        self._assembled_filled: dict[str, dict[int, torch.Tensor]] = {}
        named_modules = list(model.named_modules())
        expected_router_names = _expected_hf_moe_router_names(model, named_modules)
        router_modules = [
            (module_name, module, router_key)
            for module_name, module in named_modules
            if (router_key := _hf_moe_router_key(module_name)) is not None
        ]
        captured_router_names = {name for name, _module, _key in router_modules}
        router_keys = [key for _name, _module, key in router_modules]
        if len(router_keys) != len(set(router_keys)):
            raise RuntimeError(f"HF parity found duplicate router keys: {router_keys}")
        missing_router_names = expected_router_names - captured_router_names
        if required and missing_router_names:
            raise RuntimeError(
                "HF parity MoE routing capture is incomplete; missing "
                f"{sorted(missing_router_names)}"
            )
        if required and not captured_router_names:
            raise RuntimeError(
                "HF parity MoE case requires HF MoE routing capture, but no "
                "supported router modules were found"
            )
        for _module_name, module, router_key in router_modules:
            self._routes[router_key] = {}
            self._assembled_routes[router_key] = {}
            self._assembled_filled[router_key] = {}
            self._handles.append(
                module.register_forward_hook(self._make_hook(router_key, module))
            )

    @property
    def enabled(self) -> bool:
        return bool(self._handles)

    def set_active_micro(
        self,
        sample_index: int | None,
        micro_slot: int,
        *,
        token_uids: torch.Tensor | None = None,
        token_span: int | None = None,
    ) -> None:
        self._active_sample_index = sample_index
        self._active_micro_slot = micro_slot
        self._active_token_uids = token_uids
        self._active_token_span = token_span

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
            assembled = self._assembled_routes[router_key]
            calls = assembled if assembled else self._routes[router_key]
            if not calls:
                raise RuntimeError(f"HF parity captured no routes for '{router_key}'")
            for micro_slot, filled in self._assembled_filled[router_key].items():
                if not bool(filled.all()):
                    raise RuntimeError(
                        f"HF parity did not assemble all route rows for {router_key} "
                        f"micro {micro_slot}: {int(filled.sum())}/{int(filled.numel())}"
                    )
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
            if isinstance(output, torch.Tensor) and hasattr(
                module, "e_score_correction_bias"
            ):
                router_scores, router_indices = _glm_router_output(module, output)
            elif isinstance(output, tuple) and len(output) == 2:
                router_scores, router_indices = _nemotron_router_output(
                    module, router_key, output
                )
            elif isinstance(output, tuple) and len(output) >= 3:
                router_scores = output[1]
                router_indices = output[2]
            else:
                raise RuntimeError(
                    f"Unsupported HF router output for '{router_key}': {type(output)}"
                )
            if not isinstance(router_scores, torch.Tensor) or not isinstance(
                router_indices, torch.Tensor
            ):
                raise RuntimeError(
                    f"Expected tensor router outputs for '{router_key}', "
                    f"got scores={type(router_scores)} indices={type(router_indices)}"
                )
            indices = router_indices.detach().cpu().to(torch.int32)
            scores = router_scores.detach().cpu().to(torch.float32)
            route = RouterCallRoute(
                expert_indices=indices,
                expert_probs=scores,
                expert_mask=torch.ones_like(indices, dtype=torch.bool),
                num_experts=_hf_router_num_experts(module, router_scores),
                sample_index=self._active_sample_index,
                micro_slot=(
                    None
                    if self._active_sample_index is not None
                    else self._active_micro_slot
                ),
            )
            if self._active_token_uids is not None:
                self._assemble_route(router_key, route)
                return
            self._routes[router_key][len(self._routes[router_key])] = route

        return _hook

    def _assemble_route(self, router_key: str, route: RouterCallRoute) -> None:
        token_uids = cast(torch.Tensor, self._active_token_uids).cpu().long()
        token_span = self._active_token_span
        if token_span is None or int(token_uids.numel()) != route.num_global_tokens:
            raise RuntimeError("HF parity route path metadata does not match routes")
        micro_slot = self._active_micro_slot
        assembled = self._assembled_routes[router_key].get(micro_slot)
        filled = self._assembled_filled[router_key].get(micro_slot)
        if assembled is None:
            assembled = route.model_copy(
                update={
                    "expert_indices": torch.full(
                        (token_span, route.max_topk), -1, dtype=torch.int32
                    ),
                    "expert_probs": torch.zeros(
                        (token_span, route.max_topk), dtype=torch.float32
                    ),
                    "expert_mask": torch.zeros(
                        (token_span, route.max_topk), dtype=torch.bool
                    ),
                }
            )
            filled = torch.zeros(token_span, dtype=torch.bool)
            self._assembled_routes[router_key][micro_slot] = assembled
            self._assembled_filled[router_key][micro_slot] = filled
        assert filled is not None
        repeated = filled.index_select(0, token_uids)
        if bool(repeated.any()):
            path_rows = torch.where(repeated)[0]
            existing_rows = token_uids.index_select(0, path_rows)
            if not torch.equal(
                assembled.expert_indices.index_select(0, existing_rows),
                route.expert_indices.index_select(0, path_rows),
            ):
                raise RuntimeError("HF parity repeated path changed expert ids")
            assert assembled.expert_probs is not None
            assert route.expert_probs is not None
            if not torch.allclose(
                assembled.expert_probs.index_select(0, existing_rows),
                route.expert_probs.index_select(0, path_rows),
                rtol=3e-5,
                atol=3e-6,
            ):
                raise RuntimeError("HF parity repeated path changed expert scores")
        assembled.expert_indices.index_copy_(0, token_uids, route.expert_indices)
        assert assembled.expert_probs is not None
        assert route.expert_probs is not None
        assembled.expert_probs.index_copy_(0, token_uids, route.expert_probs)
        assert assembled.expert_mask is not None
        assert route.expert_mask is not None
        assembled.expert_mask.index_copy_(0, token_uids, route.expert_mask)
        filled.index_fill_(0, token_uids, True)


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


def _is_bridge_hf_load_hook(hook: Any) -> bool:
    fn = getattr(hook, "func", hook)
    name = getattr(fn, "__name__", "")
    qualname = getattr(fn, "__qualname__", "")
    return name in {
        "load_weights_hf_to_megatron",
        "_optimized_load_weights_hf_to_megatron",
    } or qualname.endswith(".load_weights_hf_to_megatron")


def _remove_bridge_hf_load_hook(provider_bundle: Any) -> None:
    """Disable raw checkpoint load when parity seeds from HF oracle state."""

    provider = provider_bundle.provider
    hooks = list(getattr(provider, "_pre_wrap_hooks", []))
    kept = [hook for hook in hooks if not _is_bridge_hf_load_hook(hook)]
    if len(kept) == len(hooks):
        raise RuntimeError(
            "HF parity expected a Bridge HF-load pre-wrap hook to remove"
        )
    provider._pre_wrap_hooks = kept


def _configure_hf_parity_provider_bundle(
    provider_bundle: Any,
    *,
    use_hf_reference_state: bool,
) -> None:
    if use_hf_reference_state:
        _remove_bridge_hf_load_hook(provider_bundle)
    _install_bridge_timing_debug(provider_bundle)


def _resolve_hf_reference_model_class(
    *,
    base_model: str,
    config: Any,
    auto_model_class: type[Any],
) -> type[Any]:
    auto_map = getattr(config, "auto_map", None)
    if auto_map is None:
        return auto_model_class
    if not isinstance(auto_map, Mapping):
        raise TypeError("HF reference config auto_map must be a mapping")
    class_reference = auto_map.get(auto_model_class.__name__)
    if class_reference is None:
        return auto_model_class
    if not isinstance(class_reference, str) or not class_reference:
        raise TypeError("HF reference auto_map class must be a nonempty string")
    if class_reference.count("--") > 1:
        raise RuntimeError(f"Invalid HF reference class: {class_reference!r}")
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = base_model
    if not repo_id or class_reference.count(".") != 1:
        raise RuntimeError(f"Invalid HF reference class: {class_reference!r}")
    module_name, expected_class_name = class_reference.split(".")
    if not module_name or not expected_class_name:
        raise RuntimeError(f"Invalid HF reference class: {class_reference!r}")
    local_code = Path(repo_id).is_dir()
    revision = getattr(config, "_commit_hash", None)
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise RuntimeError("HF reference config has an invalid commit hash")
    if not local_code and revision is None:
        raise RuntimeError("Remote HF reference code requires an exact revision")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.auto.auto_factory import (
        add_generation_mixin_to_remote_model,
    )

    model_class = get_class_from_dynamic_module(
        class_reference,
        repo_id,
        revision=revision,
        local_files_only=local_code,
    )
    if model_class.__name__ != expected_class_name:
        raise RuntimeError(
            "HF reference auto_map resolved the wrong class: "
            f"{model_class.__name__!r} != {expected_class_name!r}"
        )
    return add_generation_mixin_to_remote_model(model_class)


def _load_hf_model(
    *,
    base_model: str,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    allow_unvalidated_arch: bool,
) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    from art.megatron.model_support.registry import get_model_support_handler

    handler = get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    ensure_hf_reference_registered = getattr(
        handler, "ensure_hf_reference_registered", None
    )
    if ensure_hf_reference_registered is not None:
        ensure_hf_reference_registered()
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    set_hf_config_num_layers(config, num_layers)
    zero_hf_dropout_config(config)
    prepare_hf_reference_config = getattr(handler, "prepare_hf_reference_config", None)
    if prepare_hf_reference_config is not None:
        prepare_hf_reference_config(config)
    hf_reference_from_pretrained_kwargs = getattr(
        handler, "hf_reference_from_pretrained_kwargs", None
    )
    extra_kwargs = (
        hf_reference_from_pretrained_kwargs(config=config, dtype=dtype)
        if hf_reference_from_pretrained_kwargs is not None
        else {}
    )
    model_class = _resolve_hf_reference_model_class(
        base_model=base_model,
        config=config,
        auto_model_class=AutoModelForCausalLM,
    )
    model_class = handler.prepare_hf_reference_model_class(model_class)
    model = model_class.from_pretrained(
        base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    )
    if model_class is not AutoModelForCausalLM and type(model) is not model_class:
        raise RuntimeError(
            "HF reference loader returned the wrong class: "
            f"{type(model).__name__} != {model_class.__name__}"
        )
    handler.validate_hf_reference_model_precision(model, dtype=dtype)
    model.train()
    model = model.to(device)
    prepare_hf_reference_model = getattr(handler, "prepare_hf_reference_model", None)
    if prepare_hf_reference_model is not None:
        model = prepare_hf_reference_model(model)
    return model


def _collect_hf_grads(model: Any) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param)
        grads[name] = grad.detach().cpu().to(dtype=torch.float32)
    return grads


def _collect_hf_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if _is_language_hf_param_name(key)
    }


def _normalize_hf_reference_state_for_hf_parity(
    *,
    base_model: str,
    model: Any,
    state: dict[str, torch.Tensor],
    allow_unvalidated_arch: bool,
) -> dict[str, torch.Tensor]:
    from art.megatron.model_support.registry import get_model_support_handler

    handler = get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    normalize = getattr(handler, "normalize_hf_reference_state_for_hf_parity", None)
    if normalize is not None:
        normalize(state, config=model.config)
    return state


def _use_hf_reference_state_for_hf_parity(
    base_model: str, *, allow_unvalidated_arch: bool
) -> bool:
    from art.megatron.model_support.registry import get_model_support_handler

    handler = get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    enabled = getattr(handler, "use_hf_reference_state_for_hf_parity", None)
    return bool(enabled()) if enabled is not None else False


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
            layer_rows.append(
                (
                    route.expert_indices
                    if route.expert_mask is None
                    else route.expert_indices[route.expert_mask]
                ).to(torch.long)
            )
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
            selected = route.expert_indices[loss_mask]
            if route.expert_mask is not None:
                selected = selected[route.expert_mask[loss_mask]]
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
    active_router_expert_sets = {
        layer_index: set(int(row) for row in rows.reshape(-1).tolist())
        for layer_index, rows in active_router_rows.items()
        if rows.numel() > 0
    }
    for key, value in tensor_map.items():
        if match := (
            _EXPERT_WEIGHT_PATTERN.match(key)
            or _NEMOTRON_H_EXPERT_WEIGHT_PATTERN.match(key)
        ):
            layer_index = int(match.group("layer"))
            expert_index = int(match.group("expert"))
            active_experts = active_router_expert_sets.get(layer_index)
            if active_experts is not None and expert_index not in active_experts:
                continue
            if (
                layer_index == last_layer_index
                and expert_index not in loss_active_last_layer_experts
            ):
                continue
        focused_value = value
        if (
            key
            in {
                "model.language_model.embed_tokens.weight",
                "backbone.embeddings.weight",
                "model.backbone.embeddings.weight",
            }
            and active_embedding_rows.numel() > 0
        ):
            focused_value = value.index_select(0, active_embedding_rows)
        elif match := (
            _GATE_WEIGHT_PATTERN.match(key)
            or _NEMOTRON_H_GATE_WEIGHT_PATTERN.match(key)
        ):
            active_rows = active_router_rows.get(int(match.group("layer")))
            if active_rows is not None and active_rows.numel() > 0:
                focused_value = value.index_select(0, active_rows)
        focused[key] = focused_value
    return focused


def _dense_prefix_tree_attention_mask(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    position_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    context = prepare_block_mask_context(
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=position_ids,
    )
    seq_len = int(group_ids.numel())
    absolute = torch.arange(seq_len)
    group_enter = torch.from_numpy(context.group_enter_np)
    group_exit = torch.from_numpy(context.group_exit_np)
    allowed = (absolute[:, None] >= absolute[None, :]) & (
        (group_enter[None, :] <= group_enter[:, None])
        & (group_enter[:, None] < group_exit[None, :])
    )
    if sliding_window is not None:
        positions = position_ids.detach().cpu().reshape(-1)
        delta = positions[:, None] - positions[None, :]
        allowed &= (delta >= 0) & (delta < sliding_window)
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    return mask.masked_fill(allowed.to(device), 0).unsqueeze(0).unsqueeze(0)


def _hf_prefix_tree_forward_inputs(
    model: Any,
    micro: dict[str, torch.Tensor],
    *,
    actual_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
    group_ids = micro["group_ids"].reshape(-1)[:actual_len]
    parent_ids = micro["parent_ids"].reshape(-1)[:actual_len]
    position_ids = micro["position_ids"].reshape(-1)[:actual_len]
    full_mask = _dense_prefix_tree_attention_mask(
        group_ids=group_ids,
        parent_ids=parent_ids,
        position_ids=position_ids,
        device=device,
        dtype=dtype,
    )
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else config
    layer_types = tuple(getattr(text_config, "layer_types", ()))
    attention_mask: torch.Tensor | dict[str, torch.Tensor] = full_mask
    if "sliding_attention" in layer_types:
        attention_mask = {
            "full_attention": full_mask,
            "sliding_attention": _dense_prefix_tree_attention_mask(
                group_ids=group_ids,
                parent_ids=parent_ids,
                position_ids=position_ids,
                device=device,
                dtype=dtype,
                sliding_window=int(text_config.sliding_window),
            ),
        }
    return attention_mask, position_ids.unsqueeze(0).to(device=device)


def _prepare_hf_parity_megatron_micro(
    micro: dict[str, torch.Tensor],
    *,
    device: torch.device,
    provider: Any,
    model_support_handler: Any,
) -> megatron_train.PreparedSFTMicroInputs:
    prepared = megatron_train._prepare_dense_sft_micro(
        micro,
        device=device,
        provider=provider,
        model_support_handler=model_support_handler,
    )
    seq_len = int(prepared.input_ids.shape[1])
    position_ids = micro["position_ids"].reshape(-1)[:seq_len].unsqueeze(0)
    recurrent_contract, recurrent_planner_config = (
        megatron_microbatches._linear_recurrent_runtime_for_provider(
            provider, model_support_handler
        )
    )
    attention_state = create_prefix_tree_state(
        group_ids=micro["group_ids"].reshape(-1)[:seq_len].unsqueeze(0),
        parent_ids=micro["parent_ids"].reshape(-1)[:seq_len].unsqueeze(0),
        target_device=device,
        input_pos=position_ids,
        sliding_windows=megatron_microbatches._art_flex_sliding_windows(provider),
        linear_recurrent_contract=recurrent_contract,
        recurrent_planner_config=recurrent_planner_config,
        model_support_handler=model_support_handler,
        attention_head_dim=getattr(provider, "kv_channels", None),
        attention_value_head_dim=getattr(provider, "kv_channels", None),
    )
    return prepared.model_copy(
        update={
            "position_ids": position_ids.to(device=device),
            "attention_state": attention_state,
        }
    )


def _hf_requires_recurrent_prefix_paths(
    base_model: str,
    *,
    dtype: torch.dtype,
    allow_unvalidated_arch: bool,
) -> bool:
    from art.megatron.provider import prepare_provider_bundle

    provider_bundle = prepare_provider_bundle(
        base_model,
        torch_dtype=dtype,
        load_weights=False,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return (
        provider_bundle.handler.linear_recurrent_contract(provider_bundle.provider)
        is not None
    )


def _prepare_hf_reference_forward(
    model: Any,
    micro: dict[str, torch.Tensor],
    *,
    base_model: str,
    actual_len: int,
    allow_unvalidated_arch: bool,
) -> None:
    from art.megatron.model_support.registry import get_model_support_handler

    handler = get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    prepare_forward = getattr(handler, "prepare_hf_reference_forward", None)
    if prepare_forward is None:
        return
    prepare_forward(
        model,
        position_ids=micro["position_ids"].reshape(-1)[:actual_len],
        group_ids=micro["group_ids"].reshape(-1)[:actual_len],
        parent_ids=micro["parent_ids"].reshape(-1)[:actual_len],
    )


def _run_hf_sft_step(
    *,
    base_model: str,
    num_layers: int,
    micro_inputs: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    topology: ReplayParallelTopology,
    device: torch.device,
    dtype: torch.dtype,
    allow_unvalidated_arch: bool,
    require_moe_routing_replay: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    MoeRoutingReplayBundle | None,
    dict[str, torch.Tensor] | None,
]:
    _debug("loading HF model")
    model = _load_hf_model(
        base_model=base_model,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    if dtype == torch.float32:
        _install_hf_qwen35_gdn_fp32_reference(model, base_model=base_model)
    recurrent_prefix_paths = _hf_requires_recurrent_prefix_paths(
        base_model,
        dtype=dtype,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    route_capture = _HfMoeRoutingCapture(model, required=require_moe_routing_replay)
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
        attention_mask = micro["attention_mask"].reshape(-1)
        actual_len = max(int(attention_mask.sum().item()), 1)
        if recurrent_prefix_paths:
            micro_losses = _run_hf_recurrent_prefix_tree_micro(
                model=model,
                route_capture=route_capture,
                micro=micro,
                sample_index=sample_index,
                micro_slot=micro_slot,
                actual_len=actual_len,
                total_token_count=total_token_count,
                device=device,
                dtype=dtype,
            )
            trainable_losses.append(micro_losses.detach().cpu())
            loss_sum = loss_sum + micro_losses.detach().sum()
            token_count += int(micro_losses.numel())
            continue
        route_capture.set_active_micro(sample_index, micro_slot)
        _prepare_hf_reference_forward(
            model,
            micro,
            base_model=base_model,
            actual_len=actual_len,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
        input_ids = micro["input_ids"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
        labels = micro["labels"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
        hf_attention_mask, position_ids = _hf_prefix_tree_forward_inputs(
            model,
            micro,
            actual_len=actual_len,
            device=device,
            dtype=dtype,
        )
        logits = model(
            input_ids=input_ids,
            attention_mask=hf_attention_mask,
            position_ids=position_ids,
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
    hf_reference_state_dict = (
        _normalize_hf_reference_state_for_hf_parity(
            base_model=base_model,
            model=model,
            state=_collect_hf_state_dict(model),
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
        if _use_hf_reference_state_for_hf_parity(
            base_model, allow_unvalidated_arch=allow_unvalidated_arch
        )
        else None
    )
    routing_replay_bundle = route_capture.build_replay_bundle(topology=topology)
    if require_moe_routing_replay and routing_replay_bundle is None:
        raise RuntimeError("HF parity MoE case produced no routing replay bundle")
    scalar_loss = (loss_sum / max(token_count, 1)).detach().cpu().reshape(1)
    output_vector = torch.cat(trainable_losses, dim=0).to(dtype=torch.float32)
    route_capture.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _debug("finished HF step")
    return (
        output_vector,
        scalar_loss,
        grads,
        routing_replay_bundle,
        hf_reference_state_dict,
    )


def _run_hf_recurrent_prefix_tree_micro(
    *,
    model: Any,
    route_capture: _HfMoeRoutingCapture,
    micro: dict[str, torch.Tensor],
    sample_index: int | None,
    micro_slot: int,
    actual_len: int,
    total_token_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    input_ids = micro["input_ids"].reshape(-1)[:actual_len]
    labels = micro["labels"].reshape(-1)[:actual_len]
    position_ids = micro["position_ids"].reshape(-1)[:actual_len]
    shifted_labels = megatron_train.shift_tensor(labels.unsqueeze(0), -100)[0]
    expected_mask = shifted_labels != -100
    claimed_mask = torch.zeros(actual_len, dtype=torch.bool)
    claimed_targets = torch.full((actual_len,), -100, dtype=labels.dtype)
    packed_losses = torch.empty(actual_len, dtype=torch.float32)
    for path_indices in _hf_prefix_tree_paths(micro, actual_len=actual_len):
        route_capture.set_active_micro(
            sample_index,
            micro_slot,
            token_uids=path_indices,
            token_span=actual_len,
        )
        path_input_ids = input_ids.index_select(0, path_indices).unsqueeze(0).to(device)
        path_labels = labels.index_select(0, path_indices).unsqueeze(0).to(device)
        path_positions = (
            position_ids.index_select(0, path_indices).unsqueeze(0).to(device)
        )
        logits = model(
            input_ids=path_input_ids,
            attention_mask=torch.ones_like(path_input_ids, dtype=dtype),
            position_ids=path_positions,
            use_cache=False,
        ).logits
        path_shifted_labels = megatron_train.shift_tensor(path_labels, -100)[0]
        per_token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            path_shifted_labels,
            reduction="none",
            ignore_index=-100,
        )
        path_mask = path_shifted_labels != -100
        path_uids = path_indices[path_mask.cpu()]
        path_targets = path_shifted_labels[path_mask].detach().cpu()
        repeated = claimed_mask.index_select(0, path_uids)
        if bool(repeated.any()) and not torch.equal(
            claimed_targets.index_select(0, path_uids[repeated]),
            path_targets[repeated],
        ):
            raise RuntimeError("HF prefix paths assign different targets to one token")
        unclaimed = ~repeated
        selected_uids = path_uids[unclaimed]
        selected_losses = per_token_loss[path_mask][unclaimed.to(device)]
        packed_losses.index_copy_(0, selected_uids, selected_losses.detach().cpu())
        claimed_targets.index_copy_(0, selected_uids, path_targets[unclaimed])
        claimed_mask.index_fill_(0, selected_uids, True)
        if selected_losses.numel():
            (selected_losses.sum() / total_token_count).backward()
    if not torch.equal(claimed_mask, expected_mask.cpu()):
        missing = torch.where(expected_mask.cpu() & ~claimed_mask)[0].tolist()
        extra = torch.where(claimed_mask & ~expected_mask.cpu())[0].tolist()
        raise RuntimeError(
            "HF prefix paths do not preserve packed loss positions: "
            f"missing={missing} extra={extra}"
        )
    return packed_losses[expected_mask.cpu()]


def _hf_prefix_tree_paths(
    micro: dict[str, torch.Tensor], *, actual_len: int
) -> tuple[torch.Tensor, ...]:
    row = parse_prefix_tree_row(
        group_ids=micro["group_ids"].reshape(-1)[:actual_len],
        parent_ids=micro["parent_ids"].reshape(-1)[:actual_len],
    )
    if row.valid_tokens != actual_len:
        raise RuntimeError(
            f"HF prefix tree covers {row.valid_tokens}/{actual_len} valid tokens"
        )
    by_group = {segment.group_id: segment for segment in row.segments}
    parent_groups = {
        segment.parent_id
        for segment in row.segments
        if segment.parent_id != segment.group_id
    }
    paths: list[torch.Tensor] = []
    for leaf in row.segments:
        if leaf.group_id in parent_groups:
            continue
        path_segments = [by_group[group_id] for group_id in leaf.ancestors]
        path_segments.append(leaf)
        paths.append(
            torch.cat(
                [
                    torch.arange(segment.start, segment.end, dtype=torch.long)
                    for segment in path_segments
                ]
            )
        )
    return tuple(paths)


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
    use_hf_reference_state = _use_hf_reference_state_for_hf_parity(
        request.case_config.base_model,
        allow_unvalidated_arch=request.case_config.allow_unvalidated_arch,
    )
    return megatron_train.build_training_runtime(
        model_identifier=request.case_config.base_model,
        provider_torch_dtype=_dtype_for_precision(request.case_config.precision),
        provider_bundle_configure=lambda provider_bundle: (
            _configure_hf_parity_provider_bundle(
                provider_bundle,
                use_hf_reference_state=use_hf_reference_state,
            )
        ),
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


def _hf_param_names_for_mapping(mapping: Any) -> set[str]:
    names = _language_hf_param_names(mapping)
    if not names:
        return set()
    return set(names)


def _build_hf_parity_conversion_tasks(
    *,
    bridge: Any,
    model: list[Any],
    hf_keys: set[str],
) -> list[Any]:
    tasks = []
    registry_type = type(bridge._model_bridge.mapping_registry())
    lookup = registry_type.megatron_to_hf_lookup

    def permissive_lookup(registry: Any, name: str) -> Any:
        mapping = lookup(registry, name)
        if mapping is not None:
            mapping.allow_hf_name_mismatch = True
        return mapping

    with patch.object(registry_type, "megatron_to_hf_lookup", permissive_lookup):
        conversion_tasks = build_art_conversion_tasks(bridge=bridge, model=model)
    for task in conversion_tasks:
        mapping_names = _hf_param_names_for_mapping(task.mapping)
        if not mapping_names:
            tasks.append(task)
            continue
        if mapping_names & hf_keys:
            tasks.append(task)
    return tasks


def _seed_megatron_from_hf_reference_state(
    runtime: megatron_train.TrainingRuntime,
    *,
    tasks: list[Any],
    hf_reference_state_dict: dict[str, torch.Tensor],
) -> None:
    model_bridge = runtime.bridge._model_bridge
    for task in tasks:
        if task.mapping is None:
            continue
        hf_weights = model_bridge.maybe_modify_loaded_hf_weight(
            task.mapping.hf_param,
            hf_reference_state_dict,
        )
        converted_weights = task.mapping.hf_to_megatron(
            hf_weights, task.megatron_module
        )
        if isinstance(task.param_weight, torch.nn.Parameter):
            task.param_weight.data.copy_(converted_weights.to(task.param_weight.device))
        elif isinstance(task.param_weight, torch.Tensor):
            task.param_weight.copy_(converted_weights.to(task.param_weight.device))


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
    hf_state_dict_override: dict[str, torch.Tensor] | None = None,
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
    hf_state_dict = (
        hf_state_dict_override
        if hf_state_dict_override is not None
        else runtime.bridge.hf_pretrained.state
    )
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
    hf_reference_state_dict: dict[str, torch.Tensor] | None = None,
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
        )
    if hf_reference_state_dict is None:
        tasks = [
            task
            for task in build_art_conversion_tasks(
                bridge=runtime.bridge,
                model=runtime.model,
            )
            if isinstance(task.param_weight, torch.nn.Parameter)
        ]
    else:
        seed_tasks = _build_hf_parity_conversion_tasks(
            bridge=runtime.bridge,
            model=runtime.model,
            hf_keys=set(hf_reference_state_dict),
        )
        tasks = [
            task
            for task in seed_tasks
            if isinstance(task.param_weight, torch.nn.Parameter)
        ]
        _debug("seeding Megatron weights from HF oracle state")
        _seed_megatron_from_hf_reference_state(
            runtime,
            tasks=seed_tasks,
            hf_reference_state_dict=hf_reference_state_dict,
        )
    _debug("initializing Megatron optimizer state")
    megatron_train._eager_initialize_optimizer_state(runtime.optimizer)
    session = active_base_megatron_session()
    if session is not None:
        session.capture_runtime(
            runtime,
            key=BaseMegatronSessionKey(
                base_model=request.case_config.base_model,
                model_key=runtime.model_support_spec.key,
                num_layers=request.case_config.num_layers,
                precision=request.case_config.precision,
                allow_unvalidated_arch=request.case_config.allow_unvalidated_arch,
            ),
        )
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
        prepared_micro = _prepare_hf_parity_megatron_micro(
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
        if cast(torch.nn.Parameter, task.param_weight).requires_grad
        if _mapping_supports_derivative_parity(task.mapping)
        and _mapping_targets_language_only(task.mapping)
    ]
    _debug(f"retained {len(derivative_tasks)} derivative-safe conversion tasks")
    grads = _convert_megatron_tasks_to_hf(
        runtime,
        mode="grad",
        tasks=derivative_tasks,
        hf_state_dict_override=hf_reference_state_dict,
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


def _validate_distributed_process_env() -> None:
    missing = [name for name in _DISTRIBUTED_PROCESS_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"HF parity worker requires explicit distributed environment: {missing}"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            "Invalid HF parity rank environment: "
            f"rank={rank}/{world_size} local_rank={local_rank}/{local_world_size}"
        )


def _worker_run(request: HfParityRunRequest) -> None:
    _validate_distributed_process_env()
    if not torch.cuda.is_available():
        raise RuntimeError("HF parity requires at least one CUDA device")
    torch.cuda.set_device(0)
    initialize_single_rank_process_group()
    _set_deterministic_seed(request.case_config.seed)
    _configure_cuda_precision(request.case_config)
    _enable_debug_traceback_dump()

    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    trajectory_tensors = build_sft_trajectory_tensors_from_packed_tensors(
        packed_tensors
    )
    for index, trajectory in enumerate(trajectory_tensors):
        trajectory.update(
            {
                "group_ids": packed_tensors["group_ids"][index].detach().clone(),
                "parent_ids": packed_tensors["parent_ids"][index].detach().clone(),
                "position_ids": packed_tensors["input_pos"][index].detach().clone(),
            }
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
        (
            hf_outputs,
            hf_loss,
            hf_grads,
            moe_routing_replay_bundle,
            hf_reference_state_dict,
        ) = _run_hf_sft_step(
            base_model=request.case_config.base_model,
            num_layers=request.case_config.num_layers,
            micro_inputs=micro_inputs,
            sample_indices=sample_indices,
            topology=replay_topology,
            device=device,
            dtype=dtype,
            allow_unvalidated_arch=request.case_config.allow_unvalidated_arch,
            require_moe_routing_replay=request.case_config.is_moe,
        )
        megatron_outputs, megatron_loss, megatron_grads = _run_megatron_sft_step(
            request=request,
            micro_inputs=micro_inputs,
            sample_indices=sample_indices,
            device=device,
            moe_routing_replay_bundle=moe_routing_replay_bundle,
            hf_reference_state_dict=hf_reference_state_dict,
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
        from art.megatron.model_support.registry import get_model_support_handler

        handler = get_model_support_handler(
            request.case_config.base_model,
            allow_unvalidated_arch=request.case_config.allow_unvalidated_arch,
        )
        grads_rows = build_tensor_map_metric_rows(
            phase="grads",
            reference=normalized_hf_grads,
            candidate=megatron_grads,
            phase_pass_fns=_hf_parity_phase_pass_fns_for_case(request.case_config),
            group_by=getattr(handler, "hf_parity_gradient_group", None),
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
        session = active_base_megatron_session()
        if (
            (session is None or session.runtime is None)
            and torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
        ):
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
