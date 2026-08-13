from __future__ import annotations

from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import re
import types
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors.torch import load_file, save_file
import torch

from art.megatron.weights.param_name_canonicalization import canonical_art_param_name

if TYPE_CHECKING:
    from art.preprocessing.pack import PackedTensors

ROUTER_NAME_TOKEN = ".mlp.router"
ROUTER_KEY_FORMAT_VERSION = "moe_routing_replay_v4"
GLOBAL_TOKEN_UIDS_KEY = "global_token_uids"

_ROUTER_LAYER_PATTERN = re.compile(r"decoder\.layers\.(?P<layer>\d+)\.mlp\.router$")
_ROUTER_KEY_PATTERN = re.compile(r"^chunk_\d+\.layer_(?P<layer>\d+)\.mlp\.router$")
_TRACE_CHUNK_PREFIX_PATTERN = re.compile(r"^chunk(?P<chunk>\d+)\.(?P<name>.+)$")
logger = logging.getLogger(__name__)
_ACTIVE_ROUTING_REPLAY_CONTROLLER: Any | None = None


def _active_routing_replay_controller() -> Any | None:
    return _ACTIVE_ROUTING_REPLAY_CONTROLLER


@torch.compiler.disable
def _routing_with_replay_boundary(
    router_module: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    controller = _active_routing_replay_controller()
    if controller is not None:
        controller._prepare_native_target_for_router(
            router_module._art_routing_replay_router_key,
            logits=args[0],
        )
    return router_module._art_routing_replay_original(*args, **kwargs)


def _to_tensor_cpu_contiguous(
    tensor: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def _normalize_step_index(step_index: int) -> str:
    if step_index < 0:
        raise ValueError(f"step_index must be non-negative, got {step_index}")
    return f"{step_index:06d}"


def _build_tensor_key(router_key: str, call_index: int, field_name: str) -> str:
    return f"{router_key}/call_{call_index}/{field_name}"


def build_router_key_from_module_name(*, chunk_index: int, module_name: str) -> str:
    canonical_name = canonical_art_param_name(module_name)
    match = _ROUTER_LAYER_PATTERN.search(canonical_name)
    if match is None:
        raise RuntimeError(
            f"Unable to derive router key from module name '{module_name}'. "
            f"Canonicalized to '{canonical_name}', expected suffix matching "
            f"'{_ROUTER_LAYER_PATTERN.pattern}'."
        )
    layer_index = int(match.group("layer"))
    return f"chunk_{chunk_index:02d}.layer_{layer_index:04d}.mlp.router"


def _router_key_for_model_module(
    *,
    module_name: str,
    layer_prefixes: list[tuple[str, int]],
    fallback_chunk_index: int | None,
) -> str:
    for prefix, global_layer_index in layer_prefixes:
        if module_name.startswith(f"{prefix}."):
            return f"chunk_00.layer_{global_layer_index:04d}.mlp.router"
    if fallback_chunk_index is not None:
        return build_router_key_from_module_name(
            chunk_index=fallback_chunk_index,
            module_name=module_name,
        )
    raise RuntimeError(
        "PP/VPP routing replay requires every router to have an owning "
        f"TransformerLayer; router='{module_name}'"
    )


def _global_layer_prefixes(chunk: Any) -> list[tuple[str, int]]:
    from megatron.core.transformer.transformer_layer import TransformerLayer

    prefixes: dict[str, int] = {}
    for module_name, module in chunk.named_modules():
        original = getattr(module, "_orig_mod", None)
        layer = (
            original
            if isinstance(original, TransformerLayer)
            else module
            if isinstance(module, TransformerLayer)
            else None
        )
        if layer is not None:
            prefixes[module_name] = int(layer.layer_number) - 1
    return sorted(prefixes.items(), key=lambda item: len(item[0]), reverse=True)


def prepare_moe_routing_replay_boundaries(
    model_chunks: list[Any],
    *,
    pipeline_model: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Install stable eager router boundaries before model compilation."""
    if pipeline_model is None:
        from megatron.core import parallel_state as ps

        pipeline_model = len(model_chunks) > 1 or (
            ps.model_parallel_is_initialized()
            and int(ps.get_pipeline_model_parallel_world_size()) > 1
        )
    bindings: dict[str, dict[str, Any]] = {}
    for chunk_index, chunk in enumerate(model_chunks):
        layer_prefixes = _global_layer_prefixes(chunk)
        for module_name, module in chunk.named_modules():
            if ROUTER_NAME_TOKEN not in module_name or not hasattr(module, "routing"):
                continue
            router_key = _router_key_for_model_module(
                module_name=module_name,
                layer_prefixes=layer_prefixes,
                fallback_chunk_index=None if pipeline_model else chunk_index,
            )
            if router_key in bindings:
                raise RuntimeError(
                    "Multiple local model chunks own the same replay router: "
                    f"router_key='{router_key}'"
                )
            config = getattr(module, "config", None)
            if bool(getattr(config, "moe_router_fusion", False)):
                raise RuntimeError(
                    "MoE routing replay requires moe_router_fusion=False because "
                    "Megatron Core fused routing bypasses RouterReplay: "
                    f"router_key='{router_key}'"
                )
            router_replay = getattr(module, "router_replay", None)
            if router_replay is None:
                raise RuntimeError(
                    "MoE routing replay requires provider.moe_enable_routing_replay=True "
                    "before model construction: "
                    f"router_key='{router_key}'"
                )
            installed_key = getattr(
                module, "_art_routing_replay_router_key", router_key
            )
            if installed_key != router_key:
                raise RuntimeError(
                    "Routing replay boundary key changed after model construction: "
                    f"{installed_key!r} != {router_key!r}"
                )
            if not getattr(module, "_art_routing_replay_target_patched", False):
                module._art_routing_replay_original = module.routing
                module._art_routing_replay_router_key = router_key
                module.routing = types.MethodType(_routing_with_replay_boundary, module)
                module._art_routing_replay_target_patched = True
            bindings[router_key] = {
                "module": module,
                "router_replay": router_replay,
                "sequence_parallel": bool(getattr(config, "sequence_parallel", False)),
                "context_parallel_size": int(
                    getattr(config, "context_parallel_size", 1)
                ),
                "topk": int(getattr(module, "topk")),
                "chunk_index": chunk_index,
                "layer_index": _global_layer_from_router_key(router_key),
                "num_experts": int(getattr(config, "num_moe_experts", 0) or 0),
            }
    return bindings


def build_router_key_from_trace_name(trace_module_name: str) -> str:
    chunk_match = _TRACE_CHUNK_PREFIX_PATTERN.match(trace_module_name)
    if chunk_match is None:
        raise RuntimeError(
            "Forward trace router module name must start with 'chunk<idx>.'; "
            f"got '{trace_module_name}'"
        )
    return build_router_key_from_module_name(
        chunk_index=int(chunk_match.group("chunk")),
        module_name=chunk_match.group("name"),
    )


def _global_layer_from_router_key(router_key: str) -> int:
    match = _ROUTER_KEY_PATTERN.fullmatch(router_key)
    if match is None:
        raise RuntimeError(f"Invalid routing replay router key: {router_key!r}")
    return int(match.group("layer"))


class ParallelTopology(BaseModel):
    tp: int
    ep: int
    etp: int = 1
    dp: int = 1
    sp: bool = False
    cp: int = 1
    pp: int = 1
    vpp: int = 1


class RouterCallRoute(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    expert_indices: torch.Tensor
    expert_probs: torch.Tensor | None = None
    expert_mask: torch.Tensor | None = None
    num_experts: int
    sample_index: int | None = None
    micro_slot: int | None = None
    rank_token_counts: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "RouterCallRoute":
        self.expert_indices = _to_tensor_cpu_contiguous(
            self.expert_indices, dtype=torch.int32
        )
        if self.expert_probs is not None:
            self.expert_probs = _to_tensor_cpu_contiguous(
                self.expert_probs, dtype=torch.float32
            )
        if self.expert_indices.ndim != 2:
            raise RuntimeError(
                "expert_indices must have shape [num_tokens, topk], got "
                f"{tuple(self.expert_indices.shape)}"
            )
        if (
            self.expert_probs is not None
            and self.expert_probs.shape != self.expert_indices.shape
        ):
            raise RuntimeError(
                "expert_probs shape must match expert_indices shape, got "
                f"{tuple(self.expert_probs.shape)} vs {tuple(self.expert_indices.shape)}"
            )
        if self.expert_mask is not None:
            self.expert_mask = _to_tensor_cpu_contiguous(
                self.expert_mask, dtype=torch.bool
            )
            if self.expert_mask.shape != self.expert_indices.shape:
                raise RuntimeError(
                    "expert_mask shape must match expert_indices shape, got "
                    f"{tuple(self.expert_mask.shape)} vs {tuple(self.expert_indices.shape)}"
                )
        if self.num_experts <= 0:
            raise RuntimeError(f"num_experts must be >0, got {self.num_experts}")
        selected = (
            self.expert_indices
            if self.expert_mask is None
            else self.expert_indices[self.expert_mask]
        )
        if int(selected.numel()) > 0 and (
            int(selected.min().item()) < 0
            or int(selected.max().item()) >= int(self.num_experts)
        ):
            raise RuntimeError(
                "expert_indices contain ids outside [0, num_experts): "
                f"num_experts={self.num_experts}"
            )
        if self.sample_index is not None:
            self.sample_index = int(self.sample_index)
        if self.micro_slot is not None:
            self.micro_slot = int(self.micro_slot)
        if self.rank_token_counts is not None:
            counts = tuple(int(count) for count in self.rank_token_counts)
            if any(count < 0 for count in counts):
                raise RuntimeError(
                    f"rank_token_counts must be non-negative, got {counts}"
                )
            if sum(counts) != int(self.expert_indices.shape[0]):
                raise RuntimeError(
                    "rank_token_counts must sum to route token count: "
                    f"counts={counts}, tokens={int(self.expert_indices.shape[0])}"
                )
            self.rank_token_counts = counts
        return self

    @property
    def num_global_tokens(self) -> int:
        return int(self.expert_indices.shape[0])

    @property
    def max_topk(self) -> int:
        return int(self.expert_indices.shape[1])


def _router_call_key(route: RouterCallRoute) -> tuple[str, int]:
    if route.sample_index is not None:
        return ("sample", int(route.sample_index))
    if route.micro_slot is not None:
        return ("dummy_micro_slot", int(route.micro_slot))
    raise RuntimeError("Routing replay calls require sample_index or micro_slot")


class StepRouterRoutes(BaseModel):
    calls: dict[int, RouterCallRoute]

    @model_validator(mode="after")
    def _validate_calls(self) -> "StepRouterRoutes":
        if not self.calls:
            raise RuntimeError("StepRouterRoutes.calls cannot be empty")
        for call_index in self.calls:
            if call_index < 0:
                raise RuntimeError(f"call_index must be >=0, got {call_index}")
        return self


class StepRoutes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    routers: dict[str, StepRouterRoutes]
    global_token_uids: torch.Tensor

    @model_validator(mode="after")
    def _validate(self) -> "StepRoutes":
        if not self.routers:
            raise RuntimeError("StepRoutes.routers cannot be empty")
        self.global_token_uids = _to_tensor_cpu_contiguous(
            self.global_token_uids, dtype=torch.int64
        )
        if self.global_token_uids.ndim != 1:
            raise RuntimeError(
                "global_token_uids must have shape [num_global_tokens], got "
                f"{tuple(self.global_token_uids.shape)}"
            )
        if int(torch.unique(self.global_token_uids).numel()) != int(
            self.global_token_uids.numel()
        ):
            raise RuntimeError("global_token_uids must be unique per step")
        expected_tokens = int(self.global_token_uids.numel())
        token_count_by_call_key: dict[tuple[str, int], int] = {}
        for router_key, step_router in self.routers.items():
            for call_index, route in step_router.calls.items():
                call_key = _router_call_key(route)
                if route.num_global_tokens > expected_tokens:
                    raise RuntimeError(
                        "Route token count exceeds step global_token_uids span: "
                        f"router='{router_key}', call={call_index}, "
                        f"call_key={call_key}, route_tokens={route.num_global_tokens}, "
                        f"global_token_uids={expected_tokens}"
                    )
                previous_token_count = token_count_by_call_key.get(call_key)
                if (
                    previous_token_count is not None
                    and previous_token_count != route.num_global_tokens
                ):
                    raise RuntimeError(
                        "Route token count must be consistent for the same micro: "
                        f"router='{router_key}', call={call_index}, "
                        f"call_key={call_key}, expected={previous_token_count}, "
                        f"got={route.num_global_tokens}"
                    )
                token_count_by_call_key[call_key] = route.num_global_tokens
        return self


class MoeRoutingReplayBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    format_version: str = ROUTER_KEY_FORMAT_VERSION
    topology: ParallelTopology
    num_steps: int
    max_topk: int
    router_keys: list[str]
    steps: dict[int, StepRoutes] = Field(default_factory=dict)
    expert_indices: torch.Tensor | None = None
    num_experts: int | None = None
    global_grad_accumulation_sequences: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "MoeRoutingReplayBundle":
        if self.format_version != ROUTER_KEY_FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported MoE routing replay bundle format: "
                f"{self.format_version!r}; expected {ROUTER_KEY_FORMAT_VERSION!r}"
            )
        if self.num_steps <= 0:
            raise RuntimeError(f"num_steps must be >0, got {self.num_steps}")
        if self.max_topk <= 0:
            raise RuntimeError(f"max_topk must be >0, got {self.max_topk}")
        if not self.router_keys:
            raise RuntimeError("router_keys cannot be empty")
        if len(set(self.router_keys)) != len(self.router_keys):
            raise RuntimeError("router_keys must be unique")
        if self.expert_indices is not None:
            self._validate_tensor_storage()
            return self
        if (
            self.num_experts is not None
            or self.global_grad_accumulation_sequences is not None
        ):
            raise RuntimeError("Legacy replay bundles cannot carry tensor metadata")
        expected_steps = set(range(self.num_steps))
        if set(self.steps) != expected_steps:
            raise RuntimeError(
                f"steps must contain exactly {sorted(expected_steps)}, got "
                f"{sorted(self.steps)}"
            )
        router_key_set = set(self.router_keys)
        for step_index, step_routes in self.steps.items():
            if set(step_routes.routers) != router_key_set:
                raise RuntimeError(
                    f"Step {step_index} router keys differ from bundle router keys: "
                    f"step_keys={sorted(step_routes.routers)}, "
                    f"router_keys={self.router_keys}"
                )
            for router_routes in step_routes.routers.values():
                for route in router_routes.calls.values():
                    if route.max_topk > self.max_topk:
                        raise RuntimeError(
                            "Route topk exceeds bundle max_topk: "
                            f"route_topk={route.max_topk}, max_topk={self.max_topk}"
                        )
        return self

    @property
    def tensor_backed(self) -> bool:
        return self.expert_indices is not None

    def _validate_tensor_storage(self) -> None:
        indices = self.expert_indices
        assert indices is not None
        if self.steps:
            raise RuntimeError("Tensor-backed replay cannot also contain route calls")
        if indices.device.type != "cpu" or not indices.is_contiguous():
            raise RuntimeError("Tensor-backed replay requires contiguous CPU storage")
        if indices.ndim != 4 or min(map(int, indices.shape)) <= 0:
            raise RuntimeError(
                "Tensor-backed replay requires [layer, row, position, topk]"
            )
        num_experts = int(self.num_experts or 0)
        expected_dtype = torch.uint8 if num_experts <= 256 else torch.uint16
        if not 1 <= num_experts <= 65_536 or indices.dtype != expected_dtype:
            raise RuntimeError("Tensor-backed replay expert count and dtype disagree")
        accumulation = int(self.global_grad_accumulation_sequences or 0)
        if accumulation <= 0:
            raise RuntimeError("Tensor-backed replay requires positive accumulation")
        layers, sequences, _sequence_length, topk = map(int, indices.shape)
        if (
            layers != len(self.router_keys)
            or topk != self.max_topk
            or self.num_steps != math.ceil(sequences / accumulation)
        ):
            raise RuntimeError("Tensor-backed replay metadata disagrees with its shape")
        expected_keys = [
            f"chunk_00.layer_{layer:04d}.mlp.router" for layer in range(layers)
        ]
        if self.router_keys != expected_keys:
            raise RuntimeError("Tensor-backed replay router keys are not layer-major")

    @classmethod
    def from_dir(cls, bundle_dir: str | Path) -> "MoeRoutingReplayBundle":
        base_dir = Path(bundle_dir)
        manifest_path = base_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing routing replay manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != ROUTER_KEY_FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported MoE routing replay bundle format: "
                f"{manifest.get('format_version')!r}; expected "
                f"{ROUTER_KEY_FORMAT_VERSION!r}"
            )
        if manifest.get("storage") == "layer_major":
            loaded = load_file(str(base_dir / manifest["file"]))
            indices = loaded["expert_indices"].detach().clone().contiguous()
            del loaded
            return cls(
                format_version=manifest["format_version"],
                topology=ParallelTopology.model_validate(manifest["topology"]),
                num_steps=int(manifest["num_steps"]),
                max_topk=int(manifest["max_topk"]),
                router_keys=list(manifest["router_keys"]),
                expert_indices=indices,
                num_experts=int(manifest["num_experts"]),
                global_grad_accumulation_sequences=int(
                    manifest["global_grad_accumulation_sequences"]
                ),
            )
        if manifest.get("storage") != "calls":
            raise RuntimeError("Unknown MoE routing replay storage format")

        steps: dict[int, StepRoutes] = {}
        for step_index_str, step_info in manifest["steps"].items():
            step_index = int(step_index_str)
            loaded_tensors = load_file(str(base_dir / step_info["file"]))
            # Own CPU storage immediately. Safetensors CPU loads can keep
            # file-backed storage alive, which makes shared-filesystem cleanup
            # fail while long-lived Megatron ranks still hold replay bundles.
            step_tensors = {
                key: tensor.detach().clone().contiguous()
                for key, tensor in loaded_tensors.items()
            }
            del loaded_tensors
            if GLOBAL_TOKEN_UIDS_KEY not in step_tensors:
                raise RuntimeError(
                    f"Missing tensor key '{GLOBAL_TOKEN_UIDS_KEY}' for step={step_index}"
                )
            routers: dict[str, StepRouterRoutes] = {}
            for router_key, call_manifest in step_info["routers"].items():
                calls: dict[int, RouterCallRoute] = {}
                for call_index_str, call_info in call_manifest.items():
                    call_index = int(call_index_str)
                    indices_key = _build_tensor_key(
                        router_key, call_index, "expert_indices"
                    )
                    probs_key = _build_tensor_key(
                        router_key, call_index, "expert_probs"
                    )
                    mask_key = _build_tensor_key(router_key, call_index, "expert_mask")
                    if indices_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key {indices_key} in {step_info['file']}"
                        )
                    calls[call_index] = RouterCallRoute.model_construct(
                        expert_indices=step_tensors[indices_key],
                        expert_probs=step_tensors.get(probs_key),
                        expert_mask=step_tensors.get(mask_key),
                        num_experts=int(call_info["num_experts"]),
                        sample_index=call_info.get("sample_index"),
                        micro_slot=call_info.get("micro_slot"),
                        rank_token_counts=call_info.get("rank_token_counts"),
                    )
                routers[router_key] = StepRouterRoutes.model_construct(calls=calls)
            steps[step_index] = StepRoutes.model_construct(
                routers=routers,
                global_token_uids=step_tensors[GLOBAL_TOKEN_UIDS_KEY],
            )

        return cls.model_construct(
            format_version=manifest["format_version"],
            topology=ParallelTopology.model_validate(manifest["topology"]),
            num_steps=int(manifest["num_steps"]),
            max_topk=int(manifest["max_topk"]),
            router_keys=list(manifest["router_keys"]),
            steps=steps,
        )

    def to_dir(self, bundle_dir: str | Path) -> None:
        base_dir = Path(bundle_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if self.tensor_backed:
            assert self.expert_indices is not None
            tensor_file = "layer_major.safetensors"
            save_file(
                {"expert_indices": self.expert_indices}, str(base_dir / tensor_file)
            )
            manifest = {
                "format_version": self.format_version,
                "storage": "layer_major",
                "file": tensor_file,
                "topology": self.topology.model_dump(mode="json"),
                "num_steps": self.num_steps,
                "max_topk": self.max_topk,
                "router_keys": self.router_keys,
                "num_experts": self.num_experts,
                "global_grad_accumulation_sequences": (
                    self.global_grad_accumulation_sequences
                ),
            }
            with (base_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
            return
        manifest_steps: dict[str, Any] = {}

        for step_index, step_routes in sorted(self.steps.items()):
            step_name = f"step_{_normalize_step_index(step_index)}.safetensors"
            step_tensors: dict[str, torch.Tensor] = {
                GLOBAL_TOKEN_UIDS_KEY: step_routes.global_token_uids
            }
            routers_manifest: dict[str, Any] = {}
            for router_key, router_routes in sorted(step_routes.routers.items()):
                calls_manifest: dict[str, Any] = {}
                for call_index, route in sorted(router_routes.calls.items()):
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_indices")
                    ] = route.expert_indices
                    if route.expert_probs is not None:
                        step_tensors[
                            _build_tensor_key(router_key, call_index, "expert_probs")
                        ] = route.expert_probs
                    if route.expert_mask is not None:
                        step_tensors[
                            _build_tensor_key(router_key, call_index, "expert_mask")
                        ] = route.expert_mask.contiguous()
                    call_info: dict[str, Any] = {"num_experts": int(route.num_experts)}
                    if route.sample_index is not None:
                        call_info["sample_index"] = int(route.sample_index)
                    if route.micro_slot is not None:
                        call_info["micro_slot"] = int(route.micro_slot)
                    if route.rank_token_counts is not None:
                        call_info["rank_token_counts"] = [
                            int(count) for count in route.rank_token_counts
                        ]
                    calls_manifest[str(call_index)] = call_info
                routers_manifest[router_key] = calls_manifest
            save_file(step_tensors, str(base_dir / step_name))
            manifest_steps[str(step_index)] = {
                "file": step_name,
                "routers": routers_manifest,
            }

        manifest = {
            "format_version": self.format_version,
            "storage": "calls",
            "topology": self.topology.model_dump(mode="json"),
            "num_steps": self.num_steps,
            "max_topk": self.max_topk,
            "router_keys": self.router_keys,
            "steps": manifest_steps,
        }
        with (base_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)


def build_moe_routing_replay_bundle_from_packed_tensors(
    *,
    packed_tensors: PackedTensors,
    global_grad_accumulation_sequences: int,
    topology: ParallelTopology | None = None,
) -> MoeRoutingReplayBundle:
    routing_replay = packed_tensors.get("moe_routing_replay")
    if routing_replay is None:
        raise RuntimeError("Packed tensors do not contain MoE routing replay data")
    if global_grad_accumulation_sequences <= 0:
        raise RuntimeError(
            "global_grad_accumulation_sequences must be positive when building "
            f"MoE routing replay bundles, got {global_grad_accumulation_sequences}"
        )
    expert_indices = routing_replay.expert_indices
    num_layers, num_sequences, _sequence_length, topk = map(int, expert_indices.shape)

    router_keys = [
        f"chunk_00.layer_{layer_index:04d}.mlp.router"
        for layer_index in range(num_layers)
    ]
    num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
    return MoeRoutingReplayBundle(
        topology=topology or parallel_topology_from_env(),
        num_steps=num_steps,
        max_topk=topk,
        router_keys=router_keys,
        expert_indices=expert_indices,
        num_experts=routing_replay.num_experts,
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )


def parallel_topology_from_env() -> ParallelTopology:
    tp = _env_int("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", 1)
    ep = _env_int("ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE", 1)
    etp = _env_int(
        "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE",
        _env_int("ART_MEGATRON_EXPERT_TENSOR_MODEL_PARALLEL_SIZE", 1),
    )
    cp = _env_int("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", 1)
    pp = _env_int("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", 1)
    vpp = _env_int("ART_MEGATRON_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE", 1)
    world_size = _env_int("WORLD_SIZE", tp * cp * pp)
    model_parallel_size = tp * cp * pp
    if world_size % model_parallel_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} is not divisible by TP*CP*PP="
            f"{model_parallel_size}"
        )
    return ParallelTopology(
        tp=tp,
        ep=ep,
        etp=etp,
        dp=world_size // model_parallel_size,
        sp=tp > 1,
        cp=cp,
        pp=pp,
        vpp=vpp,
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _synthetic_replay_rows(
    *,
    row_positions: torch.Tensor,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    return _synthetic_replay_layer_rows(
        row_positions=row_positions,
        layer_seeds=torch.tensor([seed], dtype=torch.long),
        num_experts=num_experts,
        topk=topk,
        dtype=dtype,
    )[0]


def _synthetic_replay_layer_rows(
    *,
    row_positions: torch.Tensor,
    layer_seeds: torch.Tensor,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_experts <= 0:
        raise RuntimeError(f"num_experts must be >0, got {num_experts}")
    if topk <= 0 or topk > num_experts:
        raise RuntimeError(
            f"MoE routing topk must be in [1, num_experts], got topk={topk}, "
            f"num_experts={num_experts}"
        )
    positions = row_positions.to(device="cpu", dtype=torch.long).reshape(1, -1, 1)
    seeds = layer_seeds.to(device="cpu", dtype=torch.long).reshape(-1, 1, 1)
    offsets = torch.arange(topk, dtype=torch.long).reshape(1, 1, -1)
    rows = (seeds + (positions + 1) * 1_299_709 + offsets) % num_experts
    return rows.to(dtype=dtype)


class LocalTokenIndexer(Protocol):
    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        """Build local token uid order for current rank."""


class TopologyAwareLocalTokenIndexer:
    def __init__(self, parallel_state_module: Any | None = None) -> None:
        self._parallel_state = parallel_state_module

    def _ps(self) -> Any:
        if self._parallel_state is not None:
            return self._parallel_state
        from megatron.core import parallel_state as ps

        self._parallel_state = ps
        return ps

    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        ps = self._ps()
        local_uids = global_token_uids.to(dtype=torch.int64, device="cpu").view(1, -1)

        cp_size = int(ps.get_context_parallel_world_size())
        if context_parallel_size > 1 and cp_size > 1:
            from megatron.core.utils import get_batch_on_this_cp_rank

            local_uids = get_batch_on_this_cp_rank({"tokens": local_uids})["tokens"]

        tp_size = int(ps.get_tensor_model_parallel_world_size())
        tp_rank = int(ps.get_tensor_model_parallel_rank()) if tp_size > 1 else 0
        if sequence_parallel and tp_size > 1:
            total_tokens = int(local_uids.shape[1])
            if total_tokens != num_local_tokens:
                if total_tokens % tp_size != 0:
                    raise RuntimeError(
                        "Routing replay cannot derive sequence-parallel local token "
                        "uids from merged rows: "
                        f"total_tokens={total_tokens}, tp_size={tp_size}, "
                        f"num_local_tokens={num_local_tokens}"
                    )
                tokens_per_tp_rank = total_tokens // tp_size
                if tokens_per_tp_rank != num_local_tokens:
                    raise RuntimeError(
                        "Routing replay local token uid count mismatch after "
                        "context-parallel slicing: "
                        f"total_tokens={total_tokens}, tp_size={tp_size}, "
                        f"expected_local_tokens={num_local_tokens}, "
                        f"tp_local_tokens={tokens_per_tp_rank}"
                    )
                start = tp_rank * tokens_per_tp_rank
                local_uids = local_uids[:, start : start + tokens_per_tp_rank]

        local_uids = local_uids.reshape(-1).contiguous()
        if int(local_uids.numel()) != num_local_tokens:
            raise RuntimeError(
                "Routing replay local token uid count mismatch: "
                f"expected={num_local_tokens}, got={int(local_uids.numel())}"
            )
        return local_uids


def _router_replay_classes() -> tuple[type[Any], type[Any]]:
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    return RouterReplay, RouterReplayAction


class MoeRoutingReplayController:
    def __init__(
        self,
        *,
        bundle: MoeRoutingReplayBundle,
        strict: bool,
        local_token_indexer: LocalTokenIndexer | None = None,
        allow_recompute_reuse: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        self.bundle = bundle
        self.strict = strict
        self.allow_recompute_reuse = allow_recompute_reuse
        self.local_token_indexer = (
            local_token_indexer or TopologyAwareLocalTokenIndexer()
        )
        self._device = torch.device(device) if device is not None else None

        self._active_step_index: int | None = None
        self._active_step_samples: list[int | None] = []
        self._active_sample_index: int | None = None
        self._active_step_routes: StepRoutes | None = None
        self._active_micro_order: int | None = None
        self._active_chunk_index: int | None = None
        self._router_call_cursors: dict[str, int] = {}
        self._router_call_sequences: dict[str, list[int]] = {}
        self._router_last_call_indices: dict[str, int] = {}
        self._router_last_call_keys: dict[str, tuple[str, int] | None] = {}
        self._router_consumed_calls: dict[str, dict[tuple[str, int], int]] = {}
        self._router_reuse_counts: dict[str, int] = {}
        self._global_uid_to_row_index: dict[int, int] = {}
        self._global_uid_dense_start: int | None = None
        self._global_uid_count: int = 0
        self._local_router_keys: set[str] = set()
        self._local_router_keys_by_chunk: dict[int, set[str]] = {}
        self._router_bindings: dict[str, dict[str, Any]] = {}
        self._runtime_topology: ParallelTopology | None = None
        self._expect_recompute_reuse = False
        self._prepared_uid_sets: dict[str, torch.Tensor] = {}
        self._prepared_targets: dict[tuple[str, str, int], torch.Tensor] = {}
        self._router_prepared_target_keys: dict[str, tuple[str, int]] = {}
        self._step_targets: dict[tuple[str, str, int], torch.Tensor] = {}
        self._step_target_ready_events: dict[
            tuple[str, str, int], torch.cuda.Event
        ] = {}
        self._host_target_staging: list[torch.Tensor] = []
        self._target_copy_stream: torch.cuda.Stream | None = None
        self._active_token_uid_key: str | None = None

    def update_bundle(self, *, bundle: MoeRoutingReplayBundle, strict: bool) -> None:
        self.bundle = bundle
        self.strict = strict
        self.clear_replay_state()
        self._validate_runtime_topology()
        if self.strict:
            missing = sorted(
                router_key
                for router_key in self._local_router_keys
                if router_key not in self.bundle.router_keys
            )
            if missing:
                raise RuntimeError(
                    "Router keys from model are missing in replay bundle: "
                    f"router_keys={missing}"
                )
        self._validate_local_routes()

    def clear_replay_state(self) -> None:
        self._clear_native_router_replay_state()
        self._reset_step_state()

    def _target_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def install_router_patches(self, model_chunks: list[Any]) -> None:
        global _ACTIVE_ROUTING_REPLAY_CONTROLLER
        if self._router_bindings:
            return
        pipeline_model = self.bundle.topology.pp > 1 or len(model_chunks) > 1
        bindings = prepare_moe_routing_replay_boundaries(
            model_chunks,
            pipeline_model=pipeline_model,
        )
        for router_key, binding in bindings.items():
            chunk_index = int(binding["chunk_index"])
            if chunk_index not in self._local_router_keys_by_chunk:
                self._local_router_keys_by_chunk[chunk_index] = set()
            if self.strict and router_key not in self.bundle.router_keys:
                raise RuntimeError(
                    "Router key from model is missing in replay bundle: "
                    f"router_key='{router_key}'"
                )
            self._router_bindings[router_key] = binding
            self._local_router_keys.add(router_key)
            self._local_router_keys_by_chunk[chunk_index].add(router_key)
        self._runtime_topology = self._runtime_parallel_topology(model_chunks)
        self._validate_runtime_topology()
        self._validate_local_routes()
        self._expect_recompute_reuse = bool(self._router_bindings) and all(
            getattr(binding["module"].config, "recompute_granularity", None) == "full"
            and getattr(binding["module"].config, "recompute_method", None) == "uniform"
            and int(getattr(binding["module"].config, "recompute_num_layers", 0) or 0)
            == 1
            for binding in self._router_bindings.values()
        )
        _ACTIVE_ROUTING_REPLAY_CONTROLLER = self

    def remove_router_patches(self) -> None:
        global _ACTIVE_ROUTING_REPLAY_CONTROLLER
        if _ACTIVE_ROUTING_REPLAY_CONTROLLER is self:
            _ACTIVE_ROUTING_REPLAY_CONTROLLER = None
        self._router_bindings.clear()
        self._local_router_keys.clear()
        self._local_router_keys_by_chunk.clear()
        self._runtime_topology = None
        self._expect_recompute_reuse = False
        self._step_targets.clear()
        self._clear_native_router_replay_state()
        self._reset_step_state()

    @staticmethod
    def _runtime_parallel_topology(
        model_chunks: list[Any],
    ) -> ParallelTopology | None:
        if not torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
            return None
        from megatron.core import parallel_state as ps
        from megatron.core.utils import get_model_config

        sequence_parallel = {
            bool(getattr(get_model_config(chunk), "sequence_parallel", False))
            for chunk in model_chunks
        }
        if len(sequence_parallel) != 1:
            raise RuntimeError(
                "Model chunks disagree on sequence_parallel: "
                f"values={sorted(sequence_parallel)}"
            )
        return ParallelTopology(
            tp=int(ps.get_tensor_model_parallel_world_size()),
            ep=int(ps.get_expert_model_parallel_world_size()),
            etp=int(ps.get_expert_tensor_parallel_world_size()),
            dp=int(ps.get_data_parallel_world_size()),
            sp=sequence_parallel.pop(),
            cp=int(ps.get_context_parallel_world_size()),
            pp=int(ps.get_pipeline_model_parallel_world_size()),
            vpp=int(ps.get_virtual_pipeline_model_parallel_world_size() or 1),
        )

    def _validate_runtime_topology(self) -> None:
        if (
            self._runtime_topology is not None
            and self.bundle.topology != self._runtime_topology
        ):
            raise RuntimeError(
                "Routing replay bundle topology differs from the active trainer: "
                f"bundle={self.bundle.topology.model_dump()}, "
                f"runtime={self._runtime_topology.model_dump()}"
            )

    def _validate_local_routes(self) -> None:
        if self.bundle.tensor_backed:
            assert self.bundle.expert_indices is not None
            for router_key, binding in self._router_bindings.items():
                if router_key not in self.bundle.router_keys:
                    continue
                model_num_experts = int(binding["num_experts"])
                if model_num_experts and model_num_experts != self.bundle.num_experts:
                    raise RuntimeError(
                        "Replay expert count does not match the model router: "
                        f"router='{router_key}', replay={self.bundle.num_experts}, "
                        f"model={model_num_experts}"
                    )
                if int(binding["topk"]) != self.bundle.max_topk:
                    raise RuntimeError(
                        "Replay route topk does not match Megatron router topk: "
                        f"router='{router_key}', replay={self.bundle.max_topk}, "
                        f"router_topk={binding['topk']}"
                    )
                if int(binding["layer_index"]) >= int(
                    self.bundle.expert_indices.shape[0]
                ):
                    raise RuntimeError(
                        f"Replay has no global layer for router '{router_key}'"
                    )
            return
        for router_key, binding in self._router_bindings.items():
            if router_key not in self.bundle.router_keys:
                continue
            model_num_experts = int(binding["num_experts"])
            for step_index, step in self.bundle.steps.items():
                for call_index, route in step.routers[router_key].calls.items():
                    selected = (
                        route.expert_indices
                        if route.expert_mask is None
                        else route.expert_indices[route.expert_mask]
                    )
                    if int(selected.numel()) == 0:
                        continue
                    minimum = int(selected.min().item())
                    maximum = int(selected.max().item())
                    limit = model_num_experts or int(route.num_experts)
                    if minimum < 0 or maximum >= limit:
                        raise RuntimeError(
                            "Replay route expert id is outside the model router: "
                            f"step={step_index}, router='{router_key}', "
                            f"call={call_index}, range=[{minimum}, {maximum}], "
                            f"num_experts={limit}"
                        )

    def _active_local_router_keys(self) -> set[str]:
        if self._active_chunk_index is None:
            raise RuntimeError("Routing replay chunk is not active")
        try:
            return self._local_router_keys_by_chunk[self._active_chunk_index]
        except KeyError as exc:
            raise RuntimeError(
                f"Routing replay received unknown model chunk {self._active_chunk_index}"
            ) from exc

    def begin_micro(
        self,
        sample_index: int | None,
        micro_order: int,
        chunk_index: int = 0,
    ) -> None:
        if self._active_step_index is None:
            raise RuntimeError("Routing replay begin_micro called before set_step")
        if self.bundle.tensor_backed:
            if not 0 <= micro_order < len(self._active_step_samples):
                raise RuntimeError(
                    f"Routing replay micro order is out of range: {micro_order}"
                )
            expected_sample = self._active_step_samples[micro_order]
            if sample_index != expected_sample:
                raise RuntimeError(
                    "Routing replay micro sample differs from set_step: "
                    f"micro={micro_order}, expected={expected_sample}, "
                    f"actual={sample_index}"
                )
        self._active_sample_index = sample_index
        self._active_micro_order = micro_order
        self._active_chunk_index = chunk_index
        self._reset_staged_micro_targets()
        for router_key in sorted(self._active_local_router_keys()):
            call_indices = self._active_micro_call_indices(router_key)
            if len(call_indices) != 1:
                raise RuntimeError(
                    "Routing replay expected exactly one router call per local "
                    f"microbatch for router='{router_key}', got {call_indices}"
                )

    def set_local_input_token_uids(
        self,
        local_token_uids: torch.Tensor | None,
    ) -> None:
        self.prepare_micro_targets({"attention": local_token_uids})

    def prepare_micro_targets(
        self,
        token_uid_sets: dict[str, torch.Tensor | None],
        *,
        active_token_uid_key: str = "attention",
    ) -> None:
        if self._active_step_index is None or self._active_micro_order is None:
            raise RuntimeError(
                "Routing replay target staging requires set_step and begin_micro"
            )
        self._reset_staged_micro_targets()
        prepared_uid_sets = {
            key: self._normalize_token_uids(value)
            for key, value in token_uid_sets.items()
            if value is not None
        }
        if not prepared_uid_sets:
            raise RuntimeError("Routing replay requires at least one token UID set")
        if active_token_uid_key not in prepared_uid_sets:
            raise RuntimeError(
                "Routing replay active token UID key was not prepared: "
                f"key='{active_token_uid_key}', prepared={sorted(prepared_uid_sets)}"
            )
        self._prepared_uid_sets = prepared_uid_sets
        active_router_keys = self._active_local_router_keys()
        if not active_router_keys:
            self._active_token_uid_key = active_token_uid_key
            return
        new_target_keys: list[tuple[str, str, int]] = []
        for token_uid_key, token_uids in prepared_uid_sets.items():
            for router_key in sorted(active_router_keys):
                call_indices = self._active_micro_call_indices(router_key)
                if len(call_indices) != 1:
                    raise RuntimeError(
                        "Routing replay expected exactly one active router call while "
                        f"staging targets for router='{router_key}', got {call_indices}"
                    )
                call_index = call_indices[0]
                binding = self._router_bindings[router_key]
                target_key = (token_uid_key, router_key, call_index)
                cached_target = self._step_targets.get(target_key)
                if cached_target is not None:
                    self._prepared_targets[target_key] = cached_target
                    continue
                router_token_uids = self._token_uids_for_router_binding(
                    token_uids,
                    sequence_parallel=bool(binding["sequence_parallel"]),
                )
                target_cpu = self._explicit_target_for_router_call(
                    router_key=router_key,
                    call_index=call_index,
                    explicit_uids=router_token_uids,
                )
                self._stage_prepared_target(
                    target_key=target_key,
                    target_cpu=target_cpu,
                )
                self._step_targets[target_key] = self._prepared_targets[target_key]
                new_target_keys.append(target_key)
        self._record_target_copy_event(new_target_keys)
        self.set_active_token_uid_key(active_token_uid_key)

    def set_active_token_uid_key(self, token_uid_key: str) -> None:
        if not self._active_local_router_keys():
            self._active_token_uid_key = token_uid_key
            return
        prepared_keys = {
            key for key, _router_key, _call_index in self._prepared_targets.keys()
        }
        if token_uid_key not in prepared_keys:
            raise RuntimeError(
                "Routing replay token UID key was not staged for this micro: "
                f"key='{token_uid_key}', staged={sorted(prepared_keys)}"
            )
        self._active_token_uid_key = token_uid_key

    @staticmethod
    def _normalize_token_uids(local_token_uids: torch.Tensor) -> torch.Tensor:
        if local_token_uids.device.type != "cpu":
            raise RuntimeError(
                "Routing replay token UIDs must be CPU metadata. Passing CUDA token "
                "UIDs would force a host/device synchronization in the model path."
            )
        return local_token_uids.detach().to(dtype=torch.int64).contiguous().reshape(-1)

    def local_token_uids_for_active_dispatch(
        self,
        *,
        num_local_tokens: int,
        sequence_parallel: bool,
    ) -> torch.Tensor | None:
        if self._active_token_uid_key is None:
            return None
        token_uids = self._prepared_uid_sets.get(self._active_token_uid_key)
        if token_uids is None:
            return None
        local_uids = self._token_uids_for_router_binding(
            token_uids,
            sequence_parallel=sequence_parallel,
        )
        if int(local_uids.numel()) == int(num_local_tokens):
            return local_uids.contiguous()
        compact_uids = local_uids[local_uids >= 0]
        if int(compact_uids.numel()) == int(num_local_tokens):
            return compact_uids.contiguous()
        return None

    def set_step(
        self,
        *,
        step_index: int,
        sample_index: int | list[int | None] | None,
    ) -> None:
        if self.bundle.tensor_backed:
            self._set_tensor_step(step_index=step_index, sample_index=sample_index)
            RouterReplay, RouterReplayAction = _router_replay_classes()
            RouterReplay.clear_global_indices()
            RouterReplay.set_global_router_replay_action(
                RouterReplayAction.REPLAY_FORWARD
            )
            return
        if step_index not in self.bundle.steps:
            raise RuntimeError(
                f"Replay bundle missing step_index={step_index}. "
                f"Available steps={sorted(self.bundle.steps.keys())}"
            )
        step_routes = self.bundle.steps[step_index]
        self._active_step_index = step_index
        self._active_sample_index = (
            next((index for index in sample_index if index is not None), None)
            if isinstance(sample_index, list)
            else sample_index
        )
        self._active_micro_order = None
        self._active_chunk_index = None
        self._active_step_routes = step_routes
        self._reset_staged_micro_targets()
        self._step_targets = {}
        self._step_target_ready_events = {}
        self._host_target_staging = []
        self._router_call_cursors = {}
        self._router_call_sequences = {}
        self._router_last_call_indices = {}
        self._router_last_call_keys = {}
        self._router_consumed_calls = {}
        self._router_reuse_counts = {}
        self._global_uid_count = int(step_routes.global_token_uids.numel())
        self._global_uid_dense_start = self._dense_global_uid_start(
            step_routes.global_token_uids
        )
        self._global_uid_to_row_index = (
            {}
            if self._global_uid_dense_start is not None
            else {
                int(uid.item()): row_index
                for row_index, uid in enumerate(step_routes.global_token_uids)
            }
        )
        for router_key in sorted(self._local_router_keys):
            if router_key not in step_routes.routers:
                raise RuntimeError(
                    "Replay bundle step is missing local router key: "
                    f"step={step_index}, router='{router_key}'"
                )
            router_calls = step_routes.routers[router_key].calls
            binding_topk = int(self._router_bindings[router_key]["topk"])
            for call_index, route in router_calls.items():
                if route.expert_mask is not None and not bool(
                    route.expert_mask.all().item()
                ):
                    raise RuntimeError(
                        "masked slots are unsupported by Megatron native MoE routing "
                        f"replay: step={step_index}, router='{router_key}', "
                        f"call={call_index}"
                    )
                if route.max_topk != binding_topk:
                    raise RuntimeError(
                        "Replay route topk does not match Megatron router topk: "
                        f"step={step_index}, router='{router_key}', call={call_index}, "
                        f"route_topk={route.max_topk}, router_topk={binding_topk}"
                    )
            self._router_call_cursors[router_key] = 0
            self._router_consumed_calls[router_key] = {}
            self._router_call_sequences[router_key] = self._build_call_sequence(
                router_key=router_key,
                sample_index=sample_index,
            )
        RouterReplay, RouterReplayAction = _router_replay_classes()
        RouterReplay.clear_global_indices()
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    def _set_tensor_step(
        self,
        *,
        step_index: int,
        sample_index: int | list[int | None] | None,
    ) -> None:
        if not 0 <= step_index < self.bundle.num_steps:
            raise RuntimeError(
                f"Replay bundle missing step_index={step_index}. "
                f"Available steps={list(range(self.bundle.num_steps))}"
            )
        samples = sample_index if isinstance(sample_index, list) else [sample_index]
        if not samples:
            raise RuntimeError("Routing replay step requires at least one microbatch")
        assert self.bundle.expert_indices is not None
        accumulation = int(self.bundle.global_grad_accumulation_sequences or 0)
        start = step_index * accumulation
        stop = min(start + accumulation, int(self.bundle.expert_indices.shape[1]))
        real_samples = [sample for sample in samples if sample is not None]
        if len(real_samples) != len(set(real_samples)) or any(
            not start <= sample < stop for sample in real_samples
        ):
            raise RuntimeError(
                "Routing replay samples do not belong to the active step: "
                f"step={step_index}, span=[{start}, {stop}), samples={samples}"
            )

        self._reset_step_state()
        self._active_step_index = step_index
        self._active_step_samples = list(samples)
        self._global_uid_dense_start = 0
        self._global_uid_count = int(self.bundle.expert_indices.shape[2])
        call_sequence = list(range(len(samples)))
        for router_key in self._local_router_keys:
            self._router_call_cursors[router_key] = 0
            self._router_call_sequences[router_key] = call_sequence
            self._router_consumed_calls[router_key] = {}

    def finalize_step(self, *, expect_recompute: bool = False) -> None:
        if self._active_step_index is None:
            raise RuntimeError("finalize_step called before set_step")
        for router_key in sorted(self._local_router_keys):
            consumed = self._router_call_cursors.get(router_key, 0)
            call_sequence = self._router_call_sequences.get(router_key)
            if call_sequence is None:
                raise RuntimeError(
                    "Routing replay call sequence missing for router key: "
                    f"step={self._active_step_index}, router='{router_key}'"
                )
            if consumed != len(call_sequence):
                raise RuntimeError(
                    "Routing replay step consumption mismatch: "
                    f"step={self._active_step_index}, router='{router_key}', "
                    f"consumed={consumed}, expected={len(call_sequence)}"
                )
            if expect_recompute and self._expect_recompute_reuse:
                reused = self._router_reuse_counts.get(router_key, 0)
                if reused != len(call_sequence):
                    raise RuntimeError(
                        "Routing replay recompute consumption mismatch: "
                        f"step={self._active_step_index}, router='{router_key}', "
                        f"reused={reused}, expected={len(call_sequence)}"
                    )
        if self._router_reuse_counts:
            logger.info(
                "Routing replay reused routes for recompute: step=%s counts=%s",
                self._active_step_index,
                dict(sorted(self._router_reuse_counts.items())),
            )
        self._clear_native_router_replay_state()
        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self._active_step_index = None
        self._active_step_samples = []
        self._active_sample_index = None
        self._active_step_routes = None
        self._active_micro_order = None
        self._active_chunk_index = None
        self._router_call_cursors = {}
        self._router_call_sequences = {}
        self._router_last_call_indices = {}
        self._router_last_call_keys = {}
        self._router_consumed_calls = {}
        self._router_reuse_counts = {}
        self._reset_staged_micro_targets()
        self._step_targets = {}
        self._step_target_ready_events = {}
        self._host_target_staging = []
        self._global_uid_to_row_index = {}
        self._global_uid_dense_start = None
        self._global_uid_count = 0

    def _reset_staged_micro_targets(self) -> None:
        self._prepared_uid_sets = {}
        self._prepared_targets = {}
        self._router_prepared_target_keys = {}
        self._active_token_uid_key = None

    @staticmethod
    def _clear_native_router_replay_state() -> None:
        RouterReplay, _RouterReplayAction = _router_replay_classes()
        RouterReplay.clear_global_indices()
        RouterReplay.clear_global_router_replay_action()

    @staticmethod
    def _dense_global_uid_start(global_token_uids: torch.Tensor) -> int | None:
        num_uids = int(global_token_uids.numel())
        if num_uids == 0:
            return None
        start = int(global_token_uids[0].item())
        if num_uids == 1:
            return start
        if bool((global_token_uids[1:] == global_token_uids[:-1] + 1).all().item()):
            return start
        return None

    def _build_call_sequence(
        self,
        *,
        router_key: str,
        sample_index: int | list[int | None] | None,
    ) -> list[int]:
        if self._active_step_routes is None or self._active_step_index is None:
            raise RuntimeError("Routing replay step is not active")
        router_calls = self._active_step_routes.routers[router_key].calls
        calls_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
        for call_index, route in sorted(router_calls.items()):
            calls_by_key[_router_call_key(route)].append(call_index)
        call_sequence: list[int] = []
        for call_key in self._build_local_call_keys(sample_index=sample_index):
            matching_call_indices = calls_by_key.get(call_key)
            if not matching_call_indices:
                raise RuntimeError(
                    "Replay router call sequence is missing local micro metadata: "
                    f"step={self._active_step_index}, router='{router_key}', "
                    f"call_key={call_key}"
                )
            call_sequence.extend(matching_call_indices)
        return call_sequence

    def _build_local_call_keys(
        self,
        *,
        sample_index: int | list[int | None] | None,
    ) -> list[tuple[str, int]]:
        if not isinstance(sample_index, list):
            if sample_index is None:
                return [self._dummy_micro_call_key(local_micro_index=0)]
            return [("sample", int(sample_index))]
        return [
            self._sample_or_dummy_call_key(
                global_sample_index=global_sample_index,
                local_micro_index=local_micro_index,
            )
            for local_micro_index, global_sample_index in enumerate(sample_index)
        ]

    def _sample_or_dummy_call_key(
        self,
        *,
        global_sample_index: int | None,
        local_micro_index: int,
    ) -> tuple[str, int]:
        if global_sample_index is not None:
            return ("sample", int(global_sample_index))
        return self._dummy_micro_call_key(local_micro_index=local_micro_index)

    @staticmethod
    def _dummy_micro_call_key(*, local_micro_index: int) -> tuple[str, int]:
        from megatron.core import parallel_state as ps

        dp_rank = int(ps.get_data_parallel_rank())
        dp_world_size = int(ps.get_data_parallel_world_size())
        return ("dummy_micro_slot", local_micro_index * dp_world_size + dp_rank)

    def _active_router_call_key(self) -> tuple[str, int] | None:
        if self._active_micro_order is None:
            return None
        return self._sample_or_dummy_call_key(
            global_sample_index=self._active_sample_index,
            local_micro_index=self._active_micro_order,
        )

    def _active_micro_call_indices(self, router_key: str) -> list[int]:
        if self.bundle.tensor_backed:
            if self._active_step_index is None or self._active_micro_order is None:
                raise RuntimeError("Routing replay begin_micro called before set_step")
            return [self._active_micro_order]
        if self._active_step_routes is None:
            raise RuntimeError("Routing replay begin_micro called before set_step")
        router_calls = self._active_step_routes.routers[router_key].calls
        call_sequence = self._router_call_sequences[router_key]
        cursor = self._router_call_cursors.get(router_key, 0)
        active_call_key = self._active_router_call_key()
        consumed_call = self._router_consumed_calls.get(router_key, {}).get(
            active_call_key
        )
        if consumed_call is not None:
            return [consumed_call]
        if cursor >= len(call_sequence):
            last_index = self._router_last_call_indices.get(router_key)
            last_key = self._router_last_call_keys.get(router_key)
            if (
                active_call_key is not None
                and last_index is not None
                and last_key == active_call_key
            ):
                return [last_index]
            return []
        first_index = call_sequence[cursor]
        if active_call_key is None:
            return [first_index]
        next_key = _router_call_key(router_calls[first_index])
        last_index = self._router_last_call_indices.get(router_key)
        last_key = self._router_last_call_keys.get(router_key)
        if (
            last_index is not None
            and last_key == active_call_key
            and next_key != active_call_key
        ):
            return [last_index]
        indices: list[int] = []
        for call_index in call_sequence[cursor:]:
            if _router_call_key(router_calls[call_index]) != active_call_key:
                break
            indices.append(call_index)
        return indices

    def _next_route_call_index(self, router_key: str) -> int:
        if self.bundle.tensor_backed:
            if self._active_step_index is None or self._active_micro_order is None:
                raise RuntimeError(
                    "Routing replay router call occurred before set_step"
                )
            call_index = self._active_micro_order
            call_key = ("micro", call_index)
            consumed = self._router_consumed_calls[router_key]
            if call_key in consumed:
                if not self.allow_recompute_reuse:
                    raise RuntimeError(
                        "Routing replay recompute reuse is disabled: "
                        f"step={self._active_step_index}, router='{router_key}', "
                        f"micro={call_index}"
                    )
                self._router_reuse_counts[router_key] = (
                    self._router_reuse_counts.get(router_key, 0) + 1
                )
                return call_index
            if call_index not in self._router_call_sequences[router_key]:
                raise RuntimeError(
                    "Routing replay micro is outside the local call sequence: "
                    f"router='{router_key}', micro={call_index}"
                )
            consumed[call_key] = call_index
            self._router_call_cursors[router_key] += 1
            return call_index
        if self._active_step_routes is None:
            raise RuntimeError("Routing replay router call occurred before set_step")
        router_calls = self._active_step_routes.routers[router_key].calls
        call_sequence = self._router_call_sequences.get(router_key)
        if call_sequence is None:
            raise RuntimeError(
                "Routing replay call sequence missing for router key: "
                f"step={self._active_step_index}, router='{router_key}'"
            )
        cursor = self._router_call_cursors.get(router_key, 0)
        active_call_key = self._active_router_call_key()
        consumed_call = self._router_consumed_calls.get(router_key, {}).get(
            active_call_key
        )
        if consumed_call is not None:
            if not self.allow_recompute_reuse:
                raise RuntimeError(
                    "Routing replay recompute reuse is disabled: "
                    f"step={self._active_step_index}, router='{router_key}', "
                    f"call_key={active_call_key}"
                )
            self._router_reuse_counts[router_key] = (
                self._router_reuse_counts.get(router_key, 0) + 1
            )
            return consumed_call
        last_index = self._router_last_call_indices.get(router_key)
        last_key = self._router_last_call_keys.get(router_key)
        next_key = (
            _router_call_key(router_calls[call_sequence[cursor]])
            if cursor < len(call_sequence)
            else None
        )
        if (
            active_call_key is not None
            and last_index is not None
            and last_key == active_call_key
            and next_key != active_call_key
        ):
            if not self.allow_recompute_reuse:
                raise RuntimeError(
                    "Routing replay recompute reuse is disabled: "
                    f"step={self._active_step_index}, router='{router_key}', "
                    f"call_key={active_call_key}"
                )
            self._router_reuse_counts[router_key] = (
                self._router_reuse_counts.get(router_key, 0) + 1
            )
            return last_index
        if cursor >= len(call_sequence):
            raise RuntimeError(
                "Routing replay call cursor exceeded local call sequence: "
                f"step={self._active_step_index}, router='{router_key}', "
                f"cursor={cursor}, sequence_length={len(call_sequence)}"
            )
        call_index = call_sequence[cursor]
        self._router_call_cursors[router_key] = cursor + 1
        self._router_last_call_indices[router_key] = call_index
        call_key = _router_call_key(router_calls[call_index])
        self._router_last_call_keys[router_key] = call_key
        self._router_consumed_calls[router_key][call_key] = call_index
        return call_index

    def _prepare_native_target_for_router(
        self, router_key: str, *, logits: torch.Tensor
    ) -> None:
        if (
            self._active_step_index is None
            or self._active_micro_order is None
            or self._active_token_uid_key is None
        ):
            raise RuntimeError(
                "Routing replay router call occurred before staged targets were ready: "
                f"router='{router_key}'"
            )
        binding = self._router_bindings[router_key]
        if int(binding["chunk_index"]) != self._active_chunk_index:
            raise RuntimeError(
                "Routing replay router ran under the wrong VPP chunk: "
                f"router='{router_key}', owner={binding['chunk_index']}, "
                f"active={self._active_chunk_index}"
            )
        call_indices = self._active_micro_call_indices(router_key)
        if len(call_indices) != 1:
            raise RuntimeError(
                "Routing replay expected exactly one active router call while "
                f"preparing native replay for router='{router_key}', got {call_indices}"
            )
        call_index = self._next_route_call_index(router_key)
        if call_index != call_indices[0]:
            raise RuntimeError(
                "Routing replay cursor mismatch while preparing native replay: "
                f"router='{router_key}', expected={call_indices[0]}, "
                f"actual={call_index}"
            )
        target_key = (self._active_token_uid_key, call_index)
        staged_key = (self._active_token_uid_key, router_key, call_index)
        target = self._prepared_targets.get(staged_key)
        if target is None:
            raise RuntimeError(
                "Routing replay target was not staged before router execution: "
                f"step={self._active_step_index}, router='{router_key}', "
                f"call={call_index}, token_uid_key='{self._active_token_uid_key}'"
            )
        self._wait_for_staged_target(staged_key, target)
        if target.device.type == "cuda":
            target.record_stream(torch.cuda.current_stream(target.device))
        if self._router_prepared_target_keys.get(router_key) == target_key:
            return
        topk = int(binding["topk"])
        logit_experts = int(logits.shape[-1])
        model_num_experts = int(binding["num_experts"])
        if model_num_experts and model_num_experts != logit_experts:
            raise RuntimeError(
                "Routing replay router expert count differs from logits: "
                f"router='{router_key}', model_experts={model_num_experts}, "
                f"logit_experts={logit_experts}"
            )
        expected_tokens = int(logits.numel()) // logit_experts
        if int(target.shape[0]) != expected_tokens:
            raise RuntimeError(
                "Routing replay target token count differs from router logits: "
                f"router='{router_key}', target_tokens={int(target.shape[0])}, "
                f"logit_tokens={expected_tokens}"
            )
        if int(target.shape[1]) != topk:
            raise RuntimeError(
                "Routing replay target topk mismatch at router call: "
                f"router='{router_key}', call={call_index}, "
                f"target_topk={int(target.shape[1])}, router_topk={topk}"
            )
        router_replay = binding["router_replay"]
        router_replay.set_target_indices(target)
        router_replay.set_router_replay_action(
            _router_replay_classes()[1].REPLAY_FORWARD
        )
        self._router_prepared_target_keys[router_key] = target_key

    def _explicit_target_for_router_call(
        self,
        *,
        router_key: str,
        call_index: int,
        explicit_uids: torch.Tensor,
    ) -> torch.Tensor:
        if self.bundle.tensor_backed:
            assert self.bundle.expert_indices is not None
            num_experts = int(self.bundle.num_experts or 0)
            topk = self.bundle.max_topk
            layer_index = int(self._router_bindings[router_key]["layer_index"])
            sample_index = self._active_step_samples[call_index]
            source = (
                None
                if sample_index is None
                else self.bundle.expert_indices[layer_index, sample_index]
            )
            local_uids = explicit_uids.reshape(-1).contiguous()
            target_cpu = torch.empty(
                (int(local_uids.numel()), topk),
                dtype=(torch.uint8 if num_experts <= 256 else torch.uint16),
            )
            valid_positions = torch.nonzero(local_uids >= 0, as_tuple=False).reshape(-1)
            if int(valid_positions.numel()) > 0:
                valid_uids = local_uids[valid_positions]
                if source is None:
                    target_cpu[valid_positions] = _synthetic_replay_rows(
                        row_positions=valid_uids,
                        num_experts=num_experts,
                        topk=topk,
                        dtype=target_cpu.dtype,
                        seed=self._tensor_synthetic_seed(layer_index, call_index),
                    )
                else:
                    row_indices = self._row_indices_for_explicit_uids(
                        valid_uids=valid_uids,
                        router_key=router_key,
                        call_index=call_index,
                    )
                    target_cpu[valid_positions] = source.index_select(0, row_indices)
            invalid_positions = torch.nonzero(local_uids < 0, as_tuple=False).reshape(
                -1
            )
            if int(invalid_positions.numel()) > 0:
                target_cpu[invalid_positions] = _synthetic_replay_rows(
                    row_positions=invalid_positions,
                    num_experts=num_experts,
                    topk=topk,
                    dtype=target_cpu.dtype,
                    seed=self._tensor_synthetic_seed(layer_index, call_index),
                )
            return target_cpu.contiguous()
        if self._active_step_routes is None:
            raise RuntimeError("Routing replay explicit target used before set_step")
        route = self._active_step_routes.routers[router_key].calls[call_index]
        local_uids = explicit_uids.reshape(-1).contiguous()
        target_cpu = torch.empty(
            (int(local_uids.numel()), route.max_topk),
            dtype=torch.long,
        )
        valid_positions = torch.nonzero(local_uids >= 0, as_tuple=False).reshape(-1)
        if int(valid_positions.numel()) > 0:
            valid_uids = local_uids[valid_positions]
            row_indices = self._row_indices_for_explicit_uids(
                valid_uids=valid_uids,
                router_key=router_key,
                call_index=call_index,
            )
            target_cpu[valid_positions] = route.expert_indices.index_select(
                0,
                row_indices,
            ).to(dtype=torch.long)
        invalid_positions = torch.nonzero(local_uids < 0, as_tuple=False).reshape(-1)
        if int(invalid_positions.numel()) > 0:
            target_cpu[invalid_positions] = _synthetic_replay_rows(
                row_positions=invalid_positions,
                num_experts=route.num_experts,
                topk=route.max_topk,
                dtype=torch.long,
                seed=(int(self._active_step_index or 0) + 1) * 1_000_003
                + (call_index + 1) * 97_003,
            )
        return target_cpu.contiguous()

    def _tensor_synthetic_seed(self, layer_index: int, call_index: int) -> int:
        return (
            (int(self._active_step_index or 0) + 1) * 1_000_003
            + (layer_index + 1) * 97_003
            + (call_index + 1) * 9_176
        )

    def _row_indices_for_explicit_uids(
        self,
        *,
        valid_uids: torch.Tensor,
        router_key: str,
        call_index: int,
    ) -> torch.Tensor:
        if self._global_uid_dense_start is not None:
            row_indices = valid_uids.to(dtype=torch.long) - self._global_uid_dense_start
            out_of_range = (row_indices < 0) | (row_indices >= self._global_uid_count)
            if bool(out_of_range.any().item()):
                bad_uid = int(valid_uids[out_of_range][0].item())
                raise RuntimeError(
                    "Explicit routing replay token uid is outside the active dense "
                    f"step span: step={self._active_step_index}, "
                    f"router='{router_key}', call={call_index}, uid={bad_uid}"
                )
            return row_indices
        try:
            row_indices = [
                self._global_uid_to_row_index[int(uid)] for uid in valid_uids.tolist()
            ]
        except KeyError as exc:
            raise RuntimeError(
                "Explicit routing replay token uid is missing from the active "
                f"step map: step={self._active_step_index}, "
                f"router='{router_key}', call={call_index}, uid={exc.args[0]}"
            ) from exc
        return torch.tensor(row_indices, dtype=torch.long)

    @staticmethod
    def _token_uids_for_router_binding(
        token_uids: torch.Tensor,
        *,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        if not sequence_parallel:
            return token_uids
        from megatron.core import parallel_state as ps

        tp_size = int(ps.get_tensor_model_parallel_world_size())
        if tp_size <= 1:
            return token_uids
        tp_rank = int(ps.get_tensor_model_parallel_rank())
        token_count = int(token_uids.numel())
        local_count = (token_count + tp_size - 1) // tp_size
        start = tp_rank * local_count
        end = min(start + local_count, token_count)
        local_uids = token_uids.new_full((local_count,), -1)
        if start < token_count:
            real_uids = token_uids[start:end]
            local_uids[: int(real_uids.numel())] = real_uids
        return local_uids

    def _stage_prepared_target(
        self,
        *,
        target_key: tuple[str, str, int],
        target_cpu: torch.Tensor,
    ) -> None:
        target_cpu = target_cpu.contiguous()
        device = self._target_device()
        if device.type != "cuda":
            self._prepared_targets[target_key] = target_cpu.to(dtype=torch.long)
            return
        if self._target_copy_stream is None:
            self._target_copy_stream = torch.cuda.Stream(device=device)
        host_target = (
            target_cpu if target_cpu.is_pinned() else target_cpu.pin_memory()
        ).contiguous()
        self._host_target_staging.append(host_target)
        with torch.cuda.stream(self._target_copy_stream):
            narrow_buffer = torch.empty(
                tuple(host_target.shape),
                device=device,
                dtype=host_target.dtype,
            )
            narrow_buffer.copy_(host_target, non_blocking=True)
            buffer = narrow_buffer.to(dtype=torch.long)
            narrow_buffer.record_stream(self._target_copy_stream)
            buffer.record_stream(self._target_copy_stream)
        self._prepared_targets[target_key] = buffer

    def _record_target_copy_event(
        self,
        target_keys: list[tuple[str, str, int]],
    ) -> None:
        if self._target_copy_stream is None or not target_keys:
            return
        ready = torch.cuda.Event()
        with torch.cuda.stream(self._target_copy_stream):
            ready.record()
        for target_key in target_keys:
            self._step_target_ready_events[target_key] = ready

    def _wait_for_staged_target(
        self,
        target_key: tuple[str, str, int],
        target: torch.Tensor,
    ) -> None:
        ready = self._step_target_ready_events.get(target_key)
        if ready is None:
            return
        torch.cuda.current_stream(target.device).wait_event(ready)
