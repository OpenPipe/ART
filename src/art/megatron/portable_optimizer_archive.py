from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import json
import math
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from safetensors import safe_open
import torch

from art.megatron.tensor_snapshot import PinnedCpuSnapshotStager
from art.megatron.weights.rank_distributed_types import RankDistributedLoraStats
from art.utils.safetensors import (
    FileIdentity,
    PreparedSafetensors,
    prepare_safetensors,
    prepared_safetensors_identity,
    save_prepared_safetensors,
)

if TYPE_CHECKING:
    from art.megatron.weights.lora_publish import LocalLoraExportPlan
    from art.trainer_rank import (
        TrainerRankOptimizerLayout,
        TrainerRankOptimizerState,
    )

_COMPONENTS = ("master", "exp_avg", "exp_avg_sq")
_METADATA_KEY = "art.portable_optimizer_archive"


def portable_optimizer_semantic_contract() -> dict[str, object]:
    """Return the topology-neutral contract required for exact optimizer resume.

    Per-command hyperparameters are deliberately absent: they are checkpoint data
    in ``param_group`` and are restored exactly. This contract identifies the
    implementation and construction rules that give those values meaning.
    """
    return {
        "optimizer_implementation": (
            "transformer_engine.pytorch.optimizers.FusedAdam"
        ),
        "algorithm": "adamw",
        "bias_correction": True,
        "capturable": False,
        "trainer_rank_state_format": 1,
        "logical_archive_format": "art_logical_safetensors_v1",
        "logical_archive_metadata_format": 1,
        "parameter_state": ("master", "exp_avg", "exp_avg_sq", "step"),
        "parameter_groups": (
            "one_group_all_checkpoint_slot_parameters_in_residency_order_v1"
        ),
        "dynamic_param_group_values": ("lr", "betas", "eps", "weight_decay"),
    }

__all__ = (
    "LoadedPortableOptimizerArchive",
    "PortableOptimizerArchiveMetadata",
    "PortableOptimizerComponents",
    "PreparedPortableOptimizerArchive",
    "portable_optimizer_logical_keys_for_sites",
    "portable_optimizer_logical_tensors",
    "portable_optimizer_semantic_contract",
    "prepare_portable_optimizer_archive",
    "read_portable_optimizer_archive",
    "reconstruct_portable_optimizer_components",
    "reconstruct_trainer_rank_optimizer_state",
    "write_portable_optimizer_archive",
)


class PortableOptimizerArchiveMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    source_rank: int = Field(ge=0)
    source_world_size: int = Field(ge=1)
    logical_keys: tuple[str, ...]
    steps: dict[str, float]
    param_group: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_archive_metadata(self) -> "PortableOptimizerArchiveMetadata":
        if self.source_rank >= self.source_world_size:
            raise ValueError("portable optimizer archive rank leaves its source world")
        if self.logical_keys != tuple(sorted(set(self.logical_keys))):
            raise ValueError(
                "portable optimizer logical keys must be sorted and unique"
            )
        if any(not key for key in self.logical_keys):
            raise ValueError("portable optimizer logical keys must be nonempty")
        if set(self.steps) != set(self.logical_keys):
            raise ValueError("portable optimizer steps do not cover logical keys")
        if "params" in self.param_group:
            raise ValueError(
                "portable optimizer param-group metadata cannot contain params"
            )
        if any(not math.isfinite(step) or step < 0 for step in self.steps.values()):
            raise ValueError("portable optimizer steps must be finite and nonnegative")
        return self


class PreparedPortableOptimizerArchive(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    metadata: PortableOptimizerArchiveMetadata
    tensors: dict[str, torch.Tensor]
    exchange_stats: RankDistributedLoraStats

    @model_validator(mode="after")
    def _validate_prepared_tensors(self) -> "PreparedPortableOptimizerArchive":
        _validate_component_tensors(self.tensors, self.metadata.logical_keys)
        if self.exchange_stats.rank != self.metadata.source_rank:
            raise ValueError(
                "portable optimizer exchange rank differs from archive rank"
            )
        if self.exchange_stats.world_size != self.metadata.source_world_size:
            raise ValueError(
                "portable optimizer exchange world differs from archive world"
            )
        return self

    def identity(self, *, logical_keys: Collection[str] | None = None) -> FileIdentity:
        return prepared_safetensors_identity(_archive_payload(self, logical_keys))


class LoadedPortableOptimizerArchive(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    metadata: PortableOptimizerArchiveMetadata
    loaded_logical_keys: tuple[str, ...]
    tensors: dict[str, torch.Tensor]

    @model_validator(mode="after")
    def _validate_loaded_tensors(self) -> "LoadedPortableOptimizerArchive":
        if self.loaded_logical_keys != tuple(sorted(set(self.loaded_logical_keys))):
            raise ValueError("loaded portable optimizer keys must be sorted and unique")
        if not set(self.loaded_logical_keys).issubset(self.metadata.logical_keys):
            raise ValueError("loaded portable optimizer keys are absent from metadata")
        _validate_component_tensors(self.tensors, self.loaded_logical_keys)
        return self


class PortableOptimizerComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    master: dict[str, torch.Tensor]
    exp_avg: dict[str, torch.Tensor]
    exp_avg_sq: dict[str, torch.Tensor]
    steps: dict[str, float]
    param_group: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_components(self) -> "PortableOptimizerComponents":
        keys = set(self.master)
        if set(self.exp_avg) != keys or set(self.exp_avg_sq) != keys:
            raise ValueError("portable optimizer component mappings differ")
        if set(self.steps) != keys:
            raise ValueError("portable optimizer steps differ from component mappings")
        _validate_component_tensors(
            {
                **{f"master/{key}": value for key, value in self.master.items()},
                **{f"exp_avg/{key}": value for key, value in self.exp_avg.items()},
                **{
                    f"exp_avg_sq/{key}": value for key, value in self.exp_avg_sq.items()
                },
            },
            tuple(sorted(keys)),
        )
        if "params" in self.param_group:
            raise ValueError("portable optimizer param-group cannot contain params")
        if any(not math.isfinite(step) or step < 0 for step in self.steps.values()):
            raise ValueError("portable optimizer steps must be finite and nonnegative")
        return self


class _RankOptimizerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: dict[str, float]
    param_group: dict[str, JsonValue]
    error: str | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> "_RankOptimizerMetadata":
        if self.error is not None and (self.steps or self.param_group):
            raise ValueError("failed optimizer metadata cannot carry archive state")
        return self


class _IdentityHandler:
    key = "portable_optimizer_archive"

    @staticmethod
    def to_vllm_lora_tensors(
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return tensors, adapter_config

    @staticmethod
    def to_vllm_lora_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
        return adapter_config


def portable_optimizer_logical_tensors(
    prepared: PreparedPortableOptimizerArchive,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return immutable logical master geometry for a prepared archive shard."""
    return tuple(
        (
            key,
            tuple(prepared.tensors[f"master/{key}"].shape),
        )
        for key in prepared.metadata.logical_keys
    )


def prepare_portable_optimizer_archive(
    state: "TrainerRankOptimizerState",
    export_plan: LocalLoraExportPlan,
    *,
    group: Any | None = None,
) -> PreparedPortableOptimizerArchive:
    """Merge rank-local optimizer shards into topology-neutral logical tensors.

    A multi-rank call requires a Gloo group. Every logical key's master and both
    moments are assigned to one source rank in a single ownership/exchange pass.
    """
    from art.megatron.weights.rank_distributed_lora_publish import (
        prepare_rank_distributed_vllm_lora_source,
    )

    rank, world_size, global_rank = _rank_world(group)
    if torch.distributed.is_initialized() and (  # type: ignore[possibly-missing-attribute]
        rank != global_rank
        or world_size != int(torch.distributed.get_world_size())  # type: ignore[possibly-missing-attribute]
    ):
        raise RuntimeError("portable optimizer archives require the all-rank world")
    if world_size > 1 and str(torch.distributed.get_backend(group)) != "gloo":  # type: ignore[possibly-missing-attribute]
        raise RuntimeError("portable optimizer archive exchange requires Gloo")
    components = None
    try:
        if any(entry.kind != "regular" for entry in export_plan.entries):
            raise ValueError(
                "portable optimizer archives require an export plan without "
                "packed serving groups"
            )
        components = _optimizer_components(state, export_plan)
        masters, exp_avgs, exp_avg_sqs, steps, param_group = components
        rank_metadata = _RankOptimizerMetadata(
            steps={
                entry.key: steps[entry.source_index] for entry in export_plan.entries
            },
            param_group=param_group,
        )
    except BaseException as error:
        rank_metadata = _RankOptimizerMetadata(
            steps={},
            param_group={},
            error=f"{type(error).__name__}: {error}",
        )
    gathered_metadata = _gather_rank_metadata(rank_metadata, world_size, group)
    failures = [
        f"rank {owner}: {metadata.error}"
        for owner, metadata in enumerate(gathered_metadata)
        if metadata.error is not None
    ]
    if failures:
        raise RuntimeError(
            "portable optimizer archive validation failed: " + "; ".join(failures)
        )
    assert components is not None
    masters, exp_avgs, exp_avg_sqs, _steps, _param_group = components
    global_steps = _merge_steps(gathered_metadata)
    if any(
        metadata.param_group != rank_metadata.param_group
        for metadata in gathered_metadata
    ):
        raise RuntimeError("portable optimizer param-group config differs across ranks")

    local_tensors: dict[str, torch.Tensor] = {}
    local_metadata = []
    materialize_error: BaseException | None = None
    try:
        for prefix, sources in (
            ("master/", masters),
            ("exp_avg/", exp_avgs),
            ("exp_avg_sq/", exp_avg_sqs),
        ):
            tensors, metadata, packed, packed_metadata = export_plan.materialize(
                sources,
                owner_rank=global_rank,
                key_prefix=prefix,
                dtype_override=torch.float32,
            )
            if packed or packed_metadata:
                raise RuntimeError(
                    "portable optimizer export unexpectedly materialized packed tensors"
                )
            overlap = set(local_tensors).intersection(tensors)
            if overlap:
                raise RuntimeError(
                    "portable optimizer export produced duplicate tensors: "
                    f"{sorted(overlap)}"
                )
            local_tensors.update(tensors)
            local_metadata.extend(metadata)
    except BaseException as error:
        materialize_error = error
        local_tensors = {}
        local_metadata = []

    source = prepare_rank_distributed_vllm_lora_source(
        local_tensors=local_tensors,
        local_metadata=local_metadata,
        local_packed_tensors={},
        local_packed_metadata=(),
        handler=_IdentityHandler(),
        adapter_config={"optimizer_param_group": rank_metadata.param_group},
        conversion_group_for_key=_logical_key,
        group=group,
        metadata_group=group,
        coordinator_rank=0,
        exchange_device=torch.device("cpu"),
        stager=PinnedCpuSnapshotStager(),
        local_error=materialize_error,
    ).resolve()
    owned_keys = _logical_keys_for_tensors(source.tensors)
    return PreparedPortableOptimizerArchive(
        metadata=PortableOptimizerArchiveMetadata(
            source_rank=rank,
            source_world_size=world_size,
            logical_keys=owned_keys,
            steps={key: global_steps[key] for key in owned_keys},
            param_group=rank_metadata.param_group,
        ),
        tensors=source.tensors,
        exchange_stats=source.stats,
    )


def write_portable_optimizer_archive(
    prepared: PreparedPortableOptimizerArchive,
    path: str | Path,
    *,
    logical_keys: Collection[str] | None = None,
    identity: FileIdentity | None = None,
) -> FileIdentity:
    """Write all owned keys, or an exact selected subset, without tensor copies."""
    return save_prepared_safetensors(
        _archive_payload(prepared, logical_keys), Path(path), identity=identity
    )


def read_portable_optimizer_archive(
    path: str | Path,
    *,
    logical_keys: Collection[str] | None = None,
) -> LoadedPortableOptimizerArchive:
    """Read only the requested logical tensor triples from one source-rank archive."""
    with safe_open(Path(path), framework="pt", device="cpu") as archive:
        header_metadata = archive.metadata()
        if header_metadata is None or _METADATA_KEY not in header_metadata:
            raise RuntimeError("portable optimizer archive metadata is missing")
        metadata = PortableOptimizerArchiveMetadata.model_validate_json(
            header_metadata[_METADATA_KEY]
        )
        selected = _selected_keys(metadata.logical_keys, logical_keys)
        expected_names = _component_names(selected)
        actual_names = set(archive.keys())
        complete_names = set(_component_names(metadata.logical_keys))
        if actual_names != complete_names:
            raise RuntimeError("portable optimizer archive tensor coverage is invalid")
        tensors = {name: archive.get_tensor(name) for name in expected_names}
    return LoadedPortableOptimizerArchive(
        metadata=metadata,
        loaded_logical_keys=selected,
        tensors=tensors,
    )


def reconstruct_portable_optimizer_components(
    archives: Sequence[LoadedPortableOptimizerArchive],
) -> PortableOptimizerComponents:
    """Combine selectively loaded source archives into logical component mappings."""
    if not archives:
        raise ValueError("portable optimizer reconstruction requires an archive")
    param_group = archives[0].metadata.param_group
    master: dict[str, torch.Tensor] = {}
    exp_avg: dict[str, torch.Tensor] = {}
    exp_avg_sq: dict[str, torch.Tensor] = {}
    steps: dict[str, float] = {}
    components = {
        "master": master,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
    }
    for archive in archives:
        if archive.metadata.param_group != param_group:
            raise RuntimeError(
                "portable optimizer archives have different param groups"
            )
        for key in archive.loaded_logical_keys:
            if key in steps:
                raise RuntimeError(f"duplicate portable optimizer logical key: {key}")
            steps[key] = archive.metadata.steps[key]
            for component, mapping in components.items():
                mapping[key] = archive.tensors[f"{component}/{key}"]
    return PortableOptimizerComponents(
        master=master,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        steps=steps,
        param_group=param_group,
    )


def portable_optimizer_logical_keys_for_sites(
    sites: Sequence[tuple[Any, Any]],
) -> tuple[str, ...]:
    """Return the logical keys a prepared destination checkpoint rank requires."""
    return tuple(
        sorted(
            {
                str(key)
                for module, slot in sites
                for suffix, parameter in (
                    ("lora_A", slot.A_T),
                    ("lora_B", slot.B_T),
                )
                for key in module._expected_weight_keys_for_param(suffix, parameter)
            }
        )
    )


def reconstruct_trainer_rank_optimizer_state(
    components: PortableOptimizerComponents,
    sites: Sequence[tuple[Any, Any]],
    destination_layout: "TrainerRankOptimizerLayout",
) -> "TrainerRankOptimizerState":
    """Localize logical optimizer components into a destination v1 rank state."""
    parameters = tuple(
        (module, suffix, parameter)
        for module, slot in sites
        for suffix, parameter in (
            ("lora_A", slot.A_T),
            ("lora_B", slot.B_T),
        )
    )
    layout_parameters = destination_layout.get("parameters")
    if not isinstance(layout_parameters, Sequence) or len(layout_parameters) != len(
        parameters
    ):
        raise ValueError("destination optimizer layout differs from prepared sites")

    masters: list[torch.Tensor] = []
    optimizer_state: dict[int, dict[str, object]] = {}
    component_maps = {
        "master": components.master,
        "exp_avg": components.exp_avg,
        "exp_avg_sq": components.exp_avg_sq,
    }
    for index, (module, suffix, parameter) in enumerate(parameters):
        keys = tuple(
            str(key)
            for key in module._expected_weight_keys_for_param(suffix, parameter)
        )
        missing = set(keys).difference(components.master)
        if missing:
            raise KeyError(
                f"destination optimizer keys were not loaded: {sorted(missing)}"
            )
        parameter_steps = {components.steps[key] for key in keys} if keys else {0.0}
        if len(parameter_steps) != 1:
            raise ValueError(
                "logical optimizer steps differ within one destination parameter: "
                f"{sorted((key, components.steps[key]) for key in keys)}"
            )
        parameterization = getattr(
            parameter,
            "lora_moe_parameterization",
            getattr(module, "moe_parameterization", None),
        )
        localized: dict[str, torch.Tensor] = {}
        for component, mapping in component_maps.items():
            if keys:
                weight = module._adapter_weight(
                    {key: mapping[key] for key in keys},
                    suffix=suffix,
                    moe_parameterization=parameterization,
                )
                value = module._localized_weight(weight, into=parameter)
            else:
                value = torch.zeros(
                    tuple(parameter.shape), device="cpu", dtype=torch.float32
                )
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    "localized optimizer component differs from destination "
                    f"parameter {component}: {tuple(value.shape)} != "
                    f"{tuple(parameter.shape)}"
                )
            localized[component] = value.float().cpu().contiguous()
        masters.append(localized["master"])
        optimizer_state[index] = {
            "exp_avg": localized["exp_avg"],
            "exp_avg_sq": localized["exp_avg_sq"],
            "step": torch.tensor(next(iter(parameter_steps)), dtype=torch.float32),
        }

    return cast(
        "TrainerRankOptimizerState",
        {
            "format_version": 1,
            "layout": destination_layout,
            "master_params": tuple(masters),
            "optimizer": {
                "state": optimizer_state,
                "param_groups": [
                    {
                        **components.param_group,
                        "params": list(range(len(masters))),
                    }
                ],
            },
        },
    )


def _optimizer_components(
    state: "TrainerRankOptimizerState",
    export_plan: LocalLoraExportPlan,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[float, ...],
    dict[str, JsonValue],
]:
    if state.get("format_version") != 1:
        raise ValueError("unsupported TrainerRank optimizer state format")
    masters_value = state.get("master_params")
    optimizer_value = state.get("optimizer")
    if not isinstance(masters_value, Sequence) or not isinstance(
        optimizer_value, Mapping
    ):
        raise ValueError("TrainerRank optimizer state is incomplete")
    masters = tuple(masters_value)
    if len(masters) != export_plan.source_count:
        raise ValueError("optimizer masters differ from the LoRA export plan")
    if any(
        not isinstance(master, torch.Tensor) or master.device.type != "cpu"
        for master in masters
    ):
        raise ValueError("portable optimizer masters must be CPU tensors")
    if any(entry.source_index >= len(masters) for entry in export_plan.entries):
        raise ValueError("LoRA export plan references a missing optimizer master")

    groups = optimizer_value.get("param_groups")
    states = optimizer_value.get("state")
    if (
        not isinstance(groups, Sequence)
        or len(groups) != 1
        or not isinstance(groups[0], Mapping)
        or not isinstance(states, Mapping)
    ):
        raise ValueError("portable optimizer archives require one parameter group")
    raw_group = dict(groups[0])
    indices = raw_group.pop("params", None)
    if not isinstance(indices, Sequence) or list(indices) != list(range(len(masters))):
        raise ValueError("optimizer parameter indices differ from master order")
    if any(not isinstance(key, str) for key in raw_group):
        raise ValueError("optimizer param-group keys must be strings")
    param_group = {str(key): _as_json_value(value) for key, value in raw_group.items()}

    if any(
        type(index) is not int or index < 0 or index >= len(masters) for index in states
    ):
        raise ValueError("optimizer state references an invalid master index")
    exp_avgs: list[torch.Tensor] = []
    exp_avg_sqs: list[torch.Tensor] = []
    steps: list[float] = []
    for index, master in enumerate(masters):
        values = states.get(index, {})
        if not isinstance(values, Mapping):
            raise ValueError("optimizer parameter state must be a mapping")
        unexpected = set(values).difference({"exp_avg", "exp_avg_sq", "step"})
        if unexpected:
            raise ValueError(
                f"unsupported optimizer parameter state: {sorted(map(str, unexpected))}"
            )
        has_avg = "exp_avg" in values
        has_avg_sq = "exp_avg_sq" in values
        if has_avg != has_avg_sq:
            raise ValueError("optimizer moments must be present together")
        avg = values.get("exp_avg") if has_avg else torch.zeros_like(master)
        avg_sq = values.get("exp_avg_sq") if has_avg_sq else torch.zeros_like(master)
        if not isinstance(avg, torch.Tensor) or not isinstance(avg_sq, torch.Tensor):
            raise ValueError("optimizer moments must be tensors")
        if (
            avg.device.type != "cpu"
            or avg_sq.device.type != "cpu"
            or tuple(avg.shape) != tuple(master.shape)
            or tuple(avg_sq.shape) != tuple(master.shape)
        ):
            raise ValueError("optimizer moments differ from their master tensor")
        exp_avgs.append(avg)
        exp_avg_sqs.append(avg_sq)
        steps.append(_optimizer_step(values.get("step", 0.0)))
    return masters, tuple(exp_avgs), tuple(exp_avg_sqs), tuple(steps), param_group


def _optimizer_step(value: object) -> float:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.numel() != 1:
            raise ValueError("optimizer step tensor must be one CPU scalar")
        value = value.item()
    if type(value) not in {int, float}:
        raise ValueError("optimizer step must be numeric")
    step = float(cast(int | float, value))
    if not math.isfinite(step) or step < 0:
        raise ValueError("optimizer step must be finite and nonnegative")
    return step


def _as_json_value(value: object) -> JsonValue:
    if value is None or type(value) in {bool, int, str}:
        return cast(JsonValue, value)
    if type(value) is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError("optimizer param-group floats must be finite")
        return cast(JsonValue, value)
    if isinstance(value, (list, tuple)):
        return cast(JsonValue, [_as_json_value(item) for item in value])
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("optimizer param-group mapping keys must be strings")
        return cast(
            JsonValue, {key: _as_json_value(item) for key, item in value.items()}
        )
    raise ValueError(f"unsupported optimizer param-group value: {type(value)!r}")


def _rank_world(group: Any | None) -> tuple[int, int, int]:
    if not torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        return 0, 1, 0
    return (
        int(torch.distributed.get_rank(group)),  # type: ignore[possibly-missing-attribute]
        int(torch.distributed.get_world_size(group)),  # type: ignore[possibly-missing-attribute]
        int(torch.distributed.get_rank()),  # type: ignore[possibly-missing-attribute]
    )


def _gather_rank_metadata(
    metadata: _RankOptimizerMetadata,
    world_size: int,
    group: Any | None,
) -> tuple[_RankOptimizerMetadata, ...]:
    gathered: list[Any] = [None] * world_size
    payload = metadata.model_dump_json()
    if world_size == 1:
        gathered[0] = payload
    else:
        torch.distributed.all_gather_object(gathered, payload, group=group)  # type: ignore[possibly-missing-attribute]
    return tuple(_RankOptimizerMetadata.model_validate_json(value) for value in gathered)


def _merge_steps(metadata: Sequence[_RankOptimizerMetadata]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for rank_metadata in metadata:
        for key, step in rank_metadata.steps.items():
            previous = merged.setdefault(key, step)
            if previous != step:
                raise RuntimeError(
                    f"optimizer step differs across shards for logical key {key!r}"
                )
    return merged


def _logical_key(tensor_name: str) -> str:
    component, separator, key = tensor_name.partition("/")
    if not separator or component not in _COMPONENTS or not key:
        raise ValueError(f"invalid portable optimizer tensor name: {tensor_name!r}")
    return key


def _logical_keys_for_tensors(tensors: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    by_key: dict[str, set[str]] = {}
    for name in tensors:
        component, _, key = name.partition("/")
        _logical_key(name)
        by_key.setdefault(key, set()).add(component)
    expected = set(_COMPONENTS)
    incomplete = sorted(key for key, values in by_key.items() if values != expected)
    if incomplete:
        raise RuntimeError(
            f"portable optimizer ownership split component triples: {incomplete}"
        )
    return tuple(sorted(by_key))


def _component_names(logical_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"{component}/{key}" for component in _COMPONENTS for key in logical_keys
    )


def _validate_component_tensors(
    tensors: Mapping[str, torch.Tensor], logical_keys: Sequence[str]
) -> None:
    expected = set(_component_names(logical_keys))
    if set(tensors) != expected:
        raise ValueError("portable optimizer tensor coverage differs from logical keys")
    for name, tensor in tensors.items():
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"portable optimizer tensor must be contiguous CPU float32: {name}"
            )


def _selected_keys(
    available: Sequence[str], requested: Collection[str] | None
) -> tuple[str, ...]:
    if requested is None:
        return tuple(available)
    selected = tuple(sorted(set(requested)))
    missing = set(selected).difference(available)
    if missing:
        raise KeyError(f"portable optimizer keys are not owned here: {sorted(missing)}")
    return selected


def _archive_payload(
    prepared: PreparedPortableOptimizerArchive,
    logical_keys: Collection[str] | None,
) -> PreparedSafetensors:
    selected = _selected_keys(prepared.metadata.logical_keys, logical_keys)
    names = _component_names(selected)
    metadata = prepared.metadata.model_copy(
        update={
            "logical_keys": selected,
            "steps": {key: prepared.metadata.steps[key] for key in selected},
        }
    )
    payload = prepare_safetensors({name: prepared.tensors[name] for name in names})
    prefix = memoryview(payload.chunks[0].numpy()).cast("B")
    header_size = struct.unpack("<Q", prefix[:8])[0]
    header = json.loads(bytes(prefix[8 : 8 + header_size]))
    if "__metadata__" in header:
        raise RuntimeError("prepared safetensors unexpectedly contains metadata")
    header["__metadata__"] = {_METADATA_KEY: metadata.model_dump_json()}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    metadata_prefix = torch.frombuffer(
        bytearray(struct.pack("<Q", len(encoded)) + encoded), dtype=torch.uint8
    )
    return PreparedSafetensors((metadata_prefix, *payload.chunks[1:]))
