"""Versioned, bounded runtime dimensions for the selected plan's CPU estimator.

Eligibility is observed once, alongside the selected plan. Values are metadata,
not model tensors, callable reconstructions, or recorded memory-cost answers.
HybridEP growth and custom estimator overrides deliberately remain unsupported.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
import json
from types import MethodType, SimpleNamespace
from typing import Any, NamedTuple

from . import _gdn_memory, _impl, _memory

_MAX_BYTES = 262144
_MAX_GROUPS = 1024
_MAX_SEGMENTS = 4096
_MAX_INPUT_VALUES = 1_000_000


# Estimators bound on TrainerRank as plain functions, not methods.
_STATIC_ESTIMATORS = frozenset(
    {"_split_required_memory", "_gradient_slots", "_adapter_gradient_walk"}
)


_REFUSALS = frozenset(
    {
        "runtime_group_inventory_over_limit",
        "hybridep_runtime_facts_unsupported",
        "custom_runtime_estimator_unsupported",
        "runtime_token_inventory_over_limit",
        "runtime_segment_inventory_over_limit",
        "head_positions_unavailable",
        "runtime_facts_over_limit",
        "runtime_stage_inventory_over_limit",
        "runtime_request_inventory_over_limit",
        "runtime_shape_inventory_over_limit",
        "runtime_slot_identity_unsupported",
        "runtime_dimension_unsupported",
        "selected_request_input_mismatch",
        "runtime_request_inputs_unavailable",
    }
)


def refusal_reason(error: Exception) -> str:
    # Never retain arbitrary reader exception text, model reprs or token values.
    if (
        type(error) is ValueError
        and len(error.args) == 1
        and type(error.args[0]) is str
        and error.args[0] in _REFUSALS
    ):
        return error.args[0]
    return type(error).__name__[:64]


def _fact_budget() -> Callable[[int], None]:
    # Conservative JSON bounds, shared by capture (before copying containers)
    # and validation (before walking inventories or constructing an encoding).
    remaining = _MAX_BYTES - 256

    def reserve(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("runtime_facts_over_limit")

    return reserve


def capture(rank: Any, plan: Any) -> dict[str, Any]:
    if rank._num_layers > 1024 or not 0 < len(plan.groups) <= _MAX_GROUPS:
        raise ValueError("runtime_group_inventory_over_limit")
    if (
        getattr(rank.runtime.provider, "expert_model_parallel_size", 1) > 1
        or rank._parallel_shape.ep > 1
    ):
        raise ValueError("hybridep_runtime_facts_unsupported")
    for name in (
        "_subforward_cost",
        "_split_required_memory",
        "_estimate_required_memory_bytes_from_values",
        "_retained_memory_bytes",
        "_checkpoint_memory_floor",
        "_moe_workspace_bytes",
        "_checkpoint_moe_bytes_per_token",
        "_head_workspace_bytes",
        "_group_head_workspace_bytes",
        "_head_projection_rows",
        "_head_target_chunk_rows",
        "_plan_cost",
        "_plan_head_workspace_bytes",
        "_plan_hybridep_growth_bytes",
        "_gdn_segment_layer_bytes",
        "_one_layer_recompute",
        "_topology_key",
        "_physical_tokens",
        "_plan_group_rows",
        "_plan_retained_tokens",
        "_gradient_slots",
        "_pending_adapter_gradient_bytes",
        "_checkpoint_gradient_groups",
        "_checkpoint_adapter_gradient_bytes",
        "_adapter_gradient_walk",
    ):
        method = getattr(rank, name)
        expected = getattr(_impl.TrainerRank, name)
        supported = (
            method is expected
            if name in _STATIC_ESTIMATORS
            else type(method) is MethodType
            and method.__self__ is rank
            and method.__func__ is expected
        )
        if not supported:
            raise ValueError("custom_runtime_estimator_unsupported")
    if sum(int(g.packed.tokens.numel()) for g in plan.groups) > _MAX_INPUT_VALUES:
        raise ValueError("runtime_token_inventory_over_limit")
    for name in ("_moe_forward_stages", "_moe_gradient_stages"):
        stages = getattr(rank, name, ())
        if type(stages) is not tuple or len(stages) > 4096:
            raise ValueError("runtime_stage_inventory_over_limit")
    if len(rank.runtime.model) > 64:
        raise ValueError("runtime_shape_inventory_over_limit")
    for chunk in rank.runtime.model:
        try:
            decoder = _impl._language_model(chunk).decoder
        except (AttributeError, RuntimeError):
            continue  # The same live eligibility readers decline this family.
        if len(getattr(decoder, "layers", ())) > 1024:
            raise ValueError("runtime_shape_inventory_over_limit")
    vocabulary = _memory._head_vocabulary(rank)
    target_backward = bool(vocabulary and _memory._head_target_backward(rank))
    groups = []
    remaining = _MAX_SEGMENTS
    reserve = _fact_budget()
    head_values = _MAX_INPUT_VALUES

    def terms(checkpoint_grad: bool, ref: Any) -> list[Any]:
        coefficient, stages = _memory._moe_workspace_terms(
            rank, checkpoint_grad=checkpoint_grad, slot_ref=ref
        )
        if len(stages) > 4096:
            raise ValueError("runtime_stage_inventory_over_limit")
        reserve(64 + 72 * len(stages))
        if (
            type(coefficient) is not int
            or not 0 <= coefficient < 2**63
            or any(v >= 2**63 for stage in stages for v in stage)
        ):
            raise ValueError("runtime_dimension_unsupported")
        return [coefficient, [list(stage) for stage in stages]]

    has_grad = any(g.grad_enabled for g in plan.groups)
    for group, (physical_rows, _) in zip(
        plan.groups, rank._plan_group_rows(plan), strict=True
    ):
        segments = group.packed.segments
        remaining -= len(segments)
        if remaining < 0:
            raise ValueError("runtime_segment_inventory_over_limit")
        if len(group.items) > 4096 or len(group.request_indices) > 4096:
            raise ValueError("runtime_request_inventory_over_limit")
        name = getattr(group.slot_ref, "name", None)
        if name is not None and (type(name) is not str or len(name) > 4096):
            raise ValueError("runtime_slot_identity_unsupported")
        slot = str(group.slot_ref)
        fingerprint = None if group.layout is None else group.layout.fingerprint
        reserve(
            512
            + 12 * (len(slot) + len(fingerprint or ""))
            + 32 * len(group.request_indices)
        )
        requests = tuple(item.request for item in group.items)
        positions = group.packed.positions_by_sequence
        head_values -= sum(p.numel() for p in positions) + sum(
            r.target_tokens.numel() for r in requests if r.target_tokens is not None
        )
        if head_values < 0:
            raise ValueError("runtime_token_inventory_over_limit")
        if any(
            r.target_tokens is not None and r.target_tokens.ndim > 8 for r in requests
        ):
            raise ValueError("runtime_token_inventory_over_limit")
        # The live head estimator reads requests, but replay retains the inputs
        # selected by the planner. Refuse replacements that normalize differently
        # before reading head rows; never copy device tensors for diagnostics.
        for item in group.items:
            for selected, current in (
                (item.input_ids, item.request.input_tokens),
                (item.labels, item.request.target_tokens),
            ):
                if selected is None or current is None:
                    if selected is not current:
                        raise ValueError("selected_request_input_mismatch")
                    continue
                if selected.device.type != "cpu" or current.device.type != "cpu":
                    raise ValueError("runtime_request_inputs_unavailable")
                if selected.numel() != current.numel() or not _impl.torch.equal(
                    selected.reshape(-1),
                    current.detach().reshape(-1).to(dtype=_impl.torch.long),
                ):
                    raise ValueError("selected_request_input_mismatch")
        if vocabulary and any(p.device.type != "cpu" for p in positions):
            raise ValueError("head_positions_unavailable")
        projected = (
            rank._head_projection_rows(requests, positions=positions)
            if vocabulary
            else 0
        )
        backwards = (
            target_backward
            and group.grad_enabled
            and any(r.target_tokens is not None for r in requests)
        )
        target_rows = (
            rank._head_target_chunk_rows(requests, positions=positions)
            if backwards and any(r.logits or r.top_k is not None for r in requests)
            else projected
            if backwards
            else 0
        )
        adapter = None
        if group.grad_enabled and getattr(group.slot_ref, "name", None) is not None:
            # The live estimator reads this slot's unallocated gradient bytes
            # per decoder layer; freeze them with the selection.
            kind = getattr(group.slot_ref, "kind", None)
            if kind is not None and (type(kind) is not str or len(kind) > 64):
                raise ValueError("runtime_slot_identity_unsupported")
            pending = rank._pending_adapter_gradient_bytes((group.slot_ref,))
            if len(pending) > 1025:
                raise ValueError("runtime_shape_inventory_over_limit")
            reserve(128 + 12 * len(name) + 24 * len(pending))
            adapter = {"kind": kind, "name": name, "pending": [int(v) for v in pending]}
        model = _gdn_memory.model_shapes(rank, group.slot_ref) if has_grad else None
        if model is not None:
            if len(model[1]) > 1024:
                raise ValueError("runtime_shape_inventory_over_limit")
            reserve(128 + 512 * len(model[1]) + 256 * len(segments))
        groups.append(
            {
                "rows": physical_rows,
                "packed_rows": int(group.packed.tokens.numel()),
                "grad": group.grad_enabled,
                "slot": slot,
                "request_indices": list(group.request_indices),
                "layout_fingerprint": fingerprint,
                "forward": terms(False, group.slot_ref),
                "gradient": terms(True, group.slot_ref),
                "head_rows": projected,
                "head_target_rows": target_rows,
                "adapter": adapter,
                "gdn": None
                if model is None
                else {
                    "layers": model[0],
                    "shapes": [asdict(shape) for shape in model[1]],
                    "segments": [
                        {
                            name: getattr(s, name)
                            for name in (
                                "start",
                                "end",
                                "packed_start",
                                "group_id",
                                "parent_id",
                            )
                        }
                        for s in segments
                    ],
                },
            }
        )
    facts = {
        "version": 2,
        "checkpoint_layers": _memory._checkpoint_layers(
            rank, rank._plan_group_rows(plan)
        ),
        "checkpoint_moe_bytes_per_token": rank._checkpoint_moe_bytes_per_token(),
        "head_vocabulary": vocabulary,
        "head_target_backward": target_backward,
        "groups": groups,
    }
    # All containers above are owned, including copied stage tuples. Validate
    # primitives before bounded JSON encoding; never retain model/slot objects.
    validate(facts)
    return facts


def validate(facts: Any) -> None:
    """Accept only a finite primitive schema, never an executable reconstruction."""

    def fields(value: Any, names: set[str]) -> None:
        if type(value) is not dict or len(value) != len(names) or set(value) != names:
            raise ValueError("invalid runtime facts fields")

    def integer(value: Any, *, minimum: int = 0) -> None:
        if type(value) is not int or not minimum <= value < 2**63:
            raise ValueError("invalid runtime dimension")

    fields(
        facts,
        {
            "version",
            "checkpoint_layers",
            "checkpoint_moe_bytes_per_token",
            "head_vocabulary",
            "head_target_backward",
            "groups",
        },
    )
    if type(facts["version"]) is not int or facts["version"] != 2:
        raise ValueError("unsupported runtime facts version")
    for key in (
        "checkpoint_layers",
        "checkpoint_moe_bytes_per_token",
        "head_vocabulary",
    ):
        integer(facts[key])
    if type(facts["head_target_backward"]) is not bool:
        raise ValueError("invalid head backward eligibility")
    groups = facts["groups"]
    if type(groups) is not list or not 0 < len(groups) <= _MAX_GROUPS:
        raise ValueError("invalid runtime groups")
    remaining = _MAX_SEGMENTS
    reserve = _fact_budget()
    for group in groups:
        fields(
            group,
            {
                "rows",
                "packed_rows",
                "grad",
                "slot",
                "request_indices",
                "layout_fingerprint",
                "forward",
                "gradient",
                "head_rows",
                "head_target_rows",
                "adapter",
                "gdn",
            },
        )
        for key in ("rows", "packed_rows", "head_rows", "head_target_rows"):
            integer(group[key])
        if (
            type(group["grad"]) is not bool
            or type(group["slot"]) is not str
            or len(group["slot"]) > 4096
        ):
            raise ValueError("invalid runtime group identity")
        if (
            type(group["request_indices"]) is not list
            or len(group["request_indices"]) > 4096
        ):
            raise ValueError("invalid runtime request inventory")
        fingerprint = group["layout_fingerprint"]
        if fingerprint is not None and (
            type(fingerprint) is not str or len(fingerprint) > 128
        ):
            raise ValueError("invalid runtime layout identity")
        reserve(
            512
            + 12 * (len(group["slot"]) + len(fingerprint or ""))
            + 32 * len(group["request_indices"])
        )
        for index in group["request_indices"]:
            integer(index)
        for key in ("forward", "gradient"):
            terms = group[key]
            if (
                type(terms) is not list
                or len(terms) != 2
                or type(terms[1]) is not list
                or len(terms[1]) > 4096
            ):
                raise ValueError("invalid MoE terms")
            reserve(64 + 72 * len(terms[1]))
            integer(terms[0])
            for stage in terms[1]:
                if type(stage) is not list or len(stage) != 2:
                    raise ValueError("invalid MoE stage")
                for value in stage:
                    integer(value)
        adapter = group["adapter"]
        if adapter is not None:
            fields(adapter, {"kind", "name", "pending"})
            if (
                not group["grad"]
                or (adapter["kind"] is not None and type(adapter["kind"]) is not str)
                or len(adapter["kind"] or "") > 64
                or type(adapter["name"]) is not str
                or len(adapter["name"]) > 4096
                or type(adapter["pending"]) is not list
                or len(adapter["pending"]) > 1025
            ):
                raise ValueError("invalid adapter gradient facts")
            reserve(128 + 12 * len(adapter["name"]) + 24 * len(adapter["pending"]))
            for value in adapter["pending"]:
                integer(value)
        gdn = group["gdn"]
        if gdn is not None:
            fields(gdn, {"layers", "shapes", "segments"})
            integer(gdn["layers"], minimum=1)
            if type(gdn["shapes"]) is not list or not 0 < len(gdn["shapes"]) <= 1024:
                raise ValueError("invalid GDN shape inventory")
            if type(gdn["segments"]) is not list:
                raise ValueError("invalid GDN segments")
            remaining -= len(gdn["segments"])
            if remaining < 0:
                raise ValueError("runtime_segment_inventory_over_limit")
            reserve(128 + 512 * len(gdn["shapes"]) + 256 * len(gdn["segments"]))
            for shape in gdn["shapes"]:
                fields(shape, set(_gdn_memory.Shape.__dataclass_fields__))
                for name, value in shape.items():
                    integer(
                        value,
                        minimum=0
                        if name in {"output_lora_rank", "moe_bytes_per_row"}
                        else 1,
                    )
            for segment in gdn["segments"]:
                fields(
                    segment, {"start", "end", "packed_start", "group_id", "parent_id"}
                )
                for value in segment.values():
                    integer(value)
    if len(json.dumps(facts, separators=(",", ":"))) > _MAX_BYTES:
        raise ValueError("runtime_facts_over_limit")


def validate_tokens(inventories: Iterable[Any]) -> None:
    """Bound a whole subforward before layouts or tensors are constructed."""
    remaining = _MAX_INPUT_VALUES
    containers = 8 * _MAX_INPUT_VALUES + 8192

    def count(value: Any, depth: int = 0) -> None:
        nonlocal remaining, containers
        if depth > 8:
            raise ValueError("runtime_token_inventory_over_limit")
        if type(value) is list:
            containers -= 1
            if containers < 0 or len(value) > remaining + containers:
                raise ValueError("runtime_token_inventory_over_limit")
            for child in value:
                count(child, depth + 1)
        else:
            remaining -= 1
            if remaining < 0:
                raise ValueError("runtime_token_inventory_over_limit")
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError("invalid replay token primitive")

    for value in inventories:
        count(value)


class _ReplaySlot(NamedTuple):
    """A frozen adapter slot identity (the live LoRASlotRef's kind and name)."""

    kind: str | None
    name: str


class ReplayRank(_impl.TrainerRank):
    """The real estimator with runtime metadata readers replaced by frozen facts."""

    _facts: dict[str, Any] | None = None

    def _replay_slots(self, slot_refs: Any) -> Any:
        # Replay passes each group's index; map it to that group's frozen slot.
        if self._facts is None or slot_refs is None:
            return slot_refs
        groups = self._facts["groups"]
        return tuple(
            None
            if (adapter := groups[index]["adapter"]) is None
            else _ReplaySlot(adapter["kind"], adapter["name"])
            for index in slot_refs
        )

    def _gradient_slots(self, group_rows: Any, slot_refs: Any) -> Any:
        return _memory._gradient_slots(group_rows, self._replay_slots(slot_refs))

    def _checkpoint_gradient_groups(self, group_rows: Any, slot_refs: Any) -> Any:
        return _memory._checkpoint_gradient_groups(
            self, group_rows, self._replay_slots(slot_refs)
        )

    def _pending_adapter_gradient_bytes(self, refs: Any) -> tuple[int, ...]:
        if self._facts is None:
            return _memory._pending_adapter_gradient_bytes(self, refs)
        refs = tuple(dict.fromkeys(refs))
        if not refs:
            return ()
        if len(refs) != 1:
            raise ValueError("replayed adapter gradients are frozen per slot")
        for group in self._facts["groups"]:
            adapter = group["adapter"]
            if adapter is not None and (adapter["kind"], adapter["name"]) == tuple(
                refs[0]
            ):
                return tuple(adapter["pending"])
        return ()

    def _head_workspace_bytes(self, rows: int) -> int:
        assert self._facts is not None
        return _memory._dense_head_bytes(self._facts["head_vocabulary"], rows)

    def verify_group(
        self, group: dict[str, Any], layout: Any, records: list[dict[str, Any]]
    ) -> None:
        assert self._facts is not None
        if len(records) != len(group["request_indices"]):
            raise ValueError("runtime request membership mismatch")
        validate_tokens(
            value
            for record in records
            for value in (record["input_tokens"], record["target_tokens"])
            if value is not None
        )
        requests = tuple(
            _impl.ForwardInput(
                input_tokens=_impl.torch.tensor(
                    r["input_tokens"], dtype=_impl.torch.long, device="cpu"
                ),
                target_tokens=None
                if r["target_tokens"] is None
                else _impl.torch.tensor(
                    r["target_tokens"], dtype=_impl.torch.long, device="cpu"
                ),
                logits=r["logits"],
                top_k=r["top_k"],
                no_grad=r["no_grad"],
            )
            for r in records
        )
        from ._prefix_tree_planner import build_canonical_prefix_tree

        tree = build_canonical_prefix_tree([r.input_tokens for r in requests])
        packed = _impl.materialize_prefix_tree_layout(
            [r.input_tokens for r in requests], tree, layout, verify_shared_tokens=True
        )
        if group["gdn"] is not None and group["gdn"]["segments"] != [
            {
                name: getattr(segment, name)
                for name in ("start", "end", "packed_start", "group_id", "parent_id")
            }
            for segment in packed.segments
        ]:
            raise ValueError("GDN segment facts disagree with selected layout")
        projected = self._head_projection_rows(
            requests, positions=packed.positions_by_sequence
        )
        backwards = (
            self._facts["head_target_backward"]
            and group["grad"]
            and any(r.target_tokens is not None for r in requests)
        )
        target_rows = (
            self._head_target_chunk_rows(
                requests, positions=packed.positions_by_sequence
            )
            if backwards and any(r.logits or r.top_k is not None for r in requests)
            else projected
            if backwards
            else 0
        )
        if (projected, target_rows) != (group["head_rows"], group["head_target_rows"]):
            raise ValueError("head row facts disagree with selected requests/layout")

    def _moe_workspace_bytes(
        self, rows: int, *, checkpoint_grad: bool = False, slot_ref: Any = None
    ) -> int:
        if self._facts is None:
            return _memory._moe_workspace_bytes(
                self, rows, checkpoint_grad=checkpoint_grad, slot_ref=slot_ref
            )
        group = self._facts["groups"][0 if slot_ref is None else slot_ref]
        coefficient, stages = group["gradient" if checkpoint_grad else "forward"]
        return _memory._moe_workspace_from_terms(
            rows, (coefficient, tuple(map(tuple, stages)))
        )

    def _checkpoint_memory_floor(
        self, group_rows: Any, slot_refs: Any = None, gdn_segments: int = 0
    ) -> tuple[int, int]:
        if self._facts is None:
            return _memory._checkpoint_memory_floor(
                self, group_rows, slot_refs, gdn_segments
            )
        return _memory._checkpoint_floor_from_facts(
            self, group_rows, slot_refs, gdn_segments, self._facts["checkpoint_layers"]
        )

    def runtime_arguments(
        self, facts: Any, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        validate(facts)
        self._facts = facts
        self._moe_checkpoint_grad_bytes_per_token = facts[
            "checkpoint_moe_bytes_per_token"
        ]
        groups = facts["groups"]
        rows = tuple((g["rows"], g["grad"]) for g in groups)
        if rows != tuple(map(tuple, arguments["group_rows"])):
            raise ValueError("runtime facts disagree with selected group rows")
        if arguments.get("hybridep_growth_bytes", 0):
            raise ValueError("hybridep_runtime_facts_unsupported")
        head = max(
            max(
                _memory._dense_head_bytes(facts["head_vocabulary"], g["head_rows"]),
                3
                * _memory._dense_head_bytes(
                    facts["head_vocabulary"], g["head_target_rows"]
                ),
            )
            for g in groups
        )
        retained, workspace = 0, 0
        if any(g["grad"] for g in groups) and all(g["gdn"] is not None for g in groups):
            for index, g in enumerate(groups):
                model = g["gdn"]
                moe = self._moe_workspace_bytes(
                    g["packed_rows"], checkpoint_grad=g["grad"], slot_ref=index
                )
                saved, pending = (0, 0)
                if g["grad"]:
                    saved, pending = _gdn_memory.pending_floor(
                        self._hidden_size,
                        g["packed_rows"],
                        model["layers"],
                        tuple(_gdn_memory.Shape(**shape) for shape in model["shapes"]),
                        tuple(SimpleNamespace(**s) for s in model["segments"]),
                    )
                retained += saved
                workspace = max(workspace, moe + pending)
        return {
            **arguments,
            "group_rows": rows,
            "slot_refs": tuple(range(len(groups))),
            "head_workspace_bytes": head,
            "checkpoint_floor": (retained, workspace),
        }
