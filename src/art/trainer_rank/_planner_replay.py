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
# Replay sizes per-layer tuples from this recorded rank field; bound it as capture does.
MAX_LAYERS = 1024
_MAX_SEGMENTS = 4096
_MAX_INPUT_VALUES = 1_000_000


# Estimators bound on TrainerRank as plain functions, not methods.
_STATIC_ESTIMATORS = frozenset(
    {
        "_split_required_memory",
        "_gradient_slots",
        "_adapter_gradient_walk",
        "_adapter_gradient_head",
    }
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


# Frozen per plan; None where no checkpointed decoder would read them.
_RECOMPUTE_READERS = (
    "moe_checkpoint_state_bytes_per_token",
    "te_workspace_growth_bytes",
    "backward_row_state_bytes",
    "triton_min_rows",
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
    if (
        not 0 < rank._num_layers <= MAX_LAYERS
        or not 0 < len(plan.groups) <= _MAX_GROUPS
    ):
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
        "_adapter_gradient_head",
        "_checkpoint_input_gradient_bytes",
        "_checkpoint_gradient_covered",
        "_moe_recompute_covered_for",
        "_checkpoint_head_stage_bytes",
        "_backward_row_state_bytes",
        "_te_workspace_growth_bytes",
        "_moe_checkpoint_state_bytes_per_token",
        "_recomputed_mixer_bytes_per_token",
        "_mixer_activation_widths",
        "_triton_min_rows",
        "_plan_group_layouts",
        "_compute_group_layouts",
        "_layout_pricing_supported",
        "_layer_gdn_inputs",
        "_layout_checkpoint_floor",
        "_layout_checkpoint_rank_floors",
        "_layout_layer_boundaries",
        "_layout_adapter_gradient_bytes",
        "_checkpoint_adapter_gradient_extra",
        "_recomputed_mixer_widths",
        "_checkpoint_floor_decoder",
        "_dense_mlp_widths",
        # Producers of recorded arguments replay checks against the facts.
        "_plan_group_routed_rows",
        "_plan_head_backward_traced",
        "_head_backward_traced",
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
        coefficient, stages, shared = _memory._moe_workspace_terms(
            rank, checkpoint_grad=checkpoint_grad, slot_ref=ref
        )
        if len(stages) > 4096:
            raise ValueError("runtime_stage_inventory_over_limit")
        reserve(96 + 72 * len(stages))
        if (
            type(coefficient) is not int
            or not 0 <= coefficient < 2**63
            or any(v >= 2**63 for stage in stages for v in stage)
        ):
            raise ValueError("runtime_dimension_unsupported")
        return [coefficient, [list(stage) for stage in stages], shared]

    has_grad = any(g.grad_enabled for g in plan.groups)
    # Every rank's CP layouts come from the live runtime's CP configuration.
    plan_layouts = rank._plan_group_layouts(plan)
    for index, (group, (physical_rows, _)) in enumerate(
        zip(plan.groups, rank._plan_group_rows(plan), strict=True)
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
        if group.grad_enabled and name is not None:
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
        layout = None
        if plan_layouts is not None:
            chosen = plan_layouts[index]
            if len(chosen.attention_rows) > 64:
                raise ValueError("runtime_shape_inventory_over_limit")
            reserve(96 + 72 * len(chosen.attention_rows))
            layout = {
                "attention_rows": list(chosen.attention_rows),
                "gdn_rows": None if chosen.gdn_rows is None else list(chosen.gdn_rows),
                "attention_retained": list(chosen.attention_retained),
            }
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
                "moe_covered": group.grad_enabled
                and rank._moe_recompute_covered_for(group.slot_ref),
                "layout": layout,
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
    layers = _memory._checkpoint_layers(rank, rank._plan_group_rows(plan))
    # Per-rank layout pricing reads each decoder layer's input layout.
    inputs = rank._layer_gdn_inputs() if plan_layouts is not None and layers else ()
    if len(inputs) > MAX_LAYERS:
        raise ValueError("runtime_shape_inventory_over_limit")
    reserve(8 * len(inputs))
    # The covered dense stage for this plan's slots, and for no named slot
    # (the layout-free floor behind the input-gradient allowance).
    dense = [
        [int(v) for v in rank._dense_mlp_widths(refs)]
        for refs in (tuple(g.slot_ref for g in plan.groups), None)
    ]
    facts = {
        "version": 5,
        "checkpoint_layers": layers,
        "checkpoint_moe_bytes_per_token": rank._checkpoint_moe_bytes_per_token(),
        # Model and process readers of the recomputed layer and staged head,
        # read (like live pricing) only with a checkpointed decoder.
        **{
            name: getattr(rank, "_" + name)() if layers else None
            for name in _RECOMPUTE_READERS
        },
        "layer_gdn_inputs": list(inputs),
        "dense_widths": dense[0],
        "dense_base_widths": dense[1],
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
            "moe_checkpoint_state_bytes_per_token",
            "te_workspace_growth_bytes",
            "backward_row_state_bytes",
            "triton_min_rows",
            "layer_gdn_inputs",
            "dense_widths",
            "dense_base_widths",
            "head_vocabulary",
            "head_target_backward",
            "groups",
        },
    )
    if type(facts["version"]) is not int or facts["version"] != 5:
        raise ValueError("unsupported runtime facts version")
    for key in (
        "checkpoint_layers",
        "checkpoint_moe_bytes_per_token",
        "head_vocabulary",
    ):
        integer(facts[key])
    for key in ("dense_widths", "dense_base_widths"):
        if type(facts[key]) is not list or len(facts[key]) != 2:
            raise ValueError("invalid dense stage facts")
        for value in facts[key]:
            integer(value)
    # Live widths are both zero or both positive; a plan's named slots only
    # raise the constructor's, and only a checkpointed decoder has any.
    plan, base = facts["dense_widths"], facts["dense_base_widths"]
    if (
        any(all(pair) != any(pair) for pair in (plan, base))
        or (any(plan) and not (all(base) and plan[0] >= base[0] and plan[1] >= base[1]))
        or (any(base) and not facts["checkpoint_layers"])
    ):
        raise ValueError("invalid dense stage facts")
    for key in _RECOMPUTE_READERS:
        if not facts["checkpoint_layers"]:
            if facts[key] is not None:
                raise ValueError("invalid runtime dimension")
            continue
        # Any integer Triton setting is live-accepted (a non-positive one
        # prices no fallback chunk rows).
        integer(facts[key], minimum=-(2**63) if key == "triton_min_rows" else 0)
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
                "moe_covered",
                "layout",
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
        # Only gradient groups recompute; the live reader is not asked otherwise.
        if type(group["moe_covered"]) is not bool or (
            group["moe_covered"] and not group["grad"]
        ):
            raise ValueError("invalid MoE recompute coverage")
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
                or len(terms) != 3
                or type(terms[1]) is not list
                or len(terms[1]) > 4096
            ):
                raise ValueError("invalid MoE terms")
            reserve(96 + 72 * len(terms[1]))
            integer(terms[0])
            for stage in terms[1]:
                if type(stage) is not list or len(stage) != 2:
                    raise ValueError("invalid MoE stage")
                for value in stage:
                    integer(value)
            # The shared expert's part of the coefficient (live invariant).
            integer(terms[2])
            # Live keeps no stage or shared part beside a zero coefficient.
            if terms[2] > terms[0] or (terms[1] and not terms[0]):
                raise ValueError("invalid MoE terms")
        # Only gradient groups record a named slot's adapter. Live freezes an
        # unnamed one's gradient terms from the constructor coefficient, and
        # covers a group only with its own positive coefficient.
        if (
            group["grad"]
            and group["adapter"] is None
            and group["gradient"][0] != facts["checkpoint_moe_bytes_per_token"]
        ):
            raise ValueError("invalid MoE terms")
        if group["moe_covered"] and not (
            group["gradient"][0] and facts["checkpoint_moe_bytes_per_token"]
        ):
            raise ValueError("MoE coverage without a checkpoint coefficient")
        layout = group["layout"]
        if layout is not None:
            fields(layout, {"attention_rows", "gdn_rows", "attention_retained"})
            ranks = layout["attention_rows"]
            gdn_rows = layout["gdn_rows"]
            if (
                not group["grad"]
                or type(ranks) is not list
                or not 0 < len(ranks) <= 64
                or (
                    gdn_rows is not None
                    and (type(gdn_rows) is not list or len(gdn_rows) != len(ranks))
                )
                or type(layout["attention_retained"]) is not list
                or len(layout["attention_retained"]) != len(ranks)
            ):
                raise ValueError("invalid CP layout facts")
            reserve(96 + 72 * len(ranks))
            for value in (*ranks, *(gdn_rows or ()), *layout["attention_retained"]):
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
            # A slot without a kind (megatron-less reference) has none pending.
            if adapter["kind"] is None and any(adapter["pending"]):
                raise ValueError("invalid adapter gradient facts")
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
    # Live slot references all have a kind, or (without megatron) none do.
    kinds = {
        group["adapter"]["kind"] is None
        for group in groups
        if group["adapter"] is not None
    }
    if len(kinds) > 1:
        raise ValueError("invalid adapter gradient facts")
    # Live layouts cover every group of a plan on the same ranks, or none.
    layouts = [group["layout"] for group in groups]
    laid_out = any(layout is not None for layout in layouts)
    if laid_out and (
        any(layout is None for layout in layouts)
        or len({len(layout["attention_rows"]) for layout in layouts}) != 1
    ):
        raise ValueError("invalid CP layout facts")
    inputs = facts["layer_gdn_inputs"]
    if (
        type(inputs) is not list
        or len(inputs) > MAX_LAYERS
        or len(inputs) != (facts["checkpoint_layers"] if laid_out else 0)
        or any(type(value) is not bool for value in inputs)
    ):
        raise ValueError("invalid layer input layout facts")
    reserve(8 * len(inputs))
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
        vocabulary = self._facts["head_vocabulary"]
        return _memory._dense_head_bytes(vocabulary, rows) if rows > 0 else 0

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
        self,
        rows: int,
        *,
        routed_rows: int | None = None,
        checkpoint_grad: bool = False,
        slot_ref: Any = None,
    ) -> int:
        if self._facts is None:
            return _memory._moe_workspace_bytes(
                self,
                rows,
                routed_rows=routed_rows,
                checkpoint_grad=checkpoint_grad,
                slot_ref=slot_ref,
            )
        group = self._facts["groups"][0 if slot_ref is None else slot_ref]
        coefficient, stages, shared = group[
            "gradient" if checkpoint_grad else "forward"
        ]
        return _memory._moe_workspace_from_terms(
            rows, (coefficient, tuple(map(tuple, stages)), shared), routed_rows
        )

    def _checkpoint_memory_floor(
        self,
        group_rows: Any,
        slot_refs: Any = None,
        gdn_segments: int = 0,
        routed_rows: Any = None,
        layouts: Any = None,
    ) -> tuple[int, int]:
        if self._facts is None:
            return _memory._checkpoint_memory_floor(
                self, group_rows, slot_refs, gdn_segments, routed_rows, layouts
            )
        return _memory._checkpoint_floor_from_facts(
            self,
            group_rows,
            slot_refs,
            gdn_segments,
            self._facts["checkpoint_layers"],
            routed_rows,
            layouts,
        )

    def _dense_mlp_widths(self, slot_refs: Any = None) -> tuple[int, int]:
        if self._facts is None:
            return _memory._dense_mlp_widths(self, slot_refs)
        refs = tuple(slot_refs or ())
        if all(ref is None for ref in refs):
            stage, no_grad = self._facts["dense_base_widths"]
        elif refs == tuple(range(len(self._facts["groups"]))):
            stage, no_grad = self._facts["dense_widths"]
        else:
            raise ValueError("replayed dense widths are frozen per plan")
        return stage, no_grad

    def _layer_gdn_inputs(self) -> tuple[bool, ...]:
        if self._facts is None:
            return _memory._layer_gdn_inputs(self)
        return tuple(self._facts["layer_gdn_inputs"])

    def _moe_recompute_covered_for(self, slot_ref: Any) -> bool:
        if self._facts is None:
            return _memory._moe_recompute_covered_for(self, slot_ref)
        if slot_ref is None:
            raise ValueError("replayed MoE coverage is frozen per group")
        return self._facts["groups"][slot_ref]["moe_covered"]

    def _frozen(self, name: str) -> int:
        assert self._facts is not None
        return self._facts[name]

    def _moe_checkpoint_state_bytes_per_token(self) -> int:
        return self._frozen("moe_checkpoint_state_bytes_per_token")

    def _te_workspace_growth_bytes(self) -> int:
        return self._frozen("te_workspace_growth_bytes")

    def _backward_row_state_bytes(self) -> int:
        return self._frozen("backward_row_state_bytes")

    def _triton_min_rows(self) -> int:
        return self._frozen("triton_min_rows")

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
        # Capture refuses expert parallelism, where each group's experts see
        # its local rows (_plan_group_routed_rows).
        routed = tuple(g["rows"] for g in groups)
        recorded = arguments.get("group_routed_rows")
        if not isinstance(recorded, (list, tuple)) or tuple(recorded) != routed:
            raise ValueError("runtime facts disagree with selected routed rows")
        traced = arguments.get("head_backward_traced")
        if type(traced) is not bool:
            raise ValueError("head backward staging is not recorded")
        # Live staging needs CP2 and a target backward through a priced head.
        if traced and not (
            self._topology_key()[2] == 2
            and facts["head_target_backward"]
            and facts["head_vocabulary"]
        ):
            raise ValueError("head backward staging disagrees with recorded facts")
        # Live dense widths exist only at TP1/CP2/PP1 (_dense_mlp_widths).
        if any(facts["dense_base_widths"]) and self._topology_key()[1:] != (1, 2, 1):
            raise ValueError("dense stage facts disagree with the recorded topology")
        # Live MoE coefficients are all 0 above TP1 (_moe_output_bytes_per_token)
        # and on a dense rank, which has no MoE layers (_dense_mlp_widths).
        if (self._topology_key()[1] != 1 or any(facts["dense_base_widths"])) and (
            self._moe_output_bytes_per_token
            or self._moe_forward_stages
            or facts["checkpoint_moe_bytes_per_token"]
            or any(group[key][0] for group in groups for key in ("forward", "gradient"))
        ):
            raise ValueError("MoE facts disagree with the recorded rank")
        # An unnamed gradient group's forward terms are the constructor's.
        for group in groups:
            if (
                group["grad"]
                and group["adapter"] is None
                and (
                    group["forward"][0] != self._moe_output_bytes_per_token
                    or tuple(map(tuple, group["forward"][1]))
                    != self._moe_forward_stages
                )
            ):
                raise ValueError("invalid MoE terms")
        layouts = None
        if groups[0]["layout"] is not None:
            # Live layout pricing is CP2 at TP1/PP1 only (_layout_pricing_supported).
            _, tp, cp, pp = self._topology_key()
            if (tp, cp, pp) != (1, 2, 1) or len(
                groups[0]["layout"]["attention_rows"]
            ) != 2:
                raise ValueError("CP layout facts disagree with the recorded topology")
            layouts = tuple(
                _impl._GroupLayout(
                    attention_rows=tuple(g["layout"]["attention_rows"]),
                    gdn_rows=None
                    if g["layout"]["gdn_rows"] is None
                    else tuple(g["layout"]["gdn_rows"]),
                    attention_retained=tuple(g["layout"]["attention_retained"]),
                )
                for g in groups
            )
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
            "group_routed_rows": routed,
            "group_layouts": layouts,
            "slot_refs": tuple(range(len(groups))),
            "head_workspace_bytes": head,
            "checkpoint_floor": (retained, workspace),
        }
