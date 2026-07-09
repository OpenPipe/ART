import math
import os
import random
import time
from typing import Any, Literal, NamedTuple, cast

import torch
from typing_extensions import NotRequired, TypedDict, Unpack

from ..megatron.prefix_tree_packing import (
    estimate_prefix_tree_packed_tokens,
)
from ..megatron.prefix_tree_packing import (
    prefix_tree_pack as _prefix_tree_pack_sequences,
)
from ..types import Verbosity
from .moe_routing import (
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
    TokenRoute,
    count_route_slot_conflicts,
)
from .tokenize import TokenizedResult


class PackedTensors(TypedDict):
    tokens: torch.Tensor
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    input_pos: torch.Tensor
    assistant_mask: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    pixel_values: list[torch.Tensor | None]
    image_grid_thw: list[torch.Tensor | None]
    moe_routing_replay: PackedMoeRoutingReplay | None


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: NotRequired[tuple[int, list[int]]]
    image_grid_thw: NotRequired[tuple[int, list[int]]]


class _PackedPrefixTreeRow(TypedDict):
    token_ids: list[int]
    group_ids: list[int]
    parent_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    logprobs: list[float]
    advantages: list[float]
    weights: list[float]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    moe_routes: list[TokenRoute | None]


class _PrefixTreePackItem(NamedTuple):
    token_ids: tuple[int, ...]
    input_pos: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    advantage: float
    weight: float
    prompt_id: int
    shareable_length: int
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    moe_routes: tuple[TokenRoute | None, ...] | None


def packed_tensors_from_tokenized_results(
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
    pack_results: bool = True,
    include_moe_routing: bool = False,
) -> PackedTensors:
    return prefix_tree_pack(
        tokenized_results=tokenized_results,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        truncate_long_results=truncate_long_results,
        advantage_balance=advantage_balance,
        verbosity=verbosity,
        pack_results=pack_results,
        include_moe_routing=include_moe_routing,
    )


def prefix_tree_pack(
    *,
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
    pack_results: bool = True,
    include_moe_routing: bool = False,
) -> PackedTensors:
    rows: list[list[_PrefixTreePackItem]] = [[]]
    moe_routing_pack_stats = MoeRoutingPackStats()

    for result in tokenized_results:
        if len(result.token_ids) > seq_len and not truncate_long_results:
            if verbosity > 1:
                print("Result is too long, skipping")
            continue
        if include_moe_routing and result.moe_routed_experts is None:
            raise RuntimeError(
                "MoE routing replay from trajectories was requested, but a "
                "tokenized result has no aligned routed experts"
            )
        if sum(result.assistant_mask[result.prompt_length :]) == 0:
            if verbosity > 1:
                print("Result has no unique completion tokens, skipping")
            continue
        item = _prefix_tree_pack_item(result, seq_len=seq_len)
        if rows[-1] and (
            not pack_results
            or _packed_row_token_count([*rows[-1], item], seq_len=seq_len) > seq_len
        ):
            rows.append([])
        rows[-1].append(item)
        if truncate_long_results:
            rows[-1][-1] = _truncate_prefix_tree_pack_item(rows[-1][-1], seq_len)

    random.shuffle(rows)
    packed_rows = [
        _pack_prefix_tree_row(
            row,
            seq_len=seq_len,
            pack_results=pack_results,
            include_moe_routing=include_moe_routing,
            moe_routing_pack_stats=moe_routing_pack_stats,
        )
        for row in rows
    ]

    def pad(values: list[list], pad_value) -> list[list]:
        max_len = seq_len
        for value in values:
            value.extend([pad_value] * (max_len - len(value)))
        return values

    token_ids = [row["token_ids"] for row in packed_rows]
    group_ids = [row["group_ids"] for row in packed_rows]
    parent_ids = [row["parent_ids"] for row in packed_rows]
    input_pos = [row["input_pos"] for row in packed_rows]
    assistant_mask = [row["assistant_mask"] for row in packed_rows]
    logprobs = [row["logprobs"] for row in packed_rows]
    advantages = [row["advantages"] for row in packed_rows]
    weights = [row["weights"] for row in packed_rows]
    pixel_values = [row["pixel_values"] for row in packed_rows]
    image_grid_thw = [row["image_grid_thw"] for row in packed_rows]
    moe_routes = [row["moe_routes"] for row in packed_rows]

    assistant_mask_tensor = torch.tensor(pad(assistant_mask, 0), dtype=torch.bool)
    weights_tensor = torch.tensor(pad(weights, 0.0))
    weights_tensor = torch.where(
        assistant_mask_tensor, weights_tensor, torch.zeros_like(weights_tensor)
    )
    weights_tensor[assistant_mask_tensor] /= weights_tensor[
        assistant_mask_tensor
    ].mean()
    advantages_tensor = torch.tensor(pad(advantages, 0.0))
    advantages_tensor = torch.where(
        assistant_mask_tensor, advantages_tensor, torch.zeros_like(advantages_tensor)
    )
    if advantage_balance > 0.0:
        advantages_tensor = torch.where(
            advantages_tensor > 0,
            advantages_tensor,
            advantages_tensor * (1 - advantage_balance),
        )
    elif advantage_balance < 0.0:
        advantages_tensor = torch.where(
            advantages_tensor < 0,
            advantages_tensor,
            advantages_tensor * (1 + advantage_balance),
        )
    advantages_tensor[assistant_mask_tensor] /= (
        advantages_tensor[assistant_mask_tensor].abs()
        * weights_tensor[assistant_mask_tensor]
    ).mean()

    packed_tensors: PackedTensors = {
        "tokens": torch.tensor(pad(token_ids, pad_token_id)),
        "group_ids": torch.tensor(pad(group_ids, -1)),
        "parent_ids": torch.tensor(pad(parent_ids, -1)),
        "input_pos": torch.tensor(pad(input_pos, 0)),
        "assistant_mask": assistant_mask_tensor,
        "logprobs": torch.tensor(pad(logprobs, float("nan"))),
        "advantages": advantages_tensor,
        "weights": weights_tensor,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "moe_routing_replay": None,
    }
    if include_moe_routing:
        (
            route_tensor,
            route_mask,
            num_layers,
            topk,
            num_experts,
        ) = _tensorize_moe_routes(moe_routes, seq_len)
        moe_routing_pack_stats.packed_tokens = int(route_mask.sum().item())
        packed_tensors["moe_routing_replay"] = PackedMoeRoutingReplay(
            expert_indices=route_tensor,
            token_mask=route_mask,
            num_layers=num_layers,
            topk=topk,
            num_experts=num_experts,
            pack_stats=moe_routing_pack_stats,
        )
    return packed_tensors


def _prefix_tree_pack_item(
    result: TokenizedResult,
    *,
    seq_len: int,
) -> _PrefixTreePackItem:
    shareable_length = min(
        int(result.prompt_length),
        max(_first_trainable_token_index(result) - 1, 0),
    )
    item = _PrefixTreePackItem(
        token_ids=tuple(int(value) for value in result.token_ids),
        input_pos=tuple(int(value) for value in result.input_pos),
        assistant_mask=tuple(int(value) for value in result.assistant_mask),
        logprobs=tuple(float(value) for value in result.logprobs),
        advantage=float(result.advantage),
        weight=float(result.weight),
        prompt_id=int(result.prompt_id),
        shareable_length=shareable_length,
        pixel_values=result.pixel_values,
        image_grid_thw=result.image_grid_thw,
        moe_routes=(
            tuple(result.moe_routed_experts)
            if result.moe_routed_experts is not None
            else None
        ),
    )
    return _truncate_prefix_tree_pack_item(item, seq_len)


def _truncate_prefix_tree_pack_item(
    item: _PrefixTreePackItem,
    seq_len: int,
) -> _PrefixTreePackItem:
    if len(item.token_ids) <= seq_len:
        return item
    return _PrefixTreePackItem(
        token_ids=item.token_ids[:seq_len],
        input_pos=item.input_pos[:seq_len],
        assistant_mask=item.assistant_mask[:seq_len],
        logprobs=item.logprobs[:seq_len],
        advantage=item.advantage,
        weight=item.weight,
        prompt_id=item.prompt_id,
        shareable_length=min(item.shareable_length, seq_len),
        pixel_values=item.pixel_values,
        image_grid_thw=item.image_grid_thw,
        moe_routes=item.moe_routes[:seq_len] if item.moe_routes is not None else None,
    )


def _first_trainable_token_index(result: TokenizedResult) -> int:
    return next(
        (
            index
            for index, (is_assistant, logprob) in enumerate(
                zip(result.assistant_mask, result.logprobs, strict=True)
            )
            if bool(is_assistant) or not math.isnan(float(logprob))
        ),
        len(result.token_ids),
    )


def _packed_row_token_count(
    row: list[_PrefixTreePackItem],
    *,
    seq_len: int,
) -> int:
    if not row:
        return 0
    count = estimate_prefix_tree_packed_tokens(
        (torch.tensor(item.token_ids, dtype=torch.long) for item in row),
        max_depth=seq_len,
        shareable_lengths=(item.shareable_length for item in row),
    )
    if count is None:
        raise RuntimeError("CPU prefix-tree token estimate unexpectedly failed")
    return count


def _pack_prefix_tree_row(
    row: list[_PrefixTreePackItem],
    *,
    seq_len: int,
    pack_results: bool,
    include_moe_routing: bool,
    moe_routing_pack_stats: MoeRoutingPackStats,
) -> _PackedPrefixTreeRow:
    if not row:
        return {
            "token_ids": [],
            "group_ids": [],
            "parent_ids": [],
            "input_pos": [],
            "assistant_mask": [],
            "logprobs": [],
            "advantages": [],
            "weights": [],
            "pixel_values": None,
            "image_grid_thw": None,
            "moe_routes": [],
        }
    tree = _prefix_tree_pack_sequences(
        (torch.tensor(item.token_ids, dtype=torch.long) for item in row),
        max_depth=seq_len if pack_results else 0,
        shareable_lengths=(
            item.shareable_length if pack_results else 0 for item in row
        ),
    )
    token_ids = tree.tokens.reshape(-1).tolist()
    assigned = [False] * len(token_ids)
    input_pos = [0] * len(token_ids)
    assistant_mask = [0] * len(token_ids)
    logprobs = [float("nan")] * len(token_ids)
    advantages = [0.0] * len(token_ids)
    weights = [0.0] * len(token_ids)
    moe_routes: list[TokenRoute | None] = [None] * len(token_ids)
    for item_index, item in enumerate(row):
        packed_positions = tree.positions_by_sequence[item_index].tolist()
        for source_index, packed_index in enumerate(packed_positions):
            _validate_prefix_tree_assignment(
                item,
                source_index=source_index,
                packed_index=packed_index,
                token_ids=token_ids,
                input_pos=input_pos,
                assigned=assigned,
            )
            route = (
                item.moe_routes[source_index]
                if item.moe_routes is not None and source_index < len(item.moe_routes)
                else None
            )
            if assigned[packed_index]:
                if include_moe_routing:
                    _record_shared_route_conflict(
                        existing=moe_routes[packed_index],
                        candidate=route,
                        stats=moe_routing_pack_stats,
                    )
                continue
            assigned[packed_index] = True
            input_pos[packed_index] = item.input_pos[source_index]
            assistant_mask[packed_index] = item.assistant_mask[source_index]
            logprobs[packed_index] = item.logprobs[source_index]
            advantages[packed_index] = item.advantage
            weights[packed_index] = item.weight
            moe_routes[packed_index] = route
    return {
        "token_ids": token_ids[:seq_len],
        "group_ids": tree.group_ids.reshape(-1).tolist()[:seq_len],
        "parent_ids": tree.parent_ids.reshape(-1).tolist()[:seq_len],
        "input_pos": input_pos[:seq_len],
        "assistant_mask": assistant_mask[:seq_len],
        "logprobs": logprobs[:seq_len],
        "advantages": advantages[:seq_len],
        "weights": weights[:seq_len],
        "pixel_values": _packed_row_tensor_list(row, "pixel_values"),
        "image_grid_thw": _packed_row_tensor_list(row, "image_grid_thw"),
        "moe_routes": moe_routes[:seq_len] if include_moe_routing else [],
    }


def _validate_prefix_tree_assignment(
    item: _PrefixTreePackItem,
    *,
    source_index: int,
    packed_index: int,
    token_ids: list[int],
    input_pos: list[int],
    assigned: list[bool],
) -> None:
    if token_ids[packed_index] != item.token_ids[source_index]:
        raise RuntimeError("Prefix-tree pack token assignment mismatch")
    if not assigned[packed_index]:
        return
    if input_pos[packed_index] != item.input_pos[source_index]:
        raise RuntimeError("Prefix-tree pack cannot share mismatched input positions")
    if item.assistant_mask[source_index] or not math.isnan(item.logprobs[source_index]):
        raise RuntimeError("Prefix-tree pack attempted to share a trainable token")


def _record_shared_route_conflict(
    *,
    existing: TokenRoute | None,
    candidate: TokenRoute | None,
    stats: MoeRoutingPackStats,
) -> None:
    if existing is None or candidate is None:
        raise RuntimeError("Prefix-tree MoE route is missing")
    compared, conflicts = count_route_slot_conflicts(existing, candidate)
    stats.prefix_tree_rows += 1
    stats.prefix_tree_compared_slots += compared
    stats.prefix_tree_conflict_slots += conflicts
    stats.prefix_tree_conflict_rows += int(conflicts > 0)


def _packed_row_tensor_list(
    row: list[_PrefixTreePackItem],
    attr: Literal["pixel_values", "image_grid_thw"],
) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []
    seen_shared_prompts: set[int] = set()
    for item in row:
        tensor = getattr(item, attr)
        if tensor is None:
            continue
        if item.shareable_length > 0:
            if item.prompt_id in seen_shared_prompts:
                continue
            seen_shared_prompts.add(item.prompt_id)
        tensors.append(tensor)
    return torch.concat(tensors) if tensors else None


def _tensorize_moe_routes(
    routes_by_sequence: list[list[TokenRoute | None]],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    first_route = next(
        (
            route
            for sequence_routes in routes_by_sequence
            for route in sequence_routes
            if route is not None
        ),
        None,
    )
    if first_route is None:
        raise RuntimeError("No MoE routes were packed")
    num_layers = len(first_route)
    topk = len(first_route[0])
    max_expert_id = 0
    dense_routes: list[list[TokenRoute]] = []
    route_masks: list[list[bool]] = []
    zero_route: TokenRoute = [[0 for _ in range(topk)] for _ in range(num_layers)]
    for sequence_routes in routes_by_sequence:
        dense_sequence: list[TokenRoute] = []
        mask_sequence: list[bool] = []
        for route in sequence_routes:
            if route is None:
                dense_sequence.append(zero_route)
                mask_sequence.append(False)
                continue
            if len(route) != num_layers or any(
                len(layer_route) != topk for layer_route in route
            ):
                raise RuntimeError("Packed MoE routes must have one rectangular shape")
            max_expert_id = max(
                max_expert_id,
                max(int(expert_id) for layer in route for expert_id in layer),
            )
            dense_sequence.append(route)
            mask_sequence.append(True)
        while len(dense_sequence) < seq_len:
            dense_sequence.append(zero_route)
            mask_sequence.append(False)
        dense_routes.append(dense_sequence[:seq_len])
        route_masks.append(mask_sequence[:seq_len])
    return (
        torch.tensor(dense_routes, dtype=torch.int32),
        torch.tensor(route_masks, dtype=torch.bool),
        num_layers,
        topk,
        max(topk, max_expert_id + 1),
    )


def packed_tensors_from_dir(**kwargs: Unpack[DiskPackedTensors]) -> PackedTensors:
    os.makedirs(kwargs["dir"], exist_ok=True)
    packed_tensors = {
        key: torch.from_file(
            f"{kwargs['dir']}/{key}.pt",
            shared=True,
            size=kwargs["num_sequences"] * kwargs["sequence_length"],
            dtype=dtype,
        ).view(kwargs["num_sequences"], kwargs["sequence_length"])
        for key, dtype in {
            "tokens": torch.long,
            "group_ids": torch.long,
            "parent_ids": torch.long,
            "input_pos": torch.long,
            "assistant_mask": torch.bool,
            "logprobs": torch.float32,
            "advantages": torch.float32,
            "weights": torch.float32,
        }.items()
    }
    _add_tensor_list(packed_tensors, kwargs, "pixel_values", torch.float32)  # ty:ignore[invalid-argument-type]
    _add_tensor_list(packed_tensors, kwargs, "image_grid_thw", torch.long)  # ty:ignore[invalid-argument-type]
    return cast(PackedTensors, packed_tensors)


def _add_tensor_list(
    packed_tensors: dict[str, Any],
    disk_packed_tensors: DiskPackedTensors,
    key: str,
    dtype: torch.dtype,
) -> None:
    if info := disk_packed_tensors.get(key):
        packed_tensors[key] = []
        inner_dim, offsets = cast(tuple[int, list[int]], info)
        packed_pixel_values = torch.from_file(
            f"{disk_packed_tensors['dir']}/{key}.pt",
            shared=True,
            size=offsets[-1] * inner_dim,
            dtype=dtype,
        ).view(-1, inner_dim)
        for start, end in zip(offsets[:-1], offsets[1:]):
            packed_tensors[key].append(
                packed_pixel_values[start:end] if start < end else None
            )
    else:
        packed_tensors[key] = [None] * disk_packed_tensors["num_sequences"]


def packed_tensors_to_dir(tensors: PackedTensors, dir: str) -> DiskPackedTensors:
    os.makedirs(dir, exist_ok=True)
    disk_packed_tensors: DiskPackedTensors = {
        "dir": dir,
        "num_sequences": tensors["tokens"].shape[0],
        "sequence_length": tensors["tokens"].shape[1],
    }
    if info := _get_tensor_list_info(tensors["pixel_values"]):
        disk_packed_tensors["pixel_values"] = info
    if info := _get_tensor_list_info(tensors["image_grid_thw"]):
        disk_packed_tensors["image_grid_thw"] = info
    for key, tensor in packed_tensors_from_dir(**disk_packed_tensors).items():
        if isinstance(tensor, list):
            for i, t in enumerate(tensor):
                if t is not None:
                    t.copy_(tensors[key][i])  # ty:ignore[invalid-key, unresolved-attribute]
        else:
            tensor.copy_(tensors[key])  # type: ignore
    return disk_packed_tensors


def _get_tensor_list_info(
    tensors: list[torch.Tensor | None],
) -> tuple[int, list[int]] | None:
    inner_dims = {tensor.shape[1] for tensor in tensors if tensor is not None}
    if len(inner_dims) == 0:
        return None
    assert len(inner_dims) == 1, f"Inner dimensions of {tensors} are not the same"
    offsets = [0]
    for tensor in tensors:
        if tensor is not None:
            offsets.append(offsets[-1] + tensor.shape[0])
        else:
            offsets.append(offsets[-1])
    return inner_dims.pop(), offsets


def plot_packed_tensors(
    packed_tensors: PackedTensors, output_dir: str | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Plotting dependencies are not installed. Please install them with: "
            "pip install openpipe-art[plotting]"
        )

    plt.figure(figsize=(15, 24))

    for tensor, label, title, subplot_idx in (
        (packed_tensors["tokens"], "Token IDs", "Token IDs", 1),
        (packed_tensors["logprobs"], "Log Probabilities", "Token Log Probs", 2),
        (packed_tensors["group_ids"], "Group IDs", "Token Groups", 3),
        (packed_tensors["parent_ids"], "Parent IDs", "Parent IDs", 4),
        (packed_tensors["input_pos"], "Position", "Input Position", 5),
        (packed_tensors["assistant_mask"], "Assistant Mask", "Assistant Mask", 6),
        (packed_tensors["advantages"], "Advantages", "Token Advantages", 7),
        (packed_tensors["weights"], "Weights", "Token Weights", 8),
    ):
        plt.subplot(4, 2, subplot_idx)
        sns.heatmap(
            tensor.numpy(),
            cmap="viridis",
            cbar_kws={"label": label},
            xticklabels=False,
        )
        plt.title(title)
        plt.xlabel("Sequence Position")
        plt.ylabel("Batch")

    plt.tight_layout()
    plt.show()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/packed_tensors_plot_{int(time.time())}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
    else:
        print("No output directory specified, plot not saved")
