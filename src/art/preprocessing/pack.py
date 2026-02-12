import os
import random
import time
from typing import Any, cast

import torch
from typing_extensions import NotRequired, TypedDict, Unpack

from ..types import Verbosity
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


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: NotRequired[tuple[int, list[int]]]
    image_grid_thw: NotRequired[tuple[int, list[int]]]


def packed_tensors_from_tokenized_results(
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
) -> PackedTensors:
    # TODO: This function could potentially be optimized with vectorized operations
    token_ids: list[list[int]] = [[]]
    group_ids: list[list[int]] = [[]]
    parent_ids: list[list[int]] = [[]]
    input_pos: list[list[int]] = [[]]
    assistant_mask: list[list[int]] = [[]]
    logprobs: list[list[float]] = [[]]
    advantages: list[list[float]] = [[]]
    weights: list[list[float]] = [[]]
    pixel_values: list[list[torch.Tensor]] = [[]]
    image_grid_thw: list[list[torch.Tensor]] = [[]]

    for result in tokenized_results:
        if len(result.token_ids) > seq_len and not truncate_long_results:
            if verbosity > 1:
                print("Result is too long, skipping")
            continue
        result_without_prompt = result.without_prompt()
        if sum(result_without_prompt.assistant_mask) == 0:
            if verbosity > 1:
                print("Result has no unique completion tokens, skipping")
            continue
        if (
            len(token_ids[-1])
            + (
                len(result_without_prompt.token_ids)
                if result.prompt_id in group_ids[-1]
                else len(result.token_ids)
            )
            > seq_len
        ):
            token_ids.append([])
            group_ids.append([])
            parent_ids.append([])
            input_pos.append([])
            assistant_mask.append([])
            logprobs.append([])
            advantages.append([])
            weights.append([])
            pixel_values.append([])
            image_grid_thw.append([])
        group_id = random.randint(-(2**63), 2**63 - 1)
        if result.prompt_id in group_ids[-1]:
            result = result_without_prompt
        token_ids[-1].extend(result.token_ids)
        group_ids[-1].extend(
            [result.prompt_id] * result.prompt_length
            + [group_id] * (len(result.token_ids) - result.prompt_length)
        )
        parent_ids[-1].extend([result.prompt_id] * len(result.token_ids))
        input_pos[-1].extend(result.input_pos)
        assistant_mask[-1].extend(result.assistant_mask)
        logprobs[-1].extend(result.logprobs)
        advantages[-1].extend([result.advantage] * len(result.token_ids))
        weights[-1].extend([result.weight] * len(result.token_ids))
        if result.pixel_values is not None:
            pixel_values[-1].append(result.pixel_values)
        if result.image_grid_thw is not None:
            image_grid_thw[-1].append(result.image_grid_thw)
        if truncate_long_results:
            token_ids[-1] = token_ids[-1][:seq_len]
            group_ids[-1] = group_ids[-1][:seq_len]
            parent_ids[-1] = parent_ids[-1][:seq_len]
            input_pos[-1] = input_pos[-1][:seq_len]
            assistant_mask[-1] = assistant_mask[-1][:seq_len]
            logprobs[-1] = logprobs[-1][:seq_len]
            advantages[-1] = advantages[-1][:seq_len]
            weights[-1] = weights[-1][:seq_len]

    permutation = list(range(len(token_ids)))
    random.shuffle(permutation)
    token_ids = [token_ids[i] for i in permutation]
    group_ids = [group_ids[i] for i in permutation]
    parent_ids = [parent_ids[i] for i in permutation]
    input_pos = [input_pos[i] for i in permutation]
    assistant_mask = [assistant_mask[i] for i in permutation]
    logprobs = [logprobs[i] for i in permutation]
    advantages = [advantages[i] for i in permutation]
    weights = [weights[i] for i in permutation]
    pixel_values = [pixel_values[i] for i in permutation]
    image_grid_thw = [image_grid_thw[i] for i in permutation]

    def pad(values: list[list], pad_value) -> list[list]:
        max_len = seq_len
        for value in values:
            value.extend([pad_value] * (max_len - len(value)))
        return values

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

    return {
        "tokens": torch.tensor(pad(token_ids, pad_token_id)),
        "group_ids": torch.tensor(pad(group_ids, -1)),
        "parent_ids": torch.tensor(pad(parent_ids, -1)),
        "input_pos": torch.tensor(pad(input_pos, 0)),
        "assistant_mask": assistant_mask_tensor,
        "logprobs": torch.tensor(pad(logprobs, float("nan"))),
        "advantages": advantages_tensor,
        "weights": weights_tensor,
        "pixel_values": [
            torch.concat(tensors) if tensors else None for tensors in pixel_values
        ],
        "image_grid_thw": [
            torch.concat(tensors) if tensors else None for tensors in image_grid_thw
        ],
    }


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
