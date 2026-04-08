from dataclasses import dataclass
import importlib
import json
import os
from typing import TYPE_CHECKING, Any, Iterable
import uuid

import torch

safetensors_torch = importlib.import_module("safetensors.torch")
load_file = safetensors_torch.load_file
save_file = safetensors_torch.save_file

if TYPE_CHECKING:
    from ..preprocessing.tokenize import SFTBatch


DEFAULT_SFT_DATA_DIR = "/tmp/megatron_sft_data"


@dataclass(frozen=True)
class SerializedSFTBatches:
    sft_data_dir: str
    num_batches: int
    learning_rates: list[float]


def serialize_sft_batch_to_disk(batch: "SFTBatch", batch_dir: str) -> None:
    os.makedirs(batch_dir, exist_ok=True)
    metadata = {
        "learning_rate": batch.learning_rate,
        "num_trajectories": batch.num_trajectories,
        "num_tokens": batch.num_tokens,
        "num_trainable_tokens": batch.num_trainable_tokens,
        "num_trajectory_tensors": len(batch.trajectory_tensors),
    }
    with open(os.path.join(batch_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    for index, trajectory_tensors in enumerate(batch.trajectory_tensors):
        save_file(
            {
                key: value.squeeze(0) if value.dim() > 0 else value
                for key, value in trajectory_tensors.items()
            },
            os.path.join(batch_dir, f"trajectory_{index}.safetensors"),
        )


def materialize_sft_batches(
    batches: Iterable["SFTBatch"],
    *,
    sft_data_dir: str | None = None,
) -> SerializedSFTBatches:
    if sft_data_dir is None:
        sft_data_dir = os.path.join(DEFAULT_SFT_DATA_DIR, uuid.uuid4().hex)

    learning_rates: list[float] = []
    num_batches = 0
    for batch_index, batch in enumerate(batches):
        batch_dir = os.path.join(sft_data_dir, f"batch_{batch_index:06d}")
        serialize_sft_batch_to_disk(batch, batch_dir)
        learning_rates.append(batch.learning_rate)
        num_batches += 1

    return SerializedSFTBatches(
        sft_data_dir=sft_data_dir,
        num_batches=num_batches,
        learning_rates=learning_rates,
    )


def load_sft_batch_from_disk(
    batch_dir: str,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    with open(os.path.join(batch_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    trajectory_tensors = []
    for index in range(metadata["num_trajectory_tensors"]):
        trajectory_tensors.append(
            load_file(os.path.join(batch_dir, f"trajectory_{index}.safetensors"))
        )
    return metadata, trajectory_tensors
