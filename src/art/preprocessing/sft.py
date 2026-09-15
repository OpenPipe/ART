"""Loss-preserving SFT materialization over the shared prefix-tree row planner."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from .pack import DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH, _prefix_tree_pack_rows


class _SFTSequence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    token_ids: tuple[int, ...]
    labels: np.ndarray
    shareable_length: int
    prompt_id: int


def pack_sft_batch(
    inputs: Sequence[dict[str, torch.Tensor]], *, seq_len: int
) -> tuple[dict[str, torch.Tensor], ...]:
    """Pack one optimizer batch without filtering, truncating, or changing labels.

    Rows retain natural lengths. Supervised prediction positions are never shared;
    labels are still unshifted, as in tokenize_sft_batch's output.
    """
    items = []
    for index, tensors in enumerate(inputs):
        length = int(tensors["attention_mask"].sum().item())
        tokens = tensors["input_ids"].reshape(-1)[:length].numpy()
        labels = tensors["labels"].reshape(-1)[:length].numpy().copy()
        # An example's first token has no preceding hidden state to supervise.
        if length:
            labels[0] = -100
        targets = np.flatnonzero(labels != -100)
        items.append(
            _SFTSequence(
                token_ids=tuple(tokens.tolist()),
                labels=labels,
                shareable_length=max(int(targets[0]) - 1, 0)
                if targets.size
                else length,
                prompt_id=index,
            )
        )
    rows = []
    for sequences, plan in _prefix_tree_pack_rows(
        items,
        seq_len=seq_len,
        pack_results=True,
        min_shared_segment_length=DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH,
        # Rebuilding can change sharing at unequal supervision boundaries and
        # exceed the admitted capacity. Keep the already compacted bin geometry.
        rebuild_rows=False,
    ):
        fields = {
            name: np.empty((1, plan.length), dtype=np.int64)
            for name in ("input_ids", "labels", "group_ids", "parent_ids", "input_pos")
        }
        for segment in plan.segments:
            sequence = sequences[segment.sequence_indices[0]]
            source = slice(segment.start, segment.end)
            dest = slice(segment.packed_start, segment.packed_start + segment.length)
            fields["input_ids"][0, dest] = sequence.token_ids[source]
            fields["labels"][0, dest] = sequence.labels[source]
            fields["group_ids"][0, dest] = segment.group_id
            fields["parent_ids"][0, dest] = segment.parent_id
            fields["input_pos"][0, dest] = np.arange(segment.start, segment.end)
        rows.append(
            {
                **{name: torch.from_numpy(value) for name, value in fields.items()},
                "attention_mask": torch.ones(
                    (1, plan.length), dtype=torch.long, device="cpu"
                ),
            }
        )
    return tuple(rows)
