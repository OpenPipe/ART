import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.integration.megatron.train_inf_mismatch.output_parity import (
    build_logical_token_map,
)


def test_logical_map_excludes_masked_template_tail_token() -> None:
    packed = {
        "tokens": torch.tensor([[10, 20, 30, 99, 4, 5, 1, 0]], dtype=torch.long),
        "group_ids": torch.tensor([[7, 7, 7, 8, 8, 8, 8, -1]], dtype=torch.long),
        "parent_ids": torch.tensor([[7, 7, 7, 7, 7, 7, 7, -1]], dtype=torch.long),
        "assistant_mask": torch.tensor(
            [[False, False, False, False, True, True, False, False]]
        ),
        "logprobs": torch.tensor(
            [[math.nan, math.nan, math.nan, math.nan, -0.1, -0.2, math.nan, math.nan]],
            dtype=torch.float32,
        ),
    }

    logical_map = build_logical_token_map(packed)

    assert len(logical_map.prompts) == 1
    prompt = logical_map.prompts[0]
    assert prompt.packed_prompt_length == 3
    assert prompt.scored_token_start_index == 4
    assert prompt.token_ids == [10, 20, 30, 99, 4, 5]
    assert [token.token_id for token in logical_map.tokens] == [4, 5]
    assert [token.vllm_prompt_token_index for token in logical_map.tokens] == [4, 5]
    assert [token.art_packed_token_index for token in logical_map.tokens] == [4, 5]
