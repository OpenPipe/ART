from types import SimpleNamespace

import torch

from art.megatron.train import _globalize_context_parallel_logprob_batch


def test_empty_context_parallel_rank_participates_in_logprob_reduce(monkeypatch):
    calls = []
    group = object()
    row_range = SimpleNamespace(start=0, end=2, size=lambda: 2)
    attention_state = SimpleNamespace(
        rank_plan=SimpleNamespace(local_row_ranges=(row_range,)),
        cp_group=group,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, *, group: calls.append((tensor.shape, group)),
    )

    result = _globalize_context_parallel_logprob_batch(
        local_logprobs=[],
        attention_states=[attention_state],
        sequence_lengths=[4],
        empty_template=torch.zeros((), dtype=torch.float32),
    )

    assert calls == [(torch.Size((1, 4)), group)]
    assert len(result) == 1
    assert torch.equal(result[0], torch.zeros((1, 4)))
