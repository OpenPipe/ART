from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from test_native_streams import example, native_terms
from test_tokenize import _chat_exchange

from art.preprocessing.pack import packed_tensors_from_tokenized_results
from art.preprocessing.tokenize import tokenize_trajectory_groups
import art.trajectories as tr


def test_training_weights_and_packing_keep_each_native_causal_path() -> None:
    trajectory, tokenizer = example()
    tokenizer.name_or_path = "public/base"
    control = tr.Trajectory(
        reward=1.0,
        exchanges=tr.TrajectoryExchanges(
            chat_completions=[_chat_exchange([1, 2], [3])]
        ),
    )
    group = tr.TrajectoryGroup(trajectories=[trajectory, control])
    before = group.model_dump_json()
    results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=False,
            scale_rewards=False,
            shuffle_group_trajectories=False,
        )
    )
    assert len(results) == 3
    terms: list[tuple[tuple[int, ...], float, float, float]] = []
    for original, advantage in [(trajectory, 1.0), (control, -1.0)]:
        selected = [r for r in results if r.trajectory is original]
        assert selected
        expected = native_terms(original)
        actual = []
        for result in selected:
            assert result.advantage == advantage
            assert result.weight == pytest.approx(1 / (len(expected) + 1e-6))
            for index, sampled in enumerate(result.assistant_mask):
                if sampled:
                    prefix = tuple(result.token_ids[: index + 1])
                    actual.append((("test/model", prefix), result.logprobs[index]))
                    terms.append(
                        (prefix, result.logprobs[index], advantage, result.weight)
                    )
        assert actual == expected
    packed = packed_tensors_from_tokenized_results(
        results,
        seq_len=256,
        truncate_long_results=False,
        verbosity=0,
        min_prefix_tree_shared_segment_length=1,
    )
    # Recover causal ancestors from the actual packed group/parent structure;
    # a sibling's tokens must never appear in another source's conditioning.
    observed = []
    for row in range(packed["tokens"].shape[0]):
        tokens = packed["tokens"][row].tolist()
        groups = packed["group_ids"][row].tolist()
        parents = packed["parent_ids"][row].tolist()
        positions = packed["input_pos"][row].tolist()
        for index, sampled in enumerate(packed["assistant_mask"][row].tolist()):
            if not sampled:
                continue
            ancestry = {groups[index]}
            current = groups[index]
            while True:
                parent = parents[groups.index(current)]
                if parent in ancestry:
                    break
                ancestry.add(parent)
                current = parent
            indices = [j for j in range(index + 1) if groups[j] in ancestry]
            assert [positions[j] for j in indices] == list(range(positions[index] + 1))
            observed.append(
                (
                    tuple(tokens[j] for j in indices),
                    packed["logprobs"][row, index].item(),
                    packed["advantages"][row, index].item(),
                    packed["weights"][row, index].item(),
                )
            )
    mean_weight = sum(t[3] for t in terms) / len(terms)
    advantage_scale = sum(abs(t[2]) * t[3] / mean_weight for t in terms) / len(terms)

    def rounded(values: Any) -> Counter[Any]:
        return Counter(
            (p, round(lp, 5), round(a, 5), round(w, 5)) for p, lp, a, w in values
        )

    assert rounded(observed) == rounded(
        (p, lp, a / advantage_scale, w / mean_weight) for p, lp, a, w in terms
    )
    assert group.model_dump_json() == before


def test_public_async_dispatch_and_private_trace_agree() -> None:
    import asyncio

    from art.trajectories import _tokenize as module

    trajectory, tokenizer = example()

    async def run() -> Any:
        return (
            await tr.tokenize([trajectory], multi_history=True, tokenizer=tokenizer)
        )[0]

    public = asyncio.run(run())
    private, _ = module._tokenize_trajectory_with_trace(trajectory, tokenizer=tokenizer)
    assert public.model_dump_json() == private.model_dump_json()
