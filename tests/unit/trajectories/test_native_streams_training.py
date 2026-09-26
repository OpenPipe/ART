from __future__ import annotations

from collections import Counter
import math
from typing import Any

import pytest
from test_native_streams import example, record
from test_tokenize import _chat_exchange
import torch

from art.loss import LossInputs, loss_fn
from art.preprocessing.pack import packed_tensors_from_tokenized_results
from art.preprocessing.tokenize import tokenize_trajectory_groups
import art.trajectories as tr


@pytest.mark.parametrize("ppo", [False, True])
def test_training_loss_gradients_and_packing_keep_each_native_causal_path(
    ppo: bool,
) -> None:
    trajectory, tokenizer = example()
    tokenizer.name_or_path = "public/base"
    control = tr.Trajectory(
        reward=1.0,
        exchanges=tr.TrajectoryExchanges(
            chat_completions=[
                _chat_exchange(
                    list(
                        record(
                            trajectory.exchanges.chat_completions[0]
                        ).prompt_token_ids
                    ),
                    [3, 4, 5],
                )
            ]
        ),
    )
    group = tr.TrajectoryGroup(trajectories=[trajectory, control])
    # Include clipped and unclipped ratios for both signs of advantage.
    for original in group.trajectories:
        for exchange in original.exchanges.chat_completions:
            for index, entry in enumerate(record(exchange).logprobs.content):
                entry.logprob = (-4.8, -5.5, -7.4)[index % 3]
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
        # Enumerate original completions, independent of histories, masks or trie.
        expected = []
        claimed = set()
        for exchange in original.exchanges.chat_completions:
            choice = record(exchange)
            for index, entry in enumerate(choice.logprobs.content):
                prefix = tuple(choice.prompt_token_ids + choice.token_ids[: index + 1])
                key = (exchange.request["model"], prefix)
                if key not in claimed:
                    claimed.add(key)
                    expected.append((key, entry.logprob))
        terms.extend(
            (prefix, lp, advantage, 1 / (len(expected) + 1e-6))
            for (_, prefix), lp in expected
        )
        actual = []
        for result in selected:
            assert result.advantage == advantage
            assert result.weight == pytest.approx(1 / (len(expected) + 1e-6))
            for index, sampled in enumerate(result.assistant_mask):
                if sampled:
                    prefix = tuple(result.token_ids[: index + 1])
                    actual.append((("test/model", prefix), result.logprobs[index]))
        assert actual == expected
    packed = packed_tensors_from_tokenized_results(
        results,
        seq_len=256,
        truncate_long_results=False,
        verbosity=0,
        min_prefix_tree_shared_segment_length=1,
    )
    stats = packed["prefix_tree_packing_stats"]
    assert stats["physical_tokens"] < stats["logical_tokens"]
    # Recover causal ancestors from the actual packed group/parent structure;
    # a sibling's tokens must never appear in another source's conditioning.
    observed = []
    # A small causal softmax predictor: shared parameters make wrong context or
    # repeated ownership observable in gradients, without a transformer/backend.
    theta = torch.linspace(-0.2, 0.2, 4 * 256, dtype=torch.float64).reshape(4, 256)
    theta.requires_grad_()

    def features(prefix: tuple[int, ...]) -> torch.Tensor:
        context = prefix[:-1]
        return theta.new_tensor(
            [
                1,
                len(context) / 50,
                sum(context) / 10_000,
                sum((i + 1) * token for i, token in enumerate(context)) / 100_000,
            ]
        )

    # Ignored positions depend on parameters too: the real loss mask must remove
    # their gradients. Selected target j is predicted at packed position j - 1.
    predictions = theta.sum().expand(packed["tokens"].shape).clone()
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
            prefix = tuple(tokens[j] for j in indices)
            assert index > 0
            predictions[row, index - 1] = (features(prefix) @ theta).log_softmax(0)[
                prefix[-1]
            ]
            observed.append(
                (
                    prefix,
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
    actual_loss = loss_fn(
        LossInputs(inputs=packed), predictions, None, None, {"ppo": ppo}
    )
    actual_loss.policy_loss.backward()
    assert theta.grad is not None and theta.grad.norm() > 0

    # Closed-form per-completion loss/Jacobian: no packed fields, ART loss helper,
    # masks or autograd are used by this oracle. Only finite public LPs are used.
    expected_loss = 0.0
    expected_gradient = torch.zeros_like(theta)
    active_signs = set()
    clipped_signs = set()
    for prefix, old_lp, advantage, weight in terms:
        feature = features(prefix)
        probabilities = (feature @ theta.detach()).softmax(0)
        new_lp = probabilities[prefix[-1]].log().item()
        # The recorded LP crosses the normal float32 packing boundary.
        old_lp = torch.tensor(old_lp, dtype=torch.float32).item()
        ratio = math.exp(new_lp - old_lp)
        scale = advantage / advantage_scale * weight / mean_weight / len(terms)
        if ppo:
            clipped = (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < 0.8)
            expected_loss -= scale * (min(1.2, max(0.8, ratio)) if clipped else ratio)
            coefficient = 0.0 if clipped else scale * ratio
        else:
            clipped = ratio > 5.0
            coefficient = scale * min(5.0, ratio)
            expected_loss -= coefficient * new_lp
        if coefficient:
            active_signs.add(advantage)
        if clipped:
            clipped_signs.add(advantage)
        jacobian = -probabilities
        jacobian[prefix[-1]] += 1
        expected_gradient -= coefficient * feature[:, None] * jacobian[None, :]
    assert active_signs == clipped_signs == {-1.0, 1.0}
    # Packing normalizes coefficients in float32; the independent predictor and
    # analytic sum use float64, including cancellation between opposite signs.
    assert actual_loss.policy_loss.item() == pytest.approx(
        expected_loss, rel=2e-6, abs=1e-6
    )
    torch.testing.assert_close(theta.grad, expected_gradient, rtol=2e-6, atol=2e-8)
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
