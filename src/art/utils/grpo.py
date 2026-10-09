from collections.abc import Sequence


def rewards_have_zero_variance(rewards: Sequence[float]) -> bool:
    if len(rewards) <= 1:
        return True
    first = rewards[0]
    return all(abs(reward - first) <= 1e-12 for reward in rewards[1:])
