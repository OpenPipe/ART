import random

from art.utils.iterate_dataset import iterate_dataset


def test_iterate_dataset_is_deterministic_across_runs() -> None:
    dataset = list(range(10))

    first = [
        batch.items
        for batch in iterate_dataset(
            dataset, groups_per_step=3, num_epochs=2, use_tqdm=False
        )
    ]
    second = [
        batch.items
        for batch in iterate_dataset(
            dataset, groups_per_step=3, num_epochs=2, use_tqdm=False
        )
    ]

    assert first == second


def test_iterate_dataset_does_not_reset_global_random_state() -> None:
    dataset = list(range(10))

    random.seed(12345)
    expected = [random.random() for _ in range(3)]

    random.seed(12345)
    iterator = iterate_dataset(dataset, groups_per_step=2, num_epochs=2, use_tqdm=False)
    next(iterator)
    after_iteration = [random.random() for _ in range(3)]

    assert after_iteration == expected
