"""Unit tests for SFT utilities."""

import json
from pathlib import Path
import tempfile

import pytest

from art.utils.sft import (
    create_lr_schedule,
    create_sft_dataset_iterator,
    iterate_file,
)


# Helper to create a temporary JSONL file
def create_temp_jsonl(num_trajectories: int) -> Path:
    """Create a temporary JSONL file with dummy trajectories."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for i in range(num_trajectories):
        data = {
            "messages": [
                {"role": "user", "content": f"Message {i}"},
                {"role": "assistant", "content": f"Response {i}"},
            ],
        }
        temp_file.write(json.dumps(data) + "\n")
    temp_file.close()
    return Path(temp_file.name)


def test_iterate_file():
    """Test iterate_file reads trajectories correctly."""
    jsonl_file = create_temp_jsonl(10)

    try:
        trajectories = list(iterate_file(str(jsonl_file), epochs=1))

        assert len(trajectories) == 10

    finally:
        jsonl_file.unlink()


def test_iterate_file_multiple_epochs():
    """Test iterate_file with multiple epochs."""
    jsonl_file = create_temp_jsonl(10)

    try:
        trajectories = list(iterate_file(str(jsonl_file), epochs=3))

        # Should have 30 trajectories (10 * 3 epochs)
        assert len(trajectories) == 30

    finally:
        jsonl_file.unlink()


def test_iterate_file_with_initial_skip():
    """Test iterate_file with initial_skip for resuming."""
    jsonl_file = create_temp_jsonl(10)

    try:
        # Skip first 5 trajectories
        trajectories = list(iterate_file(str(jsonl_file), epochs=1, initial_skip=5))

        assert len(trajectories) == 5

    finally:
        jsonl_file.unlink()


def test_iterate_file_deterministic():
    """Test that iterate_file is deterministic with same seed."""
    jsonl_file = create_temp_jsonl(20)

    try:
        traj1 = list(iterate_file(str(jsonl_file), epochs=1, seed=42))
        traj2 = list(iterate_file(str(jsonl_file), epochs=1, seed=42))

        # Should get same order
        for t1, t2 in zip(traj1, traj2):
            assert t1.messages_and_choices == t2.messages_and_choices

    finally:
        jsonl_file.unlink()


def test_lr_schedule_warmup_not_zero():
    """Test that warmup doesn't start at lr=0."""
    lrs = create_lr_schedule(
        total_steps=10,
        peak_lr=1e-4,
        method="constant",
        warmup_steps=5,
        min_lr=0.0,
    )

    # First step should NOT be 0
    assert lrs[0] > 0
    # Should reach peak_lr by end of warmup
    assert lrs[4] == pytest.approx(1e-4)
    # After warmup, should stay at peak_lr (constant schedule)
    assert lrs[5] == pytest.approx(1e-4)


def test_lr_schedule_edge_cases():
    """Test LR schedule edge cases."""
    # Empty schedule
    lrs = create_lr_schedule(total_steps=0, peak_lr=1e-4)
    assert lrs == []

    # Single step
    lrs = create_lr_schedule(total_steps=1, peak_lr=1e-4)
    assert len(lrs) == 1
    assert lrs[0] == pytest.approx(1e-4)

    # Warmup steps >= total_steps (edge case)
    lrs = create_lr_schedule(total_steps=5, peak_lr=1e-4, warmup_steps=10)
    assert len(lrs) == 5
    # Should not crash and should produce valid learning rates
    assert all(lr > 0 for lr in lrs)


def test_lr_schedule_decay_methods():
    """Test that cosine and linear decay work correctly."""
    peak_lr = 1e-4
    min_lr = 1e-5

    # Linear decay: should go from peak_lr to min_lr
    lrs = create_lr_schedule(
        total_steps=5, peak_lr=peak_lr, method="linear", min_lr=min_lr
    )
    assert lrs[0] == pytest.approx(peak_lr)  # Start at peak
    assert lrs[-1] == pytest.approx(min_lr)  # End at min
    # Should be monotonically decreasing
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1]

    # Cosine decay: should go from peak_lr to min_lr
    lrs = create_lr_schedule(
        total_steps=5, peak_lr=peak_lr, method="cosine", min_lr=min_lr
    )
    assert lrs[0] == pytest.approx(peak_lr)  # Start at peak
    assert lrs[-1] == pytest.approx(min_lr)  # End at min


def test_lr_schedule_no_warmup():
    """Test schedule with warmup_steps=0."""
    lrs = create_lr_schedule(
        total_steps=5, peak_lr=1e-4, method="linear", warmup_steps=0, min_lr=0
    )
    assert len(lrs) == 5
    assert lrs[0] == pytest.approx(1e-4)  # Start at peak (no warmup)
    assert lrs[-1] == pytest.approx(0)  # End at min_lr


def _make_trajectories(n: int):
    """Create n dummy trajectories."""
    from art.trajectories import Trajectory

    return [
        Trajectory(
            messages_and_choices=[
                {"role": "user", "content": f"Message {i}"},
                {"role": "assistant", "content": f"Response {i}"},
            ],
        )
        for i in range(n)
    ]


def test_create_sft_dataset_iterator_lr_schedule_continuity():
    """Test that concatenated chunk LRs match the full schedule from create_lr_schedule."""
    trajs = _make_trajectories(100)

    # Compute the expected full schedule directly
    import math

    total_batches = math.ceil(len(trajs) / 2) * 2  # 2 epochs
    warmup_steps = int(total_batches * 0.1)
    expected_lrs = create_lr_schedule(
        total_steps=total_batches,
        peak_lr=2e-4,
        method="linear",
        warmup_steps=warmup_steps,
    )

    chunks = list(
        create_sft_dataset_iterator(
            trajs,
            chunk_size=15,
            epochs=2,
            batch_size=2,
            peak_lr=2e-4,
            seed=42,
            show_progress=False,
        )
    )

    all_lrs: list[float] = []
    for chunk in chunks:
        lr = chunk.config.learning_rate
        if isinstance(lr, list):
            all_lrs.extend(lr)
        else:
            all_lrs.append(lr)

    assert expected_lrs == all_lrs


def test_create_sft_dataset_iterator_step_tracking():
    """Test that step, epoch, and epoch_step are correct on each chunk."""
    trajs = _make_trajectories(20)
    chunks = list(
        create_sft_dataset_iterator(
            trajs,
            chunk_size=5,  # 5 batches * 2 batch_size = 10 trajectories per chunk
            epochs=2,
            batch_size=2,
            peak_lr=1e-4,
            show_progress=False,
        )
    )

    # 20 trajs, chunk_size=5 batches -> 10 trajs/chunk -> 2 chunks per epoch, 2 epochs -> 4 chunks
    assert len(chunks) == 4

    assert chunks[0].step == 0
    assert chunks[0].epoch == 0
    assert chunks[0].epoch_step == 0

    assert chunks[1].step == 5  # 10 trajs / batch_size 2 = 5 batches
    assert chunks[1].epoch == 0
    assert chunks[1].epoch_step == 5

    assert chunks[2].step == 10
    assert chunks[2].epoch == 1
    assert chunks[2].epoch_step == 0

    assert chunks[3].step == 15
    assert chunks[3].epoch == 1
    assert chunks[3].epoch_step == 5


def test_create_sft_dataset_iterator_initial_step():
    """Test that initial_step skips completed chunks."""
    trajs = _make_trajectories(100)
    all_chunks = list(
        create_sft_dataset_iterator(
            trajs,
            chunk_size=25,  # 25 batches * 2 batch_size = 50 trajectories per chunk
            epochs=1,
            batch_size=2,
            peak_lr=2e-4,
            show_progress=False,
        )
    )

    # Resume from step 25 (after first chunk of 25 batches)
    resumed_chunks = list(
        create_sft_dataset_iterator(
            trajs,
            chunk_size=25,
            epochs=1,
            batch_size=2,
            peak_lr=2e-4,
            initial_step=25,
            show_progress=False,
        )
    )

    assert len(all_chunks) == 2
    assert len(resumed_chunks) == 1
    # Resumed chunk should have the same LRs as the second full chunk
    assert resumed_chunks[0].config.learning_rate == all_chunks[1].config.learning_rate


def test_create_sft_dataset_iterator_deterministic():
    """Test that create_sft_dataset_iterator is deterministic with the same seed."""
    trajs = _make_trajectories(50)

    chunks1 = list(
        create_sft_dataset_iterator(
            trajs, chunk_size=10, epochs=2, batch_size=2, seed=42, show_progress=False
        )
    )
    chunks2 = list(
        create_sft_dataset_iterator(
            trajs, chunk_size=10, epochs=2, batch_size=2, seed=42, show_progress=False
        )
    )

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.config.learning_rate == c2.config.learning_rate
        assert c1.step == c2.step
        for t1, t2 in zip(c1.trajectories, c2.trajectories):
            assert t1.messages_and_choices == t2.messages_and_choices


def test_create_sft_dataset_iterator_empty_input():
    """Test that empty trajectories yields no chunks."""
    chunks = list(create_sft_dataset_iterator([], chunk_size=10, show_progress=False))
    assert chunks == []


def test_create_sft_dataset_iterator_single_chunk():
    """Test that chunk_size >= dataset produces one chunk with full schedule."""
    import math

    trajs = _make_trajectories(10)

    chunks = list(
        create_sft_dataset_iterator(
            trajs,
            chunk_size=100,  # larger than dataset
            epochs=1,
            batch_size=2,
            peak_lr=1e-4,
            seed=42,
            show_progress=False,
        )
    )

    total_batches = math.ceil(10 / 2)
    warmup_steps = int(total_batches * 0.1)
    expected_lrs = create_lr_schedule(
        total_steps=total_batches,
        peak_lr=1e-4,
        method="linear",
        warmup_steps=warmup_steps,
    )

    assert len(chunks) == 1
    assert chunks[0].config.learning_rate == expected_lrs
    assert chunks[0].config.batch_size == 2
    assert len(chunks[0].trajectories) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
