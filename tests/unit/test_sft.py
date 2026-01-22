"""Unit tests for SFT utilities."""

import json
from pathlib import Path
import tempfile

import pytest

from art.trajectories import Trajectory
from art.utils.sft import create_lr_schedule, create_sft_dataset_iterator, iterate_file


# Helper to create dummy trajectories
def create_dummy_trajectory(idx: int) -> Trajectory:
    """Create a dummy trajectory with a unique identifier."""
    return Trajectory(
        messages_and_choices=[
            {"role": "user", "content": f"Message {idx}"},
            {"role": "assistant", "content": f"Response {idx}"},
        ],
        reward=float(idx),
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


# ============================================================================
# Integration tests
# ============================================================================


def test_create_sft_dataset_iterator():
    """Test create_sft_dataset_iterator yields correct chunks."""
    trajectories = [create_dummy_trajectory(i) for i in range(20)]

    # batch_size=8, chunk_size=2 means each chunk has up to 2 batches of 8 trajectories
    # With 20 trajectories per epoch:
    #   - Items per chunk: 8 * 2 = 16
    #   - Chunks per epoch: ceil(20/16) = 2 (one with 16 trajs, one with 4 trajs)
    # With 3 epochs: 2 * 3 = 6 chunks total

    chunks = list(
        create_sft_dataset_iterator(
            trajectories,
            epochs=3,
            batch_size=8,  # 8 trajectories per batch
            chunk_size=2,  # 2 batches per chunk
            use_tqdm=False,
        )
    )

    # Should have 6 chunks total (2 per epoch * 3 epochs)
    assert len(chunks) == 6

    # Pattern repeats for each epoch: full chunk (16 trajs), partial chunk (4 trajs)
    assert len(chunks[0].trajectories) == 16  # Epoch 1, chunk 1
    assert len(chunks[1].trajectories) == 4  # Epoch 1, chunk 2 (partial)
    assert len(chunks[2].trajectories) == 16  # Epoch 2, chunk 1
    assert len(chunks[3].trajectories) == 4  # Epoch 2, chunk 2 (partial)
    assert len(chunks[4].trajectories) == 16  # Epoch 3, chunk 1
    assert len(chunks[5].trajectories) == 4  # Epoch 3, chunk 2 (partial)

    # Verify chunk metadata
    # Chunk 0: starts at batch 0 (16 trajectories = 2 batches)
    assert chunks[0].step == 0
    assert chunks[0].epoch == 0
    assert chunks[0].epoch_step == 0

    # Chunk 1: starts at batch 2 (after 2 batches from chunk 0)
    # chunk_start=16, global_batch_step = ceil(16/8) = 2
    assert chunks[1].step == 2
    assert chunks[1].epoch == 0
    assert chunks[1].epoch_step == 2


def test_iterate_file():
    """Test iterate_file reads trajectories correctly."""
    jsonl_file = create_temp_jsonl(10)

    try:
        # Read without shuffle
        trajectories = list(
            iterate_file(
                str(jsonl_file),
                epochs=2,
                shuffle=False,
            )
        )

        # Should have 20 trajectories (10 per epoch * 2 epochs)
        assert len(trajectories) == 20

        # Verify the content - first epoch should be in order
        for i in range(10):
            assert f"Message {i}" in str(trajectories[i].messages_and_choices)

    finally:
        jsonl_file.unlink()


def test_iterate_file_with_shuffle():
    """Test iterate_file with shuffle enabled."""
    jsonl_file = create_temp_jsonl(100)

    try:
        # Read with shuffle
        trajectories = list(
            iterate_file(
                str(jsonl_file),
                epochs=2,
                shuffle=True,
                shuffle_buffer_size=10,
            )
        )

        # Should have 200 trajectories
        assert len(trajectories) == 200

    finally:
        jsonl_file.unlink()


def test_chunk_size_validation():
    """Test that chunk_size < 1 raises an error."""
    trajectories = [create_dummy_trajectory(i) for i in range(10)]

    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        list(create_sft_dataset_iterator(trajectories, chunk_size=0, use_tqdm=False))

    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        list(create_sft_dataset_iterator(trajectories, chunk_size=-1, use_tqdm=False))


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


def test_create_sft_dataset_iterator_with_initial_step():
    """Test resuming from initial_step skips correct number of batches."""
    trajectories = [create_dummy_trajectory(i) for i in range(20)]

    # Without initial_step: should get all chunks
    all_chunks = list(
        create_sft_dataset_iterator(
            trajectories, epochs=1, batch_size=4, chunk_size=2, use_tqdm=False
        )
    )

    # With initial_step=2: should skip first 2 batches (first chunk)
    resumed_chunks = list(
        create_sft_dataset_iterator(
            trajectories,
            epochs=1,
            batch_size=4,
            chunk_size=2,
            initial_step=2,
            use_tqdm=False,
        )
    )

    # Should have fewer chunks when resuming
    assert len(resumed_chunks) < len(all_chunks)
    # First resumed chunk should start at step 2 or later
    assert resumed_chunks[0].step >= 2


def test_create_sft_dataset_iterator_epoch_shuffling():
    """Test that different epochs have different trajectory orderings."""
    trajectories = [create_dummy_trajectory(i) for i in range(10)]

    chunks = list(
        create_sft_dataset_iterator(
            trajectories,
            epochs=2,
            batch_size=10,  # One batch per epoch
            chunk_size=1,
            use_tqdm=False,
        )
    )

    # Should have 2 chunks (one per epoch)
    assert len(chunks) == 2

    # Different epochs should have different orderings (due to shuffle)
    epoch0_contents = [
        t.messages_and_choices[0]["content"] for t in chunks[0].trajectories
    ]
    epoch1_contents = [
        t.messages_and_choices[0]["content"] for t in chunks[1].trajectories
    ]
    assert epoch0_contents != epoch1_contents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
