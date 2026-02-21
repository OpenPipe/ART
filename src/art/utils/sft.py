"""Utilities for supervised fine-tuning (SFT)."""

import itertools
import json
import math
import random
from typing import TYPE_CHECKING, Generator, List, Literal, NamedTuple

if TYPE_CHECKING:
    from art.dev import TrainSFTConfig as DevTrainSFTConfig
    from art.model import TrainableModel
    from art.trajectories import Trajectory
    from art.types import TrainSFTConfig


class SFTChunk(NamedTuple):
    trajectories: "list[Trajectory]"
    config: "TrainSFTConfig"
    step: int
    epoch: int
    epoch_step: int


def _parse_jsonl_line(line: str) -> "Trajectory":
    """Parse a JSONL line into a Trajectory object.

    Args:
        line: A JSON string containing trajectory data with 'messages' and optional 'tools'.

    Returns:
        A Trajectory object with the parsed data.
    """
    from art.trajectories import Trajectory

    data = json.loads(line)
    return Trajectory(
        messages_and_choices=data.get("messages", []),
        tools=data.get("tools"),
    )


def get_file_row_count(file_path: str) -> int:
    """
    Count the number of non-empty rows in a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        Number of non-empty lines in the file

    Raises:
        ValueError: If file_path does not end with .jsonl

    Example:
        count = get_file_row_count("data.jsonl")
        print(f"Dataset has {count} items")
    """
    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def create_lr_schedule(
    total_steps: int,
    peak_lr: float,
    method: Literal["cosine", "linear", "constant"] = "linear",
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> List[float]:
    """
    Create learning rate schedule for training with optional warmup.

    Args:
        total_steps: Total number of training steps
        peak_lr: Peak learning rate
        method: Learning rate schedule method. Options:
                - "cosine": Cosine annealing from peak_lr to min_lr
                - "linear": Linear decay from peak_lr to min_lr
                - "constant": Constant learning rate (peak_lr for all steps)
        warmup_steps: Number of warmup steps (linear warmup from 0 to peak_lr)
        min_lr: Minimum learning rate (floor for decay schedules)

    Returns:
        List of learning rates for each step

    Example:
        # Cosine schedule with warmup
        lrs = create_lr_schedule(100, 1e-4, method="cosine", warmup_steps=10)

        # Use with training loop
        for step, chunk in enumerate(chunk_trajectories(...)):
            train_sft(chunk, learning_rate=lrs[step])
    """
    if total_steps <= 0:
        return []

    learning_rates = []
    decay_steps = total_steps - warmup_steps

    for step in range(total_steps):
        if step < warmup_steps:
            # Warmup: linear ramp from min_lr to peak_lr
            # Use (step + 1) so first step has lr > 0
            lr = min_lr + (peak_lr - min_lr) * ((step + 1) / warmup_steps)
        else:
            # Decay phase: progress goes from 0 to 1
            progress = (
                (step - warmup_steps) / (decay_steps - 1) if decay_steps > 1 else 0
            )
            if method == "cosine":
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            elif method == "linear":
                lr = peak_lr - (peak_lr - min_lr) * progress
            elif method == "constant":
                lr = peak_lr
            else:
                raise ValueError(
                    f"Unknown method: {method}. Choose from: cosine, linear, constant"
                )

        learning_rates.append(lr)

    return learning_rates


def create_sft_dataset_iterator(
    trajectories: "list[Trajectory]",
    chunk_size: int = 10,
    epochs: int = 1,
    batch_size: int = 2,
    peak_lr: float = 2e-4,
    schedule_type: Literal["cosine", "linear", "constant"] = "linear",
    warmup_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    initial_step: int = 0,
    show_progress: bool = True,
) -> "Generator[SFTChunk, None, None]":
    """
    Prepare trajectories in chunks for multiple model.train_sft() calls.

    Yields SFTChunk objects so that each call to model.train_sft() produces
    its own training metrics. The learning rate schedule is computed over the
    entire dataset, then sliced so that scheduling (warmup, decay) is correct
    across all chunks.

    Args:
        trajectories: List of trajectories to train on.
        chunk_size: Number of batches to process per train_sft call. Default: 10.
                    This is an internal optimization parameter and does not affect training.
        epochs: Number of times to repeat the dataset. Default: 1
        batch_size: Number of trajectories per batch. Default: 2
        peak_lr: Peak learning rate. Default: 2e-4
        schedule_type: LR schedule ("cosine", "linear", "constant"). Default: "linear"
        warmup_ratio: Fraction of total steps used for warmup. Default: 0.1
        shuffle: Whether to shuffle trajectories each epoch. Default: True
        seed: Random seed. Each epoch uses seed + epoch_number. Default: 42
        initial_step: Global batch step to resume from. Default: 0
        show_progress: Whether to display a tqdm progress bar. Default: True

    Yields:
        SFTChunk(trajectories, config, step, epoch, epoch_step).

    Example:
        for chunk in create_sft_dataset_iterator(
            trajectories=my_trajectories,
            chunk_size=10,
            epochs=3,
            batch_size=2,
            peak_lr=2e-4,
        ):
            await model.train_sft(chunk.trajectories, chunk.config)
    """
    from tqdm.auto import tqdm

    from art.types import TrainSFTConfig as SFTConfig

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    dataset_size = len(trajectories)
    if dataset_size == 0:
        return

    batches_per_epoch = math.ceil(dataset_size / batch_size)
    total_batches = batches_per_epoch * epochs
    warmup_steps = int(total_batches * warmup_ratio)

    # Compute full LR schedule across all data
    full_schedule = create_lr_schedule(
        total_steps=total_batches,
        peak_lr=peak_lr,
        method=schedule_type,
        warmup_steps=warmup_steps,
    )

    # chunk_size is in batches; compute trajectory count per chunk
    items_per_chunk = batch_size * chunk_size
    chunks_per_epoch = math.ceil(dataset_size / items_per_chunk)

    # Convert initial_step (batch-based) to initial_chunk for skipping
    initial_chunk = initial_step // chunk_size

    pbar = (
        tqdm(
            initial=initial_step, total=total_batches, desc="SFT Training", unit="step"
        )
        if show_progress
        else None
    )

    for epoch in range(epochs):
        epoch_trajs = list(trajectories)
        if shuffle:
            random.Random(seed + epoch).shuffle(epoch_trajs)

        for chunk_idx in range(chunks_per_epoch):
            global_chunk_idx = epoch * chunks_per_epoch + chunk_idx

            # Skip chunks before initial_step
            if global_chunk_idx < initial_chunk:
                continue

            chunk_start = chunk_idx * items_per_chunk
            chunk_end = min(chunk_start + items_per_chunk, dataset_size)
            chunk_trajs = epoch_trajs[chunk_start:chunk_end]

            num_batches_in_chunk = math.ceil(len(chunk_trajs) / batch_size)
            global_batch_step = epoch * batches_per_epoch + (chunk_start // batch_size)
            epoch_batch_step = chunk_start // batch_size

            chunk_lrs = full_schedule[
                global_batch_step : global_batch_step + num_batches_in_chunk
            ]

            config = SFTConfig(
                learning_rate=chunk_lrs,
                batch_size=batch_size,
            )

            yield SFTChunk(
                trajectories=chunk_trajs,
                config=config,
                step=global_batch_step,
                epoch=epoch,
                epoch_step=epoch_batch_step,
            )

            if pbar:
                pbar.update(num_batches_in_chunk)

    if pbar:
        pbar.close()


def iterate_file(
    file_path: str,
    epochs: int = 1,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
    initial_skip: int = 0,
) -> Generator["Trajectory", None, None]:
    """
    Stream trajectories from a JSONL file for one or more epochs.

    Uses buffer-based shuffling to randomize order without loading all data
    into memory. Each epoch uses a different seed for varied shuffling.

    Args:
        file_path: Path to JSONL file (one JSON object per line)
        epochs: Number of times to iterate over the file. Default: 1
        shuffle_buffer_size: Size of shuffle buffer. Default: 10000.
                             Larger values give better shuffling but use more memory.
        seed: Base random seed. Each epoch uses seed + epoch_number. Default: 42
        initial_skip: Number of trajectories to skip (for resuming). Default: 0

    Yields:
        Trajectory objects

    Raises:
        ValueError: If file_path does not end with .jsonl

    Example:
        for trajectory in iterate_file("data.jsonl", epochs=3):
            process(trajectory)
    """
    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    skipped = 0

    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        shuffle_buffer: List["Trajectory"] = []

        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                traj = _parse_jsonl_line(line)
                shuffle_buffer.append(traj)

                # Once buffer is full, start yielding randomly
                if len(shuffle_buffer) >= shuffle_buffer_size:
                    idx = rng.randint(0, len(shuffle_buffer) - 1)
                    item = shuffle_buffer.pop(idx)

                    if skipped < initial_skip:
                        skipped += 1
                    else:
                        yield item

        # Flush remaining items in shuffle buffer
        rng.shuffle(shuffle_buffer)
        for traj in shuffle_buffer:
            if skipped < initial_skip:
                skipped += 1
            else:
                yield traj


async def train_sft_from_file(
    model: "TrainableModel",
    file_path: str,
    epochs: int = 1,
    batch_size: int = 2,
    peak_lr: float = 2e-4,
    schedule_type: Literal["cosine", "linear", "constant"] = "linear",
    warmup_ratio: float = 0.1,
    initial_step: int = 0,
    final_step: int | None = None,
    _config: "DevTrainSFTConfig | None" = None,
    verbose: bool = False,
    shuffle_buffer_size: int = 10000,
) -> None:
    """
    Train a model using supervised fine-tuning from a JSONL file.

    Streams data without loading all into memory. Suitable for large files (10GB+).

    Args:
        model: The TrainableModel to fine-tune. Must be registered with a backend.
        file_path: Path to JSONL file containing training data. Each line should have:
                   - messages: List of chat messages
                   - tools: Optional list of tools
        epochs: Number of times to iterate over the dataset. Default: 1
        batch_size: Number of trajectories per batch. Default: 2
        peak_lr: Peak learning rate. Default: 2e-4
        schedule_type: Learning rate schedule ("cosine", "linear", "constant"). Default: "linear"
        warmup_ratio: Ratio of total steps for warmup (0.0 to 1.0). Default: 0.1
        initial_step: Starting step for resuming training. Default: 0
        final_step: Ending step (exclusive). If None, trains to end of dataset.
            Useful for breaking training into segments with benchmarks in between.
        _config: Experimental configuration. Use at your own risk.
        verbose: Whether to print verbose output. Default: False
        shuffle_buffer_size: Size of shuffle buffer. Default: 10000.
                             Larger values give better shuffling but use more memory.

    Example:
        await train_sft_from_file(
            model=model,
            file_path="data/train.jsonl",
            epochs=3,
            batch_size=4,
            peak_lr=2e-4,
        )
    """
    from art.types import TrainSFTConfig

    row_count = get_file_row_count(file_path)

    if verbose:
        print(f"File has {row_count} rows")

    if row_count == 0:
        if verbose:
            print("No trajectories to train on")
        return

    # Calculate total trajectories and batches
    total_trajectories = row_count * epochs
    skip_trajectories = initial_step * batch_size

    if skip_trajectories >= total_trajectories:
        if verbose:
            print(f"initial_step {initial_step} skips all trajectories")
        return

    total_batches = math.ceil(total_trajectories / batch_size)
    warmup_steps = int(total_batches * warmup_ratio)

    if final_step is not None and final_step > total_batches:
        final_step = total_batches

    # Create learning rate schedule
    full_schedule = create_lr_schedule(
        total_steps=total_batches,
        peak_lr=peak_lr,
        method=schedule_type,
        warmup_steps=warmup_steps,
    )
    learning_rates = full_schedule[initial_step:final_step]

    if verbose:
        num_training_trajectories = len(learning_rates) * batch_size
        print(f"Training {num_training_trajectories} trajectories")
        print(f"Batches: {len(learning_rates)}, batch_size: {batch_size}")
        print(f"Schedule: {schedule_type}, peak_lr: {peak_lr}")

    # Stream trajectories from file, capped to the number we need
    max_trajectories = len(learning_rates) * batch_size
    trajectories = itertools.islice(
        iterate_file(
            file_path=file_path,
            epochs=epochs,
            shuffle_buffer_size=shuffle_buffer_size,
            initial_skip=skip_trajectories,
        ),
        max_trajectories,
    )

    config = TrainSFTConfig(
        learning_rate=learning_rates,
        batch_size=batch_size,
    )

    await model.train_sft(
        trajectories,
        config,
        _config=_config,
        verbose=verbose,
    )
