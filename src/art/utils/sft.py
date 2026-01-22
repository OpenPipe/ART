"""Utilities for supervised fine-tuning (SFT)."""

from dataclasses import dataclass
import json
import math
import random
from typing import TYPE_CHECKING, Generator, List, Literal

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from art.dev import SFTConfig as DevSFTConfig
    from art.model import TrainableModel
    from art.trajectories import Trajectory
    from art.types import SFTConfig


@dataclass
class SFTDatasetChunk:
    """Container for SFT dataset chunk with trajectories, config, and step information."""

    trajectories: List["Trajectory"]
    config: "SFTConfig"
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
        reward=0.0,
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
    trajectories: List["Trajectory"],
    epochs: int = 1,
    batch_size: int = 1,
    chunk_size: int = 50,
    peak_lr: float = 2e-4,
    schedule_type: Literal["cosine", "linear", "constant"] = "linear",
    warmup_ratio: float = 0.1,
    initial_step: int = 0,
    use_tqdm: bool = True,
) -> Generator[SFTDatasetChunk, None, None]:
    """
    Create an iterator that yields SFT dataset chunks with trajectories, config, and step info.

    Combines trajectory batching with learning rate scheduling. Yields SFTDatasetChunk objects
    containing flattened trajectories, SFTConfig with learning rates, and step tracking info.

    Args:
        trajectories: List of Trajectory objects to train on
        epochs: Number of times to iterate over the trajectories. Default: 1
        batch_size: Number of trajectories per batch (one weight update per batch). Default: 1
        chunk_size: Number of batches to process per train_sft call. Default: 50.
                    This is an internal optimization parameter and does not affect training.
        peak_lr: Peak learning rate. Default: 2e-4
        schedule_type: Learning rate schedule type ("cosine", "linear", "constant"). Default: "constant"
        warmup_ratio: Ratio of total steps to use for warmup (0.0 to 1.0). Default: 0.1
        initial_step: The global training step (batch) to start from. Default: 0.
                      Useful for resuming training.
        use_tqdm: Whether to display a progress bar. Default: True

    Yields:
        SFTDatasetChunk containing:
            - trajectories: Flattened list of trajectories for this chunk
            - config: SFTConfig with custom_lr_schedule containing learning rates for each batch
            - step: Global training step (batch number) at the start of this chunk
            - epoch: Current epoch number (0-indexed)
            - epoch_step: Training step within current epoch (0-indexed)

    Example:
        trajectories = [traj1, traj2, ..., traj100]

        # Create SFT dataset iterator with constant schedule (default)
        for chunk in create_sft_dataset_iterator(
            trajectories=trajectories,
            epochs=3,
            batch_size=4,
            chunk_size=10,
            peak_lr=1e-4,
        ):
            # chunk.trajectories is a flat list of up to 40 trajectories
            # chunk.config.custom_lr_schedule is a list of learning rates (one per batch)
            # chunk.config.batch_size is 4
            # chunk.step is global training step (weight update number)
            # chunk.epoch is current epoch
            # chunk.epoch_step is training step within epoch
            await model.train_sft(chunk.trajectories, chunk.config)

        # Resume from training step 50
        for chunk in create_sft_dataset_iterator(
            trajectories=trajectories,
            epochs=3,
            batch_size=4,
            chunk_size=10,
            initial_step=50,
        ):
            # Starts from training step 50
            pass
    """
    from art.types import SFTConfig

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    dataset_size = len(trajectories)
    if dataset_size == 0:
        return

    # Calculate total batch steps (one step per batch)
    batches_per_epoch = math.ceil(dataset_size / batch_size)
    total_batch_steps = batches_per_epoch * epochs

    # Calculate warmup steps
    warmup_steps = int(total_batch_steps * warmup_ratio)

    # Create learning rate schedule (one LR per batch)
    custom_lr_schedule = create_lr_schedule(
        total_steps=total_batch_steps,
        peak_lr=peak_lr,
        method=schedule_type,
        warmup_steps=warmup_steps,
        min_lr=0.0,
    )

    # Calculate chunk iteration parameters
    items_per_chunk = batch_size * chunk_size
    chunks_per_epoch = math.ceil(dataset_size / items_per_chunk)

    # Convert initial_step (batch-based) to initial_chunk for skipping
    initial_chunk = initial_step // chunk_size

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_batch_steps,
            desc="Training SFT",
            unit="step",
        )

    for epoch in range(epochs):
        # Create indices and shuffle deterministically based on epoch
        indices = list(range(dataset_size))
        rng = random.Random(epoch)
        rng.shuffle(indices)

        for chunk_idx in range(chunks_per_epoch):
            # Calculate global chunk index for skipping
            global_chunk_idx = epoch * chunks_per_epoch + chunk_idx

            # Skip if before initial_chunk
            if global_chunk_idx < initial_chunk:
                continue

            # Get indices for this chunk
            chunk_start = chunk_idx * items_per_chunk
            chunk_end = min(chunk_start + items_per_chunk, dataset_size)
            step_indices = indices[chunk_start:chunk_end]

            # Flatten trajectories for this chunk
            chunk_trajectories: List["Trajectory"] = [
                trajectories[idx] for idx in step_indices
            ]

            # Calculate learning rates for each batch in this chunk
            chunk_lrs: List[float] = []
            num_batches_in_chunk = math.ceil(len(step_indices) / batch_size)

            # Calculate global batch step at the start of this chunk
            global_batch_step = epoch * batches_per_epoch + (chunk_start // batch_size)

            for batch_idx in range(num_batches_in_chunk):
                chunk_lrs.append(custom_lr_schedule[global_batch_step + batch_idx])

            # Create SFTConfig with custom learning rate schedule
            config = SFTConfig(
                batch_size=batch_size,
                custom_lr_schedule=chunk_lrs,
            )

            # epoch_step is the batch step within the current epoch
            epoch_batch_step = chunk_start // batch_size

            yield SFTDatasetChunk(
                trajectories=chunk_trajectories,
                config=config,
                step=global_batch_step,
                epoch=epoch,
                epoch_step=epoch_batch_step,
            )

            # Update progress bar by the number of batches in this chunk
            if progress_bar:
                progress_bar.update(num_batches_in_chunk)

    if progress_bar:
        progress_bar.close()


def iterate_file(
    file_path: str,
    epochs: int,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int | None = 42,
) -> Generator["Trajectory", None, None]:
    """
    Read JSONL file for each epoch, yielding individual Trajectory objects.

    Completes reading the entire file for one epoch before starting the next epoch.
    This ensures all trajectories from epoch N are yielded before any from epoch N+1.

    Each line should contain a dict with:
    - messages: List of chat messages
    - tools: Optional list of tools
    - reward: Optional reward (defaults to 0.0)
    - split: Optional split name (stored in metadata)
    - Any other fields will be stored in metadata

    Args:
        file_path: Path to JSONL file (one JSON object per line)
        epochs: Number of times to read through the file
        shuffle: Whether to shuffle trajectories. Defaults to True.
        shuffle_buffer_size: Size of shuffle buffer for streaming shuffle. Default: 10000.
                            Only used if shuffle=True.
        seed: Random seed for deterministic shuffling. Default: 42.
              Only used if shuffle=True.

    Yields:
        Individual Trajectory objects

    Raises:
        ValueError: If file_path does not end with .jsonl

    Example:
        # With shuffle
        for trajectory in iterate_file("data.jsonl", epochs=3, shuffle=True):
            # trajectory is a single Trajectory object
            process(trajectory)

        # No shuffle
        for trajectory in iterate_file("data.jsonl", epochs=3, shuffle=False):
            process(trajectory)
    """
    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    for epoch in range(epochs):
        # Use local Random instance to avoid modifying global random state
        if seed is not None:
            rng = random.Random(seed + epoch)
        else:
            rng = random.Random()

        if shuffle:
            # Streaming shuffle with buffer
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
                        yield shuffle_buffer.pop(idx)

            # Flush remaining items in shuffle buffer at end of epoch
            rng.shuffle(shuffle_buffer)
            for traj in shuffle_buffer:
                yield traj
        else:
            # No shuffle - sequential reading
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    yield _parse_jsonl_line(line)


async def train_sft_from_file(
    model: "TrainableModel",
    file_path: str,
    epochs: int = 1,
    batch_size: int = 1,
    chunk_size: int = 50,
    peak_lr: float = 2e-4,
    schedule_type: Literal["cosine", "linear", "constant"] = "linear",
    warmup_ratio: float = 0.1,
    initial_step: int = 0,
    use_tqdm: bool = True,
    _config: "DevSFTConfig | None" = None,
    verbose: bool = False,
) -> None:
    """
    Train a model using supervised fine-tuning from a JSONL file.

    This is a convenience function that combines iterate_file() and
    create_sft_dataset_iterator() to provide a simple interface for SFT training.

    Args:
        model: The TrainableModel to fine-tune. Must be registered with a backend.
        file_path: Path to JSONL file containing training data. Each line should have:
                   - messages: List of chat messages
                   - tools: Optional list of tools
        epochs: Number of times to iterate over the dataset. Default: 1
        batch_size: Number of trajectories per batch (one weight update per batch). Default: 1
        chunk_size: Number of batches to process per train_sft call. Default: 50.
                    This is an internal optimization parameter and does not affect training.
        peak_lr: Peak learning rate. Default: 2e-4
        schedule_type: Learning rate schedule type ("cosine", "linear", "constant"). Default: "linear"
        warmup_ratio: Ratio of total steps to use for warmup (0.0 to 1.0). Default: 0.1
        initial_step: The global training step (batch) to start from. Default: 0.
                      Useful for resuming training.
        use_tqdm: Whether to display a progress bar. Default: True
        _config: Additional experimental configuration. Use at your own risk.
        verbose: Whether to print verbose output. Default: False

    Example:
        import art
        from art.local import LocalBackend
        from art.utils.sft import train_sft_from_file

        async def main():
            backend = LocalBackend()
            model = art.TrainableModel(
                name="my-model",
                project="my-project",
                base_model="Qwen/Qwen2.5-7B-Instruct",
            )
            await model.register(backend)

            # Train with linear decay schedule
            await train_sft_from_file(
                model=model,
                file_path="data/train.jsonl",
                epochs=3,
                batch_size=4,
                peak_lr=2e-4,
                schedule_type="linear",
            )

            # Train with cosine schedule and warmup
            await train_sft_from_file(
                model=model,
                file_path="data/train.jsonl",
                epochs=1,
                batch_size=2,
                peak_lr=1e-4,
                schedule_type="cosine",
                warmup_ratio=0.1,
            )
    """
    # Load all trajectories into memory (needed for shuffling across epochs)
    trajectories = list(iterate_file(file_path, epochs=1, shuffle=False))

    if verbose:
        print(f"Loaded {len(trajectories)} trajectories from {file_path}")

    # Create dataset iterator and train
    for chunk in create_sft_dataset_iterator(
        trajectories=trajectories,
        epochs=epochs,
        batch_size=batch_size,
        chunk_size=chunk_size,
        peak_lr=peak_lr,
        schedule_type=schedule_type,
        warmup_ratio=warmup_ratio,
        initial_step=initial_step,
        use_tqdm=use_tqdm,
    ):
        await model.train_sft(
            chunk.trajectories,
            chunk.config,
            _config=_config,
            verbose=verbose,
        )
