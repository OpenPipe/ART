import math
import random
from dataclasses import dataclass
from typing import List, Generator, TypeVar, Generic, Union
from tqdm.auto import tqdm

T = TypeVar("T")


@dataclass
class DatasetBatch(Generic[T]):
    """Container for dataset batch information."""

    items: List[T]
    step: int
    epoch: int
    epoch_step: int
    total_steps: int


def adjust_lr(
    batch: DatasetBatch,
    learning_rate: float,
    warmup_length: Union[int, float] = 0,
    cooldown_length: Union[int, float] = 0,
) -> float:
    """
    Calculate the learning rate for a given batch based on the schedule.

    Args:
        batch: The DatasetBatch containing step and total_steps information.
        learning_rate: The base learning rate.
        warmup_length: Either an int (number of steps) or float (ratio of total steps). Defaults to 0.
        cooldown_length: Either an int (number of steps) or float (ratio of total steps). Defaults to 0.
                         If negative, specifies the step at which cooldown starts (e.g., -20 means cooldown starts at step 20).

    Returns:
        The adjusted learning rate for the current batch.
    """
    current_step = batch.step
    total_steps = batch.total_steps

    # Convert warmup_length to steps if it's a ratio
    if isinstance(warmup_length, float):
        warmup_steps = int(warmup_length * total_steps)
    else:
        warmup_steps = warmup_length

    # Handle cooldown_length
    if cooldown_length < 0:
        # Negative value means cooldown starts at that step
        cooldown_start = int(-cooldown_length)
        cooldown_steps = total_steps - cooldown_start
    else:
        # Convert cooldown_length to steps if it's a ratio
        if isinstance(cooldown_length, float):
            cooldown_steps = int(cooldown_length * total_steps)
        else:
            cooldown_steps = cooldown_length
        cooldown_start = total_steps - cooldown_steps

    # Ensure warmup doesn't exceed total steps
    warmup_steps = min(warmup_steps, total_steps)

    # Ensure cooldown_start is after warmup
    cooldown_start = max(cooldown_start, warmup_steps)
    cooldown_steps = total_steps - cooldown_start

    # Warmup phase
    if current_step < warmup_steps:
        return learning_rate * (current_step + 1) / warmup_steps

    # Cooldown phase
    if current_step >= cooldown_start and cooldown_steps > 0:
        steps_into_cooldown = current_step - cooldown_start
        remaining_ratio = 1.0 - (steps_into_cooldown + 1) / cooldown_steps
        return learning_rate * remaining_ratio

    # Main phase (between warmup and cooldown)
    return learning_rate


def iterate_dataset(
    dataset: List[T],
    groups_per_step: int = 1,
    num_epochs: int = 1,
    initial_step: int = 0,
    use_tqdm: bool = True,
) -> Generator[DatasetBatch[T], None, None]:
    """
    Generates batches from a dataset over multiple epochs with deterministic shuffling.

    Args:
        dataset: The list of data items.
        groups_per_step: The size of each batch. Defaults to 1.
        num_epochs: The number of times to iterate over the dataset. Defaults to 1.
        initial_step: The global step number to start from. Defaults to 0.
                           Useful for resuming training.
        use_tqdm: Whether to display a progress bar. Defaults to True.

    Yields:
        DatasetBatch: A dataclass containing:
        - items (List[T]): The list of items for the current batch.
        - epoch (int): The current epoch number (0-indexed).
        - global_step (int): The overall step number across all epochs.
        - epoch_step (int): The step number within the current epoch (0-indexed).
    """
    dataset_size = len(dataset)
    if dataset_size == 0:
        return

    steps_per_epoch = math.ceil(dataset_size / groups_per_step)
    total_steps = steps_per_epoch * num_epochs

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_steps,
            desc="Iterating dataset",
            unit="batch",
        )

    for epoch in range(num_epochs):
        # Create indices and shuffle deterministically based on epoch
        indices = list(range(dataset_size))
        random.seed(epoch)  # Ensure shuffling is the same for a given epoch
        random.shuffle(indices)

        for i in range(0, dataset_size, groups_per_step):
            epoch_step = i // groups_per_step
            # Calculate global step number before skipping
            global_step = epoch * steps_per_epoch + epoch_step

            if global_step < initial_step:
                # If using tqdm, we still need to update it even when skipping
                if progress_bar:
                    # Ensure the progress bar reflects the skipped steps accurately
                    # by setting the description or just updating.
                    # Setting n directly might be complex if initial_step > 0.
                    # A simple update() works if the bar was initialized correctly.
                    pass  # tqdm handles the initial value
                continue

            batch_indices = indices[i : i + groups_per_step]
            items = [dataset[idx] for idx in batch_indices]
            yield DatasetBatch(
                items=items,
                epoch=epoch,
                step=global_step,
                epoch_step=epoch_step,
                total_steps=total_steps,
            )

            # Update progress bar after yielding
            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()
