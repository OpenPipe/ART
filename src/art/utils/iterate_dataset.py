import math
import random
from typing import List, Generator, Tuple, TypeVar
from tqdm.auto import tqdm

T = TypeVar("T")


def iterate_dataset(
    dataset: List[T],
    batch_size: int = 1,
    num_epochs: int = 1,
    initial_iteration: int = 0,
    use_tqdm: bool = True,
) -> Generator[Tuple[List[T], int, int, int], None, None]:
    """
    Generates batches from a dataset over multiple epochs with deterministic shuffling.

    Args:
        dataset: The list of data items.
        batch_size: The size of each batch. Defaults to 1.
        num_epochs: The number of times to iterate over the dataset. Defaults to 1.
        initial_iteration: The global iteration number to start from. Defaults to 0.
                           Useful for resuming training.
        use_tqdm: Whether to display a progress bar. Defaults to True.

    Yields:
        A tuple containing:
        - batch (List[T]): The list of items for the current batch.
        - epoch (int): The current epoch number (0-indexed).
        - global_iteration (int): The overall iteration number across all epochs.
        - epoch_iteration (int): The iteration number within the current epoch (0-indexed).
    """
    dataset_size = len(dataset)
    if dataset_size == 0:
        return

    iterations_per_epoch = math.ceil(dataset_size / batch_size)
    total_iterations = iterations_per_epoch * num_epochs

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_iteration,
            total=total_iterations,
            desc="Iterating dataset",
            unit="batch",
        )

    for epoch in range(num_epochs):
        # Create indices and shuffle deterministically based on epoch
        indices = list(range(dataset_size))
        random.seed(epoch)  # Ensure shuffling is the same for a given epoch
        random.shuffle(indices)

        for i in range(0, dataset_size, batch_size):
            epoch_iteration = i // batch_size
            # Calculate global iteration number before skipping
            global_iteration = epoch * iterations_per_epoch + epoch_iteration

            if global_iteration < initial_iteration:
                # If using tqdm, we still need to update it even when skipping
                if progress_bar:
                    # Ensure the progress bar reflects the skipped iterations accurately
                    # by setting the description or just updating.
                    # Setting n directly might be complex if initial_iteration > 0.
                    # A simple update() works if the bar was initialized correctly.
                    pass  # tqdm handles the initial value
                continue

            batch_indices = indices[i : i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]
            yield batch, epoch, global_iteration, epoch_iteration

            # Update progress bar after yielding
            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()
