"""
Small utilities requiring only basic python libraries.
"""

from contextlib import contextmanager
import importlib
import logging
import time
from typing import Any, Sequence, TypeVar, cast

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    logger.info(f"Starting {key}")
    tstart = time.time()
    yield
    logger.info(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))


def dict_mean(list_of_dicts: list[dict[str, float | int]]) -> dict[str, float]:
    key2values = {}
    for d in list_of_dicts:
        for k, v in d.items():
            key2values.setdefault(k, []).append(v)
    return {k: float(np.mean(values)) for k, values in key2values.items()}


def all_same(xs: list[Any]) -> bool:
    return all(x == xs[0] for x in xs)


def lookup_func(path_to_func: str, default_module: str | None = None):
    """
    path.to.module:func_name or func_name (assumes default_module)
    """
    colon_count = path_to_func.count(":")
    if colon_count == 0 and default_module is not None:
        module_name = default_module
        func_name = path_to_func
    elif colon_count == 1:
        module_name, func_name = path_to_func.rsplit(":", 1)
    else:
        raise ValueError(f"Invalid path: {path_to_func}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def split_list(lst: Sequence[T], num_splits: int) -> list[list[T]]:
    """
    Split a sequence into a list of lists, where the sizes are as equal as possible,
    and the long and short lists are as uniformly distributed as possible.

    Args:
        lst: The sequence to split
        num_splits: Number of sublists to create

    Returns:
        A list of sublists with sizes differing by at most 1

    Raises:
        ValueError: If num_splits > len(lst) or num_splits <= 0

    Examples:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2, 3], [4, 5]]
        >>> split_list([1, 2, 3, 4, 5], 3)
        [[1, 2], [3, 4], [5]]
    """
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if num_splits > len(lst):
        raise ValueError(
            f"Cannot split list of length {len(lst)} into {num_splits} parts"
        )

    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)
    return [list(lst[edges[i] : edges[i + 1]]) for i in range(num_splits)]


def concat_lists(list_of_lists: list[list[Any]]) -> list[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def not_none(x: T | None) -> T:
    assert x is not None, f"{x=} must not be None"
    return x
