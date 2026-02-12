import asyncio
import functools
import inspect
import logging
import time
from typing import (
    Any,
    Callable,
    Optional,
    ParamSpec,
    TypeVar,
)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 0.25,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[
    [Callable[P, R]],
    Callable[P, R],
]:
    """
    Retry decorator with exponential backoff for both synchronous and asynchronous functions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called on retry with the exception and attempt number

    Returns:
        Decorated function that will retry on specified exceptions
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                last_exception = None
                current_delay = delay

                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt == max_attempts:
                            raise

                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logging.warning(
                                f"Retry {attempt}/{max_attempts} for {func.__name__} "  # ty:ignore[unresolved-attribute]
                                f"after error: {str(e)}"
                            )

                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor

                assert last_exception is not None
                raise last_exception

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception = None
                current_delay = delay

                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt == max_attempts:
                            raise

                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logging.warning(
                                f"Retry {attempt}/{max_attempts} for {func.__name__} "  # ty:ignore[unresolved-attribute]
                                f"after error: {str(e)}"
                            )

                        time.sleep(current_delay)
                        current_delay *= backoff_factor

                assert last_exception is not None
                raise last_exception

            return sync_wrapper

    return decorator
