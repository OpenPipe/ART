import copy
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def strip_logprobs(obj: Any) -> Any:
    """
    Recursively remove 'logprobs' keys from nested data structures to reduce data storage size.

    Args:
        obj: Any nested data structure

    Returns:
        The same structure with 'logprobs' keys removed, or the original
        object if deepcopy fails
    """

    try:
        copied_obj = copy.deepcopy(obj)
    except Exception as e:
        logger.warning(
            f"Failed to deepcopy object in strip_logprobs: {e}. "
            "Returning original object unchanged."
        )
        return obj

    result = _strip_logprobs(copied_obj)

    return result


def _strip_logprobs(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_logprobs(v) for k, v in obj.items() if k != "logprobs"}
    elif isinstance(obj, (list, tuple)):
        result = [_strip_logprobs(v) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if k == "logprobs":
                setattr(obj, k, None)
            else:
                setattr(obj, k, _strip_logprobs(v))
        return obj
    return obj
