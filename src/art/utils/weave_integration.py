"""Weave integration utilities for ART."""

import os
from typing import Any, Callable, TypeVar, Set
from functools import wraps
import asyncio
import inspect

# Type for async rollout functions
F = TypeVar("F", bound=Callable[..., Any])

# Track whether weave has been initialized
_weave_initialized = False
# Track functions that need weave decoration
_pending_weave_ops: Set[Callable] = set()


def rollout_op(func: F) -> F:
    """Decorator for rollout functions that ensures weave tracking.
    
    This decorator:
    1. Applies the weave.op decorator to track the rollout when weave is initialized
    2. Ensures the function returns an art.Trajectory
    3. Can be used with or without weave being initialized
    """
    # Add to pending list for later decoration
    _pending_weave_ops.add(func)
    
    # If weave is already initialized, apply decoration now
    if _weave_initialized and "WANDB_API_KEY" in os.environ:
        return _apply_weave_op(func)
    
    # Otherwise, create a wrapper that will check on each call
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if we should apply weave decoration now
        if _weave_initialized and func in _pending_weave_ops:
            _pending_weave_ops.remove(func)
            decorated = _apply_weave_op(func)
            # Replace the wrapper with the decorated function for future calls
            wrapper.__wrapped__ = decorated
            return await decorated(*args, **kwargs)
        return await func(*args, **kwargs)
    
    # Store reference to original function
    wrapper.__wrapped__ = func
    wrapper._original_func = func
    
    return wrapper


def _apply_weave_op(func: F) -> F:
    """Apply weave.op decorator to a function."""
    try:
        import weave
        # Get the original function if this is a wrapper
        original = getattr(func, '_original_func', func)
        # Check if already decorated to avoid double decoration
        if not hasattr(original, "_is_weave_op"):
            decorated = weave.op(original)
            decorated._is_weave_op = True
            return decorated
    except Exception as e:
        print(f"Failed to apply weave.op decorator: {e}")
    return func


def init_weave_with_wandb(project: str) -> None:
    """Initialize weave with the same project as wandb.
    
    Args:
        project: The wandb project name to use for weave initialization
    """
    global _weave_initialized
    if "WANDB_API_KEY" in os.environ:
        try:
            import weave
            weave.init(project)
            _weave_initialized = True
            print(f"Weave initialized for project: {project}")
            
            # Apply weave.op to any pending functions
            for func in list(_pending_weave_ops):
                _apply_weave_op(func)
                _pending_weave_ops.remove(func)
                
        except Exception as e:
            print(f"Failed to initialize weave: {e}")