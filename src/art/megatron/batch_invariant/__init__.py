"""Default trainer arithmetic; independent of batch and sequence row counts."""

from .ops import POLICY_VERSION, linear, matmul, rms_norm

__all__ = ["POLICY_VERSION", "linear", "matmul", "rms_norm"]
