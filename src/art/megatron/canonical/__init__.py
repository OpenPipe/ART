"""Versioned numerical policy for independent dense Qwen roots.

Install once, after ART's model/LoRA hooks. Keep ``execution`` open from forward
through backward (including activation recomputation). Roots must be complete,
independently padded, and have no shared prefix. The trainer must accumulate
FP32 root gradients in datum order, independent of request chunk boundaries.
"""

from .layout import POLICY_VERSION, RootLayout, current_layout, execution
from .operators import install

__all__ = [
    "POLICY_VERSION",
    "RootLayout",
    "current_layout",
    "execution",
    "install",
]
