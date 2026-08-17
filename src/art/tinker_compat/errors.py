from __future__ import annotations

import tinker


class UnsupportedCapabilityError(tinker.TinkerError, NotImplementedError):
    """The requested Tinker behavior is outside this compatibility profile."""
