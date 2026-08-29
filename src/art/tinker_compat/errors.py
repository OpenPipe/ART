import tinker


class UnsupportedCapabilityError(tinker.TinkerError):
    """The pinned Tinker SDK requested behavior outside the launch profile."""
