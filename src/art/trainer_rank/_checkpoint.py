"""Compatibility alias for the Megatron-owned checkpoint codec."""

import sys

from art.megatron import checkpoint as _checkpoint

sys.modules[__name__] = _checkpoint
