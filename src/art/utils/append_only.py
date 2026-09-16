"""Compatibility imports for ART's lightweight inference helpers."""

import sys

from art_inference import append_only as _implementation
from art_inference.append_only import *  # noqa: F403

sys.modules[__name__] = _implementation
