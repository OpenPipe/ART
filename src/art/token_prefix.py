"""Compatibility imports for ART's lightweight inference helpers."""

import sys

from art_inference import token_prefix as _implementation
from art_inference.token_prefix import *  # noqa: F403

sys.modules[__name__] = _implementation
