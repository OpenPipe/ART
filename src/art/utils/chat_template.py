"""Compatibility imports for ART's lightweight inference helpers."""

import sys

from art_inference import chat_template as _implementation
from art_inference.chat_template import *  # noqa: F403

sys.modules[__name__] = _implementation
