"""ART's shared vLLM history adapter, bundled without training dependencies."""

import sys

from ._shared import vllm as _implementation
from ._shared.vllm import *  # noqa: F403

sys.modules[__name__] = _implementation
