"""Compatibility shim for ``tinker_cookbook.renderers``."""

from tinker_cookbook.renderers import *  # noqa: F403
from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3ThinkingRenderer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
