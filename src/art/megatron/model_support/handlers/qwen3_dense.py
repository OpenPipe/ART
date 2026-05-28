from typing import Any, Sequence

from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler
from art.megatron.model_support.handlers.qwen3_common import (
    install_qwen3_text_preprocess_patch,
)
from art.megatron.model_support.spec import QWEN3_DENSE_META


class Qwen3DenseHandler(DefaultDenseHandler):
    key = QWEN3_DENSE_META.key
    native_vllm_lora_status = QWEN3_DENSE_META.native_vllm_lora_status

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        install_qwen3_text_preprocess_patch(model_chunks)


QWEN3_DENSE_HANDLER = Qwen3DenseHandler()
