from typing import Any, Sequence

from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler
from art.megatron.model_support.handlers.qwen3_common import (
    install_qwen3_text_preprocess_patch,
    qwen3_forward_kwargs,
)


class Qwen3DenseHandler(DefaultDenseHandler):
    key = "qwen3_dense"
    native_vllm_lora_status = "validated"

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        install_qwen3_text_preprocess_patch(model_chunks)

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        return qwen3_forward_kwargs(model, **kwargs)


QWEN3_DENSE_HANDLER = Qwen3DenseHandler()
