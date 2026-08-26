from types import SimpleNamespace

from art.utils.model_config import detect_chat_template_parts


def test_detects_gemma_4_template_parts() -> None:
    tokenizer = SimpleNamespace(
        chat_template="{{ '<|turn>' + role + '\\n' }}",
    )

    assert detect_chat_template_parts(tokenizer) == (
        "<|turn>user\n",
        "<|turn>model\n",
    )
