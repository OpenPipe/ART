"""Renderer for Llama 3 chat format."""

import tinker

from .base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ensure_text,
    parse_response_for_stop_token,
)


class Llama3Renderer(Renderer):
    """Renderer for Llama 3 Instruct models.

    Format::

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Note: We intentionally differ from HF's stock Llama template:

    - HF prepends "Cutting Knowledge Date..." to system messages; we don't
      (add manually if needed)

    Tool calling is NOT supported for Llama 3. The Llama 3 tool calling format
    uses bare JSON without delimiters, making it impossible to reliably distinguish
    tool calls from regular JSON content in model responses. Use a different model
    or develop your own renderer if you need tool calling.
    """

    @property
    def has_extension_property(self) -> bool:
        """Llama3 satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        header_str = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        output_str = ensure_text(message["content"]) + "<|eot_id|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
