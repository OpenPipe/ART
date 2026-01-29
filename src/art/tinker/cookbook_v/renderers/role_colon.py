"""Simple role:content format renderer."""

import tinker

from .base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolSpec,
    ensure_text,
)


class RoleColonRenderer(Renderer):
    """Simple role:content format renderer.

    Format::

        User: <content>

        Assistant: <content>

    This is basically the format used by DeepSeek R1-Zero, and similar to the format
    used by Anthropic, except that they use "Human" instead of "User".
    """

    @property
    def has_extension_property(self) -> bool:
        """RoleColon satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        header_str = message["role"].capitalize() + ":"
        output_str = " " + ensure_text(message["content"]) + "\n\n"
        # stop_overlap completes the stop sequence "\n\nUser:" for assistant messages.
        # For non-assistant messages, we use a placeholder that's never actually concatenated.
        stop_overlap_str = "User:" if message["role"] == "assistant" else "<UNUSED>"
        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        stop_overlap = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(stop_overlap_str, add_special_tokens=False)
        )
        return RenderedMessage(header=header, output=output, stop_overlap=stop_overlap)

    def get_stop_sequences(self) -> list[str]:
        return ["\n\nUser:"]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        import logging

        logger = logging.getLogger(__name__)

        str_response = self.tokenizer.decode(response)
        splitted = str_response.split("\n\nUser:")
        if len(splitted) == 1:
            logger.debug(f"Response is not a valid assistant response: {str_response}")
            return Message(role="assistant", content=str_response.strip()), False
        elif len(splitted) == 2:
            before, _after = splitted
            return Message(role="assistant", content=before.strip()), True
        else:
            raise ValueError(
                f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {len(splitted)}. "
                "You probably are using the wrong stop tokens when sampling"
            )

    @property
    def _bos_tokens(self) -> list[int]:
        bos_token_str = self.tokenizer.bos_token
        if bos_token_str is None:
            return []
        assert isinstance(bos_token_str, str)
        return self.tokenizer.encode(bos_token_str, add_special_tokens=False)

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        raise NotImplementedError("RoleColonRenderer does not support tool calling")
