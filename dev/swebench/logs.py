from langfuse import Langfuse
from langfuse.decorators import langfuse_context
from langfuse.types import SpanLevel
from logging import Handler, LogRecord
import litellm
import logging
from sweagent.agent.agents import DefaultAgent
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook


# Add Langfuse callbacks for SWE-agent litellm calls
litellm.success_callback.append("langfuse")
litellm.failure_callback.append("langfuse")


# Suppress the "dictionary changed size during iteration" errors from Langfuse
class LangfuseErrorFilter(logging.Filter):
    """Filter out specific Langfuse errors that occur due to concurrent access"""

    def filter(self, record: LogRecord) -> bool:
        # Return False to suppress the log message
        if (
            record.name == "LiteLLM"
            and "dictionary changed size during iteration" in record.getMessage()
        ):
            return False
        return True


# Add the filter to the LiteLLM logger
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.addFilter(LangfuseErrorFilter())

# Suppress urllib3 retry warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Suppress OpenAI base client retry INFO messages
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# Hook into Logger creation to suppress new loggers matching our patterns
# Only patch if not already patched
if not hasattr(logging.getLogger, "_is_patched"):
    get_logger = logging.getLogger

    def _get_logger(name: str | None = None) -> logging.Logger:
        """Patched getLogger that applies suppression to new loggers."""
        logger = get_logger(name)

        if name:
            for prefix in ("rex", "swea", "gql"):
                if name.startswith(prefix):
                    logger.setLevel(logging.CRITICAL + 1)
                    logger.propagate = False
                    logger.disabled = True
                    logger.handlers = []
                    break

        return logger

    # Mark the function as patched
    _get_logger._is_patched = True
    logging.getLogger = _get_logger


# Disable printing the patch message to reduce log noise
SaveApplyPatchHook._print_patch_message = lambda *args, **kwargs: None


def setup_agent_logger(agent: DefaultAgent) -> None:
    agent.logger.propagate = False
    agent.logger.handlers = []
    agent.logger.addHandler(
        LangfuseHandler(langfuse_context.get_current_trace_id() or "")
    )


class LangfuseHandler(Handler):
    """Custom handler to forward logs to Langfuse"""

    def __init__(self, trace_id: str) -> None:
        self.langfuse = Langfuse()
        self.trace_id = trace_id
        super().__init__()

    def emit(self, record: LogRecord) -> None:
        if record.levelname in ["DEBUG", "INFO"]:
            return
        levels: dict[str, SpanLevel] = {
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "ERROR",
        }
        self.langfuse.event(
            trace_id=self.trace_id,
            name="agent-logger",
            level=levels[record.levelname],
            status_message=record.getMessage(),
        )
        self.langfuse.flush()
