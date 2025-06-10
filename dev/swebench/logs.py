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

# Suppress urllib3 retry warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# Configuration for log suppression
SUPPRESS_CONFIG = {
    # Logger names to completely suppress
    "suppress_loggers": [
        "rex-deploy",  # All rex-deploy-ThreadPoolExecutor logs
        "swea-lm",  # All swea-lm-ThreadPoolExecutor logs
    ],
    # Exception types to suppress (will check exception class name)
    "suppress_exceptions": [
        "CommandTimeoutError",
        "BashIncorrectSyntaxError",
        "NonZeroExitCodeError",
        "FileNotFoundError",
        "TIMEOUT",  # pexpect.exceptions.TIMEOUT
    ],
    # Specific strings in messages to suppress
    "suppress_message_contains": [
        "swerex.exceptions",
        "pexpect.exceptions",
        "Retrying LM query",
        "APITimeoutError",
        "Request timed out",
    ],
    # Module/package prefixes to suppress
    "suppress_modules": [
        "swerex.",
        "pexpect.",
    ],
}


class ComprehensiveLogFilter(logging.Filter):
    """
    Comprehensive filter to suppress unwanted logs based on multiple criteria.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SUPPRESS_CONFIG

    def filter(self, record: LogRecord) -> bool:
        # Check logger name prefixes
        for logger_prefix in self.config.get("suppress_loggers", []):
            if record.name.startswith(logger_prefix):
                return False

        # Check module name
        if hasattr(record, "module"):
            for module_prefix in self.config.get("suppress_modules", []):
                if record.module.startswith(module_prefix):
                    return False

        # Check message content
        message = record.getMessage()
        for substring in self.config.get("suppress_message_contains", []):
            if substring in message:
                return False

        # Check exception info
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_type:
                exc_name = exc_type.__name__
                # Check against suppressed exception types
                for exc_pattern in self.config.get("suppress_exceptions", []):
                    if exc_pattern in exc_name:
                        return False

                # Also check the full module path of the exception
                exc_module = exc_type.__module__
                for module_prefix in self.config.get("suppress_modules", []):
                    if exc_module.startswith(module_prefix):
                        return False

        return True


# Apply the comprehensive filter to root logger
logging.getLogger().addFilter(ComprehensiveLogFilter())

# Additionally, you can set specific loggers to higher levels
# to avoid processing their logs at all
for logger_name in SUPPRESS_CONFIG.get("suppress_loggers", []):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)


# Disable printing the patch message to reduce log noise
SaveApplyPatchHook._print_patch_message = lambda *args, **kwargs: None


def setup_agent_logger(agent: DefaultAgent) -> None:
    agent.logger.propagate = False
    agent.logger.handlers = []
    agent.logger.addHandler(
        LangfuseHandler(langfuse_context.get_current_trace_id() or "")
    )


class LangfuseHandler(Handler):
    """
    Custom handler to forward logs to Langfuse
    """

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
