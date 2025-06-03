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


# # Custom filter to suppress swerex and rex-deploy related critical logs
# class SuppressSwerexLogsFilter(logging.Filter):
#     def filter(self, record):
#         # Suppress logs from rex-deploy loggers
#         if record.name.startswith("rex-deploy"):
#             return False
#         # Suppress swerex exception logs
#         if "swerex.exceptions" in record.getMessage() or "swerex.exceptions" in str(
#             record.exc_info
#         ):
#             return False
#         # Suppress CommandTimeoutError and BashIncorrectSyntaxError logs
#         if any(
#             error in record.getMessage()
#             for error in [
#                 "CommandTimeoutError",
#                 "BashIncorrectSyntaxError",
#                 "pexpect.exceptions.TIMEOUT",
#             ]
#         ):
#             return False
#         return True


# # Apply the filter to the root logger to catch all logs
# logging.getLogger().addFilter(SuppressSwerexLogsFilter())

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
