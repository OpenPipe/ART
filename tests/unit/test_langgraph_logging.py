from pathlib import Path

import pytest

pytest.importorskip("langchain_openai")
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from art import Trajectory  # noqa: E402
from art.langgraph.llm_wrapper import create_messages_from_logs  # noqa: E402
from art.langgraph.logging import FileLogger


class NonSerializable:
    pass


def test_file_logger_keeps_structured_logs_in_memory(tmp_path: Path):
    log_path = tmp_path / "rollout"
    logger = FileLogger(str(log_path))
    entry = {"input": NonSerializable(), "output": NonSerializable()}

    logger.log("completion-id", entry)

    assert logger.load_logs() == [("completion-id", entry)]
    assert not log_path.with_suffix(".pkl").exists()
    assert log_path.read_text().startswith("completion-id: ")


def test_file_logger_load_logs_returns_copy(tmp_path: Path):
    logger = FileLogger(str(tmp_path / "rollout"))
    logger.log("completion-id", {"output": "ok"})

    logs = logger.load_logs()
    logs.append(("other-id", {"output": "mutated"}))

    assert logger.load_logs() == [("completion-id", {"output": "ok"})]


def test_create_messages_from_logs_reads_in_memory_entries(tmp_path: Path):
    logger = FileLogger(str(tmp_path / "rollout"))
    logger.log(
        "completion-id",
        {
            "input": [HumanMessage(content="hello")],
            "output": AIMessage(content="hi"),
            "tools": None,
        },
    )

    trajectory = create_messages_from_logs(logger, Trajectory())

    assert trajectory.messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
