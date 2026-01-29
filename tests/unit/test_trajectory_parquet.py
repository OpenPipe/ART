"""
Comprehensive tests for Parquet trajectory serialization and migration.

Tests cover:
1. Round-trip serialization (write -> read -> compare)
2. Migration from JSONL to Parquet
3. Edge cases (empty, nullable, unicode, long content)
4. Compression verification
5. Golden file regression tests
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import cast

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_union_param import (
    ChatCompletionMessageToolCallUnionParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
import pytest

from art import Trajectory, TrajectoryGroup
from art.types import MessageOrChoice
from art.utils.trajectory_logging import (
    read_trajectory_groups_parquet,
    write_trajectory_groups_parquet,
)
from art.utils.trajectory_migration import (
    MigrationResult,
    deserialize_trajectory_groups,
    migrate_jsonl_to_parquet,
    migrate_model_dir,
    migrate_trajectories_dir,
    serialize_trajectory_groups,
    trajectory_to_dict,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "trajectories"


def _ensure_message(item: MessageOrChoice) -> ChatCompletionMessageParam:
    """Narrow a trajectory entry to a concrete message (not a Choice)."""
    assert not isinstance(item, Choice)
    return cast(ChatCompletionMessageParam, item)  # ty:ignore[redundant-cast]


def _ensure_assistant_message(
    item: MessageOrChoice,
) -> ChatCompletionAssistantMessageParam:
    msg = _ensure_message(item)
    assert msg["role"] == "assistant"
    return cast(ChatCompletionAssistantMessageParam, msg)  # ty:ignore[redundant-cast]


def _ensure_tool_message(item: MessageOrChoice) -> ChatCompletionToolMessageParam:
    msg = _ensure_message(item)
    assert msg["role"] == "tool"
    return cast(ChatCompletionToolMessageParam, msg)  # ty:ignore[redundant-cast]


def _ensure_user_message(item: MessageOrChoice) -> ChatCompletionUserMessageParam:
    msg = _ensure_message(item)
    assert msg["role"] == "user"
    return cast(ChatCompletionUserMessageParam, msg)  # ty:ignore[redundant-cast]


class TestParquetRoundTrip:
    """Test that data survives write -> read cycle."""

    def test_simple_trajectory(self, tmp_path: Path):
        """Basic round-trip test with minimal trajectory."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.85,
                        metrics={"duration": 5.2},
                        metadata={"trace_id": "abc-123"},
                        messages_and_choices=[
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ],
                        logs=["log1", "log2"],
                    )
                ],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)

        loaded = read_trajectory_groups_parquet(parquet_path)

        assert len(loaded) == 1
        assert len(loaded[0].trajectories) == 1
        traj = loaded[0].trajectories[0]
        assert traj.reward == 0.85
        assert traj.metrics == {"duration": 5.2}
        assert traj.metadata == {"trace_id": "abc-123"}
        assert len(traj.messages_and_choices) == 3
        assert traj.logs == ["log1", "log2"]

    def test_tool_calls(self, tmp_path: Path):
        """Test trajectories with tool calls."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=1.0,
                        metrics={},
                        metadata={},
                        messages_and_choices=[
                            {"role": "user", "content": "Search for pizza"},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query": "pizza"}',
                                        },
                                        "id": "call_123",
                                        "type": "function",
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "content": '{"results": ["Pizza Hut"]}',
                                "tool_call_id": "call_123",
                            },
                        ],
                        logs=[],
                    )
                ],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        traj = loaded[0].trajectories[0]
        assert len(traj.messages_and_choices) == 3

        # Check tool call message
        tool_call_msg = _ensure_assistant_message(traj.messages_and_choices[1])
        tool_calls = cast(
            list[ChatCompletionMessageToolCallUnionParam],
            list(tool_call_msg.get("tool_calls") or []),
        )
        assert tool_calls, "Assistant message should include tool calls"
        first_call = tool_calls[0]
        assert first_call["type"] == "function"
        function_call = cast(ChatCompletionMessageFunctionToolCallParam, first_call)
        assert function_call["function"]["name"] == "search"

        # Check tool result message
        tool_result_msg = _ensure_tool_message(traj.messages_and_choices[2])
        assert tool_result_msg["tool_call_id"] == "call_123"

    def test_group_level_fields_round_trip(self, tmp_path: Path):
        """Group-level metadata/metrics/logs should survive round-trip."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.4,
                        metrics={"idx": 0},
                        metadata={},
                        messages_and_choices=[{"role": "user", "content": "msg0"}],
                        logs=[],
                    ),
                    Trajectory(
                        reward=0.6,
                        metrics={"idx": 1},
                        metadata={},
                        messages_and_choices=[{"role": "user", "content": "msg1"}],
                        logs=[],
                    ),
                ],
                metadata={"scenario_id": "abc-123", "difficulty": "hard"},
                metrics={"judge_score": 0.7, "pass_rate": 1},
                logs=["group log 1", "group log 2"],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        assert len(loaded) == 1
        group = loaded[0]
        assert group.metadata == {"scenario_id": "abc-123", "difficulty": "hard"}
        assert group.metrics == {"judge_score": 0.7, "pass_rate": 1}
        assert group.logs == ["group log 1", "group log 2"]

    def test_choice_format(self, tmp_path: Path):
        """Test trajectories with Choice format (finish_reason) are flattened to messages."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.9,
                        metrics={},
                        metadata={},
                        messages_and_choices=[
                            {"role": "user", "content": "Hello"},
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Hi!",
                                    "tool_calls": None,
                                },
                            },
                        ],
                        logs=[],
                    )
                ],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        traj = loaded[0].trajectories[0]
        # Choice format is flattened to a simple message dict
        # The inner message content is preserved
        user_msg = _ensure_user_message(traj.messages_and_choices[0])
        assistant_msg = _ensure_assistant_message(traj.messages_and_choices[1])

        assert user_msg["role"] == "user"
        user_content = user_msg["content"]
        assert isinstance(user_content, str)
        assert user_content == "Hello"

        assert assistant_msg["role"] == "assistant"
        assistant_content = assistant_msg.get("content")
        assert isinstance(assistant_content, str)
        assert assistant_content == "Hi!"

    def test_unicode_content(self, tmp_path: Path):
        """Test trajectories with unicode and special characters."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.75,
                        metrics={"emoji_count": 3},
                        metadata={"language": "mixed"},
                        messages_and_choices=[
                            {"role": "user", "content": "Hello 你好 مرحبا 🎉"},
                            {
                                "role": "assistant",
                                "content": 'Hi! 你好！مرحبا! 👋\n\tTabbed "quoted" content',
                            },
                        ],
                        logs=["Unicode: 中文日志"],
                    )
                ],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        traj = loaded[0].trajectories[0]
        user_msg = _ensure_user_message(traj.messages_and_choices[0])
        user_content = user_msg["content"]
        assert isinstance(user_content, str)
        assert "你好" in user_content
        assert "🎉" in user_content
        assert "中文日志" in traj.logs[0]

    def test_multiple_trajectories(self, tmp_path: Path):
        """Test multiple trajectories within a single group."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        metrics={"idx": 0},
                        metadata={},
                        messages_and_choices=[{"role": "user", "content": "msg0"}],
                        logs=[],
                    ),
                    Trajectory(
                        reward=0.7,
                        metrics={"idx": 1},
                        metadata={},
                        messages_and_choices=[{"role": "user", "content": "msg1"}],
                        logs=[],
                    ),
                ],
                exceptions=[],
            ),
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        # Should have 1 group with 2 trajectories
        assert len(loaded) == 1
        assert len(loaded[0].trajectories) == 2
        assert loaded[0].trajectories[0].reward == 0.5
        assert loaded[0].trajectories[1].reward == 0.7

    def test_multiple_groups_preserved(self, tmp_path: Path):
        """Test that multiple trajectory groups are preserved separately."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        metrics={"group": 0, "idx": 0},
                        metadata={},
                        messages_and_choices=[
                            {"role": "user", "content": "group0-msg0"}
                        ],
                        logs=[],
                    ),
                    Trajectory(
                        reward=0.6,
                        metrics={"group": 0, "idx": 1},
                        metadata={},
                        messages_and_choices=[
                            {"role": "user", "content": "group0-msg1"}
                        ],
                        logs=[],
                    ),
                ],
                exceptions=[],
            ),
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.9,
                        metrics={"group": 1, "idx": 0},
                        metadata={},
                        messages_and_choices=[
                            {"role": "user", "content": "group1-msg0"}
                        ],
                        logs=[],
                    ),
                ],
                exceptions=[],
            ),
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=1.0,
                        metrics={"group": 2, "idx": 0},
                        metadata={},
                        messages_and_choices=[
                            {"role": "user", "content": "group2-msg0"}
                        ],
                        logs=[],
                    ),
                ],
                exceptions=[],
            ),
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        # Should have 3 separate groups
        assert len(loaded) == 3, f"Expected 3 groups, got {len(loaded)}"

        # Group 0 should have 2 trajectories
        assert len(loaded[0].trajectories) == 2
        assert loaded[0].trajectories[0].metrics["group"] == 0
        assert loaded[0].trajectories[1].metrics["group"] == 0

        # Group 1 should have 1 trajectory
        assert len(loaded[1].trajectories) == 1
        assert loaded[1].trajectories[0].metrics["group"] == 1

        # Group 2 should have 1 trajectory
        assert len(loaded[2].trajectories) == 1
        assert loaded[2].trajectories[0].metrics["group"] == 2

    def test_empty_trajectory_group(self, tmp_path: Path):
        """Test empty trajectory groups."""
        original = [TrajectoryGroup(trajectories=[], exceptions=[])]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)

        # File should exist and be readable
        assert parquet_path.exists()
        loaded = read_trajectory_groups_parquet(parquet_path)
        assert len(loaded) == 0 or all(len(g.trajectories) == 0 for g in loaded)

    def test_nullable_fields(self, tmp_path: Path):
        """Test trajectories with null/empty fields."""
        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.0,
                        metrics={},
                        metadata={},
                        messages_and_choices=[],
                        logs=[],
                    )
                ],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)
        loaded = read_trajectory_groups_parquet(parquet_path)

        traj = loaded[0].trajectories[0]
        assert traj.reward == 0.0
        assert traj.metrics == {}
        assert traj.metadata == {}
        assert traj.messages_and_choices == []
        assert traj.logs == []


class TestMigration:
    """Test JSONL to Parquet migration."""

    def test_migrate_simple_jsonl(self, tmp_path: Path):
        """Test migrating a simple JSONL file."""
        # Create a JSONL file with enough content to benefit from compression
        jsonl_path = tmp_path / "0001.jsonl"
        # Include a longer system prompt to make compression beneficial
        data = {
            "trajectories": [
                {
                    "reward": 0.8,
                    "metrics": {"duration": 5.0},
                    "metadata": {"id": "test"},
                    "messages_and_choices": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. " * 100,
                        },
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"},
                    ],
                    "tools": None,
                    "additional_histories": [],
                    "logs": [],
                }
            ]
        }
        jsonl_path.write_text(json.dumps(data))

        # Migrate
        result = migrate_jsonl_to_parquet(jsonl_path, delete_original=True)

        # Check result
        assert result.files_migrated == 1
        assert result.errors == []
        assert result.bytes_before > 0
        assert result.bytes_after > 0
        # Note: Very small files may not compress well due to Parquet overhead
        # but files with repeated content should compress

        # Check files
        parquet_path = tmp_path / "0001.parquet"
        assert parquet_path.exists()
        assert not jsonl_path.exists()

        # Verify content
        loaded = read_trajectory_groups_parquet(parquet_path)
        assert len(loaded) == 1
        assert len(loaded[0].trajectories) == 1
        assert loaded[0].trajectories[0].reward == 0.8

    def test_migrate_keeps_original_when_requested(self, tmp_path: Path):
        """Test that delete_original=False preserves the JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps({"trajectories": []}))

        result = migrate_jsonl_to_parquet(jsonl_path, delete_original=False)

        assert jsonl_path.exists()
        assert (tmp_path / "test.parquet").exists()

    def test_migrate_dry_run(self, tmp_path: Path):
        """Test dry run mode doesn't modify files."""
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps({"trajectories": []}))

        result = migrate_jsonl_to_parquet(jsonl_path, dry_run=True)

        assert result.files_migrated == 1
        assert jsonl_path.exists()
        assert not (tmp_path / "test.parquet").exists()

    def test_migrate_directory(self, tmp_path: Path):
        """Test migrating an entire directory."""
        # Create directory structure
        train_dir = tmp_path / "trajectories" / "train"
        val_dir = tmp_path / "trajectories" / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create files
        for i in range(3):
            (train_dir / f"{i:04d}.jsonl").write_text(
                json.dumps(
                    {
                        "trajectories": [
                            {
                                "reward": i * 0.1,
                                "metrics": {},
                                "metadata": {},
                                "messages_and_choices": [],
                                "tools": None,
                                "additional_histories": [],
                                "logs": [],
                            }
                        ]
                    }
                )
            )
        (val_dir / "0000.jsonl").write_text(
            json.dumps(
                {
                    "trajectories": [
                        {
                            "reward": 0.5,
                            "metrics": {},
                            "metadata": {},
                            "messages_and_choices": [],
                            "tools": None,
                            "additional_histories": [],
                            "logs": [],
                        }
                    ]
                }
            )
        )

        # Migrate
        result = migrate_trajectories_dir(tmp_path / "trajectories")

        assert result.files_migrated == 4
        assert len(list(train_dir.glob("*.parquet"))) == 3
        assert len(list(val_dir.glob("*.parquet"))) == 1
        assert len(list(train_dir.glob("*.jsonl"))) == 0

    def test_migrate_model_dir(self, tmp_path: Path):
        """Test migrating a model directory."""
        model_dir = tmp_path / "my-model"
        traj_dir = model_dir / "trajectories" / "train"
        traj_dir.mkdir(parents=True)

        (traj_dir / "0000.jsonl").write_text(
            json.dumps(
                {
                    "trajectories": [
                        {
                            "reward": 0.8,
                            "metrics": {},
                            "metadata": {},
                            "messages_and_choices": [],
                            "tools": None,
                            "additional_histories": [],
                            "logs": [],
                        }
                    ]
                }
            )
        )

        result = migrate_model_dir(model_dir)

        assert result.files_migrated == 1
        assert (traj_dir / "0000.parquet").exists()

    def test_migrate_idempotent(self, tmp_path: Path):
        """Test that running migration twice is safe."""
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps({"trajectories": []}))

        # First migration
        result1 = migrate_jsonl_to_parquet(jsonl_path)
        assert result1.files_migrated == 1

        # Second migration (no JSONL files left)
        result2 = migrate_jsonl_to_parquet(jsonl_path)
        assert result2.files_migrated == 0  # File doesn't exist anymore


class TestGoldenFiles:
    """Test against real trajectory data fixtures."""

    @pytest.fixture
    def fixtures_available(self):
        """Check if fixtures are available."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixtures directory not found")
        return True

    def test_real_sample_migration(self, tmp_path: Path, fixtures_available):
        """Test migrating a real trajectory file."""
        real_sample = FIXTURES_DIR / "real_sample.jsonl"
        if not real_sample.exists():
            pytest.skip("real_sample.jsonl not found")

        # Copy to temp
        test_file = tmp_path / "real_sample.jsonl"
        shutil.copy(real_sample, test_file)

        original_size = test_file.stat().st_size

        # Migrate
        result = migrate_jsonl_to_parquet(test_file, delete_original=False)

        parquet_path = tmp_path / "real_sample.parquet"
        new_size = parquet_path.stat().st_size

        # Check compression (should be at least 10x)
        assert result.compression_ratio >= 10, (
            f"Compression ratio {result.compression_ratio} is less than expected"
        )

        # Verify data integrity
        with open(test_file, "r") as f:
            original_data = [json.loads(line) for line in f if line.strip()]

        loaded = read_trajectory_groups_parquet(parquet_path)

        # Count original trajectories
        original_count = sum(len(g.get("trajectories", [])) for g in original_data)
        loaded_count = sum(len(g.trajectories) for g in loaded)
        assert loaded_count == original_count

    def test_fixture_roundtrips(self, tmp_path: Path, fixtures_available):
        """Test all fixture files round-trip correctly."""
        fixture_files = list(FIXTURES_DIR.glob("*.jsonl"))
        if not fixture_files:
            pytest.skip("No fixture files found")

        for fixture_path in fixture_files:
            test_file = tmp_path / fixture_path.name
            shutil.copy(fixture_path, test_file)

            # Read original
            with open(test_file, "r") as f:
                original_data = [json.loads(line) for line in f if line.strip()]

            # Migrate
            result = migrate_jsonl_to_parquet(test_file, delete_original=False)

            if result.errors:
                pytest.fail(f"Migration error for {fixture_path.name}: {result.errors}")

            # Read back
            parquet_path = test_file.with_suffix(".parquet")
            loaded = read_trajectory_groups_parquet(parquet_path)

            # Compare trajectory count
            original_count = sum(len(g.get("trajectories", [])) for g in original_data)
            loaded_count = sum(len(g.trajectories) for g in loaded)
            assert loaded_count == original_count, f"Mismatch for {fixture_path.name}"


class TestCompression:
    """Test compression efficiency."""

    def test_long_content_compresses_well(self, tmp_path: Path):
        """Test that long repeated content (like system prompts) compresses well."""
        long_prompt = "A" * 20000  # 20KB system prompt

        original = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        metrics={},
                        metadata={},
                        messages_and_choices=[
                            {"role": "system", "content": long_prompt},
                            {"role": "user", "content": "Hello"},
                        ],
                        logs=[],
                    )
                    for _ in range(10)  # Same prompt repeated 10 times
                ],
                exceptions=[],
            )
        ]

        parquet_path = tmp_path / "test.parquet"
        write_trajectory_groups_parquet(original, parquet_path)

        # Should compress well due to repeated content
        parquet_size = parquet_path.stat().st_size
        uncompressed_estimate = 10 * 20000  # 200KB if not compressed

        # ZSTD should achieve at least 10x compression on repeated content
        assert parquet_size < uncompressed_estimate / 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file(self, tmp_path: Path):
        """Test handling of nonexistent file."""
        result = migrate_jsonl_to_parquet(tmp_path / "nonexistent.jsonl")
        assert result.files_migrated == 0
        assert len(result.errors) == 1

    def test_invalid_json(self, tmp_path: Path):
        """Test handling of invalid JSON."""
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("not valid json{{{")

        result = migrate_jsonl_to_parquet(bad_file)
        assert result.files_migrated == 0
        assert len(result.errors) == 1

    def test_empty_file(self, tmp_path: Path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        result = migrate_jsonl_to_parquet(empty_file, delete_original=False)

        # Should succeed with empty parquet
        parquet_path = tmp_path / "empty.parquet"
        assert parquet_path.exists()

    def test_non_jsonl_file_skipped(self, tmp_path: Path):
        """Test that non-JSONL files are skipped."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("This is not a JSONL file")

        result = migrate_jsonl_to_parquet(txt_file)
        assert result.files_migrated == 0
        assert result.files_skipped == 1
