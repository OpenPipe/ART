"""
Tests for frontend trajectory logging (Model.log() implementation).

Tests verify:
1. Parquet files written by Model.log() are readable by existing infrastructure
2. history.jsonl format is compatible with existing readers
3. File paths match LocalBackend locations exactly
4. Metrics are calculated and prefixed correctly
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from art import Model, TrainableModel, Trajectory, TrajectoryGroup
from art.utils.trajectory_logging import read_trajectory_groups_parquet


class TestFrontendLoggingCompatibility:
    """Test that trajectories logged via frontend are readable by existing infra."""

    @pytest.fixture
    def sample_trajectories(self) -> list[Trajectory]:
        """Create sample trajectories for testing."""
        return [
            Trajectory(
                reward=0.8,
                metrics={"duration": 5.2, "tokens": 100},
                metadata={"trace_id": "abc-123"},
                messages_and_choices=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                logs=["log1", "log2"],
            ),
            Trajectory(
                reward=0.9,
                metrics={"duration": 3.1, "tokens": 50},
                metadata={"trace_id": "def-456"},
                messages_and_choices=[
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                logs=[],
            ),
        ]

    @pytest.fixture
    def sample_trajectory_groups(
        self, sample_trajectories: list[Trajectory]
    ) -> list[TrajectoryGroup]:
        """Create sample trajectory groups for testing."""
        return [
            TrajectoryGroup(
                trajectories=[sample_trajectories[0]],
                exceptions=[],
            ),
            TrajectoryGroup(
                trajectories=[sample_trajectories[1]],
                exceptions=[],
            ),
        ]

    @pytest.mark.asyncio
    async def test_parquet_readable_by_read_trajectory_groups_parquet(
        self, tmp_path: Path, sample_trajectory_groups: list[TrajectoryGroup]
    ):
        """Direct parquet reader compatibility."""
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        # Mock get_step to return 0 for non-trainable model
        await model.log(sample_trajectory_groups, split="val")

        # Verify readable by existing utility
        parquet_path = (
            tmp_path / "test-project/models/test-model/trajectories/val/0000.parquet"
        )
        assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"

        loaded = read_trajectory_groups_parquet(parquet_path)
        assert len(loaded) == 2
        assert loaded[0].trajectories[0].reward == 0.8
        assert loaded[1].trajectories[0].reward == 0.9

    @pytest.mark.asyncio
    async def test_parquet_schema_preserved(
        self, tmp_path: Path, sample_trajectory_groups: list[TrajectoryGroup]
    ):
        """Verify parquet schema contains expected fields."""
        import pyarrow.parquet as pq

        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        await model.log(sample_trajectory_groups, split="val")

        parquet_path = (
            tmp_path / "test-project/models/test-model/trajectories/val/0000.parquet"
        )
        table = pq.read_table(parquet_path)

        # Check expected columns exist
        expected_columns = [
            "group_index",
            "group_metadata",
            "group_metrics",
            "group_logs",
            "reward",
            "metrics",
            "metadata",
            "tools",
            "logs",
            "messages",
        ]
        for col in expected_columns:
            assert col in table.column_names, f"Missing column: {col}"


class TestHistoryJsonlCompatibility:
    """Test history.jsonl format compatibility."""

    @pytest.fixture
    def sample_trajectory_groups(self) -> list[TrajectoryGroup]:
        """Create sample trajectory groups for testing."""
        return [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.8,
                        metrics={"custom_metric": 42.0},
                        messages_and_choices=[
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi!"},
                        ],
                    ),
                    Trajectory(
                        reward=0.6,
                        metrics={"custom_metric": 38.0},
                        messages_and_choices=[
                            {"role": "user", "content": "Bye"},
                            {"role": "assistant", "content": "Goodbye!"},
                        ],
                    ),
                ],
                exceptions=[],
            )
        ]

    @pytest.mark.asyncio
    async def test_history_jsonl_format(
        self, tmp_path: Path, sample_trajectory_groups: list[TrajectoryGroup]
    ):
        """Verify history.jsonl has correct format for downstream readers."""
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        await model.log(sample_trajectory_groups, split="val")

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        assert history_path.exists()

        with open(history_path) as f:
            entry = json.loads(f.readline())

        # Verify required fields
        assert "step" in entry
        assert "recorded_at" in entry
        assert "val/reward" in entry  # Prefixed metric

    @pytest.mark.asyncio
    async def test_history_readable_by_polars(
        self, tmp_path: Path, sample_trajectory_groups: list[TrajectoryGroup]
    ):
        """Verify history.jsonl is readable by pl.read_ndjson (used by delete_checkpoints)."""
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        await model.log(sample_trajectory_groups, split="val")

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        df = pl.read_ndjson(str(history_path))

        assert "step" in df.columns
        assert "val/reward" in df.columns
        assert "val/reward_std_dev" in df.columns

    @pytest.mark.asyncio
    async def test_history_appends_entries(
        self, tmp_path: Path, sample_trajectory_groups: list[TrajectoryGroup]
    ):
        """Verify multiple log calls append to history.jsonl."""
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        # Log twice
        await model.log(sample_trajectory_groups, split="val")
        await model.log(sample_trajectory_groups, split="train")

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        df = pl.read_ndjson(str(history_path))

        # Should have 2 entries
        assert len(df) == 2

        # Check both splits are present
        columns = df.columns
        assert any("val/" in col for col in columns)
        assert any("train/" in col for col in columns)


class TestPathStructure:
    """Test that file paths match LocalBackend locations exactly."""

    @pytest.mark.asyncio
    async def test_file_locations_match_localbackend(self, tmp_path: Path):
        """Verify files are written to expected paths."""
        model = Model(
            name="mymodel",
            project="myproj",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        trajectories = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        messages_and_choices=[{"role": "user", "content": "test"}],
                    )
                ],
                exceptions=[],
            )
        ]

        await model.log(trajectories, split="val")

        # Verify exact paths
        assert (
            tmp_path / "myproj/models/mymodel/trajectories/val/0000.parquet"
        ).exists()
        assert (tmp_path / "myproj/models/mymodel/history.jsonl").exists()

    @pytest.mark.asyncio
    async def test_step_numbering_format(self, tmp_path: Path):
        """Verify step numbers are zero-padded to 4 digits."""
        # Create a mock trainable model with step > 0
        model = TrainableModel(
            name="mymodel",
            project="myproj",
            base_model="gpt-4",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        # Mock the backend and get_step
        mock_backend = MagicMock()
        mock_backend._get_step = AsyncMock(return_value=42)
        model._backend = mock_backend

        trajectories = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        messages_and_choices=[{"role": "user", "content": "test"}],
                    )
                ],
                exceptions=[],
            )
        ]

        await model.log(trajectories, split="train")

        # Verify zero-padded step in filename
        assert (
            tmp_path / "myproj/models/mymodel/trajectories/train/0042.parquet"
        ).exists()


class TestMetricCalculation:
    """Test metric calculation and formatting."""

    @pytest.mark.asyncio
    async def test_metric_prefixes(self, tmp_path: Path):
        """Verify metrics are prefixed with split name."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        trajectories = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.7,
                        metrics={"custom": 1.0},
                        messages_and_choices=[{"role": "user", "content": "test"}],
                    )
                ],
                exceptions=[],
            )
        ]

        await model.log(trajectories, split="val")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        # All metrics should be prefixed (except step and recorded_at)
        metric_keys = [k for k in entry.keys() if k not in ["step", "recorded_at"]]
        assert all(k.startswith("val/") for k in metric_keys), (
            f"Not all metrics prefixed: {metric_keys}"
        )

    @pytest.mark.asyncio
    async def test_standard_metrics_present(self, tmp_path: Path):
        """Verify standard metrics (reward, exception_rate, reward_std_dev) are computed."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        trajectory_groups = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.8,
                        messages_and_choices=[{"role": "user", "content": "test1"}],
                    ),
                    Trajectory(
                        reward=0.6,
                        messages_and_choices=[{"role": "user", "content": "test2"}],
                    ),
                ],
                exceptions=[],
            )
        ]

        await model.log(trajectory_groups, split="val")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert "val/reward" in entry
        assert "val/exception_rate" in entry
        assert "val/reward_std_dev" in entry

        # Check reward average is correct
        assert entry["val/reward"] == 0.7  # (0.8 + 0.6) / 2

    @pytest.mark.asyncio
    async def test_group_metric_aggregation(self, tmp_path: Path):
        """Verify group-level metrics are aggregated once per group."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        trajectory_groups = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.8,
                        messages_and_choices=[{"role": "user", "content": "a"}],
                    )
                ],
                metrics={"judge_score": 0.2},
                exceptions=[],
            ),
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.6,
                        messages_and_choices=[{"role": "user", "content": "b"}],
                    )
                ],
                metrics={"judge_score": 0.6},
                exceptions=[],
            ),
        ]

        await model.log(trajectory_groups, split="val")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["val/group_metric_judge_score"] == 0.4

    @pytest.mark.asyncio
    async def test_exception_rate_calculation(self, tmp_path: Path):
        """Verify exception_rate is calculated correctly for successful trajectories."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        # TrajectoryGroup stores trajectories and exceptions separately
        # The Model.log() iterates over the group which yields trajectories and exceptions
        trajectory_groups = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        messages_and_choices=[{"role": "user", "content": "test"}],
                    )
                ],
                exceptions=[],
            )
        ]

        await model.log(trajectory_groups, split="val")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        # All successful trajectories = 0% exception rate
        assert entry["val/exception_rate"] == 0.0


class TestWandbIntegration:
    """Test wandb integration logic (without mocking wandb itself)."""

    @pytest.mark.asyncio
    async def test_wandb_not_called_without_api_key(self, tmp_path: Path):
        """Verify _get_wandb_run returns None without WANDB_API_KEY."""
        # Ensure WANDB_API_KEY is not set
        env_backup = os.environ.get("WANDB_API_KEY")
        if "WANDB_API_KEY" in os.environ:
            del os.environ["WANDB_API_KEY"]

        try:
            model = Model(
                name="test",
                project="test",
                base_path=str(tmp_path),
            )

            # Verify _get_wandb_run returns None when no API key
            result = model._get_wandb_run()
            assert result is None
        finally:
            if env_backup is not None:
                os.environ["WANDB_API_KEY"] = env_backup

    def test_should_log_wandb_logic_default(self, tmp_path: Path):
        """Test the should_log_wandb logic with default report_metrics."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=None,  # Default
        )

        # With no API key and default report_metrics, should not log
        env_backup = os.environ.get("WANDB_API_KEY")
        if "WANDB_API_KEY" in os.environ:
            del os.environ["WANDB_API_KEY"]
        try:
            should_log = (
                model.report_metrics is None and "WANDB_API_KEY" in os.environ
            ) or (model.report_metrics is not None and "wandb" in model.report_metrics)
            assert should_log is False
        finally:
            if env_backup is not None:
                os.environ["WANDB_API_KEY"] = env_backup

    def test_should_log_wandb_logic_with_key(self, tmp_path: Path):
        """Test the should_log_wandb logic with WANDB_API_KEY present."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=None,  # Default
        )

        # With API key and default report_metrics, should log
        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
            should_log = (
                model.report_metrics is None and "WANDB_API_KEY" in os.environ
            ) or (model.report_metrics is not None and "wandb" in model.report_metrics)
            assert should_log is True

    def test_should_log_wandb_logic_explicit_wandb(self, tmp_path: Path):
        """Test the should_log_wandb logic with explicit wandb in report_metrics."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=["wandb"],
        )

        # With explicit wandb in report_metrics, should log regardless of env var
        should_log = (
            model.report_metrics is None and "WANDB_API_KEY" in os.environ
        ) or (model.report_metrics is not None and "wandb" in model.report_metrics)
        assert should_log is True

    def test_should_log_wandb_logic_empty_list(self, tmp_path: Path):
        """Test the should_log_wandb logic with empty report_metrics list."""
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],  # Explicit empty list
        )

        # With empty report_metrics, should not log even with API key
        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
            should_log = (
                model.report_metrics is None and "WANDB_API_KEY" in os.environ
            ) or (model.report_metrics is not None and "wandb" in model.report_metrics)
            assert should_log is False


class TestModelAttributes:
    """Test new Model attributes."""

    def test_base_path_default(self):
        """Verify base_path defaults to '.art'."""
        model = Model(name="test", project="test")
        assert model.base_path == ".art"

    def test_base_path_custom(self):
        """Verify base_path can be customized."""
        model = Model(name="test", project="test", base_path="/custom/path")
        assert model.base_path == "/custom/path"

    def test_report_metrics_default(self):
        """Verify report_metrics defaults to None."""
        model = Model(name="test", project="test")
        assert model.report_metrics is None

    def test_report_metrics_custom(self):
        """Verify report_metrics can be customized."""
        model = Model(name="test", project="test", report_metrics=["wandb", "custom"])
        assert model.report_metrics == ["wandb", "custom"]
