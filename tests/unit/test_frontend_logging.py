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
from art.local.backend import LocalBackend
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
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
        metric_keys = [
            k
            for k in entry.keys()
            if k
            not in [
                "step",
                "recorded_at",
                "training_step",
                "time/wall_clock_sec",
            ]
        ]
        assert all(k.startswith(("val/", "data/")) for k in metric_keys), (
            f"Not all metrics routed into taxonomy namespaces: {metric_keys}"
        )
        assert entry["training_step"] == 0
        assert entry["time/wall_clock_sec"] >= 0

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

        assert entry["val/group_judge_score"] == 0.4

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

    @pytest.mark.asyncio
    async def test_exception_rate_counts_group_exceptions(self, tmp_path: Path):
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
                        reward=0.5,
                        messages_and_choices=[{"role": "user", "content": "test"}],
                    )
                ],
                exceptions=[ValueError("boom")],
            )
        ]

        await model.log(trajectory_groups, split="val")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["val/exception_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_generator_of_trajectories_is_consumed_once(self, tmp_path: Path):
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        def trajectories():
            yield Trajectory(
                reward=1.0,
                metrics={"custom": 1.0},
                messages_and_choices=[{"role": "user", "content": "first"}],
            )
            yield Trajectory(
                reward=3.0,
                metrics={"custom": 3.0},
                messages_and_choices=[{"role": "user", "content": "second"}],
            )

        await model.log(trajectories(), split="val")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["val/reward"] == pytest.approx(2.0)
        assert entry["val/custom"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_train_trajectory_metrics_default_to_train_prefix(
        self, tmp_path: Path
    ):
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
                        metrics={
                            "custom_score": 1.0,
                            "reward/prefixed": 2.0,
                        },
                        messages_and_choices=[{"role": "user", "content": "test"}],
                    )
                ],
                exceptions=[],
            )
        ]

        await model.log(trajectories, split="train")

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["train/reward"] == 0.7
        assert entry["train/exception_rate"] == 0.0
        assert entry["train/custom_score"] == 1.0
        assert entry["reward/prefixed"] == 2.0

    @pytest.mark.asyncio
    async def test_train_logs_add_default_data_metrics_from_trajectory_groups(
        self, tmp_path: Path
    ):
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
                        reward=0.8,
                        messages_and_choices=[{"role": "user", "content": "a"}],
                    ),
                    Trajectory(
                        reward=0.2,
                        messages_and_choices=[{"role": "user", "content": "b"}],
                    ),
                ],
                metadata={"scenario_id": "scenario-1"},
            ),
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.5,
                        messages_and_choices=[{"role": "user", "content": "c"}],
                    )
                ],
                exceptions=[],
                metadata={"scenario_id": "scenario-2"},
            ),
        ]

        await model.log(trajectories, split="train", step=1)

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        merged: dict[str, float] = {}
        for row in rows:
            merged.update(row)

        assert merged["data/step_num_scenarios"] == pytest.approx(2.0)
        assert merged["data/step_num_trajectories"] == pytest.approx(3.0)
        assert merged["data/step_num_groups_submitted"] == pytest.approx(2.0)
        assert merged["data/step_num_groups_trainable"] == pytest.approx(1.0)
        assert merged["data/cum/num_unique_scenarios"] == pytest.approx(2.0)
        assert "train/num_groups_submitted" not in merged
        assert "train/num_groups_trainable" not in merged
        assert "train/num_trajectories" not in merged

    @pytest.mark.asyncio
    async def test_costs_are_logged_in_hierarchical_taxonomy(self, tmp_path: Path):
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        await model.log(
            trajectories=None,
            split="train",
            step=1,
            metrics={
                "costs/train/prefill": 0.2,
                "costs/train/sample": 0.3,
            },
        )
        await model.log(
            trajectories=None,
            split="train",
            step=2,
            metrics={
                "costs/train/prefill": 0.1,
            },
        )

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert first["costs/train/prefill"] == pytest.approx(0.2)
        assert first["costs/train/sample"] == pytest.approx(0.3)
        assert first["costs/train"] == pytest.approx(0.5)
        assert first["costs/all"] == pytest.approx(0.5)
        assert first["costs/cum/all"] == pytest.approx(0.5)

        assert second["costs/train/prefill"] == pytest.approx(0.1)
        assert second["costs/cum/train/prefill"] == pytest.approx(0.3)
        assert second["costs/cum/train"] == pytest.approx(0.6)
        assert second["costs/cum/all"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_cost_cumulative_persists_across_model_recreation(
        self, tmp_path: Path
    ):
        model_1 = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        await model_1.log(
            trajectories=None,
            split="train",
            step=1,
            metrics={"costs/train/prefill": 0.25},
        )

        model_2 = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        await model_2.log(
            trajectories=None,
            split="train",
            step=2,
            metrics={"costs/train/prefill": 0.75},
        )

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert first["costs/cum/train/prefill"] == pytest.approx(0.25)
        assert second["costs/cum/train/prefill"] == pytest.approx(1.0)
        assert second["costs/cum/all"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_metrics_builder_loads_resume_state_before_builder_use(
        self, tmp_path: Path
    ):
        model_1 = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        model_1.metrics_builder().add_data(scenario_ids=["scenario-a"])
        await model_1.log(trajectories=None, split="train", step=1, metrics={})

        model_2 = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        model_2.metrics_builder().add_data(scenario_ids=["scenario-b"])
        await model_2.log(trajectories=None, split="train", step=2, metrics={})

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert first["data/cum/num_unique_scenarios"] == pytest.approx(1.0)
        assert second["data/cum/num_unique_scenarios"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_direct_time_and_data_metrics_get_cumulative_variants(
        self, tmp_path: Path
    ):
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        await model.log(
            trajectories=None,
            split="train",
            step=1,
            metrics={
                "time/step_actor_s": 1.5,
                "data/step_actor_tokens": 10,
            },
        )

        history_path = tmp_path / "test/models/test/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["time/step_actor_s"] == pytest.approx(1.5)
        assert entry["time/cum/actor_s"] == pytest.approx(1.5)
        assert entry["data/step_actor_tokens"] == pytest.approx(10)
        assert entry["data/cum/actor_tokens"] == pytest.approx(10)

    @pytest.mark.asyncio
    async def test_log_without_new_builder_metrics_skips_extra_taxonomy_row(
        self, tmp_path: Path
    ):
        model = Model(
            name="test",
            project="test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        model.metrics_builder().add_data(scenario_ids=["scenario-a"])
        await model.log(
            trajectories=None,
            split="train",
            step=1,
            metrics={
                "time/step_trainer_s": 2.0,
                "data/step_trainer_tokens": 20.0,
            },
        )
        await model.log(
            trajectories=None,
            split="train",
            step=2,
            metrics={"loss/train": 1.0},
        )

        history_path = tmp_path / "test/models/test/history.jsonl"
        rows = [json.loads(line) for line in history_path.open() if line.strip()]

        assert len(rows) == 2
        assert rows[0]["throughput/avg_trainer_tok_per_s"] == pytest.approx(10.0)
        assert rows[0]["data/cum/num_unique_scenarios"] == pytest.approx(1.0)
        assert rows[1]["loss/train"] == pytest.approx(1.0)
        assert "throughput/avg_trainer_tok_per_s" not in rows[1]
        assert "data/cum/num_unique_scenarios" not in rows[1]


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


class TestLocalBackendAutomaticMetrics:
    @pytest.mark.asyncio
    async def test_train_logs_automatic_wall_time_and_gpu_cost(
        self, tmp_path: Path
    ) -> None:
        backend = LocalBackend(gpu_cost_per_hour_usd=3.0)

        with patch("art.model.time.monotonic", side_effect=[100.0, 106.0, 111.0]):
            model = TrainableModel(
                name="test-model",
                project="test-project",
                base_model="Qwen/Qwen3-4B-Instruct-2507",
                base_path=str(tmp_path),
                report_metrics=[],
                _internal_config={"trainer_gpu_ids": [0]},
            )
            model._backend = backend

            await model.log(
                trajectories=None,
                split="train",
                step=1,
                metrics={"loss/train": 1.0},
            )
            await model.log(
                trajectories=None,
                split="train",
                step=2,
                metrics={"loss/train": 0.5},
            )

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        rows = [json.loads(line) for line in history_path.open() if line.strip()]

        first_gpu_cost = 6.0 * 3.0 / 3600.0
        second_gpu_cost = 5.0 * 3.0 / 3600.0

        assert rows[0]["time/step_wall_s"] == pytest.approx(6.0)
        assert rows[0]["costs/gpu"] == pytest.approx(first_gpu_cost)
        assert rows[0]["costs/all"] == pytest.approx(first_gpu_cost)
        assert rows[0]["costs/cum/gpu"] == pytest.approx(first_gpu_cost)

        assert rows[1]["time/step_wall_s"] == pytest.approx(5.0)
        assert rows[1]["costs/gpu"] == pytest.approx(second_gpu_cost)
        assert rows[1]["costs/cum/gpu"] == pytest.approx(
            first_gpu_cost + second_gpu_cost
        )
        assert rows[1]["costs/cum/all"] == pytest.approx(
            first_gpu_cost + second_gpu_cost
        )

    @pytest.mark.asyncio
    async def test_unknown_local_gpu_skips_cost_but_keeps_wall_time(
        self, tmp_path: Path
    ) -> None:
        backend = LocalBackend()

        with patch("art.model.time.monotonic", side_effect=[50.0, 55.0]):
            with patch("art.local.backend.torch.cuda.is_available", return_value=True):
                with patch("art.local.backend.torch.cuda.device_count", return_value=1):
                    with patch(
                        "art.local.backend.torch.cuda.get_device_name",
                        return_value="NVIDIA A100-SXM4-80GB",
                    ):
                        model = TrainableModel(
                            name="test-model",
                            project="test-project",
                            base_model="Qwen/Qwen3-4B-Instruct-2507",
                            base_path=str(tmp_path),
                            report_metrics=[],
                            _internal_config={"trainer_gpu_ids": [0]},
                        )
                        model._backend = backend
                        await model.log(
                            trajectories=None,
                            split="train",
                            step=1,
                            metrics={"loss/train": 1.0},
                        )

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["time/step_wall_s"] == pytest.approx(5.0)
        assert "costs/gpu" not in entry
        assert "costs/all" not in entry


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


class TestTrainSFTMetricsAggregation:
    """Test that train_sft aggregates metrics and logs once (same as RL)."""

    @pytest.mark.asyncio
    async def test_train_sft_aggregates_metrics(self, tmp_path: Path):
        """Verify train_sft aggregates metrics from multiple batches into one log entry."""
        model = TrainableModel(
            name="test-sft",
            project="test-project",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            base_path=str(tmp_path),
        )

        # Mock the backend to yield multiple batch metrics
        mock_backend = MagicMock()

        async def mock_train_sft(*args, **kwargs):
            # Simulate 3 batches with different metrics
            yield {
                "loss/train": 1.0,
                "loss/learning_rate": 1e-4,
                "loss/grad_norm": 0.5,
            }
            yield {
                "loss/train": 0.8,
                "loss/learning_rate": 1e-4,
                "loss/grad_norm": 0.4,
            }
            yield {
                "loss/train": 0.6,
                "loss/learning_rate": 1e-4,
                "loss/grad_norm": 0.3,
            }

        mock_backend._train_sft = mock_train_sft
        mock_backend._get_step = AsyncMock(return_value=1)  # Step after training
        model._backend = mock_backend

        # Create dummy trajectories
        trajectories = [
            Trajectory(
                reward=0.0,
                messages_and_choices=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ],
            )
            for _ in range(3)
        ]

        # Run train_sft
        await model.train_sft(trajectories)

        # Verify history.jsonl has exactly ONE entry (not 3)
        history_path = tmp_path / "test-project/models/test-sft/history.jsonl"
        assert history_path.exists(), "history.jsonl should be created"

        with open(history_path) as f:
            lines = f.readlines()

        assert len(lines) == 1, f"Expected 1 log entry, got {len(lines)}"

        entries = [json.loads(line) for line in lines]
        merged: dict[str, float] = {}
        for entry in entries:
            merged.update(entry)

        assert all(entry["step"] == 1 for entry in entries)
        assert merged["loss/train"] == pytest.approx(0.8)  # (1.0 + 0.8 + 0.6) / 3
        assert merged["loss/grad_norm"] == pytest.approx(0.4)  # (0.5 + 0.4 + 0.3) / 3
        assert merged["time/step_trainer_s"] >= 0
        assert merged["time/cum/trainer_s"] >= 0

    @pytest.mark.asyncio
    async def test_train_sft_single_step_increment(self, tmp_path: Path):
        """Verify train_sft results in single step increment regardless of batch count."""
        model = TrainableModel(
            name="test-sft-step",
            project="test-project",
            base_model="gpt-4",
            base_path=str(tmp_path),
        )

        mock_backend = MagicMock()

        async def mock_train_sft(*args, **kwargs):
            # Simulate 5 batches
            for i in range(5):
                yield {"loss": 1.0 - i * 0.1}

        mock_backend._train_sft = mock_train_sft
        mock_backend._get_step = AsyncMock(return_value=1)  # Step is 1 after training
        model._backend = mock_backend

        trajectories = [
            Trajectory(
                reward=0.0,
                messages_and_choices=[{"role": "user", "content": f"msg{i}"}],
            )
            for i in range(10)
        ]

        await model.train_sft(trajectories)

        # Verify only one log entry at step 1
        history_path = tmp_path / "test-project/models/test-sft-step/history.jsonl"
        df = pl.read_ndjson(str(history_path))

        assert len(df) == 1, "Should have exactly 1 log entry"
        assert set(df["step"].to_list()) == {1}, "Step should be 1 (single increment)"

    @pytest.mark.asyncio
    async def test_train_sft_no_metrics_when_empty(self, tmp_path: Path):
        """Verify train_sft handles empty training gracefully."""
        model = TrainableModel(
            name="test-sft-empty",
            project="test-project",
            base_model="gpt-4",
            base_path=str(tmp_path),
        )

        mock_backend = MagicMock()

        async def mock_train_sft(*args, **kwargs):
            # No batches yielded (empty training)
            return
            yield  # Make it a generator

        mock_backend._train_sft = mock_train_sft
        model._backend = mock_backend

        trajectories = []

        await model.train_sft(trajectories)

        # Verify no history.jsonl created (no metrics to log)
        history_path = tmp_path / "test-project/models/test-sft-empty/history.jsonl"
        assert not history_path.exists(), (
            "No history.jsonl should be created for empty training"
        )


class TestGradientStepMetrics:
    @pytest.mark.asyncio
    async def test_local_backend_train_returns_gradient_step_count(
        self, tmp_path: Path
    ):
        model = TrainableModel(
            name="test-backend-train",
            project="test-project",
            base_model="gpt-4",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        backend = LocalBackend(path=str(tmp_path))

        async def mock_train_model(*args, **kwargs):
            for loss in (1.0, 0.8):
                yield {
                    "loss/train": loss,
                    TRAIN_GRADIENT_STEPS_KEY: 2.0,
                }

        backend._train_model = mock_train_model  # type: ignore[method-assign]
        backend._get_step = AsyncMock(return_value=1)  # type: ignore[method-assign]

        groups = [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=1.0,
                        messages_and_choices=[
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "hi"},
                        ],
                    )
                ]
            )
        ]

        result = await backend.train(model, groups, save_checkpoint=False)

        assert result.metrics[TRAIN_GRADIENT_STEPS_KEY] == pytest.approx(2.0)
