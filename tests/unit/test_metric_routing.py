import json
import os
from pathlib import Path
import types
from unittest.mock import MagicMock, patch

import pytest

from art import Model


class TestMetricRoutingBaseline:
    def test_log_metrics_routes_known_sections_without_split_prefix(
        self, tmp_path: Path
    ) -> None:
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        model._log_metrics(
            {
                "reward/mean": 0.9,
                "custom": 1.0,
                "checkpoint/foo": 1.5,
                "rewardish/value": 2.0,
            },
            split="train",
            step=7,
        )

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["reward/mean"] == 0.9
        assert entry["train/custom"] == 1.0
        assert entry["train/checkpoint/foo"] == 1.5
        assert entry["train/rewardish/value"] == 2.0
        assert entry["training_step"] == 7
        assert entry["time/wall_clock_sec"] >= 0

    def test_get_wandb_run_registers_taxonomy_sections(self, tmp_path: Path) -> None:
        fake_run = MagicMock()
        fake_run._is_finished = False

        fake_wandb = types.SimpleNamespace()
        fake_wandb.init = MagicMock(return_value=fake_run)
        fake_wandb.define_metric = MagicMock()
        fake_wandb.Settings = lambda **kwargs: kwargs

        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"wandb": fake_wandb}):
                model = Model(
                    name="test-model",
                    project="test-project",
                    base_path=str(tmp_path),
                )
                run = model._get_wandb_run()

        assert run is fake_run
        define_calls = [
            (call.args, call.kwargs) for call in fake_run.define_metric.call_args_list
        ]
        assert define_calls == [
            (("training_step",), {}),
            (("time/wall_clock_sec",), {}),
            (("sft/gradient_step",), {}),
            (("reward/*",), {"step_metric": "training_step"}),
            (("loss/*",), {"step_metric": "training_step"}),
            (("throughput/*",), {"step_metric": "training_step"}),
            (("costs/*",), {"step_metric": "training_step"}),
            (("time/*",), {"step_metric": "training_step"}),
            (("data/*",), {"step_metric": "training_step"}),
            (("sft/*",), {"step_metric": "sft/gradient_step"}),
            (("train/*",), {"step_metric": "training_step"}),
            (("val/*",), {"step_metric": "training_step"}),
            (("test/*",), {"step_metric": "training_step"}),
            (("discarded/*",), {"step_metric": "training_step"}),
        ]

    def test_log_metrics_defines_nested_cost_keys_with_training_step(
        self, tmp_path: Path
    ) -> None:
        fake_run = MagicMock()
        fake_run._is_finished = False
        fake_run.config = MagicMock()

        fake_wandb = types.SimpleNamespace()
        fake_wandb.init = MagicMock(return_value=fake_run)
        fake_wandb.define_metric = MagicMock()
        fake_wandb.Settings = lambda **kwargs: kwargs

        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"wandb": fake_wandb}):
                model = Model(
                    name="test-model",
                    project="test-project",
                    base_path=str(tmp_path),
                    report_metrics=["wandb"],
                )
                model._log_metrics(
                    {
                        "costs/train/sample": 0.1,
                        "costs/cum/train/prefill": 0.2,
                    },
                    split="train",
                    step=1,
                )

        define_calls = [
            (call.args, call.kwargs) for call in fake_run.define_metric.call_args_list
        ]
        assert (
            ("costs/train/sample",),
            {"step_metric": "training_step"},
        ) in define_calls
        assert (
            ("costs/cum/train/prefill",),
            {"step_metric": "training_step"},
        ) in define_calls
        fake_run.log.assert_called_once()
        logged_metrics = fake_run.log.call_args.args[0]
        assert logged_metrics["costs/train/sample"] == 0.1
        assert logged_metrics["costs/cum/train/prefill"] == 0.2
        assert logged_metrics["training_step"] == 1
        assert "time/wall_clock_sec" in logged_metrics
        assert fake_run.log.call_args.kwargs == {}

    def test_update_wandb_config_seeds_wandb_init(self, tmp_path: Path) -> None:
        fake_run = MagicMock()
        fake_run._is_finished = False
        fake_run.config = MagicMock()

        fake_wandb = types.SimpleNamespace()
        fake_wandb.init = MagicMock(return_value=fake_run)
        fake_wandb.define_metric = MagicMock()
        fake_wandb.Settings = lambda **kwargs: kwargs

        payload = {
            "experiment": {"learning_rate": 1e-5, "batch_size": 4},
            "dataset": {"split": "train"},
        }

        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"wandb": fake_wandb}):
                model = Model(
                    name="test-model",
                    project="test-project",
                    base_path=str(tmp_path),
                )
                model.update_wandb_config(payload)
                run = model._get_wandb_run()

        assert run is fake_run
        init_kwargs = fake_wandb.init.call_args.kwargs
        assert init_kwargs["config"] == payload
        assert "allow_val_change" not in init_kwargs
        fake_run.config.update.assert_called_once_with(payload)

    def test_update_wandb_config_updates_active_run(self, tmp_path: Path) -> None:
        fake_run = MagicMock()
        fake_run._is_finished = False
        fake_run.config = MagicMock()

        fake_wandb = types.SimpleNamespace()
        fake_wandb.init = MagicMock(return_value=fake_run)
        fake_wandb.define_metric = MagicMock()
        fake_wandb.Settings = lambda **kwargs: kwargs

        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"wandb": fake_wandb}):
                model = Model(
                    name="test-model",
                    project="test-project",
                    base_path=str(tmp_path),
                )
                model.update_wandb_config({"experiment": {"learning_rate": 1e-5}})
                _ = model._get_wandb_run()
                fake_run.config.update.reset_mock()

                model.update_wandb_config(
                    {"experiment": {"learning_rate": 1e-5, "batch_size": 8}},
                )
                model.update_wandb_config(
                    {"dataset": {"split": "train"}},
                )

        assert fake_run.config.update.call_count == 2
        assert fake_run.config.update.call_args_list[0].args == (
            {"experiment": {"learning_rate": 1e-5, "batch_size": 8}},
        )
        assert fake_run.config.update.call_args_list[1].args == (
            {
                "experiment": {"learning_rate": 1e-5, "batch_size": 8},
                "dataset": {"split": "train"},
            },
        )

    def test_update_wandb_config_rejects_conflicting_values(
        self, tmp_path: Path
    ) -> None:
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
        )

        model.update_wandb_config({"experiment": {"learning_rate": 1e-5}})

        with pytest.raises(
            ValueError,
            match="Conflicting value for 'experiment.learning_rate'",
        ):
            model.update_wandb_config({"experiment": {"learning_rate": 2e-5}})
