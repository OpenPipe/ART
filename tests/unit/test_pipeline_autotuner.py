import pytest

from art.pipeline_tuner import PipelineAutotuneConfig
from art.pipeline_tuner.autotune import (
    PipelineAutotuner,
    _group_vllm_metric_rows,
    _vllm_load_stats,
    build_initial_settings,
    recommended_queue_size,
)
from art.pipeline_tuner.config import (
    PipelineMetric,
    PipelineTuneSettings,
    TunerDecision,
    TunerWindowStats,
)
from art.pipeline_tuner.worker_controller import RolloutWorkerController


class _PendingTask:
    def done(self) -> bool:
        return False


def test_build_initial_settings_uses_art_owned_defaults() -> None:
    settings = build_initial_settings(
        config=PipelineAutotuneConfig(),
        inference_gpu_count=2,
        policy_age_limit_steps=3.0,
    )

    assert settings.num_rollout_workers == 16
    assert settings.min_batch_size == 8
    assert settings.max_batch_size == 8
    assert settings.target_groups_per_step == 8
    assert settings.queue_maxsize == 12


def test_recommended_queue_keeps_one_batch_and_policy_age_bound() -> None:
    assert (
        recommended_queue_size(
            target_groups_per_step=8,
            limit_steps_off_policy=3.0,
            num_rollout_workers=16,
            running_reserve_fraction=0.75,
        )
        == 12
    )
    assert (
        recommended_queue_size(
            target_groups_per_step=8,
            limit_steps_off_policy=3.0,
            num_rollout_workers=64,
            running_reserve_fraction=0.75,
        )
        == 8
    )


def test_worker_controller_keeps_live_workers_when_at_target() -> None:
    controller = RolloutWorkerController(object(), target_workers=2)
    controller._tasks = {0: _PendingTask(), 1: _PendingTask()}  # type: ignore[dict-item]

    controller._reconcile()

    assert controller._retiring == set()


def test_vllm_load_stats_uses_request_second_pressure() -> None:
    metrics: list[PipelineMetric] = []
    rows = [
        (0.0, 0.0, 0.0),
        (64.0, 0.0, 0.0),
        (128.0, 16.0, 16.0),
        (0.0, 0.0, 0.0),
    ]
    for idx, (running, waiting, waiting_capacity) in enumerate(rows):
        t_s = float(idx)
        metrics.extend(
            [
                PipelineMetric(
                    name="vllm/num_requests_running", value=running, t_s=t_s
                ),
                PipelineMetric(
                    name="vllm/num_requests_waiting", value=waiting, t_s=t_s
                ),
                PipelineMetric(
                    name="vllm/num_requests_waiting_capacity",
                    value=waiting_capacity,
                    t_s=t_s,
                ),
                PipelineMetric(name="vllm/max_num_seqs", value=256.0, t_s=t_s),
            ]
        )

    stats = _vllm_load_stats(metrics, window_start_s=0.0, window_end_s=4.0)

    assert stats.capacity_wait_frac == 0.25
    assert stats.capacity_wait_request_s == 16.0
    assert stats.running_request_s == 192.0
    assert stats.pressure == 16.0 / 192.0
    assert stats.capacity_wait_area == 16.0 / 256.0 / 4.0
    assert stats.running_area == (64.0 + 128.0) / 256.0 / 4.0
    assert stats.idle_frac == 0.5


def test_vllm_load_stats_groups_scrape_burst_timestamps() -> None:
    metrics = [
        PipelineMetric(name="vllm/num_requests_running", value=10.0, t_s=1.000),
        PipelineMetric(name="vllm/num_requests_waiting", value=8.0, t_s=1.001),
        PipelineMetric(name="vllm/num_requests_waiting_capacity", value=8.0, t_s=1.002),
        PipelineMetric(name="vllm/num_requests_running", value=10.0, t_s=2.000),
        PipelineMetric(name="vllm/num_requests_waiting", value=0.0, t_s=2.001),
        PipelineMetric(name="vllm/num_requests_waiting_capacity", value=0.0, t_s=2.002),
    ]

    stats = _vllm_load_stats(metrics, window_start_s=1.0, window_end_s=3.0)

    assert stats.capacity_wait_request_s == 8.0
    assert stats.running_request_s == 20.0
    assert stats.pressure == 0.4
    assert (
        len(
            _group_vllm_metric_rows([(rec.t_s, rec.name, rec.value) for rec in metrics])
        )
        == 2
    )


def test_low_vllm_pressure_increases_workers_when_trainer_underfed() -> None:
    tuner = PipelineAutotuner(
        config=PipelineAutotuneConfig(),
        settings=PipelineTuneSettings(
            num_rollout_workers=16,
            min_batch_size=8,
            max_batch_size=8,
            queue_maxsize=12,
            target_groups_per_step=8,
        ),
        model_name="test",
        backend_name="MegatronBackend",
        packed_sequence_length=122880,
        inference_gpu_count=2,
        policy_age_limit_steps=3.0,
    )
    decision = tuner._decide(
        TunerWindowStats(
            start_step=4,
            end_step=7,
            trainer_idle_frac=0.30,
            vllm_capacity_wait_frac=0.20,
            vllm_active_frac=0.40,
            vllm_pressure=0.25,
            vllm_capacity_wait_area=0.014,
            vllm_running_area=0.05,
            vllm_idle_frac=0.61,
            queue_freshness_pressure=0.30,
            token_weighted_policy_age_steps_mean=1.0,
            groups_per_step_mean=8.0,
            group_pack_token_samples=[15000.0] * 32,
        )
    )

    assert decision.state == "inference_under_train_under"
    assert decision.action == "increase_workers"
    assert decision.updated.num_rollout_workers == 20


def test_vllm_pressure_overloaded_train_underfed_holds_without_immediate_warning() -> (
    None
):
    tuner = PipelineAutotuner(
        config=PipelineAutotuneConfig(),
        settings=PipelineTuneSettings(
            num_rollout_workers=16,
            min_batch_size=8,
            max_batch_size=8,
            queue_maxsize=12,
            target_groups_per_step=8,
        ),
        model_name="test",
        backend_name="MegatronBackend",
        packed_sequence_length=122880,
        inference_gpu_count=2,
        policy_age_limit_steps=3.0,
    )
    decision = tuner._decide(
        TunerWindowStats(
            start_step=4,
            end_step=7,
            trainer_idle_frac=0.40,
            vllm_pressure=1.0,
            vllm_capacity_wait_area=0.08,
            vllm_running_area=0.60,
            vllm_idle_frac=0.05,
            queue_freshness_pressure=0.20,
            token_weighted_policy_age_steps_mean=1.0,
            groups_per_step_mean=8.0,
            group_pack_token_samples=[15000.0] * 32,
        )
    )

    assert decision.state == "inference_over_train_under"
    assert decision.recommendations == []


def test_stable_holds_emit_inference_gpu_warning() -> None:
    settings = PipelineTuneSettings(
        num_rollout_workers=16,
        min_batch_size=8,
        max_batch_size=8,
        queue_maxsize=12,
        target_groups_per_step=8,
    )
    tuner = PipelineAutotuner(
        config=PipelineAutotuneConfig(),
        settings=settings,
        model_name="test",
        backend_name="MegatronBackend",
        packed_sequence_length=122880,
        inference_gpu_count=2,
        policy_age_limit_steps=3.0,
    )
    for step in range(5):
        tuner.decisions.append(
            TunerDecision(
                step=step + 1,
                state="inference_over_train_under",
                action="hold",
                reason="test",
                previous=settings,
                updated=settings,
                stats=TunerWindowStats(
                    start_step=step + 1,
                    end_step=step + 1,
                    trainer_idle_frac=0.60,
                    vllm_pressure=1.0,
                    group_pack_token_samples=[15000.0] * 8,
                ),
            )
        )

    with pytest.warns(UserWarning, match="increase inference GPUs"):
        tuner._emit_stable_recommendations(tuner.decisions[-1])

    assert any(
        "increase inference GPUs" in item
        for item in tuner.decisions[-1].recommendations
    )
