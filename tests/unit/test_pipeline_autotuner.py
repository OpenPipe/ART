from art.pipeline_tuner import PipelineAutotuneConfig
from art.pipeline_tuner.autotune import build_initial_settings, recommended_queue_size
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
