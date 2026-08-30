from types import SimpleNamespace

import pytest

from art.pipeline_tuner import PipelineAutotuneConfig
from art.pipeline_tuner.attachment import (
    PipelineAutotunerAttachment,
    VllmMetricPollHealth,
)
from art.pipeline_tuner.autotune import PipelineAutotuner
from art.pipeline_tuner.config import (
    PipelineTuneSettings,
    TunerDecision,
    TunerWindowStats,
)


def _decision() -> TunerDecision:
    settings = PipelineTuneSettings(
        num_rollout_workers=16,
        min_batch_size=8,
        max_batch_size=8,
        queue_maxsize=12,
        target_groups_per_step=8,
    )
    return TunerDecision(
        step=7,
        state="test",
        action="hold",
        reason="test",
        previous=settings,
        updated=settings,
        stats=TunerWindowStats(
            start_step=4,
            end_step=7,
            window_start_s=1.0,
            window_end_s=5.0,
        ),
    )


def test_queue_backpressure_does_not_mask_trainer_underfeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = PipelineTuneSettings(
        num_rollout_workers=12,
        min_batch_size=4,
        max_batch_size=4,
        queue_maxsize=4,
        target_groups_per_step=4,
    )
    tuner = PipelineAutotuner(
        config=PipelineAutotuneConfig(worker_load_change_windows=1),
        settings=settings,
        model_name="test",
        backend_name="test",
        packed_sequence_length=32768,
        target_packed_sequences=2,
        inference_gpu_count=2,
        policy_age_limit_steps=4.0,
    )
    monkeypatch.setattr(
        tuner,
        "_settings_with_recomputed_queue",
        lambda current, _stats, *, adapt_target: current,
    )

    decision = tuner._decide(
        TunerWindowStats(
            start_step=4,
            end_step=5,
            trainer_underfeed_score=0.3,
            vllm_pressure=0.4,
            queue_put_wait_frac=0.8,
        )
    )

    assert decision.action == "increase_workers"
    assert decision.updated.num_rollout_workers > settings.num_rollout_workers


@pytest.mark.parametrize("timeouts,raises", [(1, False), (2, True)])
def test_metric_poll_health_contract(timeouts: int, raises: bool) -> None:
    attachment = PipelineAutotunerAttachment(PipelineAutotuneConfig())
    attachment._poll_health = [
        VllmMetricPollHealth(t_s=float(index + 1), timed_out=index < timeouts)
        for index in range(3)
    ]

    if raises:
        with pytest.raises(RuntimeError, match="metric polls timed out"):
            attachment._raise_if_unhealthy_metric_window(_decision())
    else:
        attachment._raise_if_unhealthy_metric_window(_decision())


@pytest.mark.asyncio
async def test_required_vllm_metric_contract() -> None:
    class Backend:
        async def collect_train_step_vllm_metrics(self, _model: object):
            return {"vllm/num_requests_running": 1.0}

    attachment = PipelineAutotunerAttachment(PipelineAutotuneConfig())
    attachment.trainer = SimpleNamespace(backend=Backend(), model=object())

    with pytest.raises(RuntimeError, match="missing"):
        await attachment._collect_required_serving_metrics()


@pytest.mark.asyncio
@pytest.mark.parametrize("training_failed", [False, True])
async def test_sampler_failure_does_not_mask_training_failure(
    training_failed: bool,
) -> None:
    attachment = PipelineAutotunerAttachment(PipelineAutotuneConfig())
    attachment._sampler_error = RuntimeError("metrics endpoint closed")

    if training_failed:
        await attachment.on_stop(training_failed=True)
    else:
        with pytest.raises(RuntimeError, match="metrics sampler failed"):
            await attachment.on_stop(training_failed=False)
