from types import SimpleNamespace

from art_vllm_runtime.metrics import _ArtRuntimeMetricsState


def _scheduler_stats(
    *, drafts: int, draft_tokens: int, accepted_tokens: int
) -> SimpleNamespace:
    return SimpleNamespace(
        num_running_reqs=1,
        num_waiting_reqs=0,
        num_skipped_waiting_reqs=0,
        kv_cache_usage=0.25,
        prefix_cache_stats=SimpleNamespace(queries=0, hits=0),
        connector_prefix_cache_stats=None,
        spec_decoding_stats=SimpleNamespace(
            num_drafts=drafts,
            num_draft_tokens=draft_tokens,
            num_accepted_tokens=accepted_tokens,
        ),
    )


def test_fast_metrics_accumulate_speculative_decode_work() -> None:
    state = _ArtRuntimeMetricsState()

    state.record(
        _scheduler_stats(drafts=3, draft_tokens=3, accepted_tokens=2),
        None,
        engine_idx=0,
    )
    state.record(
        _scheduler_stats(drafts=4, draft_tokens=4, accepted_tokens=3),
        None,
        engine_idx=0,
    )

    metrics = state.snapshot()["metrics"]
    assert metrics["spec_decode_drafts_total"] == 7.0
    assert metrics["spec_decode_draft_tokens_total"] == 7.0
    assert metrics["spec_decode_accepted_tokens_total"] == 5.0
