"""Low-overhead ART metrics snapshot for the dedicated vLLM runtime."""

from __future__ import annotations

import threading
import time
from typing import Any

from vllm.v1.metrics.loggers import StatLoggerBase

from art_vllm_runtime.fast_metrics import FastMetricsSharedWriter
from art_vllm_runtime.runtime_usage import record_finished_requests


class _ArtRuntimeMetricsState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._record_count = 0
        self._last_update_unix_s = 0.0
        self._writer: FastMetricsSharedWriter | None = None
        self._engine_gauges: dict[int, dict[str, float]] = {}
        self._engine_configs: dict[int, dict[str, float]] = {}
        self._counters = {
            "prompt_tokens_total": 0.0,
            "prompt_tokens_computed_total": 0.0,
            "prompt_tokens_cached_total": 0.0,
            "prompt_tokens_local_cache_hit_total": 0.0,
            "prompt_tokens_external_kv_transfer_total": 0.0,
            "generation_tokens_total": 0.0,
            "spec_decode_drafts_total": 0.0,
            "spec_decode_draft_tokens_total": 0.0,
            "spec_decode_accepted_tokens_total": 0.0,
            "prefix_cache_queries_total": 0.0,
            "prefix_cache_hits_total": 0.0,
            "external_prefix_cache_queries_total": 0.0,
            "external_prefix_cache_hits_total": 0.0,
            "num_preempted_reqs_total": 0.0,
            "policy_cache_salted_lora_requests_total": 0.0,
            "policy_cache_unsalted_lora_requests_total": 0.0,
            "policy_cache_waiting_requests_updated_total": 0.0,
            "policy_cache_started_waiting_requests_skipped_total": 0.0,
        }

    def configure(self, vllm_config: Any, *, engine_idx: int) -> None:
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        engine_config: dict[str, float] = {}
        for metric_name, obj, attr in (
            ("max_num_seqs", scheduler_config, "max_num_seqs"),
            ("max_num_batched_tokens", scheduler_config, "max_num_batched_tokens"),
            ("max_model_len", model_config, "max_model_len"),
            ("world_size", parallel_config, "world_size_across_dp"),
        ):
            value = getattr(obj, attr, None)
            if isinstance(value, (int, float)):
                engine_config[metric_name] = float(value)
        max_num_scheduled_tokens = getattr(
            scheduler_config, "max_num_scheduled_tokens", None
        )
        if isinstance(max_num_scheduled_tokens, (int, float)):
            engine_config["max_num_scheduled_tokens"] = float(max_num_scheduled_tokens)
        elif "max_num_batched_tokens" in engine_config:
            engine_config["max_num_scheduled_tokens"] = engine_config[
                "max_num_batched_tokens"
            ]
        with self._lock:
            self._engine_configs[engine_idx] = engine_config
            self._publish_locked()

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        *,
        engine_idx: int,
    ) -> None:
        now = time.time()
        with self._lock:
            self._record_count += 1
            self._last_update_unix_s = now
            if scheduler_stats is not None:
                waiting_capacity = float(scheduler_stats.num_waiting_reqs)
                waiting_deferred = float(scheduler_stats.num_skipped_waiting_reqs)
                self._engine_gauges[engine_idx] = {
                    "running": float(scheduler_stats.num_running_reqs),
                    "waiting": waiting_capacity + waiting_deferred,
                    "waiting_capacity": waiting_capacity,
                    "waiting_deferred": waiting_deferred,
                    "kv_cache_usage": float(scheduler_stats.kv_cache_usage),
                }
                self._counters["prefix_cache_queries_total"] += float(
                    scheduler_stats.prefix_cache_stats.queries
                )
                self._counters["prefix_cache_hits_total"] += float(
                    scheduler_stats.prefix_cache_stats.hits
                )
                connector = scheduler_stats.connector_prefix_cache_stats
                if connector is not None:
                    self._counters["external_prefix_cache_queries_total"] += float(
                        connector.queries
                    )
                    self._counters["external_prefix_cache_hits_total"] += float(
                        connector.hits
                    )
                spec_decode = scheduler_stats.spec_decoding_stats
                if spec_decode is not None:
                    self._counters["spec_decode_drafts_total"] += float(
                        spec_decode.num_drafts
                    )
                    self._counters["spec_decode_draft_tokens_total"] += float(
                        spec_decode.num_draft_tokens
                    )
                    self._counters["spec_decode_accepted_tokens_total"] += float(
                        spec_decode.num_accepted_tokens
                    )
            if iteration_stats is not None:
                record_finished_requests(iteration_stats)
                prompt_stats = iteration_stats.prompt_token_stats
                self._counters["prompt_tokens_total"] += float(
                    iteration_stats.num_prompt_tokens
                )
                self._counters["prompt_tokens_computed_total"] += float(
                    prompt_stats.computed
                )
                self._counters["prompt_tokens_cached_total"] += float(
                    prompt_stats.cached_tokens
                )
                self._counters["prompt_tokens_local_cache_hit_total"] += float(
                    prompt_stats.local_cache_hit
                )
                self._counters["prompt_tokens_external_kv_transfer_total"] += float(
                    prompt_stats.external_kv_transfer
                )
                self._counters["generation_tokens_total"] += float(
                    iteration_stats.num_generation_tokens
                )
                self._counters["num_preempted_reqs_total"] += float(
                    iteration_stats.num_preempted_reqs
                )
            self._publish_locked()

    def _metrics_locked(self) -> dict[str, float]:
        gauges = list(self._engine_gauges.values())
        engine_configs = list(self._engine_configs.values())
        metrics = dict(self._counters)
        prefix_queries = metrics["prefix_cache_queries_total"]
        external_prefix_queries = metrics["external_prefix_cache_queries_total"]
        max_model_lens = [
            item["max_model_len"] for item in engine_configs if "max_model_len" in item
        ]
        metrics.update(
            {
                "prefix_cache_hit_rate": (
                    metrics["prefix_cache_hits_total"] / prefix_queries
                    if prefix_queries > 0
                    else 0.0
                ),
                "external_prefix_cache_hit_rate": (
                    metrics["external_prefix_cache_hits_total"]
                    / external_prefix_queries
                    if external_prefix_queries > 0
                    else 0.0
                ),
                "num_requests_running": sum(item["running"] for item in gauges),
                "num_requests_waiting": sum(item["waiting"] for item in gauges),
                "num_requests_waiting_capacity": sum(
                    item["waiting_capacity"] for item in gauges
                ),
                "num_requests_waiting_deferred": sum(
                    item["waiting_deferred"] for item in gauges
                ),
                "kv_cache_usage_perc": max(
                    (item["kv_cache_usage"] for item in gauges), default=0.0
                ),
                "max_num_seqs": sum(
                    item.get("max_num_seqs", 0.0) for item in engine_configs
                ),
                "max_num_batched_tokens": sum(
                    item.get("max_num_batched_tokens", 0.0) for item in engine_configs
                ),
                "max_num_scheduled_tokens": sum(
                    item.get("max_num_scheduled_tokens", 0.0) for item in engine_configs
                ),
                "max_model_len": max(max_model_lens, default=0.0),
                "world_size": max(
                    (item.get("world_size", 0.0) for item in engine_configs),
                    default=0.0,
                ),
            }
        )
        return metrics

    def _publish_locked(self) -> None:
        if self._writer is not None:
            self._writer.publish(
                last_update_unix_s=self._last_update_unix_s,
                record_count=self._record_count,
                engine_count=len(self._engine_gauges),
                metrics=self._metrics_locked(),
            )

    def set_writer(self, writer: FastMetricsSharedWriter | None) -> None:
        with self._lock:
            self._writer = writer
            self._publish_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "source": "art_vllm_runtime",
                "last_update_unix_s": self._last_update_unix_s,
                "record_count": self._record_count,
                "engine_count": len(self._engine_gauges),
                "metrics": self._metrics_locked(),
            }

    def record_policy_cache_salt_audit(
        self, *, lora_request: bool, salted: bool
    ) -> None:
        if not lora_request:
            return
        key = (
            "policy_cache_salted_lora_requests_total"
            if salted
            else "policy_cache_unsalted_lora_requests_total"
        )
        with self._lock:
            self._counters[key] += 1.0
            self._publish_locked()

    def record_policy_cache_waiting_update(
        self, *, updated: int, skipped_started: int
    ) -> None:
        with self._lock:
            self._counters["policy_cache_waiting_requests_updated_total"] += float(
                updated
            )
            self._counters["policy_cache_started_waiting_requests_skipped_total"] += (
                float(skipped_started)
            )
            self._publish_locked()


_STATE = _ArtRuntimeMetricsState()


class ArtRuntimeStatLogger(StatLoggerBase):
    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.engine_index = engine_index
        _STATE.configure(vllm_config, engine_idx=engine_index)

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int | None = None,
    ) -> None:
        del mm_cache_stats
        _STATE.record(
            scheduler_stats,
            iteration_stats,
            engine_idx=self.engine_index if engine_idx is None else engine_idx,
        )

    def log_engine_initialized(self) -> None:
        return None


def get_art_metrics_snapshot() -> dict[str, Any]:
    return _STATE.snapshot()


def set_fast_metrics_writer(writer: FastMetricsSharedWriter | None) -> None:
    _STATE.set_writer(writer)


def record_policy_cache_salt_audit(*, lora_request: bool, salted: bool) -> None:
    _STATE.record_policy_cache_salt_audit(lora_request=lora_request, salted=salted)


def record_policy_cache_waiting_update(*, updated: int, skipped_started: int) -> None:
    _STATE.record_policy_cache_waiting_update(
        updated=updated,
        skipped_started=skipped_started,
    )
