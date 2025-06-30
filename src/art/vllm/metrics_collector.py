"""Collect metrics from vLLM's Prometheus endpoint and log to wandb."""

import asyncio
from typing import Dict, Optional, Any, DefaultDict
from collections import defaultdict
import httpx
from wandb.wandb_run import Run as WandbRun
from prometheus_client.parser import text_string_to_metric_families
import math


class VLLMMetricsCollector:
    """Collects metrics from vLLM's /metrics endpoint and logs to wandb."""

    def __init__(
        self,
        base_url: str,
        wandb_run: Optional[WandbRun] = None,
        polling_interval: int = 15,
        timeout: int = 10,
    ):
        """
        Initialize the metrics collector.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000")
            wandb_run: Wandb run object for logging metrics
            polling_interval: Seconds between metric collections
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.wandb_run = wandb_run
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        if self.wandb_run is not None:
            self.wandb_run.define_metric("vllm/*", step_metric="_runtime")

    async def start(self):
        """Start the metrics collection task."""
        if self._task is not None:
            raise RuntimeError("Metrics collector is already running")

        self._stop_event.clear()
        self._task = asyncio.create_task(self._collect_metrics_loop())

    async def stop(self):
        """Stop the metrics collection task."""
        if self._task is None:
            return

        self._stop_event.set()
        await self._task
        self._task = None

    async def _collect_metrics_loop(self):
        """Main loop for collecting metrics."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while not self._stop_event.is_set():
                try:
                    await self._collect_and_log_metrics(client)
                except Exception as e:
                    print(f"Error collecting vLLM metrics: {e}")

                # Wait for the next collection interval or stop event
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.polling_interval
                    )
                except asyncio.TimeoutError:
                    continue

    async def _collect_and_log_metrics(self, client: httpx.AsyncClient):
        """Collect metrics from endpoint and log to wandb."""
        response = await client.get(f"{self.base_url}/metrics")
        response.raise_for_status()

        metrics = self._parse_prometheus_metrics(response.text)

        # Log to wandb if run is available
        if self.wandb_run is not None:
            if metrics:
                # Let wandb auto-log _runtime and _step; we don't pass an explicit step
                # so that charts use _runtime as configured in `define_metric`.
                self.wandb_run.log(metrics)

    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Extract latency p50/p99 and concurrency gauges from Prometheus text."""

        running: Optional[float] = None
        waiting: Optional[float] = None

        # Aggregate histogram buckets across all models
        bucket_totals: DefaultDict[float, float] = defaultdict(float)

        for family in text_string_to_metric_families(metrics_text):
            if family.name in {
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
            }:
                total = sum(float(s.value) for s in family.samples)
                if family.name == "vllm:num_requests_running":
                    running = total
                else:
                    waiting = total
            elif family.name == "vllm:e2e_request_latency_seconds":
                for sample in family.samples:
                    # Only consider the histogram *buckets*; ignore _count/_sum etc.
                    if not sample.name.endswith("_bucket"):
                        continue

                    le = sample.labels.get("le", "+Inf")
                    bound = float("inf") if le == "+Inf" else float(le)
                    bucket_totals[bound] += float(sample.value)

        p50 = p99 = None
        if bucket_totals:
            bounds_sorted = sorted(bucket_totals.keys())
            counts_sorted = [bucket_totals[b] for b in bounds_sorted]
            total = counts_sorted[-1]

            if total > 0:

                def _quantile(q: float) -> float:
                    target = total * q
                    for b, c in zip(bounds_sorted, counts_sorted):
                        if c >= target:
                            return b
                    return float("nan")

                p50 = _quantile(0.5)
                p99 = _quantile(0.99)

        # Use "/" in metric names for wandb (vllm/...)
        out: Dict[str, Any] = {}
        if running is not None:
            out["vllm/num_requests_running"] = running
        if waiting is not None:
            out["vllm/num_requests_waiting"] = waiting
        if p50 is not None and not math.isnan(p50):
            out["vllm/e2e_request_latency_seconds_p50"] = p50
        if p99 is not None and not math.isnan(p99):
            out["vllm/e2e_request_latency_seconds_p99"] = p99

        return out
