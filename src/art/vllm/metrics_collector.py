"""Collect metrics from vLLM's Prometheus endpoint and log to wandb."""

import asyncio
import re
from typing import Dict, Optional, Any
import httpx
from wandb.wandb_run import Run as WandbRun


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
            wandb_metrics = {
                "vllm/concurrent_requests": metrics.get("vllm:num_requests_running", 0),
                "vllm/queued_requests": metrics.get("vllm:num_requests_waiting", 0),
                "vllm/gpu_cache_usage_perc": metrics.get(
                    "vllm:gpu_cache_usage_perc", 0
                ),
            }

            # Add request latency metrics if available
            if "vllm:request_latency_seconds" in metrics:
                wandb_metrics["vllm/request_latency_p50"] = metrics[
                    "vllm:request_latency_seconds"
                ].get("0.5", 0)
                wandb_metrics["vllm/request_latency_p99"] = metrics[
                    "vllm:request_latency_seconds"
                ].get("0.99", 0)

            self.wandb_run.log(wandb_metrics)

    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus format metrics into a dictionary."""
        metrics = {}

        # Parse simple gauge metrics
        gauge_pattern = r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)"

        # Parse histogram/summary metrics with labels
        histogram_pattern = (
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*){(.+?)}\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)"
        )

        for line in metrics_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try simple gauge pattern first
            match = re.match(gauge_pattern, line)
            if match:
                metric_name, value = match.groups()
                metrics[metric_name] = float(value)
                continue

            # Try histogram pattern with labels
            match = re.match(histogram_pattern, line)
            if match:
                metric_name, labels, value = match.groups()

                # Parse quantile from labels for latency metrics
                if "quantile=" in labels:
                    quantile_match = re.search(r'quantile="([0-9.]+)"', labels)
                    if quantile_match:
                        quantile = quantile_match.group(1)
                        if metric_name not in metrics:
                            metrics[metric_name] = {}
                        metrics[metric_name][quantile] = float(value)
                else:
                    # For other labeled metrics, just use the base name
                    metrics[metric_name] = float(value)

        return metrics
