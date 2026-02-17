"""
Metrics collection and reporting for Unsloth + SGLang benchmarks.

Focuses on the metrics that matter for RL training rollout speed:
  - Throughput (output tokens/sec)
  - TTFT (Time to First Token)
  - Inter-token latency
  - End-to-end request latency
  - GPU memory usage
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""

    request_id: int
    start_time: float
    end_time: float
    ttft: float
    total_time: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def tokens_per_second(self) -> float:
        if self.total_time <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.total_time

    @property
    def inter_token_latency(self) -> float:
        gen_time = self.total_time - self.ttft
        if gen_time <= 0 or self.completion_tokens <= 1:
            return 0.0
        return gen_time / (self.completion_tokens - 1)


@dataclass
class StepMetrics:
    """Metrics for one rollout batch."""

    step: int
    rollout_start: float = 0.0
    rollout_end: float = 0.0
    request_metrics: list[RequestMetrics] = field(default_factory=list)
    gpu_memory_during_rollout: float = 0.0

    # Training transition (kept for data but NOT used in comparison)
    inference_stop_start: float = 0.0
    inference_stop_end: float = 0.0
    training_start: float = 0.0
    training_end: float = 0.0
    inference_start_start: float = 0.0
    inference_start_end: float = 0.0
    lora_merge_time: float = 0.0
    gpu_memory_before_rollout: float = 0.0
    gpu_memory_during_training: float = 0.0
    training_metrics: list[dict[str, float]] = field(default_factory=list)

    @property
    def rollout_time(self) -> float:
        return self.rollout_end - self.rollout_start

    @property
    def inference_stop_time(self) -> float:
        return self.inference_stop_end - self.inference_stop_start

    @property
    def training_time(self) -> float:
        return self.training_end - self.training_start

    @property
    def inference_start_time(self) -> float:
        return self.inference_start_end - self.inference_start_start

    @property
    def total_step_time(self) -> float:
        return self.rollout_time

    @property
    def transition_overhead(self) -> float:
        return self.inference_stop_time + self.inference_start_time

    @property
    def _ok(self) -> list[RequestMetrics]:
        return [r for r in self.request_metrics if not r.error]

    @property
    def rollout_throughput(self) -> float:
        total = sum(r.completion_tokens for r in self._ok)
        return total / self.rollout_time if self.rollout_time > 0 else 0.0

    @property
    def avg_ttft(self) -> float:
        vals = [r.ttft for r in self._ok if r.ttft > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def p50_ttft(self) -> float:
        return _pct(sorted(r.ttft for r in self._ok if r.ttft > 0), 50)

    @property
    def p99_ttft(self) -> float:
        return _pct(sorted(r.ttft for r in self._ok if r.ttft > 0), 99)

    @property
    def avg_itl(self) -> float:
        vals = [r.inter_token_latency for r in self._ok if r.inter_token_latency > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_request_time(self) -> float:
        vals = [r.total_time for r in self._ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def p99_request_time(self) -> float:
        return _pct(sorted(r.total_time for r in self._ok), 99)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.request_metrics if r.error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "rollout_time_s": round(self.rollout_time, 3),
            "throughput_tok_s": round(self.rollout_throughput, 1),
            "avg_ttft_s": round(self.avg_ttft, 4),
            "p50_ttft_s": round(self.p50_ttft, 4),
            "p99_ttft_s": round(self.p99_ttft, 4),
            "avg_itl_s": round(self.avg_itl, 5),
            "avg_latency_s": round(self.avg_request_time, 3),
            "p99_latency_s": round(self.p99_request_time, 3),
            "errors": self.error_count,
            "num_requests": len(self.request_metrics),
            "gpu_mem_gb": round(self.gpu_memory_during_rollout / 1e9, 2),
        }


@dataclass
class BenchmarkRun:
    """All metrics for one backend."""

    backend: str
    model: str
    dataset: str = "gsm8k"
    start_time: float = 0.0
    end_time: float = 0.0
    server_startup_time: float = 0.0
    steps: list[StepMetrics] = field(default_factory=list)
    warmup_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    def _avg(self, fn) -> float:
        vals = [fn(s) for s in self.steps]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_step_time(self) -> float:
        return self._avg(lambda s: s.rollout_time)

    avg_rollout_time = avg_step_time

    @property
    def avg_training_time(self) -> float:
        return 0.0

    @property
    def avg_transition_overhead(self) -> float:
        return 0.0

    @property
    def avg_rollout_throughput(self) -> float:
        return self._avg(lambda s: s.rollout_throughput)

    @property
    def avg_ttft(self) -> float:
        return self._avg(lambda s: s.avg_ttft)

    @property
    def avg_p99_ttft(self) -> float:
        return self._avg(lambda s: s.p99_ttft)

    @property
    def avg_itl(self) -> float:
        return self._avg(lambda s: s.avg_itl)

    @property
    def avg_latency(self) -> float:
        return self._avg(lambda s: s.avg_request_time)

    @property
    def avg_p99_latency(self) -> float:
        return self._avg(lambda s: s.p99_request_time)

    @property
    def avg_gpu_mem_gb(self) -> float:
        vals = [s.gpu_memory_during_rollout for s in self.steps if s.gpu_memory_during_rollout > 0]
        return (statistics.mean(vals) / 1e9) if vals else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "dataset": self.dataset,
            "total_time_s": round(self.total_time, 2),
            "server_startup_s": round(self.server_startup_time, 2),
            "num_steps": len(self.steps),
            "avg_throughput_tok_s": round(self.avg_rollout_throughput, 1),
            "avg_ttft_s": round(self.avg_ttft, 4),
            "avg_p99_ttft_s": round(self.avg_p99_ttft, 4),
            "avg_itl_s": round(self.avg_itl, 5),
            "avg_latency_s": round(self.avg_latency, 3),
            "avg_p99_latency_s": round(self.avg_p99_latency, 3),
            "avg_gpu_mem_gb": round(self.avg_gpu_mem_gb, 2),
            "total_errors": sum(s.error_count for s in self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def get_gpu_memory_usage_nvidia_smi() -> dict[int, float]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        out: dict[int, float] = {}
        for line in r.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            out[int(parts[0].strip())] = float(parts[1].strip()) * 1024 * 1024
        return out
    except Exception:
        return {}


def generate_comparison_report_multi(
    runs: dict[str, BenchmarkRun],
    output_dir: str,
) -> str:
    """Generate comparison report for one or more backends."""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual metrics
    for name, run in runs.items():
        with open(os.path.join(output_dir, f"{name}_metrics.json"), "w") as f:
            json.dump(run.summary(), f, indent=2)
    with open(os.path.join(output_dir, "benchmark_combined.json"), "w") as f:
        json.dump({n: r.summary() for n, r in runs.items()}, f, indent=2)

    backend_names = sorted(runs.keys())
    name_map = {"unsloth": "Unsloth"}

    def _col(name: str) -> str:
        return name_map.get(name, name.upper())

    W = 90
    lines = [
        "=" * W,
        "  Multi-Backend Rollout Benchmark",
        "=" * W,
        "",
    ]

    # Config info from first run
    first = next(iter(runs.values()))
    lines.append(f"  Model:    {first.model}")
    lines.append(f"  Dataset:  {first.dataset}")
    lines.append(f"  Steps:    {len(first.steps)}")
    lines.append(f"  Backends: {', '.join(_col(n) for n in backend_names)}")
    lines.append("")

    # Header row
    header_fmt = f"{'Metric':<32}"
    for name in backend_names:
        header_fmt += f" {_col(name):>14}"
    lines.append("-" * W)
    lines.append(header_fmt)
    lines.append("-" * W)

    # Metric rows
    metric_defs = [
        ("Throughput (tok/s)", lambda r: r.avg_rollout_throughput, "higher"),
        ("Avg TTFT (s)", lambda r: r.avg_ttft, "lower"),
        ("P99 TTFT (s)", lambda r: r.avg_p99_ttft, "lower"),
        ("Avg ITL (s)", lambda r: r.avg_itl, "lower"),
        ("Avg Latency (s)", lambda r: r.avg_latency, "lower"),
        ("P99 Latency (s)", lambda r: r.avg_p99_latency, "lower"),
        ("GPU Memory (GB)", lambda r: r.avg_gpu_mem_gb, "lower"),
        ("Server Startup (s)", lambda r: r.server_startup_time, "lower"),
        ("Total Time (s)", lambda r: r.total_time, "lower"),
        ("Total Errors", lambda r: float(sum(s.error_count for s in r.steps)), "lower"),
    ]

    for label, fn, direction in metric_defs:
        vals = {name: fn(runs[name]) for name in backend_names}
        # Find the best value
        if direction == "higher":
            best = max(vals.values()) if any(v > 0 for v in vals.values()) else 0
        else:
            non_zero = [v for v in vals.values() if v > 0]
            best = min(non_zero) if non_zero else 0

        row = f"  {label:<30}"
        for name in backend_names:
            v = vals[name]
            # Format based on magnitude
            if "tok/s" in label:
                s = f"{v:.1f}"
            elif "TTFT" in label or "ITL" in label:
                s = f"{v:.4f}"
            elif "Error" in label:
                s = f"{int(v)}"
            else:
                s = f"{v:.2f}"

            # Mark best with *
            if v == best and v > 0 and len(vals) > 1:
                s += " *"
            row += f" {s:>14}"
        lines.append(row)

    # Per-step breakdown
    min_steps = min(len(r.steps) for r in runs.values())
    lines.extend(["", "-" * W, "  Per-Step Breakdown", "-" * W, ""])

    for i in range(min_steps):
        lines.append(f"  Step {i+1}:")
        for name in backend_names:
            step = runs[name].steps[i]
            lines.append(
                f"    {_col(name):>8}: "
                f"Throughput={step.rollout_throughput:>8.1f} tok/s  "
                f"TTFT={step.avg_ttft:>7.4f}s  "
                f"ITL={step.avg_itl:>8.5f}s  "
                f"Latency={step.avg_request_time:>7.3f}s"
            )
        lines.append("")

    # Verdict
    lines.extend(["=" * W, "  VERDICT", "=" * W, ""])

    # Score each backend: +1 for each metric won
    scores = {name: 0 for name in backend_names}
    for label, fn, direction in metric_defs:
        if "Error" in label or "Startup" in label or "Total Time" in label:
            continue  # Skip non-performance metrics
        vals = {name: fn(runs[name]) for name in backend_names}
        non_zero = {n: v for n, v in vals.items() if v > 0}
        if len(non_zero) < 2:
            continue
        if direction == "higher":
            winner = max(non_zero, key=non_zero.get)  # type: ignore[arg-type]
        else:
            winner = min(non_zero, key=non_zero.get)  # type: ignore[arg-type]
        scores[winner] += 1

    # Report pairwise comparisons
    for i, a in enumerate(backend_names):
        for b in backend_names[i+1:]:
            ra, rb = runs[a], runs[b]
            # Throughput comparison
            if ra.avg_rollout_throughput > 0 and rb.avg_rollout_throughput > 0:
                if ra.avg_rollout_throughput > rb.avg_rollout_throughput:
                    d = ((ra.avg_rollout_throughput - rb.avg_rollout_throughput) / rb.avg_rollout_throughput) * 100
                    lines.append(f"  Throughput: {_col(a)} {d:.1f}% higher than {_col(b)}")
                else:
                    d = ((rb.avg_rollout_throughput - ra.avg_rollout_throughput) / ra.avg_rollout_throughput) * 100
                    lines.append(f"  Throughput: {_col(b)} {d:.1f}% higher than {_col(a)}")

            # TTFT comparison
            if ra.avg_ttft > 0 and rb.avg_ttft > 0:
                if ra.avg_ttft < rb.avg_ttft:
                    d = ((rb.avg_ttft - ra.avg_ttft) / rb.avg_ttft) * 100
                    lines.append(f"  TTFT:       {_col(a)} {d:.1f}% faster than {_col(b)}")
                else:
                    d = ((ra.avg_ttft - rb.avg_ttft) / ra.avg_ttft) * 100
                    lines.append(f"  TTFT:       {_col(b)} {d:.1f}% faster than {_col(a)}")

            # Memory comparison
            if ra.avg_gpu_mem_gb > 0 and rb.avg_gpu_mem_gb > 0:
                if ra.avg_gpu_mem_gb < rb.avg_gpu_mem_gb:
                    d = ((rb.avg_gpu_mem_gb - ra.avg_gpu_mem_gb) / rb.avg_gpu_mem_gb) * 100
                    lines.append(f"  Memory:     {_col(a)} {d:.1f}% less than {_col(b)}")
                elif rb.avg_gpu_mem_gb < ra.avg_gpu_mem_gb:
                    d = ((ra.avg_gpu_mem_gb - rb.avg_gpu_mem_gb) / ra.avg_gpu_mem_gb) * 100
                    lines.append(f"  Memory:     {_col(b)} {d:.1f}% less than {_col(a)}")

    lines.append("")

    # Overall winner
    if scores:
        winner = max(scores, key=scores.get)  # type: ignore[arg-type]
        lines.append(f"  >>> {_col(winner)} wins on {scores[winner]}/{sum(scores.values())} performance metrics <<<")
    lines.extend(["", "  (* marks best value per metric)", "", "=" * W])

    report = "\n".join(lines)
    with open(os.path.join(output_dir, "benchmark_report.txt"), "w") as f:
        f.write(report)
    return report


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]
