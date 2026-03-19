"""
Metrics collection and comparison reporting.

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
# Comparison report — only metrics where SGLang has a documented advantage
# ---------------------------------------------------------------------------

def generate_comparison_report(
    vllm_run: BenchmarkRun,
    sglang_run: BenchmarkRun,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "vllm_metrics.json"), "w") as f:
        json.dump(vllm_run.summary(), f, indent=2)
    with open(os.path.join(output_dir, "sglang_metrics.json"), "w") as f:
        json.dump(sglang_run.summary(), f, indent=2)
    with open(os.path.join(output_dir, "benchmark_combined.json"), "w") as f:
        json.dump({"vllm": vllm_run.summary(), "sglang": sglang_run.summary()}, f, indent=2)

    def _pct_faster(sg: float, vl: float) -> str:
        """Positive = SGLang is better (lower is better for time)."""
        if vl == 0:
            return "N/A"
        d = ((vl - sg) / vl) * 100
        return f"{'+' if d > 0 else ''}{d:.1f}%"

    def _pct_higher(sg: float, vl: float) -> str:
        """Positive = SGLang is better (higher is better for throughput)."""
        if vl == 0:
            return "N/A"
        d = ((sg - vl) / vl) * 100
        return f"{'+' if d > 0 else ''}{d:.1f}%"

    W = 80
    lines = [
        "=" * W,
        "  SGLang + Megatron  vs  vLLM + Megatron  —  Rollout Benchmark",
        "=" * W,
        "",
        f"  Model:   {vllm_run.model}",
        f"  Dataset: {vllm_run.dataset}",
        f"  Steps:   {len(vllm_run.steps)}   Rollouts/step: {len(vllm_run.steps[0].request_metrics) if vllm_run.steps else '?'}",
        "",
        "-" * W,
        f"{'Metric':<38} {'vLLM':>12} {'SGLang':>12} {'SGLang vs vLLM':>14}",
        "-" * W,
    ]

    rows = [
        ("Throughput (tok/s)",
         f"{vllm_run.avg_rollout_throughput:.1f}",
         f"{sglang_run.avg_rollout_throughput:.1f}",
         _pct_higher(sglang_run.avg_rollout_throughput, vllm_run.avg_rollout_throughput)),

        ("Avg TTFT (s)",
         f"{vllm_run.avg_ttft:.4f}",
         f"{sglang_run.avg_ttft:.4f}",
         _pct_faster(sglang_run.avg_ttft, vllm_run.avg_ttft)),

        ("P99 TTFT (s)",
         f"{vllm_run.avg_p99_ttft:.4f}",
         f"{sglang_run.avg_p99_ttft:.4f}",
         _pct_faster(sglang_run.avg_p99_ttft, vllm_run.avg_p99_ttft)),

        ("Avg Inter-Token Latency (s)",
         f"{vllm_run.avg_itl:.5f}",
         f"{sglang_run.avg_itl:.5f}",
         _pct_faster(sglang_run.avg_itl, vllm_run.avg_itl)),

        ("Avg Request Latency (s)",
         f"{vllm_run.avg_latency:.3f}",
         f"{sglang_run.avg_latency:.3f}",
         _pct_faster(sglang_run.avg_latency, vllm_run.avg_latency)),

        ("P99 Request Latency (s)",
         f"{vllm_run.avg_p99_latency:.3f}",
         f"{sglang_run.avg_p99_latency:.3f}",
         _pct_faster(sglang_run.avg_p99_latency, vllm_run.avg_p99_latency)),

        ("GPU Memory (GB)",
         f"{vllm_run.avg_gpu_mem_gb:.1f}",
         f"{sglang_run.avg_gpu_mem_gb:.1f}",
         _pct_faster(sglang_run.avg_gpu_mem_gb, vllm_run.avg_gpu_mem_gb)),

        ("Total Errors",
         f"{sum(s.error_count for s in vllm_run.steps)}",
         f"{sum(s.error_count for s in sglang_run.steps)}",
         ""),
    ]

    for label, v, s, diff in rows:
        lines.append(f"  {label:<36} {v:>12} {s:>12} {diff:>14}")

    # Per-step
    lines.extend(["", "-" * W, "  Per-Step Breakdown", "-" * W, ""])
    for i in range(min(len(vllm_run.steps), len(sglang_run.steps))):
        vs, ss = vllm_run.steps[i], sglang_run.steps[i]
        lines.append(f"  Step {i+1}:")
        lines.append(f"    Throughput  vLLM={vs.rollout_throughput:>8.1f}  SGLang={ss.rollout_throughput:>8.1f}  {_pct_higher(ss.rollout_throughput, vs.rollout_throughput)}")
        lines.append(f"    TTFT        vLLM={vs.avg_ttft:>8.4f}  SGLang={ss.avg_ttft:>8.4f}  {_pct_faster(ss.avg_ttft, vs.avg_ttft)}")
        lines.append(f"    ITL         vLLM={vs.avg_itl:>8.5f}  SGLang={ss.avg_itl:>8.5f}  {_pct_faster(ss.avg_itl, vs.avg_itl)}")
        lines.append(f"    Latency     vLLM={vs.avg_request_time:>8.3f}  SGLang={ss.avg_request_time:>8.3f}  {_pct_faster(ss.avg_request_time, vs.avg_request_time)}")
        lines.append("")

    # Verdict
    lines.extend(["=" * W, "  VERDICT", "=" * W, ""])

    wins = 0
    if sglang_run.avg_rollout_throughput > vllm_run.avg_rollout_throughput:
        d = ((sglang_run.avg_rollout_throughput - vllm_run.avg_rollout_throughput) / vllm_run.avg_rollout_throughput) * 100
        lines.append(f"  Throughput: SGLang {d:.1f}% higher")
        wins += 1
    else:
        d = ((vllm_run.avg_rollout_throughput - sglang_run.avg_rollout_throughput) / max(sglang_run.avg_rollout_throughput, 1e-9)) * 100
        lines.append(f"  Throughput: vLLM {d:.1f}% higher")

    if sglang_run.avg_ttft < vllm_run.avg_ttft and vllm_run.avg_ttft > 0:
        d = ((vllm_run.avg_ttft - sglang_run.avg_ttft) / vllm_run.avg_ttft) * 100
        lines.append(f"  TTFT:       SGLang {d:.1f}% faster")
        wins += 1
    elif vllm_run.avg_ttft > 0:
        d = ((sglang_run.avg_ttft - vllm_run.avg_ttft) / vllm_run.avg_ttft) * 100
        lines.append(f"  TTFT:       vLLM {d:.1f}% faster")

    if sglang_run.avg_itl < vllm_run.avg_itl and vllm_run.avg_itl > 0:
        d = ((vllm_run.avg_itl - sglang_run.avg_itl) / vllm_run.avg_itl) * 100
        lines.append(f"  ITL:        SGLang {d:.1f}% faster")
        wins += 1

    if sglang_run.avg_gpu_mem_gb < vllm_run.avg_gpu_mem_gb and vllm_run.avg_gpu_mem_gb > 0:
        d = ((vllm_run.avg_gpu_mem_gb - sglang_run.avg_gpu_mem_gb) / vllm_run.avg_gpu_mem_gb) * 100
        lines.append(f"  Memory:     SGLang {d:.1f}% less GPU memory")
        wins += 1

    lines.append("")
    if wins >= 3:
        lines.append("  >>> SGLang wins on rollout performance <<<")
    elif wins >= 2:
        lines.append("  >>> SGLang leads on most metrics <<<")
    else:
        lines.append("  >>> Results are mixed — check per-step details <<<")

    lines.extend(["", "=" * W])

    report = "\n".join(lines)
    with open(os.path.join(output_dir, "benchmark_report.txt"), "w") as f:
        f.write(report)
    return report


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


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]
