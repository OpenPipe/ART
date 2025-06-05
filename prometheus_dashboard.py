#!/usr/bin/env python3
"""
Simple Terminal Prometheus vLLM Dashboard
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
from collections import deque

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich import box


class PrometheusClient:
    def __init__(self, base_url: str = "http://0.0.0.0:9090"):
        self.base_url = base_url

    def get_metric_value(self, query: str) -> Optional[float]:
        """Get a single metric value"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/query", params={"query": query}, timeout=5
            )
            response.raise_for_status()
            result = response.json()
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
        except:
            pass
        return None


class Dashboard:
    def __init__(self):
        self.console = Console()
        self.client = PrometheusClient()
        self.history = deque(maxlen=30)

    def create_sparkline(self, values: List[float], width: int = 20) -> str:
        """Create a simple sparkline chart"""
        if not values or all(v == 0 for v in values):
            return "░" * width

        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return "▄" * width

        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        chars = " ▁▂▃▄▅▆▇█"

        # Ensure we generate exactly 'width' characters
        sparkline = ""
        if len(values) >= width:
            # If we have more values than width, sample evenly
            step = len(values) / width
            for i in range(width):
                idx = int(i * step)
                if idx < len(normalized):
                    char_idx = int(normalized[idx] * (len(chars) - 1))
                    sparkline += chars[char_idx]
        else:
            # If we have fewer values than width, interpolate or repeat
            for i in range(width):
                idx = int(i * len(values) / width)
                if idx < len(normalized):
                    char_idx = int(normalized[idx] * (len(chars) - 1))
                    sparkline += chars[char_idx]

        return sparkline[:width].ljust(width)  # Ensure exact width

    def collect_metrics(self) -> Dict:
        """Collect all metrics"""
        prompt_rate = (
            self.client.get_metric_value("rate(vllm:prompt_tokens_total[1m])") or 0
        )
        generation_rate = (
            self.client.get_metric_value("rate(vllm:generation_tokens_total[1m])") or 0
        )
        running_requests = int(
            self.client.get_metric_value("vllm:num_requests_running") or 0
        )
        swapped_requests = int(
            self.client.get_metric_value("vllm:num_requests_swapped") or 0
        )
        waiting_requests = int(
            self.client.get_metric_value("vllm:num_requests_waiting") or 0
        )

        # Fix cache usage (convert from 0-1 to percentage)
        cache_raw = self.client.get_metric_value("vllm:gpu_cache_usage_perc")
        gpu_cache = (cache_raw * 100) if cache_raw is not None else 0
        cpu_cache = self.client.get_metric_value("vllm:cpu_cache_usage_perc")
        cpu_cache = (cpu_cache * 100) if cpu_cache is not None else 0

        metrics = {
            "prompt_tokens": prompt_rate,
            "generation_tokens": generation_rate,
            "total_tokens": prompt_rate + generation_rate,
            "running_requests": running_requests,
            "swapped_requests": swapped_requests,
            "waiting_requests": waiting_requests,
            "total_requests": running_requests + swapped_requests + waiting_requests,
            "gpu_cache": gpu_cache,
            "cpu_cache": cpu_cache,
            "timestamp": datetime.now(),
        }

        self.history.append(metrics)
        return metrics

    def create_dashboard(self, metrics: Dict) -> Panel:
        """Create the main dashboard"""
        table = Table(
            show_header=True, header_style="bold cyan", box=box.ROUNDED, expand=True
        )
        table.add_column("Metric", style="white", width=25, no_wrap=True)
        table.add_column("Current", style="green", justify="right", width=12)
        table.add_column("Trend", style="yellow", ratio=2)

        # Calculate sparkline width based on console width
        console_width = self.console.width
        # Account for: Panel borders (2), table borders (2), column separators (3),
        # fixed columns (20 + 12 = 32), plus extra buffer for padding
        # Adding 6 extra characters as buffer to prevent overflow
        available_width = console_width - 2 - 2 - 3 - 32 - 12
        # Ensure minimum width
        sparkline_width = max(20, available_width)

        # Token throughput
        prompt_history = [h["prompt_tokens"] for h in self.history]
        generation_history = [h["generation_tokens"] for h in self.history]
        total_history = [h["total_tokens"] for h in self.history]

        table.add_row(
            "🚀 Prompt Tokens/sec",
            f"{metrics['prompt_tokens']:.1f}",
            self.create_sparkline(prompt_history, width=sparkline_width),
        )
        table.add_row(
            "🚀 Generation Tokens/sec",
            f"{metrics['generation_tokens']:.1f}",
            self.create_sparkline(generation_history, width=sparkline_width),
        )
        # table.add_row(
        #     "🚀 [bold]Total Tokens/sec[/bold]",
        #     f"[bold]{metrics['total_tokens']:.1f}[/bold]",
        #     self.create_sparkline(total_history, width=sparkline_width),
        # )

        table.add_row("", "", "")  # Spacer

        # Request states
        running_history = [h["running_requests"] for h in self.history]
        swapped_history = [h["swapped_requests"] for h in self.history]
        waiting_history = [h["waiting_requests"] for h in self.history]

        table.add_row(
            "📋 Running Requests",
            str(metrics["running_requests"]),
            self.create_sparkline(running_history, width=sparkline_width),
        )
        # table.add_row(
        #     "📋 Swapped Requests",
        #     str(metrics["swapped_requests"]),
        #     self.create_sparkline(swapped_history, width=sparkline_width),
        # )
        table.add_row(
            "📋 Pending Requests",
            str(metrics["waiting_requests"]),
            self.create_sparkline(waiting_history, width=sparkline_width),
        )
        # table.add_row(
        #     "📋 [bold]Total Requests[/bold]",
        #     f"[bold]{metrics['total_requests']}[/bold]",
        #     "",
        # )

        table.add_row("", "", "")  # Spacer

        # Cache usage
        gpu_cache_history = [h["gpu_cache"] for h in self.history]
        cpu_cache_history = [h["cpu_cache"] for h in self.history]

        table.add_row(
            "💾 GPU KV Cache",
            f"{metrics['gpu_cache']:.1f}%",
            self.create_sparkline(gpu_cache_history, width=sparkline_width),
        )
        # table.add_row(
        #     "💾 CPU KV Cache",
        #     f"{metrics['cpu_cache']:.1f}%",
        #     self.create_sparkline(cpu_cache_history, width=sparkline_width),
        # )

        timestamp = metrics["timestamp"].strftime("%H:%M:%S")
        return Panel(
            table,
            title=f"[bold magenta]vLLM Metrics Dashboard[/bold magenta] - {timestamp}",
            border_style="blue",
        )

    def run(self):
        """Run the dashboard"""
        with Live(console=self.console, refresh_per_second=2, screen=True) as live:
            while True:
                try:
                    metrics = self.collect_metrics()
                    dashboard = self.create_dashboard(metrics)
                    live.update(dashboard)
                    time.sleep(2)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(5)


if __name__ == "__main__":
    Dashboard().run()
