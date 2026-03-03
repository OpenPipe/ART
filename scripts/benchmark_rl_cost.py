#!/usr/bin/env python3
"""
RL Training Cost Comparison: SGLang vs vLLM

Uses the REAL just-the-facts example from ART:
- Scrapes actual news articles (500-2000 token prefixes)
- RULER reward model (OpenPipe's relative scoring) for differentiated rewards
- Conciseness penalty to break ties (realistic for summarization)
- Real GRPO training with backend.train()

This is the authentic ART training loop - no synthetic data.

Key Features:
    1. RULER Integration: Scores trajectories relative to each other, providing
       variance that allows GRPO to learn. Solves "training never runs" problem.
    
    2. Decoupled Generation/Scoring: Generate many rollouts (e.g., 32) but score
       them in smaller RULER groups (e.g., 8) for meaningful relative comparison.
    
    3. Training Effectiveness Tracking: Tracks steps_trained vs steps_skipped,
       and reports cost_per_training_update - the metric companies actually care about.

Requirements:
    - OPENROUTER_API_KEY env var (for reward model calls)
    - OPENAI_API_KEY env var (for RULER judge - uses gpt-4o-mini by default)
    - newspaper3k, aiohttp, beautifulsoup4, lxml (pip install with .[sglang])

Usage:
    # Run with RULER (default, recommended)
    # Generates 32 rollouts, scores in groups of 8
    python scripts/benchmark_rl_cost.py --backend sglang --output results_sglang.json
    python scripts/benchmark_rl_cost.py --backend vllm --output results_vllm.json
    
    # Custom generation/scoring sizes (decouple generation from RULER scoring)
    python scripts/benchmark_rl_cost.py --backend sglang --rollouts-per-step 64 --ruler-group-size 8
    
    # Debug RULER scoring
    python scripts/benchmark_rl_cost.py --backend sglang --ruler-debug
    
    # Use a different RULER judge model
    python scripts/benchmark_rl_cost.py --backend sglang --ruler-judge openai/gpt-4o
    
    # Disable RULER (not recommended - may skip training steps)
    python scripts/benchmark_rl_cost.py --backend sglang --no-ruler
    
    # Compare results
    python scripts/benchmark_rl_cost.py --compare results_sglang.json results_vllm.json
"""

# IMPORTANT: Import unsloth BEFORE any other ML libraries to prevent early CUDA initialization.
# This must happen before importing transformers, torch, vllm, or the art package.
# See: https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
import os
os.environ["IMPORT_UNSLOTH"] = "1"  # Tell art package to import unsloth early

try:
    import unsloth  # noqa: F401
except ImportError:
    pass  # unsloth not installed, continue without it

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import art

from openai.types.chat.chat_completion import Choice

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "just-the-facts"))

from art.rewards import ruler_score_group

# GPU hourly costs (USD)
GPU_COSTS = {
    "H100": 3.50,
    "A100_80GB": 2.50,
    "A100_40GB": 1.80,
    "A10G": 1.00,
    "L4": 0.70,
    "default": 2.00,
}

# Custom RULER rubric for summarization with conciseness emphasis
SUMMARIZATION_RUBRIC = """
- A summary that accurately captures all key facts from the article should score higher than one that misses important information.
- A summary with NO hallucinated facts should score significantly higher than one that adds information not in the original article.
- CONCISENESS MATTERS: Between two equally accurate summaries, the shorter one that still captures all key points should score higher. Verbose or padded summaries should be penalized.
- Neutral, unbiased language should score higher than language showing political or emotional bias.
- If one summary is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
"""


def apply_conciseness_penalty(group: "art.TrajectoryGroup", target_words: int = 200, penalty_per_50_words: float = 0.05) -> "art.TrajectoryGroup":
    """
    Apply a conciseness penalty to break ties between similar RULER scores.
    
    For summarization, shorter summaries (that still capture the facts) are better.
    This adds differentiation when RULER gives similar scores.
    
    Args:
        group: TrajectoryGroup with RULER scores already applied
        target_words: Ideal summary length in words
        penalty_per_50_words: Penalty for each 50 words over target
    
    Returns:
        The same group with adjusted rewards
    """
    for traj in group.trajectories:
        # Get the summary text from the last assistant message
        messages = traj.messages()
        summary_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                summary_text = msg.get("content", "")
                break
        
        if summary_text:
            word_count = len(summary_text.split())
            traj.metrics["word_count"] = word_count
            
            # Calculate penalty for being over target
            if word_count > target_words:
                excess_words = word_count - target_words
                penalty = (excess_words / 50) * penalty_per_50_words
                penalty = min(penalty, 0.2)  # Cap penalty at 0.2
                
                traj.metrics["conciseness_penalty"] = penalty
                traj.reward = max(0.0, traj.reward - penalty)
            else:
                traj.metrics["conciseness_penalty"] = 0.0
    
    return group


async def score_with_ruler_and_conciseness(
    group: "art.TrajectoryGroup",
    judge_model: str = "openai/gpt-4o-mini",
    debug: bool = False,
) -> "art.TrajectoryGroup | None":
    """
    Score trajectories using RULER, then apply conciseness penalty.
    
    This provides:
    1. Differentiated scores via RULER's relative ranking
    2. Additional variance via conciseness penalty to break ties
    """
    # First, score with RULER using our summarization rubric
    scored_group = await ruler_score_group(
        group,
        judge_model=judge_model,
        rubric=SUMMARIZATION_RUBRIC,
        swallow_exceptions=True,
        debug=debug,
    )
    
    if scored_group is None:
        return None
    
    # Then apply conciseness penalty to break ties
    return apply_conciseness_penalty(scored_group)


@dataclass
class TimingStats:
    """Accumulated timing statistics."""
    total_rollout_time: float = 0.0
    total_train_time: float = 0.0
    rollout_counts: list[float] = field(default_factory=list)
    train_counts: list[float] = field(default_factory=list)
    tokens_generated: int = 0
    rollouts_completed: int = 0
    steps_completed: int = 0
    # Track actual training vs skipped steps
    steps_trained: int = 0  # Steps where gradient update ran
    steps_skipped: int = 0  # Steps skipped due to low variance
    trained_step_times: list[float] = field(default_factory=list)  # Only trained steps


@dataclass 
class BenchmarkResult:
    """Complete benchmark results."""
    backend: str
    model: str
    gpu_type: str
    num_steps: int
    rollouts_per_step: int
    
    # Timing breakdown
    total_time_seconds: float
    total_rollout_time_seconds: float
    total_train_time_seconds: float
    avg_rollout_time_seconds: float
    avg_train_time_seconds: float
    rollout_pct: float
    train_pct: float
    
    # Throughput
    rollouts_completed: int
    rollouts_per_second: float
    tokens_generated: int
    tokens_per_second: float
    
    # Cost
    gpu_hours: float
    estimated_cost_usd: float
    cost_per_1k_rollouts_usd: float
    
    # Training effectiveness (the metrics companies care about)
    steps_trained: int  # Steps where gradient update actually ran
    steps_skipped: int  # Steps skipped due to low reward variance
    training_efficiency_pct: float  # steps_trained / total_steps * 100
    cost_per_training_update_usd: float  # Cost per actual gradient update
    avg_train_time_trained_steps: float  # Avg train time for steps that ran
    
    # Training quality metrics
    avg_reward: float
    avg_fact_recall: float
    avg_hallucination: float


def get_gpu_info() -> tuple[str, float]:
    """Get GPU type and hourly cost."""
    gpu_type = "default"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            for key in GPU_COSTS:
                if key.lower().replace("_", "") in name.replace("-", "").replace(" ", ""):
                    gpu_type = key
                    break
    except Exception:
        pass
    return gpu_type, GPU_COSTS.get(gpu_type, GPU_COSTS["default"])


async def run_benchmark(
    backend_type: str,
    model_name: str,
    num_steps: int,
    rollouts_per_step: int,
    ruler_group_size: int = 8,  # RULER scores groups of this size
    use_ruler: bool = True,
    ruler_judge_model: str = "openai/gpt-4o-mini",
    ruler_debug: bool = False,
) -> BenchmarkResult:
    """Run actual ART training with the just-the-facts example."""
    
    import art
    from art import TrajectoryGroup
    from art.utils.strip_logprobs import strip_logprobs
    import weave
    
    # Import real just-the-facts components
    from just_the_facts.rollout import rollout, FactsScenario
    from just_the_facts.scenarios import train_scenarios
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment variable required for reward model")
    
    weave.init(f"rl-cost-benchmark-{backend_type}", global_postprocess_output=strip_logprobs)
    
    gpu_type, gpu_cost = get_gpu_info()
    
    num_ruler_groups = (rollouts_per_step + ruler_group_size - 1) // ruler_group_size
    
    print(f"\n{'='*70}")
    print(f"ART RL Cost Benchmark: {backend_type.upper()}")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hr)")
    print(f"Steps: {num_steps}")
    print(f"Generation: {rollouts_per_step} rollouts/step")
    print(f"RULER scoring: {num_ruler_groups} groups of {ruler_group_size} (decoupled from generation)" if use_ruler else "Reward: LLM checks only (coarse)")
    print(f"Reward: RULER ({ruler_judge_model}) + conciseness penalty" if use_ruler else "")
    print(f"Using just-the-facts with real articles")
    print(f"{'='*70}\n")
    
    # Initialize backend
    if backend_type == "sglang":
        from art.sglang_backend import SGLangBackend
        backend = SGLangBackend()
    else:
        from art.local import LocalBackend
        backend = LocalBackend()
    
    # Time-sharing mode: vLLM and Unsloth share GPU 0
    # vLLM sleeps during training, Unsloth offloads during inference
    model = art.TrainableModel(
        name=f"facts-bench-{backend_type}",
        project="rl-cost-benchmark",
        base_model=model_name,
        _internal_config={
            "engine_args": {
                "gpu_memory_utilization": 0.80,
            },
        },
    )
    
    print("=" * 60)
    print("BENCHMARK CODE VERSION: 2026-02-01-v2")
    print("=" * 60)
    print("Registering model...")
    await model.register(backend)
    
    # Test vLLM server connectivity with retries
    print(f"Model inference URL: {model.inference_base_url}")
    print(f"Model inference name: {model.inference_model_name}")
    print(f"Model name: {model.name}")
    print(f"Testing vLLM server connectivity...")
    from openai import AsyncOpenAI
    test_client = AsyncOpenAI(
        api_key=model.inference_api_key or "dummy",
        base_url=model.inference_base_url,
    )
    for attempt in range(5):
        try:
            test_resp = await test_client.chat.completions.create(
                model=model.name,
                messages=[{"role": "user", "content": "Say 'hello'"}],
                max_tokens=5,
            )
            print(f"vLLM server OK: {test_resp.choices[0].message.content}")
            break
        except Exception as e:
            print(f"vLLM server test attempt {attempt+1}/5 FAILED: {type(e).__name__}: {e}")
            if attempt < 4:
                print("  Waiting 5 seconds before retry...")
                await asyncio.sleep(5)
            else:
                print("  vLLM server not responding after 5 attempts. Continuing anyway...")
    
    stats = TimingStats()
    all_rewards = []
    all_fact_recall = []
    all_hallucination = []
    
    total_start = time.perf_counter()
    
    # Use scenarios from just-the-facts (real news article URLs)
    scenarios_to_use = train_scenarios[:num_steps]  # One scenario per step
    
    for step, scenario in enumerate(scenarios_to_use):
        step_start = time.perf_counter()
        print(f"\n--- Step {step + 1}/{num_steps} ---")
        print(f"  Article: {scenario.article_url[:60]}...")
        
        # === ROLLOUT PHASE ===
        # All rollouts share the same article = same long prefix
        # DECOUPLED: Generate rollouts_per_step rollouts, but score in groups of ruler_group_size
        # This allows generating more samples for efficiency while keeping RULER groups
        # small enough for meaningful relative comparisons
        rollout_start = time.perf_counter()
        
        # Calculate how many RULER groups we need
        num_groups = (rollouts_per_step + ruler_group_size - 1) // ruler_group_size
        
        train_groups = await art.gather_trajectory_groups(
            (
                TrajectoryGroup(
                    rollout(model, scenario) 
                    for _ in range(min(ruler_group_size, rollouts_per_step - (group_idx * ruler_group_size)))
                )
                for group_idx in range(num_groups)
            ),
            # Use RULER + conciseness penalty for differentiated rewards
            # Each group is scored independently, enabling relative ranking within smaller batches
            after_each=lambda group: (
                score_with_ruler_and_conciseness(
                    group,
                    judge_model=ruler_judge_model,
                    debug=ruler_debug,
                )
                if use_ruler
                else None
            ),
            pbar_desc=f"step {step+1} rollouts ({num_groups} groups of {ruler_group_size})",
            max_exceptions=3,
        )
        
        rollout_time = time.perf_counter() - rollout_start
        stats.total_rollout_time += rollout_time
        stats.rollout_counts.append(rollout_time)
        
        # Collect metrics
        step_rollouts = 0
        step_tokens = 0
        step_rewards = []
        step_ruler_scores = []
        step_fact_recall = []
        step_hallucination = []
        step_word_counts = []
        
        # Debug: show what's in each group
        print(f"  [DEBUG] Got {len(train_groups)} groups")
        for i, group in enumerate(train_groups):
            print(f"  [DEBUG] Group {i}: {len(group.trajectories)} trajectories, {len(group.exceptions)} exceptions")
            # Print any exceptions that occurred
            if group.exceptions:
                for exc in group.exceptions:
                    print(f"    - {exc.type}: {exc.message}")
                    # Extract APIStatusError details from traceback
                    if exc.traceback and "APIStatusError" in exc.traceback:
                        print(f"    [Extracting APIStatusError details from traceback...]")
                        for line in exc.traceback.split('\n'):
                            if "status_code" in line.lower() or "error" in line.lower() or "api" in line.lower():
                                print(f"      {line.strip()}")
                    # Print last 10 lines of traceback for more context
                    if exc.traceback:
                        tb_lines = exc.traceback.strip().split('\n')
                        print(f"    [Full traceback last 10 lines:]")
                        for line in tb_lines[-10:]:
                            print(f"      {line}")
            for traj in group.trajectories:
                step_rollouts += 1
                step_rewards.append(traj.reward)
                
                # Collect RULER-specific metrics
                if "ruler_score" in traj.metrics:
                    step_ruler_scores.append(traj.metrics["ruler_score"])
                if "word_count" in traj.metrics:
                    step_word_counts.append(traj.metrics["word_count"])
                
                # Collect original check metrics (preserved in independent_reward flow)
                if "fact_recall" in traj.metrics:
                    step_fact_recall.append(traj.metrics["fact_recall"])
                if "hallucinated_facts" in traj.metrics:
                    step_hallucination.append(traj.metrics["hallucinated_facts"])
                
                # Token counting from response content
                for item in traj.messages_and_choices:
                    if isinstance(item, Choice):
                        content = getattr(item.message, 'content', None)
                        if content:
                            step_tokens += len(content) // 4
        
        stats.rollouts_completed += step_rollouts
        stats.tokens_generated += step_tokens
        all_rewards.extend(step_rewards)
        all_fact_recall.extend(step_fact_recall)
        all_hallucination.extend(step_hallucination)
        
        avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
        avg_recall = sum(step_fact_recall) / len(step_fact_recall) if step_fact_recall else 0
        reward_variance = (sum((r - avg_reward)**2 for r in step_rewards) / len(step_rewards)) if len(step_rewards) > 1 else 0
        
        print(f"  Rollouts: {step_rollouts} in {rollout_time:.2f}s ({step_rollouts/rollout_time:.1f}/s)")
        if step_rewards:
            print(f"  Reward: avg={avg_reward:.3f}, var={reward_variance:.4f}, range=[{min(step_rewards):.3f}, {max(step_rewards):.3f}]")
        else:
            print(f"  Reward: No successful rollouts - check exceptions above")
        if step_ruler_scores:
            print(f"  RULER scores: avg={sum(step_ruler_scores)/len(step_ruler_scores):.3f}")
        if step_word_counts:
            print(f"  Word counts: avg={sum(step_word_counts)//len(step_word_counts)}, range=[{min(step_word_counts)}, {max(step_word_counts)}]")
        
        # === TRAINING PHASE ===
        train_start = time.perf_counter()
        
        # RULER + conciseness penalty provides differentiated rewards,
        # so scale_rewards=True allows GRPO to learn from the variance
        result = await backend.train(
            model,
            train_groups,
            learning_rate=1e-6,  # Matches just-the-facts config
            scale_rewards=True,  # Enable reward scaling (RULER provides variance to scale)
            verbose=False,
        )
        
        train_time = time.perf_counter() - train_start
        stats.total_train_time += train_time
        stats.train_counts.append(train_time)
        stats.steps_completed += 1
        
        # Detect if training actually ran or was skipped
        # A skipped step has no loss or very fast time (just checkpoint overhead)
        loss = result.metrics.get('loss')
        step_trained = loss is not None and train_time > 2.0  # Real training takes >2s
        
        if step_trained:
            stats.steps_trained += 1
            stats.trained_step_times.append(train_time)
            print(f"  Training: {train_time:.2f}s, loss: {loss:.4f} [TRAINED]")
        else:
            stats.steps_skipped += 1
            skip_reason = "no loss" if loss is None else f"fast ({train_time:.1f}s, likely no gradient)"
            print(f"  Training: {train_time:.2f}s, loss: {loss} [SKIPPED - {skip_reason}]")
        
        step_time = time.perf_counter() - step_start
        print(f"  Step total: {step_time:.2f}s")
    
    total_time = time.perf_counter() - total_start
    
    print("\nShutting down...")
    await backend.close()
    
    # Calculate final metrics
    gpu_hours = total_time / 3600
    estimated_cost = gpu_hours * gpu_cost
    cost_per_1k = (estimated_cost / stats.rollouts_completed) * 1000 if stats.rollouts_completed > 0 else 0
    
    # THE METRIC COMPANIES CARE ABOUT: cost per actual training update
    cost_per_training_update = estimated_cost / stats.steps_trained if stats.steps_trained > 0 else float('inf')
    training_efficiency = (stats.steps_trained / num_steps * 100) if num_steps > 0 else 0
    avg_train_time_trained = (
        sum(stats.trained_step_times) / len(stats.trained_step_times) 
        if stats.trained_step_times else 0
    )
    
    return BenchmarkResult(
        backend=backend_type,
        model=model_name,
        gpu_type=gpu_type,
        num_steps=num_steps,
        rollouts_per_step=rollouts_per_step,
        total_time_seconds=total_time,
        total_rollout_time_seconds=stats.total_rollout_time,
        total_train_time_seconds=stats.total_train_time,
        avg_rollout_time_seconds=stats.total_rollout_time / num_steps if num_steps > 0 else 0,
        avg_train_time_seconds=stats.total_train_time / num_steps if num_steps > 0 else 0,
        rollout_pct=stats.total_rollout_time / total_time * 100 if total_time > 0 else 0,
        train_pct=stats.total_train_time / total_time * 100 if total_time > 0 else 0,
        rollouts_completed=stats.rollouts_completed,
        rollouts_per_second=stats.rollouts_completed / stats.total_rollout_time if stats.total_rollout_time > 0 else 0,
        tokens_generated=stats.tokens_generated,
        tokens_per_second=stats.tokens_generated / stats.total_rollout_time if stats.total_rollout_time > 0 else 0,
        gpu_hours=gpu_hours,
        estimated_cost_usd=estimated_cost,
        cost_per_1k_rollouts_usd=cost_per_1k,
        # Training effectiveness metrics
        steps_trained=stats.steps_trained,
        steps_skipped=stats.steps_skipped,
        training_efficiency_pct=training_efficiency,
        cost_per_training_update_usd=cost_per_training_update,
        avg_train_time_trained_steps=avg_train_time_trained,
        # Quality metrics
        avg_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        avg_fact_recall=sum(all_fact_recall) / len(all_fact_recall) if all_fact_recall else 0,
        avg_hallucination=sum(all_hallucination) / len(all_hallucination) if all_hallucination else 0,
    )


def print_results(r: BenchmarkResult) -> None:
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {r.backend.upper()}")
    print(f"{'='*70}")
    print(f"Model: {r.model}")
    print(f"GPU: {r.gpu_type}")
    
    print(f"\n⏱️  TIME BREAKDOWN:")
    print(f"  Total: {r.total_time_seconds:.1f}s ({r.total_time_seconds/60:.1f} min)")
    print(f"  Rollouts: {r.total_rollout_time_seconds:.1f}s ({r.rollout_pct:.1f}%)")
    print(f"  Training: {r.total_train_time_seconds:.1f}s ({r.train_pct:.1f}%)")
    print(f"  Avg rollout/step: {r.avg_rollout_time_seconds:.2f}s")
    print(f"  Avg train/step: {r.avg_train_time_seconds:.2f}s")
    
    print(f"\n🚀 THROUGHPUT:")
    print(f"  Rollouts: {r.rollouts_completed} total")
    print(f"  Rollouts/sec: {r.rollouts_per_second:.2f}")
    print(f"  Tokens/sec: {r.tokens_per_second:.0f}")
    
    print(f"\n📊 QUALITY:")
    print(f"  Avg reward: {r.avg_reward:.3f}")
    print(f"  Avg fact recall: {r.avg_fact_recall:.3f}")
    print(f"  Avg hallucination: {r.avg_hallucination:.3f}")
    
    print(f"\n🎯 TRAINING EFFECTIVENESS:")
    print(f"  Steps trained: {r.steps_trained}/{r.num_steps} ({r.training_efficiency_pct:.1f}% efficiency)")
    print(f"  Steps skipped: {r.steps_skipped} (no gradient update)")
    if r.steps_trained > 0:
        print(f"  Avg train time (trained steps only): {r.avg_train_time_trained_steps:.2f}s")
    
    print(f"\n💰 COST:")
    print(f"  GPU hours: {r.gpu_hours:.4f}")
    print(f"  Estimated cost: ${r.estimated_cost_usd:.4f}")
    print(f"  Cost/1K rollouts: ${r.cost_per_1k_rollouts_usd:.4f}")
    if r.steps_trained > 0:
        print(f"  Cost/training update: ${r.cost_per_training_update_usd:.4f}  ← THE METRIC THAT MATTERS")
    else:
        print(f"  Cost/training update: N/A (no steps trained!)")
    
    print(f"{'='*70}\n")


def compare_results(sglang_file: str, vllm_file: str) -> None:
    """Compare two benchmark results."""
    with open(sglang_file) as f:
        sg = json.load(f)
    with open(vllm_file) as f:
        vl = json.load(f)
    
    def delta(sg_val: float, vl_val: float, lower_is_better: bool = True) -> str:
        if vl_val == 0:
            return "N/A"
        pct = (vl_val - sg_val) / vl_val * 100
        if lower_is_better:
            return f"{pct:+.1f}%" if pct > 0 else f"{pct:.1f}%"
        else:
            return f"{-pct:+.1f}%" if pct < 0 else f"{-pct:.1f}%"
    
    print(f"\n{'='*80}")
    print("COMPARISON: SGLang vs vLLM (just-the-facts benchmark)")
    print(f"{'='*80}")
    print(f"Model: {sg['model']}")
    print(f"Steps: {sg['num_steps']}, Rollouts/step: {sg['rollouts_per_step']}")
    
    print(f"\n{'Metric':<35} {'vLLM':>15} {'SGLang':>15} {'Δ SGLang':>12}")
    print("-" * 80)
    
    # Time breakdown
    print(f"{'Total time (s)':<35} {vl['total_time_seconds']:>15.1f} {sg['total_time_seconds']:>15.1f} {delta(sg['total_time_seconds'], vl['total_time_seconds']):>12}")
    print(f"{'Rollout time (s)':<35} {vl['total_rollout_time_seconds']:>15.1f} {sg['total_rollout_time_seconds']:>15.1f} {delta(sg['total_rollout_time_seconds'], vl['total_rollout_time_seconds']):>12}")
    print(f"{'Train time (s)':<35} {vl['total_train_time_seconds']:>15.1f} {sg['total_train_time_seconds']:>15.1f} {delta(sg['total_train_time_seconds'], vl['total_train_time_seconds']):>12}")
    print(f"{'Rollout % of total':<35} {vl['rollout_pct']:>14.1f}% {sg['rollout_pct']:>14.1f}%")
    
    print()
    print(f"{'Rollouts/sec':<35} {vl['rollouts_per_second']:>15.2f} {sg['rollouts_per_second']:>15.2f} {delta(sg['rollouts_per_second'], vl['rollouts_per_second'], False):>12}")
    print(f"{'Tokens/sec':<35} {vl['tokens_per_second']:>15.0f} {sg['tokens_per_second']:>15.0f} {delta(sg['tokens_per_second'], vl['tokens_per_second'], False):>12}")
    
    print()
    print(f"{'Cost/1K rollouts ($)':<35} {vl['cost_per_1k_rollouts_usd']:>15.4f} {sg['cost_per_1k_rollouts_usd']:>15.4f} {delta(sg['cost_per_1k_rollouts_usd'], vl['cost_per_1k_rollouts_usd']):>12}")
    
    # Training effectiveness - THE METRICS THAT MATTER
    print(f"\n{'='*80}")
    print("TRAINING EFFECTIVENESS (the metrics companies care about)")
    print("-" * 80)
    print(f"{'Steps trained':<35} {vl['steps_trained']:>15} {sg['steps_trained']:>15}")
    print(f"{'Steps skipped':<35} {vl['steps_skipped']:>15} {sg['steps_skipped']:>15}")
    print(f"{'Training efficiency %':<35} {vl['training_efficiency_pct']:>14.1f}% {sg['training_efficiency_pct']:>14.1f}%")
    
    if sg['steps_trained'] > 0 and vl['steps_trained'] > 0:
        print(f"{'Cost/training update ($)':<35} {vl['cost_per_training_update_usd']:>15.4f} {sg['cost_per_training_update_usd']:>15.4f} {delta(sg['cost_per_training_update_usd'], vl['cost_per_training_update_usd']):>12}")
    else:
        print(f"{'Cost/training update ($)':<35} {'N/A':>15} {'N/A':>15} (some backend had 0 trained steps)")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHT: RadixAttention benefit on rollout generation")
    print("(All rollouts per step share the same article = long shared prefix)")
    
    rollout_speedup = (vl['total_rollout_time_seconds'] - sg['total_rollout_time_seconds']) / vl['total_rollout_time_seconds'] * 100 if vl['total_rollout_time_seconds'] > 0 else 0
    
    if rollout_speedup > 0:
        print(f"\n  SGLang is {rollout_speedup:.1f}% faster on rollout generation")
        print(f"  This is where RadixAttention's prefix caching helps")
    else:
        print(f"\n  vLLM is {-rollout_speedup:.1f}% faster on rollout generation")
    
    # Cost savings at scale
    print(f"\n📈 PROJECTED SAVINGS AT SCALE:")
    for scale_name, rollouts in [("10K rollouts", 10000), ("100K rollouts", 100000), ("1M rollouts", 1000000)]:
        vl_cost = vl['cost_per_1k_rollouts_usd'] * (rollouts / 1000)
        sg_cost = sg['cost_per_1k_rollouts_usd'] * (rollouts / 1000)
        savings = vl_cost - sg_cost
        if savings > 0:
            print(f"  {scale_name}: Save ${savings:.2f} ({savings/vl_cost*100:.1f}%)")
        else:
            print(f"  {scale_name}: Extra ${-savings:.2f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="ART RL Training Cost Benchmark (just-the-facts)")
    parser.add_argument("--backend", choices=["sglang", "vllm"], help="Backend to benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--num-steps", type=int, default=5, help="Training steps (1 article per step)")
    parser.add_argument("--rollouts-per-step", type=int, default=32, 
                        help="Total rollouts to generate per step (default: 32)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("SGLANG", "VLLM"), help="Compare results")
    
    # RULER configuration - DECOUPLED from generation
    parser.add_argument("--ruler-group-size", type=int, default=8,
                        help="RULER scores groups of this size (default: 8). "
                             "Decoupled from --rollouts-per-step to allow generating more samples "
                             "while keeping scoring groups small for meaningful relative comparison.")
    parser.add_argument("--no-ruler", action="store_true",
                        help="Disable RULER (use coarse LLM checks only - may cause training to skip)")
    parser.add_argument("--ruler-judge", default="openai/gpt-4o-mini",
                        help="RULER judge model (default: openai/gpt-4o-mini)")
    parser.add_argument("--ruler-debug", action="store_true",
                        help="Print RULER judge reasoning for debugging")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    if not args.backend:
        parser.error("--backend required unless using --compare")
    
    result = asyncio.run(run_benchmark(
        args.backend,
        args.model,
        args.num_steps,
        args.rollouts_per_step,
        ruler_group_size=args.ruler_group_size,
        use_ruler=not args.no_ruler,
        ruler_judge_model=args.ruler_judge,
        ruler_debug=args.ruler_debug,
    ))
    
    print_results(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
