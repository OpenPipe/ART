#!/usr/bin/env python3
"""
2048 Game RL Rollout Benchmark: SGLang vs vLLM

Real RL task showing where SGLang's prefix caching helps:
- System prompt is shared across ALL moves in ALL games
- Each game is multi-turn (10-50 moves)
- Perfect use case for RadixAttention

Usage:
    python scripts/benchmark_2048_rollout.py --backend vllm --output results_vllm.json
    python scripts/benchmark_2048_rollout.py --backend sglang --output results_sglang.json
    python scripts/benchmark_2048_rollout.py --compare results_sglang.json results_vllm.json
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import aiohttp

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "2048"))

# GPU hourly costs (USD)
GPU_COSTS = {
    "H100": 3.50,
    "A100_80GB": 2.50,
    "A100_40GB": 1.80,
    "A10G": 1.00,
    "L4": 0.70,
    "default": 2.00,
}

SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"


@dataclass
class BenchmarkResult:
    """Benchmark results for 2048 rollout comparison."""
    backend: str
    model: str
    gpu_type: str
    num_games: int
    completed_games: int
    failed_games: int
    
    # Timing
    total_time_seconds: float
    
    # Game metrics
    total_moves: int
    avg_moves_per_game: float
    total_wins: int
    win_rate: float
    
    # Throughput
    moves_per_second: float
    games_per_second: float
    
    # Cost
    gpu_hours: float
    estimated_cost_usd: float
    cost_per_100_games_usd: float


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


async def wait_for_server(host: str, port: int, timeout: float = 180.0) -> None:
    """Wait for server to be ready."""
    start_time = time.time()
    print("Waiting for server", end="", flush=True)
    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        print(" ready!")
                        return
        except Exception:
            pass
        print(".", end="", flush=True)
        await asyncio.sleep(2)
    raise TimeoutError(f"\nServer did not start within {timeout} seconds")


def start_vllm_server(model_name: str) -> subprocess.Popen:
    """Start vLLM server for 2048 benchmark."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "16",  # Sequential execution
        "--enable-prefix-caching",
    ]
    print(f"Starting vLLM server")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def start_sglang_server(model_name: str) -> subprocess.Popen:
    """Start SGLang server for 2048 benchmark."""
    sglang_python = sys.executable
    if os.path.exists(".venv-sglang-server/bin/python"):
        sglang_python = os.path.abspath(".venv-sglang-server/bin/python")
        print(f"Using SGLang server venv: {sglang_python}")
    
    cmd = [
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--mem-fraction-static", "0.90",
        "--max-running-requests", "16",  # Sequential execution
        "--max-total-tokens", "32768",
    ]
    print(f"Starting SGLang server")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Stop server subprocess."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


async def run_benchmark(
    backend_type: str,
    model_name: str,
    num_games: int,
) -> BenchmarkResult:
    """Run 2048 rollout benchmark."""
    
    import art
    from rollout import rollout as _original_rollout
    
    # Wrapper to limit max moves per game
    async def rollout_with_limit(model, step, is_validation, max_moves=30):
        """Rollout with move limit to prevent infinite games."""
        import openai
        from utils import (
            WINNING_VALUE,
            apply_agent_move,
            check_game_finished,
            generate_game,
            max_cell_value,
            render_board,
            total_board_value,
        )
        import math
        
        game = generate_game()
        move_number = 0
        
        trajectory = art.Trajectory(
            messages_and_choices=[
                {
                    "role": "system",
                    "content": "You are an excellent 2048 player. Always choose the move most likely to lead to combine cells to eventually reach the number 2048. Optional moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>",
                },
            ],
            metadata={
                "game_id": game["id"],
                "step": step,
                "validation": is_validation,
            },
            reward=0,
        )
        
        while move_number < max_moves:
            trajectory.messages_and_choices.append(
                {"role": "user", "content": render_board(game)}
            )
            
            client = model.openai_client()
            try:
                chat_completion = await client.chat.completions.create(
                    max_completion_tokens=128,
                    messages=trajectory.messages(),
                    model=model.name,
                )
            except Exception as e:
                trajectory.metrics["invalid_move"] = 1
                trajectory.reward = -1
                break
            
            choice = chat_completion.choices[0]
            content = choice.message.content
            assert isinstance(content, str)
            trajectory.messages_and_choices.append(choice)
            
            try:
                apply_agent_move(game, content)
                move_number += 1
            except ValueError:
                trajectory.metrics["invalid_move"] = 1
                trajectory.reward = -1
                break
            
            if check_game_finished(game):
                trajectory.metrics["invalid_move"] = 0
                break
        
        max_value = max_cell_value(game)
        board_value = total_board_value(game)
        agent_won = max_value == WINNING_VALUE
        trajectory.metrics["max_value"] = max_value
        trajectory.metrics["board_value"] = board_value
        trajectory.metrics["num_moves"] = move_number
        trajectory.metrics["win"] = agent_won
        
        if agent_won:
            trajectory.reward = 2
        else:
            max_value_reward = (math.log(max_value, 2) - 1) / (math.log(WINNING_VALUE, 2) - 1)
            board_value_reward = (math.log(board_value, 2) - 1) / (math.log(WINNING_VALUE * 16, 2) - 1)
            trajectory.reward = max_value_reward + (board_value_reward * 0.2)
        
        return trajectory
    
    rollout = rollout_with_limit
    
    gpu_type, gpu_cost = get_gpu_info()
    
    print(f"\n{'='*60}")
    print(f"2048 Game Benchmark: {backend_type.upper()} (ROLLOUTS ONLY)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hr)")
    print(f"Games: {num_games}")
    print(f"{'='*60}\n")
    
    # Kill any existing servers
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    await asyncio.sleep(2)
    
    # Start server
    print(f"Starting {backend_type} server...")
    if backend_type == "sglang":
        server_proc = start_sglang_server(model_name)
    else:
        server_proc = start_vllm_server(model_name)
    
    try:
        await wait_for_server(SERVER_HOST, SERVER_PORT)
        
        # Create model pointing to local server
        model = art.Model(
            name=model_name,
            project="2048-benchmark",
            inference_api_key="dummy",
            inference_base_url=f"http://{SERVER_HOST}:{SERVER_PORT}/v1",
            inference_model_name=model_name,
        )
        
        # Warm up
        print("Warming up...")
        await rollout(model, step=0, is_validation=False, max_moves=30)
        print("Warm-up complete.\n")
        
        # Run games SEQUENTIALLY (simple and reliable)
        print(f"Playing {num_games} games sequentially:")
        total_start = time.perf_counter()
        all_trajectories = []
        failed_count = 0
        
        for game_idx in range(num_games):
            print(f"  Game {game_idx + 1}/{num_games}...", end="", flush=True)
            game_start = time.perf_counter()
            
            # Progress updater - print dot every 2 seconds to show it's alive
            progress_task = None
            progress_stopped = asyncio.Event()
            
            async def show_progress():
                while not progress_stopped.is_set():
                    try:
                        await asyncio.wait_for(progress_stopped.wait(), timeout=2.0)
                        break
                    except asyncio.TimeoutError:
                        elapsed = time.perf_counter() - game_start
                        print(f"[{elapsed:.0f}s]", end="", flush=True)
            
            try:
                progress_task = asyncio.create_task(show_progress())
                
                # 45 second timeout per game (max 30 moves × ~1s/move = ~30s + margin)
                traj = await asyncio.wait_for(
                    rollout(model, step=game_idx, is_validation=False, max_moves=30),
                    timeout=45.0
                )
                
                progress_stopped.set()
                await progress_task
                
                game_time = time.perf_counter() - game_start
                moves = traj.metrics.get("num_moves", 0)
                max_val = traj.metrics.get("max_value", 0)
                won = "🏆" if traj.metrics.get("win", False) else ""
                print(f" ✓ {moves} moves, max={max_val} in {game_time:.1f}s {won}")
                all_trajectories.append(traj)
            except asyncio.TimeoutError:
                progress_stopped.set()
                if progress_task:
                    await progress_task
                print(f" ✗ timeout (45s)")
                failed_count += 1
            except Exception as e:
                progress_stopped.set()
                if progress_task:
                    await progress_task
                print(f" ✗ {type(e).__name__}")
                failed_count += 1
        
        total_time = time.perf_counter() - total_start
        
        print(f"\n✓ {len(all_trajectories)}/{num_games} games completed", end="")
        if failed_count > 0:
            print(f" ({failed_count} failed)")
        else:
            print()
        
    finally:
        print("\nShutting down server...")
        stop_server(server_proc)
    
    # Calculate metrics
    completed_games = len(all_trajectories)
    total_moves = sum(t.metrics.get("num_moves", 0) for t in all_trajectories)
    total_wins = sum(1 for t in all_trajectories if t.metrics.get("win", False))
    
    gpu_hours = total_time / 3600
    estimated_cost = gpu_hours * gpu_cost
    cost_per_100 = (estimated_cost / completed_games) * 100 if completed_games > 0 else 0
    
    return BenchmarkResult(
        backend=backend_type,
        model=model_name,
        gpu_type=gpu_type,
        num_games=num_games,
        completed_games=completed_games,
        failed_games=failed_count,
        total_time_seconds=total_time,
        total_moves=total_moves,
        avg_moves_per_game=total_moves / completed_games if completed_games > 0 else 0,
        total_wins=total_wins,
        win_rate=total_wins / completed_games * 100 if completed_games > 0 else 0,
        moves_per_second=total_moves / total_time if total_time > 0 else 0,
        games_per_second=completed_games / total_time if total_time > 0 else 0,
        gpu_hours=gpu_hours,
        estimated_cost_usd=estimated_cost,
        cost_per_100_games_usd=cost_per_100,
    )


def print_results(r: BenchmarkResult) -> None:
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {r.backend.upper()}")
    print(f"{'='*60}")
    print(f"Model: {r.model}")
    print(f"GPU: {r.gpu_type}")
    
    print(f"\n🎮 GAMES:")
    print(f"  Attempted: {r.num_games}")
    print(f"  Completed: {r.completed_games} ({r.completed_games/r.num_games*100:.1f}%)")
    if r.failed_games > 0:
        print(f"  Failed: {r.failed_games} (timeout or error)")
    print(f"  Wins: {r.total_wins} ({r.win_rate:.1f}%)")
    print(f"  Total moves: {r.total_moves}")
    print(f"  Avg moves/game: {r.avg_moves_per_game:.1f}")
    
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Total time: {r.total_time_seconds:.1f}s")
    print(f"  Games/sec: {r.games_per_second:.2f}")
    print(f"  Moves/sec: {r.moves_per_second:.1f}")
    
    print(f"\n💰 COST:")
    print(f"  GPU hours: {r.gpu_hours:.4f}")
    print(f"  Total cost: ${r.estimated_cost_usd:.4f}")
    print(f"  Cost/100 games: ${r.cost_per_100_games_usd:.4f}")
    
    print(f"{'='*60}\n")


def compare_results(sglang_file: str, vllm_file: str) -> None:
    """Compare SGLang vs vLLM on 2048."""
    with open(sglang_file) as f:
        sg = json.load(f)
    with open(vllm_file) as f:
        vl = json.load(f)
    
    print(f"\n{'='*70}")
    print("2048 Game: SGLang vs vLLM Comparison")
    print(f"{'='*70}")
    print(f"Model: {sg['model']}")
    print(f"Games attempted: {sg['num_games']}")
    print(f"vLLM completed: {vl['completed_games']}/{vl['num_games']} ({vl['completed_games']/vl['num_games']*100:.1f}%)")
    print(f"SGLang completed: {sg['completed_games']}/{sg['num_games']} ({sg['completed_games']/sg['num_games']*100:.1f}%)")
    
    print(f"\n{'Metric':<30} {'vLLM':>15} {'SGLang':>15} {'Difference':>12}")
    print("-" * 70)
    
    # Time
    time_savings = (vl['total_time_seconds'] - sg['total_time_seconds']) / vl['total_time_seconds'] * 100
    print(f"{'Total time (s)':<30} {vl['total_time_seconds']:>15.1f} {sg['total_time_seconds']:>15.1f} {time_savings:>11.1f}%")
    
    # Throughput
    speed_gain = (sg['moves_per_second'] - vl['moves_per_second']) / vl['moves_per_second'] * 100
    print(f"{'Moves/sec':<30} {vl['moves_per_second']:>15.1f} {sg['moves_per_second']:>15.1f} {speed_gain:>+11.1f}%")
    
    # Cost
    cost_savings = (vl['cost_per_100_games_usd'] - sg['cost_per_100_games_usd']) / vl['cost_per_100_games_usd'] * 100
    print(f"{'Cost/100 games ($)':<30} {vl['cost_per_100_games_usd']:>15.4f} {sg['cost_per_100_games_usd']:>15.4f} {cost_savings:>11.1f}%")
    
    # Game performance
    print(f"\n{'Game Performance':<30} {'vLLM':>15} {'SGLang':>15}")
    print("-" * 70)
    print(f"{'Win rate %':<30} {vl['win_rate']:>15.1f} {sg['win_rate']:>15.1f}")
    print(f"{'Avg moves/game':<30} {vl['avg_moves_per_game']:>15.1f} {sg['avg_moves_per_game']:>15.1f}")
    
    # Headline
    print(f"\n{'='*70}")
    if cost_savings > 0:
        print(f"📊 SGLang saves {cost_savings:.1f}% on 2048 RL rollout costs")
        print(f"   (System prompt shared across ~{int(vl['avg_moves_per_game']) * vl['num_games']} moves)")
    else:
        print(f"📊 vLLM is {-cost_savings:.1f}% cheaper for this workload")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="2048 Game RL Rollout Benchmark")
    parser.add_argument("--backend", choices=["sglang", "vllm"], help="Backend to benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model to use")
    parser.add_argument("--num-games", type=int, default=20, help="Number of games to play (default: 20)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("SGLANG", "VLLM"), help="Compare two result files")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    if not args.backend:
        parser.error("--backend required unless using --compare")
    
    result = asyncio.run(run_benchmark(
        args.backend,
        args.model,
        args.num_games,
    ))
    
    print_results(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
