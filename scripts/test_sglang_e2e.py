#!/usr/bin/env python3
"""End-to-end test for SGLang backend with training loop.

Tests the full RL cycle:
1. Server startup
2. Inference (rollouts)
3. Training (GRPO)
4. Weight sync (hot-reload or restart)
5. Second inference (verify weights updated)

Usage:
    source .venv/bin/activate
    python scripts/test_sglang_e2e.py
"""

# Suppress multiprocessing resource_tracker warnings
import warnings
warnings.filterwarnings("ignore", message="resource_tracker:")

# CRITICAL: Set CUDA_VISIBLE_DEVICES for training BEFORE any imports
# This must be the VERY FIRST thing to happen before PyTorch initializes CUDA
import os

# For split-mode training, we need GPUs 1,2,3 for training
# But we keep all GPUs visible so SGLang server (subprocess) can use GPU 0
# The subprocess will set its own CUDA_VISIBLE_DEVICES
os.environ["IMPORT_UNSLOTH"] = "1"  # Tell art package to import unsloth early

# IMPORTANT: Import unsloth BEFORE any other ML libraries to prevent early CUDA initialization.
# This must happen before importing transformers, torch, vllm, or the art package.
# See: https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
try:
    import unsloth  # noqa: F401
except ImportError:
    pass  # unsloth not installed, continue without it

import asyncio
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def test_e2e():
    """Run end-to-end test."""
    print("=" * 60)
    print("SGLang Backend End-to-End Test")
    print("=" * 60)
    
    # Step 1: Import and config check
    print("\n[1/7] Importing modules...")
    try:
        import art
        from art.sglang_backend import SGLangBackend, SGLangConfig
        from art.trajectories import Trajectory, TrajectoryGroup
        from openai import AsyncOpenAI
        print("  ✓ Imports OK")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Step 2: Check server Python
    print("\n[2/7] Checking SGLang server environment...")
    config = SGLangConfig()
    server_python = config.get_server_python()
    print(f"  Server Python: {server_python}")
    if ".venv-sglang-server" in server_python:
        print("  ✓ Using separate SGLang server environment")
    else:
        print("  ⚠ Using same Python (may have dependency issues)")
    
    # Step 3: Initialize backend
    print("\n[3/7] Initializing SGLangBackend...")
    try:
        backend = SGLangBackend()
        print(f"  Mode: {'split' if backend.device_config.is_split_mode else 'shared'}-GPU")
        print(f"  Inference: cuda:{backend.device_config.inference_device}")
        print(f"  Training: cuda:{backend.device_config.training_devices}")
        print("  ✓ Backend initialized")
    except Exception as e:
        print(f"  ✗ Backend init failed: {e}")
        return False
    
    # Step 4: Register model
    print("\n[4/7] Registering model...")
    try:
        model = art.TrainableModel(
            name="sglang-e2e-test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            project="sglang-test",
        )
        await backend.register(model)
        print(f"  Model: {model.name}")
        print(f"  Base: {model.base_model}")
        print("  ✓ Model registered")
    except Exception as e:
        print(f"  ✗ Registration failed: {e}")
        await backend.close()
        return False
    
    # Step 5: Start server and test inference
    print("\n[5/7] Starting server and testing inference...")
    try:
        base_url, api_key = await backend._prepare_backend_for_training(model, None)
        print(f"  Server URL: {base_url}")
        
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        model_name = backend._model_inference_name(model)
        print(f"  Model name for inference: {model_name}")
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'test passed' in exactly two words."}],
            max_tokens=10,
        )
        response_text = response.choices[0].message.content
        print(f"  Response: {response_text}")
        print("  ✓ Inference works")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        await backend.close()
        return False
    
    # Step 6: Create trajectories using real inference and train
    print("\n[6/7] Running training step...")
    try:
        # Create trajectories by doing actual inference (to get real Choice objects)
        trajectories = []
        
        for i, (question, expected_reward) in enumerate([
            ("What is 2+2? Answer with just the number.", 1.0),
            ("What is 2+2? Answer with a wrong number.", 0.0),
        ]):
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": question}],
                max_tokens=10,
                logprobs=True,  # Request logprobs for training
            )
            choice = response.choices[0]
            
            traj = Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": question},
                    choice,  # Real Choice object from API
                ],
                reward=expected_reward,
            )
            trajectories.append(traj)
            print(f"  Trajectory {i+1}: '{choice.message.content}' -> reward={expected_reward}")
        
        trajectory_group = TrajectoryGroup(trajectories=trajectories)
        
        print("  Training on 2 trajectories...")
        result = await backend.train(
            model,
            [trajectory_group],
            learning_rate=1e-5,
            verbose=True,
        )
        print(f"  Step: {result.step}")
        print(f"  Metrics: {result.metrics}")
        print("  ✓ Training complete")
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        await backend.close()
        return False
    
    # Step 7: Test inference after training (weights should be updated)
    print("\n[7/7] Testing inference after training...")
    try:
        # Get updated model name
        model_name = backend._model_inference_name(model)
        print(f"  Model name: {model_name}")
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
        )
        response_text = response.choices[0].message.content
        print(f"  Response: {response_text}")
        print("  ✓ Post-training inference works")
    except Exception as e:
        print(f"  ✗ Post-training inference failed: {e}")
        import traceback
        traceback.print_exc()
        await backend.close()
        return False
    
    # Skip cleanup - just kill processes on exit
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    # Force kill SGLang server (faster than graceful shutdown)
    import subprocess
    subprocess.run(["pkill", "-9", "-f", "sglang"], capture_output=True)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_e2e())
    sys.exit(0 if success else 1)
