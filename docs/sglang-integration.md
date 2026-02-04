# SGLang Backend Integration

ART supports SGLang as an alternative inference engine to vLLM. SGLang offers
potentially faster inference for agent trajectories due to its RadixAttention
prefix caching mechanism.

## Architecture

### Multi-GPU Split Mode (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-GPU Split Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0: SGLang Inference Server                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • RadixAttention cache (PERSISTENT across training)       │ │
│  │  • OpenAI-compatible API on localhost:8000                 │ │
│  │  • LoRA hot-reload via /update_weights_from_lora           │ │
│  │  • No restart needed = cache stays warm                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  GPU 1+: Training (Unsloth/GRPO)                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • PEFT/LoRA model                                         │ │
│  │  • Optimizer states                                        │ │
│  │  • Gradient computation                                    │ │
│  │  • Checkpoint saving                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Weight Sync: Hot-reload via HTTP API (~5-10s)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Single-GPU Fallback Mode

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single-GPU Shared Mode                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0: Time-multiplexed                                        │
│                                                                  │
│  [Inference Phase]                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  SGLang Server running                                      │ │
│  │  Training model offloaded to CPU                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         ↓ Stop server                           │
│  [Training Phase]                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Training model on GPU                                      │ │
│  │  SGLang server stopped                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         ↓ Restart server                        │
│  [Inference Phase]                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  SGLang Server running (cache cleared)                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Weight Sync: Server restart (~30-60s, cache lost)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why SGLang?

| Feature | vLLM | SGLang | Benefit for RL |
|---------|------|--------|----------------|
| Prefix Caching | PagedAttention | RadixAttention (automatic LRU) | Better multi-turn perf |
| Cache Persistence | Manual | Automatic | Less memory management |
| Scheduling | Continuous batching | Zero-overhead | Lower latency |
| Structured Outputs | Native | Optimized | Faster tool calls |
| Weight Updates | LoRA add | Hot-reload API | No restart needed |

**Key benefit**: SGLang's RadixAttention automatically caches common prefixes across
requests. For RL training where many rollouts share the same system prompt and context,
this provides significant speedups.

## Installation

**CRITICAL**: SGLang requires a TWO-environment architecture due to torchao version conflicts.

### Quick Setup (Recommended)
```bash
# Run the setup script (creates both environments)
chmod +x scripts/setup_sglang.sh
./scripts/setup_sglang.sh
```

### Manual Setup
```bash
# 1. Main training environment (ART + Unsloth)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[sglang]"
deactivate

# 2. SGLang server environment (ISOLATED - no ART)
python3.11 -m venv .venv-sglang-server
source .venv-sglang-server/bin/activate
pip install "sglang[srt]>=0.5.5"
deactivate

# 3. Activate main env to run training
source .venv/bin/activate
```

The SGLang backend automatically detects `.venv-sglang-server` and uses it for the inference server subprocess.

## Usage

### Basic Usage (Auto-detect GPUs)

```python
from art.sglang_backend import SGLangBackend
import art

model = art.TrainableModel(
    name="my-model",
    base_model="Qwen/Qwen2.5-3B-Instruct",
    project="my-project",
)

# Auto-detects GPU count:
# - 2+ GPUs: split mode (recommended)
# - 1 GPU: shared mode (fallback)
backend = SGLangBackend()
await backend.register(model)

# Everything else works like LocalBackend
result = await backend.train(model, trajectory_groups)
```

### Explicit Device Configuration

```python
from art.sglang_backend import SGLangBackend, DeviceConfig, SGLangConfig

# 2-GPU setup
backend = SGLangBackend(
    inference_device=0,      # SGLang on GPU 0
    training_devices=[1],    # Training on GPU 1
)

# 4-GPU setup with multi-GPU training
backend = SGLangBackend(
    inference_device=0,
    training_devices=[1, 2, 3],
)

# Custom SGLang configuration
backend = SGLangBackend(
    sglang_config=SGLangConfig(
        mem_fraction_static=0.85,
        weight_sync_method="lora",  # or "disk", "restart"
        flush_cache_on_sync=False,  # Keep cache warm
        tensor_parallel_size=1,
    )
)
```

### With vLLM (Default Behavior)

```python
import art

# Default LocalBackend uses vLLM
backend = art.LocalBackend()
await backend.register(model)
```

## Configuration Reference

### DeviceConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_device` | int | 0 | GPU index for SGLang server |
| `training_devices` | list[int] | [1] | GPU indices for training |
| `auto_detect` | bool | True | Auto-detect available GPUs |

### SGLangConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mem_fraction_static` | float | 0.9 | GPU memory for SGLang (0.0-1.0) |
| `disable_radix_cache` | bool | False | Disable RadixAttention (NOT recommended) |
| `max_loras_per_batch` | int | 4 | Max LoRA adapters to batch |
| `context_length` | int | None | Max context (None = model default) |
| `weight_sync_method` | str | "lora" | "lora", "disk", or "restart" |
| `flush_cache_on_sync` | bool | False | Clear KV cache on weight sync |
| `server_timeout` | float | 120.0 | Server startup timeout (seconds) |
| `tensor_parallel_size` | int | 1 | TP size for large models |

## Weight Synchronization Methods

| Method | Speed | Cache | Best For |
|--------|-------|-------|----------|
| `lora` | ~5-10s | Preserved | Multi-GPU, frequent training |
| `disk` | ~10-20s | Preserved | Large checkpoints |
| `restart` | ~30-60s | Lost | Single-GPU fallback |






## Benchmarking Your Setup

```bash
# In vLLM environment
source .venv-vllm/bin/activate
python scripts/benchmark_inference.py --engine vllm --model Qwen/Qwen2.5-3B-Instruct

# In SGLang environment
source .venv-sglang/bin/activate
python scripts/benchmark_inference.py --engine sglang --model Qwen/Qwen2.5-3B-Instruct
```

## Troubleshooting

### "SGLang is not installed"

```bash
source .venv-sglang/bin/activate
pip install openpipe-art[sglang]
```

### Server timeout errors

```python
backend = SGLangBackend(
    sglang_config=SGLangConfig(server_timeout=180.0)
)
```

Or via environment:
```bash
export ART_SERVER_TIMEOUT=180
```

### CUDA out of memory

```python
backend = SGLangBackend(
    sglang_config=SGLangConfig(mem_fraction_static=0.8)
)
```

### Check server logs

```bash
cat .art/<project>/<model>/logs/sglang.log
```

## References

- [verl SGLang integration](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)
- [SGLang weight sync optimization (slime)](https://hebiao064.github.io/rl-weight-sync)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
