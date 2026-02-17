# Unsloth + SGLang: MoE-Optimized RL Training Benchmark

Benchmark for the Unsloth + SGLang backend that combines SGLang for inference with Unsloth for MoE training. Uses a **dedicated GPU split** where inference and training run on separate GPUs for zero sleep/wake overhead, with a **persistent training worker** that keeps the model loaded across steps.

---

## Architecture — Dedicated GPU Split (Default)

```
┌──────────────────────────────────────────────────────────────────┐
│  4-GPU Setup (Recommended)                                       │
│                                                                  │
│  ┌─ GPUs 0, 2 (TP=2) ────────────┐  ┌─ GPU 1 ──────────────┐   │
│  │  SGLang Server                  │  │  Unsloth Training     │   │
│  │  • Always active (no sleep)     │  │  • Dedicated GPU      │   │
│  │  • TP=2 inference               │  │  • Persistent worker  │   │
│  │                                 │  │    (model loaded once) │   │
│  │  ┌──────────┐  ┌────────────┐   │  │  • LoRA + Optimizer   │   │
│  │  │  TP=2    │  │  LoRA      │   │  │  • ART loss function  │   │
│  │  │  Model   │  │  Hot-reload│   │  │                       │   │
│  │  │  Shards  │  │  < 0.1s    │   │  │  GPU 3: idle          │   │
│  │  └──────────┘  └────────────┘   │  └───────────────────────┘   │
│  └─────────────────────────────────┘                              │
│                                                                   │
│  ✓ No sleep/wake overhead                                         │
│  ✓ SGLang stays active during training                            │
│  ✓ Persistent worker — model loaded once, reused across steps     │
│  ✓ TP must be power of 2 (vocab size constraint)                  │
│  ✓ Generation is 70-90% of RL time → more inference GPUs = win    │
└───────────────────────────────────────────────────────────────────┘
```

### Auto-Detected GPU Splits

TP must be a power of 2 (model vocab sizes like Qwen3's 151936 are divisible by 1,2,4,8 but NOT 3).

| GPUs Available | Inference GPUs | TP Size | Training GPU | Mode |
|:-:|:-:|:-:|:-:|:-:|
| 8 | 0, 2, 3, 4 | 4 | 1 | **Dedicated** |
| 4 | 0, 2 | 2 | 1 | **Dedicated** |
| 3 | 0, 2 | 2 | 1 | **Dedicated** |
| 2 | 0 | 1 | 1 | **Dedicated** |
| 1 | 0 | 1 | — | Shared (sleep/wake) |

GPU 1 is chosen as primary training GPU to keep GPU 0 as the primary SGLang rank.

### Key Features

- **Dedicated GPU split** — inference and training on separate GPUs, zero sleep/wake overhead
- **Persistent training worker** — model loaded once at step 1, reused for all subsequent steps (~0s model load overhead on steps 2+)
- **Auto-detected** — optimal split computed from available GPU count
- **~12x faster MoE training** via Unsloth Triton kernels
- **~35% less VRAM** via Split LoRA approach
- **LoRA hot-reload** for weight sync (<0.1s)

### Shared Mode (Single GPU Fallback)

When only 1 GPU is available, falls back to the verl-style sleep/wake pattern where SGLang releases GPU memory before training and reclaims it after. This adds ~5-15s overhead per step.

---

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | End-to-end benchmark runner |
| `config.py` | Benchmark configuration + GPU split helper |
| `metrics_collector.py` | Metrics collection and reporting |
| `sglang_server.py` | SGLang server lifecycle management (supports GPU pinning) |
| `unsloth_sglang_service.py` | Unsloth + SGLang service with dedicated/shared GPU modes |
| `setup_environments.sh` | Environment setup script |

---

## Training Loop

### Dedicated Mode (2+ GPUs, default)

**Step 1 (cold start):**

1. **Rollout** — SGLang generates on inference GPUs (always active, TP=2)
2. **Data pipeline** — ART preprocessing tokenizes/packs into packed tensors
3. **Spawn worker** — on dedicated training GPU (`CUDA_VISIBLE_DEVICES`)
4. **Load model** — base model + LoRA adapter (~50s one-time cost)
5. **Train** — ART loss on packed tensors
6. **Save LoRA** — adapter saved to disk
7. **Load LoRA** — hot-reload adapter into SGLang (<0.1s)

**Steps 2+ (persistent worker):**

1. **Rollout** — SGLang generates (never stops)
2. **Data pipeline** — tokenize/pack
3. **Train** — reuse persistent worker (model already loaded, ~0s overhead)
4. **Save LoRA** + **Load LoRA** — save and hot-reload

No sleep/wake. SGLang never stops. Worker stays alive until benchmark end.

### Shared Mode (1 GPU fallback)

1. **Rollout** — SGLang generates completions
2. **Data pipeline** — tokenize/pack
3. **Sleep** — SGLang releases GPU memory
4. **Spawn subprocess** → **Train** → **Save LoRA** → **Kill**
5. **Wake** — SGLang restores GPU memory
6. **Load LoRA** — hot-reload

---

## Running the Benchmark

```bash
# Setup environments
bash benchmarks/sglang_vs_vllm/setup_environments.sh

# Run with auto-detected GPU split (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python benchmarks/sglang_vs_vllm/run_benchmark.py \
    --sglang-python ~/.venvs/sglang-bench/bin/python \
    --num-steps 10 --num-rollouts 64 --dataset gsm8k

# Explicit GPU split: inference on GPUs 0,2 (TP=2), training on GPU 1
uv run python benchmarks/sglang_vs_vllm/run_benchmark.py \
    --inference-gpus 0,2 --training-gpus 1 \
    --sglang-python ~/.venvs/sglang-bench/bin/python

# Force shared mode (sleep/wake) even with multiple GPUs
uv run python benchmarks/sglang_vs_vllm/run_benchmark.py \
    --training-gpus -1 \
    --sglang-python ~/.venvs/sglang-bench/bin/python
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Model to benchmark |
| `--dataset` | `agentic` | Dataset: gsm8k, sharegpt, agentic, math, synthetic |
| `--num-steps` | `3` | Number of RL training steps |
| `--num-rollouts` | `16` | Rollouts per step |
| `--inference-gpus` | auto | Comma-separated GPU IDs for SGLang inference (e.g. `0,2,3`) |
| `--training-gpus` | auto | Comma-separated GPU IDs for training (e.g. `1`), `-1` for shared mode |
| `--tp` | `0` (auto) | Tensor parallel size (overridden by `--inference-gpus` count) |
| `--unsloth-lora-rank` | `1` | LoRA rank for Unsloth training |
| `--unsloth-moe-backend` | `auto` | MoE backend: auto, grouped_mm (H100+), unsloth_triton (A100) |
| `--unsloth-port` | `8300` | SGLang inference server port |
| `--gpu-memory-utilization` | `0.7` | GPU memory fraction for SGLang |

GSM8K test set (1,319 questions) is downloaded automatically on first run and cached locally.

---

## Trade-Offs vs Distributed Training

| | Unsloth + SGLang (this) | Distributed (Megatron) |
|---|---|---|
| **Inference** | N-1 GPUs (TP=2 on 4 GPUs) | N GPUs (TP=4) |
| **Training** | 1 GPU (persistent worker) | N GPUs (tensor parallel) |
| **Model reload per step** | **0s** (steps 2+) | ~50-80s (sleep/wake + resharding) |
| **Sleep/wake overhead** | None (dedicated split) | Yes (each step) |
| **Training throughput** | Single GPU | Linear scaling across N GPUs |
| **Setup complexity** | Simple | Complex |
| **Best for** | Rapid prototyping, MoE models | Production, large-scale |

The persistent worker eliminates model reload overhead on steps 2+, which partially compensates for using fewer inference GPUs. Unsloth is single-GPU by design (no tensor parallelism for training), so the dedicated GPU split with persistent worker is the optimal configuration.

---

## Credits

- [ART (OpenPipe)](https://github.com/OpenPipe/ART) — The codebase this is built on
- [verl (Volcano Engine)](https://github.com/volcengine/verl) — Reference for the SGLang integration pattern
- [SGLang](https://github.com/sgl-project/sglang) — Inference engine
- [Unsloth](https://unsloth.ai/) — MoE-optimized training
