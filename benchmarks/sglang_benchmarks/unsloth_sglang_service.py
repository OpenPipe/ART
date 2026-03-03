"""
Unsloth + SGLang service — MoE training with dedicated GPU split.

Training pipeline:
  - LoRA config: rank=1, alpha=32, targets attention modules
  - Loss function: art.loss.loss_fn with on_policy_correction=True
  - Data pipeline: ART's packed tensors (tokenize_trajectory_groups +
    packed_tensors_from_tokenized_results), saved to disk
  - Optimizer: AdamW(lr=5e-6, betas=(0.9, 0.99), weight_decay=0.1, clip_grad=0.1)

GPU allocation modes:

  DEDICATED (multi-GPU, recommended for 2+ GPUs):
    SGLang inference and Unsloth training run on SEPARATE GPUs.
    Example with 4 GPUs:
      - GPUs 0,2: SGLang with TP=2   (fast inference)
      - GPUs 1,3: Unsloth DDP x2     (parallel training)
    Benefits:
      - NO sleep/wake overhead (GPUs never shared)
      - SGLang stays fully active during training
      - Spare GPUs used for DDP training (near-linear speedup)
      - Generation is 70-90% of RL time, so more inference GPUs = real speedup

  SHARED (single-GPU fallback):
    SGLang and Unsloth time-share the SAME GPU via sleep/wake.
    Used when only 1 GPU is available.

Training loop — dedicated mode (per step):
  1. generate()       — SGLang active on inference GPUs
  2. spawn subprocess — on dedicated training GPU (CUDA_VISIBLE_DEVICES)
  3. init_model()     — load base model + previous LoRA checkpoint
  4. train            — ART loss on packed tensors
  5. save_lora()      — save adapter to disk
  6. KILL subprocess  — free training GPU memory (Unsloth holds tensor caches)
  7. load_lora()      — hot-reload adapter into SGLang (<2s)
  (no sleep/wake needed — GPUs are separate)

Training loop — shared mode (per step):
  1. generate()       — SGLang active
  2. sleep()          — SGLang releases KV cache AND weights
  3. spawn subprocess — fresh CUDA context
  4. init_model() → train → save_lora()
  5. KILL subprocess  — process death frees ALL GPU memory
  6. wake_up()        — SGLang restores base weights + KV cache
  7. load_lora()      — hot-reload adapter

Reference:
  - https://unsloth.ai/docs/new/faster-moe
  - https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import socket
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import torch

from .sglang_server import SGLangServer, SGLangServerConfig

logger = logging.getLogger(__name__)


def _is_vllm_healthy() -> bool:
    """Return True if vLLM's C extension loads without ABI errors."""
    try:
        import vllm._C  # noqa: F401
        return True
    except (ImportError, OSError, AttributeError):
        return False


class _StubModule(types.ModuleType):
    """A module whose public attributes are no-op callables returning None.

    Used to mock ``unsloth_zoo.vllm_utils`` when vLLM's C extension is broken.
    Any function imported from the mock (e.g. ``_get_torchao_fp8_config``)
    will be a harmless no-op.

    Dunder attributes (``__file__``, ``__path__``, ``__spec__``, …) are NOT
    mocked — Python's ``inspect`` module iterates ``sys.modules`` and accesses
    ``__file__`` on every module.  If ``__file__`` returns a callable instead
    of a string, ``inspect.getsourcefile()`` crashes with
    ``AttributeError: 'function' object has no attribute 'endswith'``.
    """

    def __getattr__(self, name: str):
        # Let dunder lookups raise AttributeError so inspect/importlib
        # treat this module as one without source (like builtins).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return None
        return _noop


def _patch_vllm_for_unsloth_import() -> None:
    """Make Unsloth importable even when vLLM's C extension is broken.

    Unsloth + unsloth_zoo have deep vLLM imports at module load time:
      1. ``unsloth/__init__.py`` → ``fix_vllm_guided_decoding_params()``
         chains into ``vllm._C`` (ABI crash).
      2. ``unsloth_zoo/vllm_utils.py`` → ``import vllm.model_executor.layers...``
         chains deep into vLLM quantization/fused_moe layers that call
         ``torch.ops._C`` custom ops (which aren't registered if _C failed).

    On cloud GPU images where vLLM was compiled against a different PyTorch
    ABI (e.g. vLLM 0.15.1 + PyTorch 2.10.0), these imports crash.

    Since we use SGLang (not vLLM) for inference, we:
      1. Create a dummy ``vllm._C`` module
      2. Pre-populate ``sys.modules["unsloth_zoo.vllm_utils"]`` with a stub
         so the *real* module (which does ``import vllm.model_executor...``)
         is never loaded

    vLLM inference (if used in a separate process) is unaffected — each
    subprocess has its own module state.
    """
    if _is_vllm_healthy():
        return  # vLLM works fine, no mocking needed

    logger.info(
        "vLLM C extension is broken (ABI mismatch with PyTorch). "
        "Mocking vllm internals for Unsloth import — we use SGLang, not vLLM."
    )

    # 1. Dummy vllm._C so shallow imports don't crash
    sys.modules["vllm._C"] = types.ModuleType("vllm._C")

    # 2. Mock unsloth_zoo.vllm_utils BEFORE Unsloth imports it.
    #    This prevents the real module from loading, which means the deep
    #    vllm.model_executor import chain never executes.
    if "unsloth_zoo.vllm_utils" not in sys.modules:
        sys.modules["unsloth_zoo.vllm_utils"] = _StubModule("unsloth_zoo.vllm_utils")


def _gc_and_empty_cuda_cache(n: int = 3) -> None:
    for _ in range(n):
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Unsloth Training State — lives for one step in a short-lived subprocess
# ---------------------------------------------------------------------------

@dataclass
class UnslothTrainingState:
    """Holds model, tokenizer, and optimizer for one training step.

    Created fresh each step in a new subprocess. The subprocess is killed
    after training, which is the only reliable way to free GPU memory
    (Unsloth's monkey-patching holds module-level tensor caches).
    """

    model: Any  # PeftModelForCausalLM after FastLanguageModel.get_peft_model()
    tokenizer: Any
    optimizer: torch.optim.Optimizer


# ---------------------------------------------------------------------------
# Training Worker — runs in a persistent subprocess via mp_actors
# ---------------------------------------------------------------------------

class UnslothTrainingWorker:
    """Training worker — runs in a persistent subprocess via mp_actors.

    Uses the SAME training pipeline as the Megatron backend:
      - LoRA config: rank=1, alpha=32, targets attention modules
      - Loss: art.loss.loss_fn with on_policy_correction=True
      - Data: ART packed tensors loaded from disk

    GPU lifecycle (matches Megatron):
      Each step: load model → train → save → DESTROY model.
      Unsloth's monkey-patching holds hidden references to GPU tensors that
      prevent model.to("cpu") from releasing memory. Full destruction + gc
      is the only reliable way to free GPU memory for SGLang wake_up.

    Communication with the parent process is via mp_actors proxy (pickle over
    multiprocessing queues). Only lightweight data crosses the boundary:
      - packed_tensors_dir: str (path to packed tensors on disk)
      - metrics: dict[str, float] (~1KB)
      - checkpoint paths: str
    """

    def __init__(
        self,
        base_model: str,
        output_dir: str,
        lora_rank: int = 1,
        lora_alpha: int = 32,
        max_seq_length: int = 8192,
        learning_rate: float = 5e-6,
        moe_backend: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.moe_backend = moe_backend
        self.load_in_4bit = load_in_4bit
        self._state: UnslothTrainingState | None = None
        self._last_checkpoint: str | None = None
        self._vllm_patched: bool = False

    async def init_model(self) -> dict[str, Any]:
        """Load model to GPU. Called at the start of each training step.

        On the first call, loads from scratch. On subsequent calls, loads
        the base model + previous LoRA checkpoint weights (optimizer state
        is re-initialized — same as Megatron which starts a fresh process
        each step).
        """
        if self.moe_backend != "auto":
            os.environ["UNSLOTH_MOE_BACKEND"] = self.moe_backend

        if not self._vllm_patched:
            _patch_vllm_for_unsloth_import()
            self._vllm_patched = True

        from unsloth import FastLanguageModel

        logger.info(f"Loading model: {self.base_model}")
        logger.info(f"  lora_rank={self.lora_rank}  max_seq_length={self.max_seq_length}")
        logger.info(f"  load_in_4bit={self.load_in_4bit}  moe_backend={self.moe_backend}")
        if self._last_checkpoint:
            logger.info(f"  resuming LoRA from: {self._last_checkpoint}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            # Only target attention modules for MoE models.
            # gate/up/down_proj exist in EVERY expert, so targeting them
            # multiplies params by num_experts (52M vs 3M for rank=1).
            # Megatron's LoRA applies to shared layers differently, so
            # attention-only matches the effective behavior for MoE.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        # Resume LoRA weights from previous step (if any)
        if self._last_checkpoint:
            adapter_file = os.path.join(self._last_checkpoint, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                from safetensors.torch import load_file
                lora_state = load_file(adapter_file)
                model.load_state_dict(lora_state, strict=False)
                logger.info(f"  resumed {len(lora_state)} LoRA tensors from checkpoint")

        FastLanguageModel.for_training(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=0.1,
        )

        n_params = sum(p.numel() for p in trainable)
        logger.info(f"Unsloth ready — {n_params:,} trainable params")

        self._state = UnslothTrainingState(model=model, tokenizer=tokenizer, optimizer=optimizer)
        return {"trainable_params": n_params}

    async def train_on_packed_tensors(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Train using ART's packed tensors and loss function.

        Matches Megatron's training loop exactly:
          - Same packed tensor format (tokens, logprobs, advantages, etc.)
          - Same loss function (art.loss.loss_fn with on_policy_correction=True)
          - Same optimizer (AdamW, clip_grad=0.1)

        The packed tensors are created by ART's preprocessing pipeline
        (tokenize_trajectory_groups + packed_tensors_from_tokenized_results)
        in the benchmark runner, then saved to disk. This method loads them
        and runs the training loop.
        """
        from art.preprocessing.pack import packed_tensors_from_dir
        from art.loss import loss_fn, shift_tensor

        state = self._state
        assert state is not None

        device = next(state.model.parameters()).device
        state.model.train()

        if lr is not None:
            for pg in state.optimizer.param_groups:
                pg["lr"] = lr

        packed = packed_tensors_from_dir(
            dir=packed_tensors_dir,
            num_sequences=num_sequences,
            sequence_length=sequence_length,
        )

        total_loss = 0.0
        n_seqs = 0
        completion_tokens = 0

        state.optimizer.zero_grad()
        t0 = time.perf_counter()

        for idx in range(num_sequences):
            inputs = {
                key: value[idx:idx + 1].to(device)
                for key, value in packed.items()
                if isinstance(value, torch.Tensor)
            }

            tokens = inputs["tokens"]
            batch_size, seq_len = tokens.shape
            completion_tokens += int(inputs["assistant_mask"].sum().item())

            attn_mask = (inputs["group_ids"] != -1).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = state.model(
                    input_ids=tokens,
                    position_ids=inputs["input_pos"],
                    attention_mask=attn_mask,
                )
                logits = outputs.logits

                labels = shift_tensor(tokens, 0)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                new_logprobs = log_probs.gather(
                    dim=-1, index=labels.unsqueeze(-1),
                ).squeeze(-1)

                experimental_config = {"on_policy_correction": True}
                loss_result = loss_fn(
                    inputs, new_logprobs, ref_logprobs=None, entropies=None,
                    experimental_config=experimental_config,
                )

                loss = loss_result.mean_policy_loss / num_sequences

            loss.backward()

            total_loss += loss_result.mean_policy_loss.item()
            n_seqs += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in state.model.parameters() if p.requires_grad],
            max_norm=0.1,
        )
        state.optimizer.step()
        state.optimizer.zero_grad()

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_seqs, 1)
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"  trained: loss={avg_loss:.4f}  {completion_tokens / elapsed:.0f} tok/s  "
            f"VRAM={gpu_mem_gb:.1f}GB  {elapsed:.2f}s (ART loss, packed tensors)"
        )

        return {
            "loss": avg_loss,
            "training_time_s": elapsed,
            "tokens_per_sec": completion_tokens / elapsed,
            "gpu_memory_gb": gpu_mem_gb,
            "total_tokens": completion_tokens,
            "batch_size": n_seqs,
            "seq_len": sequence_length,
        }

    async def save_lora(self, step: int) -> str:
        """Save LoRA adapter via PEFT save_pretrained (standard format)."""
        assert self._state is not None

        ckpt = os.path.join(self.output_dir, "checkpoints", f"{step:04d}")
        os.makedirs(ckpt, exist_ok=True)

        self._state.model.save_pretrained(ckpt)
        # NOTE: do NOT save tokenizer here.  tokenizer.save_pretrained()
        # writes added_tokens.json to the same directory.  SGLang's
        # LoRAConfig reads that file and treats it as LoRA vocabulary
        # additions, making can_support() fail because the memory pool
        # has lora_added_tokens_size=0.  The Megatron backend also does
        # NOT save the tokenizer alongside the adapter.
        # SGLang uses its own tokenizer — the adapter only needs
        # adapter_config.json + adapter_model.safetensors.

        adapter = os.path.join(ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter):
            mb = os.path.getsize(adapter) / 1e6
            logger.info(f"LoRA saved: {ckpt} ({mb:.1f} MB)")
        else:
            logger.warning(f"adapter_model.safetensors not found in {ckpt}")

        # Track for next step's init_model to resume from
        self._last_checkpoint = ckpt
        return ckpt



# ---------------------------------------------------------------------------
# Main Service
# ---------------------------------------------------------------------------

@dataclass
class UnslothSGLangService:
    """Unsloth MoE training + SGLang inference with dedicated GPU split.

    Uses ART's data pipeline and loss function for identical training behavior.

    GPU allocation (auto-detected from num_gpus if not specified):
      8 GPUs:  inference=[0,2,3,4] TP=4, training=[1,5,6,7] (DDP x4)
      4 GPUs:  inference=[0,2]     TP=2, training=[1,3]     (DDP x2)
      3 GPUs:  inference=[0,2]     TP=2, training=[1]
      2 GPUs:  inference=[0]       TP=1, training=[1]
      1 GPU:   shared mode — sleep/wake (no split)

    In dedicated mode, SGLang stays fully active during training.
    No sleep/wake overhead. When multiple training GPUs are available,
    DDP is used for near-linear training speedup.
    """

    model_name: str
    base_model: str
    output_dir: str
    sglang_python: str = "python"
    port: int = 8300
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.7
    max_running_requests: int = 256
    log_dir: str = ""

    # GPU split — None means auto-detect from available GPUs.
    # inference_gpus: list of physical GPU IDs for SGLang (e.g. [0, 2, 3])
    # training_gpus: list of physical GPU IDs for Unsloth (e.g. [1] or [1, 3] for DDP)
    # When training_gpus is [-1], shared mode is used (sleep/wake on same GPUs).
    inference_gpus: list[int] | None = None
    training_gpus: list[int] | None = None

    # Unsloth config
    lora_rank: int = 1
    lora_alpha: int = 32
    max_seq_length: int = 8192
    learning_rate: float = 5e-6
    # "auto" lets Unsloth pick: grouped_mm (H100+), unsloth_triton (A100), native_torch
    moe_backend: str = "auto"
    load_in_4bit: bool = False  # MoE nn.Parameter doesn't support bnb 4bit yet

    # Internal state
    _server: SGLangServer | None = None
    _worker: Any = None  # mp_actors proxy to UnslothTrainingWorker in subprocess
    _latest_step: int = 0
    _is_sleeping: bool = False
    _active_lora_name: str | None = None
    _last_checkpoint: str | None = None  # LoRA checkpoint from previous step

    def __post_init__(self) -> None:
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        # Let Unsloth auto-select, or override
        if self.moe_backend != "auto":
            os.environ["UNSLOTH_MOE_BACKEND"] = self.moe_backend

        # Auto-detect GPU split if not specified
        if self.inference_gpus is None or self.training_gpus is None:
            self._auto_detect_gpu_split()

    def _auto_detect_gpu_split(self) -> None:
        """Auto-detect optimal inference/training GPU split.

        TP size must be a power of 2 — most models have vocab sizes that
        are multiples of powers of 2, but NOT arbitrary numbers like 3.
        (e.g. Qwen3's vocab_size=151936 is divisible by 1,2,4,8 but NOT 3)

        Strategy for N GPUs:
          N >= 2: GPU 1 = primary training GPU, remaining GPUs for inference.
                  TP = largest power of 2 that fits in remaining GPUs.
                  Spare GPUs (beyond TP) are added to training for DDP.
          N == 1: shared mode (training_gpus=[-1], sleep/wake fallback)

        Examples:
          8 GPUs: TP=4, inference=[0,2,3,4], training=[1,5,6,7] (DDP x4)
          4 GPUs: TP=2, inference=[0,2],      training=[1,3]     (DDP x2)
          3 GPUs: TP=2, inference=[0,2],      training=[1]
          2 GPUs: TP=1, inference=[0],        training=[1]
          1 GPU:  shared mode (sleep/wake)
        """
        num_gpus = torch.cuda.device_count()

        if num_gpus >= 2:
            primary_training_gpu = 1
            non_training = [i for i in range(num_gpus) if i != primary_training_gpu]

            # Largest power of 2 that fits
            tp = 1
            while tp * 2 <= len(non_training):
                tp *= 2

            self.inference_gpus = non_training[:tp]
            self.tensor_parallel_size = tp

            # Spare GPUs become additional training GPUs (DDP)
            spare = non_training[tp:]
            self.training_gpus = [primary_training_gpu] + spare

            if len(self.training_gpus) > 1:
                logger.info(
                    f"GPU split auto-detected ({num_gpus} GPUs): "
                    f"inference={self.inference_gpus} (TP={tp}), "
                    f"training={self.training_gpus} (DDP x{len(self.training_gpus)})"
                )
            else:
                logger.info(
                    f"GPU split auto-detected ({num_gpus} GPUs): "
                    f"inference={self.inference_gpus} (TP={tp}), "
                    f"training=GPU {self.training_gpus[0]}"
                )
        else:
            # Single GPU — shared mode
            self.training_gpus = [-1]
            self.inference_gpus = []
            logger.info("Single GPU detected — using shared mode (sleep/wake)")

    @property
    def _dedicated_gpus(self) -> bool:
        """True if inference and training run on separate GPUs (no sleep/wake)."""
        return (
            self.training_gpus is not None
            and len(self.training_gpus) > 0
            and self.training_gpus[0] >= 0
            and bool(self.inference_gpus)
        )

    # ------------------------------------------------------------------
    # SGLang server — start ONCE, never restart
    # ------------------------------------------------------------------

    def _create_server(self) -> SGLangServer:
        # Pin SGLang to inference GPUs when using dedicated GPU split
        cuda_vis = None
        if self._dedicated_gpus:
            cuda_vis = ",".join(str(g) for g in self.inference_gpus)  # type: ignore[union-attr]

        return SGLangServer(SGLangServerConfig(
            model_path=self.base_model,
            served_model_name=self.base_model,
            port=self.port,
            host="0.0.0.0",
            tensor_parallel_size=self.tensor_parallel_size,
            mem_fraction_static=self.gpu_memory_utilization,
            max_running_requests=self.max_running_requests,
            python_executable=self.sglang_python,
            log_file=os.path.join(self.log_dir, "sglang.log"),
            trust_remote_code=True,
            enable_p2p_check=True,
            chunked_prefill_size=32768,
            # memory_saver only needed for shared mode (sleep/wake)
            enable_memory_saver=not self._dedicated_gpus,
            enable_lora=True,
            max_lora_rank=8,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            cuda_visible_devices=cuda_vis,
        ))

    def _spawn_worker(self, last_checkpoint: str | None = None) -> Any:
        """Spawn a FRESH training subprocess on the dedicated training GPU.

        Each training step gets a brand new process. When that process is
        killed after training, ALL GPU memory is released — guaranteed by
        the OS.

        In dedicated GPU split mode, CUDA_VISIBLE_DEVICES is set so the
        training subprocess only sees the training GPU (e.g. GPU 1).
        This ensures Unsloth loads the model on the correct device and
        doesn't interfere with SGLang's inference GPUs.

        NOTE: This path is for single-GPU training only. For multi-GPU DDP,
        see _train_step_ddp() which uses torchrun instead.
        """
        from mp_actors import move_to_child_process

        # Pin training to the dedicated GPU before spawning
        if self._dedicated_gpus:
            gpu_id = self.training_gpus[0]  # type: ignore[index]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Training subprocess pinned to GPU {gpu_id}")

        worker = UnslothTrainingWorker(
            base_model=self.base_model,
            output_dir=self.output_dir,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            max_seq_length=self.max_seq_length,
            learning_rate=self.learning_rate,
            moe_backend=self.moe_backend,
            load_in_4bit=self.load_in_4bit,
        )
        worker._last_checkpoint = last_checkpoint
        proxy = move_to_child_process(
            worker,
            log_file=os.path.join(self.log_dir, "unsloth_worker.log"),
            process_name="unsloth-trainer",
        )

        # Restore parent's CUDA_VISIBLE_DEVICES so it doesn't affect
        # the parent process (which doesn't use CUDA directly).
        if self._dedicated_gpus:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        return proxy

    def _kill_worker(self) -> None:
        """Kill the training subprocess, releasing ALL GPU memory."""
        if self._worker is not None:
            from mp_actors import close_proxy
            t0 = time.perf_counter()
            try:
                close_proxy(self._worker)
            except Exception:
                pass
            self._worker = None
            elapsed = time.perf_counter() - t0
            logger.info(f"Training subprocess killed in {elapsed:.2f}s (GPU memory released)")

    async def start(self) -> float:
        """Start SGLang server on inference GPUs.

        In dedicated mode, SGLang is pinned to inference GPUs and stays
        fully active during training (no sleep/wake). Training runs on
        a separate dedicated GPU.
        """
        self._server = self._create_server()
        startup = await self._server.start()

        if self._dedicated_gpus:
            logger.info(
                f"SGLang ready — {self.base_model} on :{self.port} "
                f"(startup {startup:.1f}s, TP={self.tensor_parallel_size}, "
                f"inference GPUs={self.inference_gpus})"
            )
            train_desc = (
                f"training={self.training_gpus} (DDP x{len(self.training_gpus)})"
                if len(self.training_gpus) > 1  # type: ignore[arg-type]
                else f"training=GPU {self.training_gpus[0]}"  # type: ignore[index]
            )
            logger.info(
                f"Dedicated GPU split: inference={self.inference_gpus}, "
                f"{train_desc} (NO sleep/wake needed)"
            )
        else:
            logger.info(
                f"SGLang ready — {self.base_model} on :{self.port} "
                f"(startup {startup:.1f}s, shared mode with sleep/wake)"
            )
        return startup

    async def stop(self) -> None:
        """Stop everything. Called once at benchmark end."""
        self._kill_worker()
        if self._server is not None:
            await self._server.stop()
            self._server = None
        _gc_and_empty_cuda_cache()

    # ------------------------------------------------------------------
    # verl-style sleep / wake (identical to sglang backend)
    # ------------------------------------------------------------------

    async def sleep(self) -> float:
        """Release GPU memory so Unsloth can train."""
        if self._server is None or not self._server.is_running:
            return 0.0
        t0 = time.perf_counter()
        await self._server.sleep(tags=["kv_cache", "weights"])
        self._is_sleeping = True
        elapsed = time.perf_counter() - t0
        logger.info(f"SGLang asleep (kv_cache + weights freed) — {elapsed:.2f}s")
        return elapsed

    async def wake_up(self) -> float:
        """Restore GPU memory after training (with retry)."""
        if self._server is None or not self._server.is_running:
            return 0.0
        t0 = time.perf_counter()
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._server.wake_up(tags=["kv_cache", "weights"])
                self._is_sleeping = False
                elapsed = time.perf_counter() - t0
                logger.info(f"SGLang awake (kv_cache + weights restored) — {elapsed:.2f}s")
                return elapsed
            except Exception as e:
                if attempt < max_attempts:
                    wait = 5 * attempt
                    logger.warning(
                        f"wake_up() attempt {attempt}/{max_attempts} failed: {e}  "
                        f"— waiting {wait}s before retry"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"wake_up() failed after {max_attempts} attempts: {e}")
                    return 0.0

    # ------------------------------------------------------------------
    # LoRA hot-reload (save is now in UnslothTrainingWorker)
    # ------------------------------------------------------------------

    async def _load_lora(self, lora_path: str, step: int) -> float:
        """Hot-reload LoRA into SGLang (<2s)."""
        if self._server is None:
            return 0.0

        adapter = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter):
            logger.warning(f"No adapter at {adapter}")
            return 0.0

        name = f"{self.model_name}@step{step}"
        elapsed = await self._server.load_lora_adapter(
            lora_path=lora_path, lora_name=name, flush_cache=False,
        )
        if elapsed < 0:
            logger.error("load_lora_adapter failed — base weights intact but not updated")
            return 0.0

        self._active_lora_name = name
        logger.info(f"LoRA hot-reload: '{name}' in {elapsed:.2f}s")
        return elapsed

    # ------------------------------------------------------------------
    # Full step: sleep → train → save → wake → load_lora
    # ------------------------------------------------------------------

    async def train_step(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """One complete training step.

        Dispatches based on GPU configuration:
          - Multiple training GPUs → DDP via torchrun
          - Single dedicated GPU   → mp_actors subprocess
          - No dedicated GPU       → shared mode (sleep/wake)
        """
        use_ddp = (
            self._dedicated_gpus
            and self.training_gpus
            and len(self.training_gpus) > 1
            and num_sequences >= len(self.training_gpus)
        )
        if use_ddp:
            return await self._train_step_ddp(
                packed_tensors_dir, num_sequences, sequence_length, lr,
            )
        elif self._dedicated_gpus:
            return await self._train_step_dedicated(
                packed_tensors_dir, num_sequences, sequence_length, lr,
            )
        else:
            return await self._train_step_shared(
                packed_tensors_dir, num_sequences, sequence_length, lr,
            )

    async def _train_step_dedicated(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Training on a dedicated GPU — NO sleep/wake needed.

        SGLang stays fully active on inference GPUs while Unsloth trains
        on its own GPU. This eliminates all sleep/wake overhead.

        The worker is kept alive across steps (persistent mode) so the
        model is loaded only once. Since training GPUs are separate from
        inference GPUs, there is no need to free training GPU memory
        between steps.

        Step 1: spawn → init_model → train → save → lora_reload
        Step N: train → save → lora_reload  (worker reused, ~0s model load)
        """
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # No sleep needed — SGLang runs on separate GPUs
        timings["sleep_s"] = 0.0

        # 1. Spawn worker + load model (only on first step)
        if self._worker is None:
            t = time.perf_counter()
            self._worker = self._spawn_worker(last_checkpoint=self._last_checkpoint)

            init_result = await self._worker.init_model()
            n_params = init_result.get("trainable_params", "?")
            logger.info(
                f"Unsloth worker on GPU {self.training_gpus[0]} — "  # type: ignore[index]
                f"{n_params:,} trainable params (persistent)"
            )
            timings["model_load_s"] = time.perf_counter() - t
        else:
            logger.info(
                f"Reusing persistent worker on GPU {self.training_gpus[0]}"  # type: ignore[index]
            )
            timings["model_load_s"] = 0.0

        # 2. Train
        train_metrics = await self._worker.train_on_packed_tensors(
            packed_tensors_dir, num_sequences, sequence_length, lr,
        )

        # 3. Save LoRA
        t = time.perf_counter()
        self._latest_step += 1
        ckpt = await self._worker.save_lora(self._latest_step)
        self._last_checkpoint = ckpt
        timings["save_s"] = time.perf_counter() - t

        # Worker stays alive — no kill in dedicated mode.
        # Cleaned up in stop() at benchmark end.
        timings["kill_s"] = 0.0

        # No wake needed — SGLang never slept
        timings["wake_s"] = 0.0

        # 4. Hot-reload LoRA into SGLang
        timings["lora_reload_s"] = await self._load_lora(ckpt, self._latest_step)

        timings["total_overhead_s"] = time.perf_counter() - t_total

        reused = timings["model_load_s"] == 0.0
        logger.info(
            f"Step {self._latest_step} done (dedicated GPU) — "
            f"train={train_metrics['training_time_s']:.1f}s  "
            f"overhead={timings['total_overhead_s']:.1f}s  "
            f"({'persistent worker' if reused else 'fresh worker'})"
        )

        return {**train_metrics, **timings}

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port for DDP master."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    async def _train_step_ddp(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Training on multiple GPUs via DDP (torchrun).

        Launches train_ddp.py via torchrun with CUDA_VISIBLE_DEVICES
        set to the training GPUs. Communication is via files on disk:
          - Input: packed tensors (already saved by benchmark runner)
          - Output: LoRA checkpoint + metrics JSON

        Loop:
          1. Write config JSON for the DDP script
          2. Launch torchrun subprocess
          3. Wait for completion, read metrics JSON
          4. Hot-reload LoRA into SGLang
        """
        assert self.training_gpus is not None and len(self.training_gpus) > 1

        timings: dict[str, float] = {}
        t_total = time.perf_counter()
        timings["sleep_s"] = 0.0  # No sleep needed — dedicated GPUs

        self._latest_step += 1

        # 1. Write config for the DDP training script
        ddp_config = {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "max_seq_length": self.max_seq_length,
            "learning_rate": self.learning_rate,
            "moe_backend": self.moe_backend,
            "load_in_4bit": self.load_in_4bit,
            "last_checkpoint": self._last_checkpoint,
            "packed_tensors_dir": packed_tensors_dir,
            "num_sequences": num_sequences,
            "sequence_length": sequence_length,
            "lr": lr,
            "step_number": self._latest_step,
            "results_file": os.path.join(
                self.log_dir, f"ddp_results_step{self._latest_step:04d}.json"
            ),
        }
        config_file = os.path.join(
            self.log_dir, f"ddp_config_step{self._latest_step:04d}.json"
        )
        with open(config_file, "w") as f:
            json.dump(ddp_config, f, indent=2)

        # 2. Launch torchrun
        n_gpus = len(self.training_gpus)
        cuda_vis = ",".join(str(g) for g in self.training_gpus)
        master_port = self._find_free_port()

        train_script = os.path.join(
            os.path.dirname(__file__), "train_ddp.py"
        )

        # Use the same python as the current venv
        python_exe = sys.executable

        cmd = [
            python_exe, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            f"--master_port={master_port}",
            train_script,
            "--config", config_file,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_vis
        # Ensure project root is on PYTHONPATH
        project_root = str(Path(__file__).parent.parent.parent)
        extra_paths = [project_root, os.path.join(project_root, "src")]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            extra_paths + ([existing] if existing else [])
        )

        logger.info(
            f"DDP training: {n_gpus} GPUs {self.training_gpus}, "
            f"master_port={master_port}"
        )

        t = time.perf_counter()
        stderr_log = os.path.join(
            self.log_dir, f"ddp_stderr_step{self._latest_step:04d}.log"
        )

        # Run torchrun in a thread to avoid blocking the event loop
        stdout_log = os.path.join(
            self.log_dir, f"ddp_stdout_step{self._latest_step:04d}.log"
        )

        def _run_torchrun():
            with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
                return subprocess.run(cmd, env=env, stdout=fout, stderr=ferr)

        loop = asyncio.get_event_loop()
        proc_result = await loop.run_in_executor(None, _run_torchrun)

        timings["torchrun_s"] = time.perf_counter() - t

        if proc_result.returncode != 0:
            # Read stderr for error details
            err_msg = ""
            if os.path.exists(stderr_log):
                with open(stderr_log) as f:
                    err_msg = f.read()[-2000:]  # last 2000 chars
            raise RuntimeError(
                f"DDP training failed (exit code {proc_result.returncode}). "
                f"Stderr: {err_msg}"
            )

        # 3. Read results from DDP script
        results_file = ddp_config["results_file"]
        if not os.path.exists(results_file):
            raise RuntimeError(
                f"DDP training completed but no results file at {results_file}"
            )

        with open(results_file) as f:
            train_metrics = json.load(f)

        ckpt = train_metrics.pop("checkpoint")
        self._last_checkpoint = ckpt
        timings["save_s"] = 0.0  # save timing is included in torchrun_s

        # No kill needed — torchrun subprocess already exited
        timings["kill_s"] = 0.0
        timings["wake_s"] = 0.0

        # 4. Hot-reload LoRA into SGLang
        timings["lora_reload_s"] = await self._load_lora(ckpt, self._latest_step)
        timings["total_overhead_s"] = time.perf_counter() - t_total
        timings["model_load_s"] = 0.0  # included in torchrun_s

        logger.info(
            f"Step {self._latest_step} done (DDP x{n_gpus}) — "
            f"train={train_metrics['training_time_s']:.1f}s  "
            f"overhead={timings['total_overhead_s']:.1f}s  "
            f"(no sleep/wake)"
        )

        return {**train_metrics, **timings}

    async def _train_step_shared(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Training on shared GPU — sleep/wake cycle (single-GPU fallback).

        Loop:
          1. sleep()          — SGLang releases KV cache + weights
          2. spawn worker     — new subprocess (fresh CUDA context)
          3. init_model()     — load base model + previous LoRA checkpoint
          4. train            — ART loss on packed tensors
          5. save_lora()      — save adapter to disk
          6. KILL worker      — subprocess dies, ALL GPU memory freed
          7. wake_up()        — SGLang restores KV cache + weights
          8. load_lora()      — hot-reload adapter (<2s)
        """
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # 1. Sleep SGLang — free GPU for training
        timings["sleep_s"] = await self.sleep()

        # 2. Spawn worker
        t = time.perf_counter()
        self._worker = self._spawn_worker(last_checkpoint=self._last_checkpoint)

        # 3. Load model
        init_result = await self._worker.init_model()
        n_params = init_result.get("trainable_params", "?")
        logger.info(f"Unsloth worker model loaded — {n_params:,} trainable params")
        timings["model_load_s"] = time.perf_counter() - t

        # 4. Train
        train_metrics = await self._worker.train_on_packed_tensors(
            packed_tensors_dir, num_sequences, sequence_length, lr,
        )

        # 5. Save LoRA
        t = time.perf_counter()
        self._latest_step += 1
        ckpt = await self._worker.save_lora(self._latest_step)
        self._last_checkpoint = ckpt
        timings["save_s"] = time.perf_counter() - t

        # 6. Kill worker
        t = time.perf_counter()
        self._kill_worker()
        timings["kill_s"] = time.perf_counter() - t

        await asyncio.sleep(2)

        # 7. Wake SGLang
        timings["wake_s"] = await self.wake_up()

        # 8. Hot-reload LoRA
        timings["lora_reload_s"] = await self._load_lora(ckpt, self._latest_step)

        # 9. Health check
        if self._server is not None and not self._server.is_running:
            logger.warning("SGLang server died after wake — restarting...")
            t = time.perf_counter()
            try:
                await self._server.stop()
            except Exception:
                pass
            self._server = self._create_server()
            await self._server.start()
            self._active_lora_name = None
            timings["restart_s"] = time.perf_counter() - t
            logger.warning(f"SGLang restarted in {timings['restart_s']:.1f}s (no LoRA)")

        timings["total_overhead_s"] = time.perf_counter() - t_total

        logger.info(
            f"Step {self._latest_step} done (shared GPU) — "
            f"train={train_metrics['training_time_s']:.1f}s  "
            f"overhead={timings['total_overhead_s']:.1f}s"
        )

        return {**train_metrics, **timings}

    # ------------------------------------------------------------------
    # Properties for the benchmark runner
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://0.0.0.0:{self.port}/v1"

    @property
    def inference_model_name(self) -> str:
        """Model name for inference requests via the OpenAI-compatible API.

        SGLang v0.5.3+ uses "base-model:adapter-name" syntax for
        /v1/chat/completions when a LoRA adapter is active.
        Falls back to base model name when no adapter is loaded.
        """
        if self._active_lora_name:
            return f"{self.base_model}:{self._active_lora_name}"
        return self.base_model
