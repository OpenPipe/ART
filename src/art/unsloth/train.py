"""Unsloth training runtime and public API.

Public cross-repo API consumed by serverless-training:
- create_unsloth_train_context
- run_unsloth_rl_training
- run_unsloth_sft_training
"""

import asyncio
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import gc
import os
import time
from typing import Any, AsyncIterator, Callable, Iterable, Literal, cast

from datasets import Dataset
import nest_asyncio
import peft
from peft.peft_model import PeftModel
import torch
from torch.optim import Optimizer
from transformers import GenerationMixin, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from .. import dev, types
from ..loss import loss_fn, shift_tensor
from ..preprocessing.inputs import TrainInputs, create_train_inputs
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from ..preprocessing.tokenize import SFTBatch
from ..types import TrainConfig

nest_asyncio.apply()

__all__ = [
    "CausalLM",
    "StopTrainingLoop",
    "UnslothTrainContext",
    "create_unsloth_train_context",
    "gc_and_empty_cuda_cache",
    "run_unsloth_rl_training",
    "run_unsloth_sft_training",
]

_TRAIN_TASK_SHUTDOWN_TIMEOUT_S = 5.0

_UPSTREAM_TRAIN_METRIC_KEYS = {
    "reward": "reward",
    "reward_std_dev": "reward_std_dev",
    "exception_rate": "exception_rate",
    "policy_loss": "loss/train",
    "loss": "loss/train",
    "entropy": "loss/entropy",
    "kl_div": "loss/kl_div",
    "kl_policy_ref": "loss/kl_policy_ref",
    "grad_norm": "loss/grad_norm",
    "learning_rate": "loss/learning_rate",
    "num_groups_submitted": "data/step_num_groups_submitted",
    "num_groups_trainable": "data/step_num_groups_trainable",
    "num_trajectories": "data/step_num_trajectories",
    "num_trainable_tokens": "data/step_trainer_tokens",
    "train_tokens": "data/step_trainer_tokens",
    "num_datums": "data/step_num_datums",
}


class StopTrainingLoop(Exception):
    """Signal that the background trainer loop should exit cleanly."""


class _StopTrainInputs:
    """Sentinel used to stop the background trainer loop cleanly."""


_STOP_TRAIN_INPUT = _StopTrainInputs()
_TrainLoopInput = TrainInputs | _StopTrainInputs


class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass


@dataclass
class UnslothTrainContext:
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: peft.peft_model.PeftModelForCausalLM
    trainer: GRPOTrainer
    inputs_queue: asyncio.Queue[_TrainLoopInput]
    results_queue: asyncio.Queue[dict[str, float]]
    train_task: asyncio.Task[None] | None = None
    warmup_pending: bool = True
    last_training_mode: Literal["sft", "rl"] | None = None
    _is_offloaded: bool = False
    _pinned_buffers: dict[str, torch.Tensor] | None = None

    def offload_to_cpu(self) -> None:
        if self._is_offloaded:
            return

        if self._pinned_buffers is None:
            self._pinned_buffers = {}

        for name, param in self.peft_model.named_parameters():
            if param.device.type != "cuda":
                continue
            if (
                name not in self._pinned_buffers
                or self._pinned_buffers[name].shape != param.shape
            ):
                self._pinned_buffers[name] = torch.empty(
                    param.shape,
                    dtype=param.dtype,
                    device="cpu",
                    pin_memory=True,
                )
            self._pinned_buffers[name].copy_(param.data, non_blocking=True)
            param.data = self._pinned_buffers[name]

        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for param_id, state in optimizer.state.items():
                for key, value in state.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device.type != "cuda"
                    ):
                        continue
                    buffer_key = f"opt_{id(param_id)}_{key}"
                    if (
                        buffer_key not in self._pinned_buffers
                        or self._pinned_buffers[buffer_key].shape != value.shape
                    ):
                        self._pinned_buffers[buffer_key] = torch.empty(
                            value.shape,
                            dtype=value.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                    self._pinned_buffers[buffer_key].copy_(value, non_blocking=True)
                    state[key] = self._pinned_buffers[buffer_key]

        torch.cuda.synchronize()
        self._is_offloaded = True
        gc_and_empty_cuda_cache()

    def reload_to_gpu(self, device: str = "cuda:0") -> None:
        if not self._is_offloaded:
            return

        for _, param in self.peft_model.named_parameters():
            if param.device.type != "cpu":
                continue
            gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
            gpu_tensor.copy_(param.data, non_blocking=True)
            param.data = gpu_tensor

        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device.type != "cpu"
                    ):
                        continue
                    gpu_tensor = torch.empty(
                        value.shape, dtype=value.dtype, device=device
                    )
                    gpu_tensor.copy_(value, non_blocking=True)
                    state[key] = gpu_tensor

        torch.cuda.synchronize()
        self._is_offloaded = False

    async def load_lora_adapter(self, lora_path: str) -> None:
        try:
            await self.results_queue.join()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            import importlib

            load_safetensors = importlib.import_module("safetensors.torch").load_file
        except Exception:
            load_safetensors = None  # type: ignore[assignment]

        state_dict = None
        st_path = os.path.join(lora_path, "adapter_model.safetensors")
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        alt_st_path = os.path.join(lora_path, "model.safetensors")
        alt_bin_path = os.path.join(lora_path, "pytorch_model.bin")
        try:
            if os.path.exists(st_path) and load_safetensors is not None:
                state_dict = load_safetensors(st_path, device="cpu")
            elif os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")  # type: ignore[call-arg]
            elif os.path.exists(alt_st_path) and load_safetensors is not None:
                state_dict = load_safetensors(alt_st_path, device="cpu")
            elif os.path.exists(alt_bin_path):
                state_dict = torch.load(alt_bin_path, map_location="cpu")  # type: ignore[call-arg]
            else:
                raise FileNotFoundError(f"No adapter weights found in {lora_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load LoRA adapter weights: {exc}") from exc

        with torch.no_grad():
            self.peft_model.zero_grad(set_to_none=True)
            optimizer = getattr(self.trainer, "optimizer", None)
            if optimizer is not None:
                optimizer = getattr(optimizer, "optimizer", optimizer)
                if hasattr(optimizer, "zero_grad"):
                    optimizer.zero_grad(set_to_none=True)  # type: ignore[arg-type]
                if hasattr(optimizer, "state") and isinstance(optimizer.state, dict):
                    optimizer.state.clear()

        try:
            try:
                from peft.utils.save_and_load import (
                    set_peft_model_state_dict as _set_peft_model_state_dict,
                )
            except Exception:
                from peft import (
                    set_peft_model_state_dict as _set_peft_model_state_dict,  # type: ignore
                )

            active_adapter = getattr(self.peft_model, "active_adapter", "default")
            _set_peft_model_state_dict(
                self.peft_model,
                state_dict,
                adapter_name=active_adapter,
            )
            self.peft_model.set_adapter(active_adapter)
        except Exception as exc:
            raise RuntimeError(f"Failed to set LoRA weights in-place: {exc}") from exc

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    async def load_optimizer_state(self, checkpoint_dir: str) -> None:
        try:
            await self.results_queue.join()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            if not isinstance(self.trainer.optimizer, Optimizer):
                raise RuntimeError("Trainer optimizer is not initialized")
            self.trainer.optimizer.load_state_dict(optimizer_state)

    def save_lora_adapter(self, lora_path: str) -> None:
        self.trainer.save_model(lora_path)

    def save_optimizer_state(self, checkpoint_dir: str) -> None:
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if not isinstance(self.trainer.optimizer, Optimizer):
            raise RuntimeError("Trainer optimizer is not initialized")
        torch.save(self.trainer.optimizer.state_dict(), optimizer_path)

    async def stop_background_training(
        self,
        *,
        timeout_s: float = _TRAIN_TASK_SHUTDOWN_TIMEOUT_S,
    ) -> None:
        train_task = self.train_task
        self.train_task = None
        if train_task is None or train_task.done():
            return

        self.inputs_queue.put_nowait(_STOP_TRAIN_INPUT)
        try:
            await asyncio.wait_for(train_task, timeout=timeout_s)
        except asyncio.TimeoutError:
            train_task.cancel()


def _canonicalize_upstream_metric_key(metric: str) -> str:
    if "/" in metric:
        return metric
    if metric == "tokens_per_second":
        return ""
    if metric.startswith("group_metric_"):
        return f"group_{metric[len('group_metric_') :]}"
    return _UPSTREAM_TRAIN_METRIC_KEYS.get(metric, metric)


def _canonicalize_upstream_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        canonical_key: float(value)
        for key, value in metrics.items()
        if (canonical_key := _canonicalize_upstream_metric_key(key))
    }


async def train(
    trainer: "GRPOTrainer",
    results_queue: asyncio.Queue[dict[str, float]],
) -> None:
    _compute_loss = trainer.compute_loss
    _log = trainer.log
    trainer.compute_loss = get_compute_loss_fn(trainer)
    trainer.log = get_log_fn(trainer, results_queue)  # ty:ignore[invalid-assignment]
    # Ensure we have a metrics container in the expected format
    try:
        is_dict = isinstance(getattr(trainer, "_metrics", None), dict)
        is_train_dict = is_dict and isinstance(trainer._metrics.get("train"), dict)
    except Exception:
        is_train_dict = False
    if not is_train_dict:
        trainer._metrics = {"train": defaultdict(list)}
    try:
        trainer.train()
    except StopTrainingLoop:
        return
    finally:
        trainer.compute_loss = _compute_loss
        trainer.log = _log  # ty:ignore[invalid-assignment]


def get_compute_loss_fn(trainer: "GRPOTrainer") -> Callable[..., torch.Tensor]:
    def compute_loss(
        model: "PeftModel",
        inputs: "TrainInputs",
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        config: TrainConfig = inputs.pop("config")  # type: ignore
        _config: dev.TrainConfig = inputs.pop("_config")  # type: ignore
        return_new_logprobs: bool = inputs.pop("return_new_logprobs", False)  # type: ignore

        num_trajectories_learning_rate_multiplier = (
            torch.unique(inputs["group_ids"]).numel()
            - torch.unique(inputs["parent_ids"]).numel()
        ) ** _config.get("num_trajectories_learning_rate_multiplier_power", 0.0)
        if optimizer := trainer.optimizer:
            optimizer = getattr(optimizer, "optimizer", optimizer)
            if param_groups := getattr(optimizer, "param_groups"):
                for param_group in param_groups:
                    param_group["lr"] = (
                        config.learning_rate * num_trajectories_learning_rate_multiplier
                    )
                    # param_group["betas"] = config.betas
                    # if param_group.get("weight_decay"):
                    #     param_group["weight_decay"] = config.weight_decay

        if inputs.get("pixel_values") and inputs["pixel_values"][0] is not None:
            inputs["pixel_values"] = inputs["pixel_values"][0]  # type: ignore
        else:
            del inputs["pixel_values"]  # type: ignore
        if inputs.get("image_grid_thw") and inputs["image_grid_thw"][0] is not None:
            inputs["image_grid_thw"] = inputs["image_grid_thw"][0]  # type: ignore
        else:
            del inputs["image_grid_thw"]  # type: ignore

        # Move tensors to the correct device
        inputs = {
            key: tensor.to(trainer.accelerator.device)  # type: ignore
            for key, tensor in inputs.items()
        }  # ty:ignore[invalid-assignment]

        accelerate_mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION")
        force_float32 = os.environ.get("UNSLOTH_FORCE_FLOAT32")

        if (
            accelerate_mixed_precision is None
            or accelerate_mixed_precision == "fp16"
            or force_float32 == "1"
        ):
            dtype_for_autocasting = torch.float16
        else:
            dtype_for_autocasting = torch.bfloat16

        batch_size, seq_len = inputs["tokens"].size()
        attn_bias = calculate_attn_bias(
            batch_size,
            seq_len,
            trainer.accelerator.device,
            inputs["group_ids"],
            inputs["parent_ids"],
            dtype_for_autocasting,
        )

        # Calculate log probabilities
        lm_head_t = cast(
            torch.Tensor,
            trainer.model.get_output_embeddings().weight.t(),  # type: ignore
        )  # Shape [H, V]
        next_input_ids = shift_tensor(inputs["tokens"], 0)
        chunk_size = _config.get("logprob_calculation_chunk_size", 1024)
        # Assert that sequence length is evenly divisible by the chunk size
        assert seq_len % chunk_size == 0, (
            f"Sequence length ({seq_len}) must be evenly divisible by chunk size ({chunk_size})"
        )
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        forward_kwargs = {}
        if "pixel_values" in inputs:
            forward_kwargs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
        new_logprobs, entropies = calculate_logprobs(
            dtype_for_autocasting,
            trainer,
            inputs["tokens"],
            attn_bias,
            forward_kwargs,
            next_input_ids,
            lm_head_t,
            chunk_size=chunk_size,
            inference_mode=return_new_logprobs,
            no_grad=return_new_logprobs,
            reference_logprobs=False,
        )
        if return_new_logprobs:
            return torch.nn.functional.pad(new_logprobs[:, :-1], (1, 0), value=0.0)
        if config.kl_penalty_coef > 0.0:
            ref_adapter = _config.get("kl_ref_adapter_path")
            ref_logprobs, _ = calculate_logprobs(
                dtype_for_autocasting,
                trainer,
                inputs["tokens"],
                attn_bias,
                forward_kwargs,
                next_input_ids,
                lm_head_t,
                chunk_size=chunk_size,
                # Can't use inference_mode with a custom adapter — inference
                # tensors don't track version counters, which breaks unsloth's
                # LoRA kernels. Use no_grad instead.
                inference_mode=ref_adapter is None,
                no_grad=ref_adapter is not None,
                reference_logprobs=True,
                reference_adapter_name=ref_adapter,
            )
        else:
            ref_logprobs = None
        del attn_bias

        loss = loss_fn(
            inputs,
            new_logprobs,
            ref_logprobs,
            entropies,
            _config,
        )

        trainer._metrics["train"]["loss/learning_rate"].append(config.learning_rate)
        trainer._metrics["train"]["loss/train"].append(loss.policy_loss.item())
        if loss.entropy is not None:
            trainer._metrics["train"]["loss/entropy"].append(loss.entropy.item())
        if loss.kl_policy_ref is not None:
            trainer._metrics["train"]["loss/kl_policy_ref"].append(
                loss.kl_policy_ref.item()
            )
        return loss.policy_loss

    return compute_loss


def get_log_fn(
    trainer: Any, results_queue: asyncio.Queue[dict[str, float]]
) -> Callable[..., None]:
    def log(logs: dict[str, float], start_time: float | None = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in trainer._metrics["train"].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". Normalize them into the `val/...` taxonomy instead.
        if next(iter(logs.keys())).startswith("eval_"):
            normalized_metrics = {f"val/{key}": val for key, val in metrics.items()}
            normalized_logs = {
                f"val/{_canonicalize_upstream_metric_key(key[len('eval_') :])}": val
                for key, val in logs.items()
            }
            results_queue.put_nowait({**normalized_metrics, **normalized_logs})
        else:
            results_queue.put_nowait(
                {**_canonicalize_upstream_metrics(logs), **metrics}
            )
        trainer._metrics["train"].clear()

    return log


def calculate_attn_bias(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = calculate_mask(batch_size, seq_len, device, group_ids, parent_ids)
    # Use the same dtype as autocast to save memory and avoid dtype conversions
    attn_bias = torch.where(
        mask,
        torch.tensor(
            0.0,
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            float("-inf"),
            dtype=dtype,
            device=device,
        ),
    )
    del mask
    return attn_bias


def calculate_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> torch.Tensor:
    causal_mask = (
        torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        )
        .unsqueeze(0)
        .expand(batch_size, seq_len, seq_len)
    )
    group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    parent_mask = parent_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    mask = causal_mask & (group_mask | parent_mask)
    return mask


@contextmanager
def _use_adapter(trainer: "GRPOTrainer", adapter_path: str):
    """Context manager that switches to a named LoRA adapter, then restores the original."""
    # Sanitize the path to a valid module name (no dots allowed by PyTorch)
    safe_name = adapter_path.replace(".", "_").replace("/", "_")
    peft_model = trainer.accelerator.unwrap_model(
        trainer.model, keep_fp32_wrapper=False
    )
    if safe_name not in peft_model.peft_config:
        peft_model.load_adapter(adapter_path, adapter_name=safe_name)
    previous_adapter = peft_model.active_adapter
    if isinstance(previous_adapter, list):
        previous_adapter = previous_adapter[0]
    peft_model.set_adapter(safe_name)
    try:
        yield
    finally:
        peft_model.set_adapter(previous_adapter)


def calculate_logprobs(
    dtype_for_autocast: torch.dtype,
    trainer: "GRPOTrainer",
    input_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    forward_kwargs: dict[str, torch.Tensor],
    next_input_ids: torch.Tensor,
    lm_head_t: torch.Tensor,
    chunk_size: int,
    inference_mode: bool,
    no_grad: bool,
    reference_logprobs: bool,
    reference_adapter_name: str | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # Returns (log_probs, entropy) both shape [B, S]
    if reference_logprobs and reference_adapter_name is not None:
        adapter_ctx = _use_adapter(trainer, reference_adapter_name)
    elif reference_logprobs:
        adapter_ctx = trainer.accelerator.unwrap_model(
            trainer.model, keep_fp32_wrapper=False
        ).disable_adapter()
    else:
        adapter_ctx = nullcontext()
    with (
        torch.inference_mode() if inference_mode else nullcontext(),
        torch.no_grad() if no_grad else nullcontext(),
        adapter_ctx,
        torch.amp.autocast_mode.autocast(device_type="cuda", dtype=dtype_for_autocast),
    ):
        hidden_states = trainer.model(  # type: ignore
            input_ids=input_ids, causal_mask=causal_mask, **forward_kwargs
        ).logits  # Shape [B, S, H]
    return _calculate_logprobs(lm_head_t, hidden_states, next_input_ids, chunk_size)


def _calculate_logprobs(
    lm_head_t: torch.Tensor,  # Shape [H, V]
    hidden_states: torch.Tensor,  # Shape [B, S, H]
    next_input_ids: torch.Tensor,  # Shape [B, S]
    chunk_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # Returns (log_probs, entropy) both shape [B, S]
    batch_size, seq_len, _ = hidden_states.shape
    # Output shape is [B, S]
    log_probs = torch.empty(
        (batch_size, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    entropy = torch.empty(
        (batch_size, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # Ensure lm_head_t is in the same dtype as hidden_states
    lm_head_t = lm_head_t.to(hidden_states.dtype)

    # Chunk over sequence length S using Python range
    for i in range(0, seq_len, chunk_size):
        chunk_hs = hidden_states[:, i : i + chunk_size, :]  # [B, chunk_size, H]
        chunk_input_ids = next_input_ids[:, i : i + chunk_size]  # [B, chunk_size]
        chunk_logits = torch.matmul(chunk_hs, lm_head_t)  # [B, chunk_size, V]
        chunk_selected_logits = torch.gather(
            chunk_logits, dim=-1, index=chunk_input_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, chunk_size]
        chunk_logsumexp = torch.logsumexp(chunk_logits, dim=-1)  # [B, chunk_size]
        log_probs[:, i : i + chunk_size] = chunk_selected_logits - chunk_logsumexp

        # Compute entropy for the chunk
        log_probs_full = chunk_logits - chunk_logsumexp.unsqueeze(-1)
        chunk_entropy = (-torch.exp(log_probs_full) * log_probs_full).sum(
            dim=-1
        )  # [B, chunk_size]
        entropy[:, i : i + chunk_size] = chunk_entropy

        del (
            chunk_hs,
            chunk_input_ids,
            chunk_logits,
            chunk_selected_logits,
            chunk_logsumexp,
            log_probs_full,
            chunk_entropy,
        )
    del hidden_states
    return log_probs, entropy


def gc_and_empty_cuda_cache(n: int = 3) -> None:
    [gc.collect() >= 0 and torch.cuda.empty_cache() for _ in range(n)]


def create_unsloth_train_context(
    *,
    init_args: dict[str, Any],
    peft_args: dict[str, Any],
    trainer_args: dict[str, Any],
    use_fast_model: bool = False,
) -> UnslothTrainContext:
    import unsloth

    loader_cls = unsloth.FastModel if use_fast_model else unsloth.FastLanguageModel
    model, tokenizer = cast(
        tuple[CausalLM, PreTrainedTokenizerBase],
        loader_cls.from_pretrained(**init_args),
    )

    if (
        hasattr(model, "peft_config")
        and getattr(model, "peft_config", None) is not None
    ):
        peft_model = cast(peft.peft_model.PeftModelForCausalLM, model)
    else:
        peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            loader_cls.get_peft_model(model, **peft_args),
        )

    if not hasattr(peft_model, "warnings_issued"):
        peft_model.warnings_issued = {}  # type: ignore[attr-defined]

    trainer = GRPOTrainer(
        model=peft_model,  # type: ignore[arg-type]
        reward_funcs=[],
        args=GRPOConfig(**trainer_args),
        train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(10_000_000)]),
        processing_class=tokenizer,
    )
    if trainer.optimizer is None:
        trainer.create_optimizer()

    inputs_queue: asyncio.Queue[_TrainLoopInput] = asyncio.Queue()
    results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

    def _async_prepare_inputs(*_: Any, **__: Any) -> dict[str, torch.Tensor]:
        async def get_inputs() -> _TrainLoopInput:
            return await inputs_queue.get()

        inputs = asyncio.run(get_inputs())
        if isinstance(inputs, _StopTrainInputs):
            raise StopTrainingLoop()
        return cast(dict[str, torch.Tensor], inputs)

    trainer._prepare_inputs = _async_prepare_inputs

    return UnslothTrainContext(
        model=model,
        tokenizer=tokenizer,
        peft_model=peft_model,
        trainer=trainer,
        inputs_queue=inputs_queue,
        results_queue=results_queue,
    )


def _get_trainer_optimizer(ctx: UnslothTrainContext) -> Optimizer:
    optimizer = cast(Optimizer | None, getattr(ctx.trainer, "optimizer", None))
    if optimizer is None:
        raise RuntimeError("Trainer optimizer must be initialized before training")
    return optimizer


def _reset_optimizer_if_mode_changed(
    ctx: UnslothTrainContext,
    mode: Literal["sft", "rl"],
) -> None:
    mode_changed = ctx.last_training_mode is not None and ctx.last_training_mode != mode
    if mode_changed:
        _get_trainer_optimizer(ctx).state.clear()
    ctx.last_training_mode = mode


def _precalculate_new_logprobs(
    ctx: UnslothTrainContext,
    packed_tensors: PackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
) -> torch.Tensor:
    return torch.cat(
        [
            ctx.trainer.compute_loss(
                ctx.peft_model,
                TrainInputs(  # ty:ignore[missing-typed-dict-key]
                    **{
                        key: value[offset : offset + 1]
                        for key, value in packed_tensors.items()
                        if isinstance(value, torch.Tensor)
                    },
                    pixel_values=packed_tensors["pixel_values"][offset : offset + 1],
                    image_grid_thw=packed_tensors["image_grid_thw"][
                        offset : offset + 1
                    ],
                    config=config,
                    _config=_config,
                    return_new_logprobs=True,
                ),
            )
            for offset in range(0, packed_tensors["tokens"].shape[0])
        ]
    ).to("cpu")


async def run_unsloth_rl_training(
    ctx: UnslothTrainContext,
    disk_packed_tensors: DiskPackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
    verbose: bool = False,
) -> AsyncIterator[dict[str, float]]:
    _reset_optimizer_if_mode_changed(ctx, "rl")
    optimizer = _get_trainer_optimizer(ctx)
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = 0.1

    packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
    await ctx.results_queue.join()

    if ctx.train_task is None:
        ctx.train_task = asyncio.create_task(
            train(
                trainer=ctx.trainer,
                results_queue=ctx.results_queue,
            )
        )

    warmup = ctx.warmup_pending
    precalculate_logprobs = _config.get("precalculate_logprobs", False)

    for offset in range(0, packed_tensors["tokens"].shape[0]):
        for _ in range(2 if warmup else 1):
            if precalculate_logprobs and not warmup:
                packed_tensors["original_logprobs"] = packed_tensors["logprobs"]  # type: ignore[index]
                packed_tensors["logprobs"] = _precalculate_new_logprobs(
                    ctx,
                    packed_tensors,
                    config,
                    _config,
                )
                precalculate_logprobs = False

            ctx.inputs_queue.put_nowait(
                create_train_inputs(packed_tensors, offset, config, _config, warmup)
            )

            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(ctx.results_queue.get()),
                    ctx.train_task,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if verbose:
                print(
                    "Done waiting for a result from the queue or for the training task to, presumably, raise an exception"
                )
            for task in done:
                result = task.result()
                assert result is not None, "The training task should never finish."
                ctx.results_queue.task_done()
                if warmup:
                    gc_and_empty_cuda_cache()
                    await asyncio.sleep(0.1)
                    warmup = False
                    ctx.warmup_pending = False
                else:
                    yield result


async def run_unsloth_sft_training(
    ctx: UnslothTrainContext,
    batches: Iterable[SFTBatch],
    verbose: bool = False,
    *,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> AsyncIterator[dict[str, float]]:
    _reset_optimizer_if_mode_changed(ctx, "sft")
    optimizer = _get_trainer_optimizer(ctx)

    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay

    ctx.peft_model.train()
    device = next(ctx.peft_model.parameters()).device

    for batch_idx, batch in enumerate(batches):
        batch_start_time = time.perf_counter()
        batch_loss = 0.0

        for param_group in optimizer.param_groups:
            param_group["lr"] = batch.learning_rate

        num_trainable_tokens = torch.tensor(
            batch.num_trainable_tokens,
            dtype=torch.long,
            device=device,
        )

        for trajectory_tensor in batch.trajectory_tensors:
            input_ids = trajectory_tensor["input_ids"].to(device)
            attention_mask = trajectory_tensor["attention_mask"].to(device)
            labels = trajectory_tensor["labels"].to(device)

            outputs = ctx.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_items_in_batch=num_trainable_tokens,
            )
            loss = outputs.loss
            loss.backward()
            batch_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            ctx.peft_model.parameters(),
            max_grad_norm,
        ).item()

        optimizer.step()
        optimizer.zero_grad()

        batch_time = time.perf_counter() - batch_start_time
        tokens_per_second = batch.num_tokens / batch_time if batch_time > 0 else 0.0

        if verbose:
            print(
                f"Batch {batch_idx}: loss={batch_loss:.4f}, lr={batch.learning_rate:.2e}, "
                f"grad_norm={grad_norm:.4f}, tok/s={tokens_per_second:.1f}"
            )

        yield {
            "loss": batch_loss,
            "learning_rate": batch.learning_rate,
            "grad_norm": grad_norm,
            "num_trajectories": float(batch.num_trajectories),
            "num_tokens": float(batch.num_tokens),
            "num_trainable_tokens": float(batch.num_trainable_tokens),
            "tokens_per_second": tokens_per_second,
        }
