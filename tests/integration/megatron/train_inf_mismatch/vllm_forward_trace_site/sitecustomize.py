from __future__ import annotations

import builtins
import functools
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

_REAL_IMPORT = builtins.__import__
_LOCK = threading.Lock()
_PATCHED: set[str] = set()
_CALL_INDEX = 0
_LAYER_RE = re.compile(r"model\.layers\.\d+")


def _trace_dir() -> Path | None:
    raw = os.environ.get("ART_VLLM_FORWARD_TRACE_DIR")
    return Path(raw) if raw else None


def _event(kind: str, **payload: Any) -> None:
    trace_dir = _trace_dir()
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "kind": kind,
        "pid": os.getpid(),
        "time": time.time(),
        **payload,
    }
    with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _next_index() -> int:
    global _CALL_INDEX
    with _LOCK:
        value = _CALL_INDEX
        _CALL_INDEX += 1
        return value


def _primary_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _primary_tensor(item)
            if isinstance(tensor, torch.Tensor):
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _primary_tensor(item)
            if isinstance(tensor, torch.Tensor):
                return tensor
    return None


def _primary_input(name: str, inputs: Any) -> Any:
    import torch

    if (
        _LAYER_RE.fullmatch(name)
        or name.endswith(".self_attn")
        or name.endswith(".attention")
    ) and isinstance(inputs, tuple):
        for item in inputs[1:]:
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                return item
    return _primary_tensor(inputs)


def _save_tensor(
    trace_dir: Path, call_index: int, field: str, tensor: Any
) -> str | None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    max_rows = int(os.environ.get("ART_VLLM_FORWARD_TRACE_MAX_ROWS", "768"))
    if tensor.ndim > 0 and int(tensor.shape[0]) > max_rows:
        return None
    rel_path = Path("tensors") / f"{call_index:06d}_{field}.pt"
    path = trace_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    return str(rel_path)


def _should_capture(name: str) -> bool:
    if name == "model.embed_tokens" or name == "model.norm":
        return True
    if _LAYER_RE.fullmatch(name):
        return True
    if os.environ.get("ART_VLLM_FORWARD_TRACE_DETAIL") != "1":
        return False
    return (
        name.endswith(".input_layernorm")
        or name.endswith(".self_attn")
        or name.endswith(".qkv_proj")
        or name.endswith(".q_norm")
        or name.endswith(".k_norm")
        or name.endswith(".o_proj")
        or name.endswith(".post_attention_layernorm")
        or name.endswith(".mlp")
        or name.endswith(".gate_up_proj")
        or name.endswith(".down_proj")
    )


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _make_hook(name: str):
    def _hook(module: Any, inputs: Any, output: Any) -> None:
        trace_dir = _trace_dir()
        if trace_dir is None:
            return
        call_index = _next_index()
        primary_input = _primary_input(name, inputs)
        primary_output = _primary_tensor(output)
        _event(
            "module",
            call_index=call_index,
            module_name=name,
            module_type=module.__class__.__name__,
            primary_input_shape=_shape(primary_input),
            primary_output_shape=_shape(primary_output),
            primary_input_path=_save_tensor(
                trace_dir, call_index, "primary_input", primary_input
            ),
            primary_output_path=_save_tensor(
                trace_dir, call_index, "primary_output", primary_output
            ),
        )

    return _hook


def _register_model_hooks(model: Any) -> None:
    if getattr(model, "_art_vllm_forward_trace_registered", False):
        return
    names: list[str] = []
    for name, module in model.named_modules():
        if _should_capture(name):
            module.register_forward_hook(_make_hook(name))
            names.append(name)
    setattr(model, "_art_vllm_forward_trace_registered", True)
    _event("registered_module_hooks", module_names=names)


def _patch_causal_lm_class(module: Any, class_name: str) -> None:
    key = f"{module.__name__}.{class_name}"
    if key in _PATCHED or not hasattr(module, class_name):
        return
    cls = getattr(module, class_name)
    original_init = cls.__init__
    original_compute_logits = getattr(cls, "compute_logits", None)

    @functools.wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if _trace_dir() is not None:
            _register_model_hooks(self)

    cls.__init__ = __init__

    if original_compute_logits is not None:

        @functools.wraps(original_compute_logits)
        def compute_logits(self: Any, hidden_states: Any, *args: Any, **kwargs: Any):
            output = original_compute_logits(self, hidden_states, *args, **kwargs)
            trace_dir = _trace_dir()
            if trace_dir is not None:
                call_index = _next_index()
                _event(
                    "compute_logits",
                    call_index=call_index,
                    module_name="compute_logits",
                    module_type=self.__class__.__name__,
                    primary_input_shape=_shape(hidden_states),
                    primary_output_shape=_shape(output),
                    primary_input_path=_save_tensor(
                        trace_dir, call_index, "primary_input", hidden_states
                    ),
                    primary_output_path=(
                        _save_tensor(trace_dir, call_index, "primary_output", output)
                        if os.environ.get("ART_VLLM_FORWARD_TRACE_SAVE_LOGITS") == "1"
                        else None
                    ),
                )
            return output

        cls.compute_logits = compute_logits

    _PATCHED.add(key)
    _event("patched_class", target=key)


def _maybe_patch(name: str, module: Any) -> None:
    if _trace_dir() is None:
        return
    if name == "vllm.model_executor.models.qwen3":
        _patch_causal_lm_class(module, "Qwen3ForCausalLM")
    elif name == "vllm.model_executor.models.qwen3_moe":
        _patch_causal_lm_class(module, "Qwen3MoeForCausalLM")


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _REAL_IMPORT(name, globals, locals, fromlist, level)
    if level == 0:
        _maybe_patch(name, module)
    return module


builtins.__import__ = _import  # ty: ignore[invalid-assignment]


def _patch_loop() -> None:
    import sys

    while True:
        if _trace_dir() is not None:
            for name, module in list(sys.modules.items()):
                _maybe_patch(name, module)
        time.sleep(0.1)


threading.Thread(target=_patch_loop, daemon=True).start()
_event("sitecustomize_active")
