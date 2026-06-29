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
_LORA_PATCHED = False
_LORA_PATCHING = False
_LORA_DUMPED_SET_LORA: set[str] = set()


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


def _lora_internal_trace_enabled() -> bool:
    return os.environ.get("ART_VLLM_LORA_INTERNAL_TRACE") == "1"


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
    rel_path = Path("tensors") / f"{os.getpid()}_{call_index:06d}_{field}.pt"
    path = trace_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    return str(rel_path)


def _save_internal_tensor(
    trace_dir: Path,
    *,
    stem: str,
    field: str,
    tensor: Any,
) -> str | None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    rel_path = Path("lora_internal") / f"{os.getpid()}_{safe_stem}_{field}.pt"
    path = trace_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    return str(rel_path)


def _save_tensor_list(
    trace_dir: Path,
    *,
    stem: str,
    field: str,
    tensors: Any,
) -> list[dict[str, Any]]:
    import torch

    if not isinstance(tensors, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for index, tensor in enumerate(tensors):
        rows.append(
            {
                "index": index,
                "shape": _shape(tensor),
                "path": _save_internal_tensor(
                    trace_dir,
                    stem=stem,
                    field=f"{field}_{index}",
                    tensor=tensor,
                )
                if isinstance(tensor, torch.Tensor)
                else None,
            }
        )
    return rows


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
        or name.endswith(".self_attn.attn")
        or name.endswith(".attn.attn")
        or name.endswith(".qkv_proj")
        or name.endswith(".q_norm")
        or name.endswith(".k_norm")
        or name.endswith(".o_proj")
        or name.endswith(".post_attention_layernorm")
        or name.endswith(".mlp")
        or name.endswith(".mlp.router")
        or name.endswith(".gate_up_proj")
        or name.endswith(".down_proj")
    )


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _attention_input_paths(
    trace_dir: Path, call_index: int, name: str, inputs: Any
) -> dict[str, Any]:
    if not (
        name.endswith(".attn.attn") or name.endswith(".self_attn.attn")
    ) or not isinstance(inputs, tuple):
        return {}
    fields: dict[str, Any] = {}
    for index, field in enumerate(("query", "key", "value")):
        if index >= len(inputs):
            break
        fields[f"{field}_input_path"] = _save_tensor(
            trace_dir,
            call_index,
            f"{field}_input",
            inputs[index],
        )
        fields[f"{field}_input_shape"] = _shape(inputs[index])
    return fields


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
            **_attention_input_paths(trace_dir, call_index, name, inputs),
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


def _module_prefix(module: Any) -> str:
    base_layer = getattr(module, "base_layer", None)
    return str(getattr(base_layer, "prefix", ""))


def _is_qkv_or_o_proj(module: Any) -> bool:
    prefix = _module_prefix(module)
    return ".attn.qkv_proj" in prefix or ".attn.o_proj" in prefix


def _is_first_layer_attention_proj(module: Any) -> bool:
    prefix = _module_prefix(module)
    return prefix in {
        "model.layers.0.attn.qkv_proj",
        "model.layers.0.attn.o_proj",
    }


def _patch_lora_layers(module: Any | None = None) -> None:
    global _LORA_PATCHED, _LORA_PATCHING
    if _LORA_PATCHED or _LORA_PATCHING:
        return
    if not _lora_internal_trace_enabled():
        return
    _LORA_PATCHING = True
    try:
        import torch

        if module is None:
            import sys

            module = sys.modules.get("vllm.lora.layers.column_parallel_linear")
        if module is None:
            return
        ColumnParallelLinearWithLoRA = getattr(
            module,
            "ColumnParallelLinearWithLoRA",
            None,
        )
        MergedColumnParallelLinearWithLoRA = getattr(
            module,
            "MergedColumnParallelLinearWithLoRA",
            None,
        )
        if ColumnParallelLinearWithLoRA is None:
            return
        if MergedColumnParallelLinearWithLoRA is None:
            return
    except Exception as exc:
        _event("lora_internal_patch_failed", error=repr(exc))
        return
    finally:
        _LORA_PATCHING = False

    original_set_lora = MergedColumnParallelLinearWithLoRA.set_lora
    original_forward = ColumnParallelLinearWithLoRA.forward

    @functools.wraps(original_set_lora)
    def set_lora(self: Any, index: int, lora_a: Any, lora_b: Any):
        result = original_set_lora(self, index, lora_a, lora_b)
        trace_dir = _trace_dir()
        if trace_dir is not None and _is_qkv_or_o_proj(self):
            prefix = _module_prefix(self)
            key = f"{os.getpid()}:{prefix}:{index}"
            if key not in _LORA_DUMPED_SET_LORA:
                _LORA_DUMPED_SET_LORA.add(key)
                stem = f"{prefix}.slot{index}"
                _event(
                    "lora_internal_set_lora",
                    module_name=prefix,
                    module_type=self.__class__.__name__,
                    slot_index=index,
                    tp_rank=getattr(self, "tp_rank", None),
                    tp_size=getattr(self, "tp_size", None),
                    output_slices=list(getattr(self, "output_slices", ())),
                    output_ids=list(getattr(self, "output_ids", ())),
                    input_size=getattr(self, "input_size", None),
                    lora_config_fully_sharded=getattr(
                        getattr(self, "lora_config", None),
                        "fully_sharded_loras",
                        None,
                    ),
                    input_lora_a=_save_tensor_list(
                        trace_dir, stem=stem, field="input_lora_a", tensors=lora_a
                    ),
                    input_lora_b=_save_tensor_list(
                        trace_dir, stem=stem, field="input_lora_b", tensors=lora_b
                    ),
                    stacked_lora_a=_save_tensor_list(
                        trace_dir,
                        stem=stem,
                        field="stacked_lora_a",
                        tensors=getattr(self, "lora_a_stacked", ()),
                    ),
                    stacked_lora_b=_save_tensor_list(
                        trace_dir,
                        stem=stem,
                        field="stacked_lora_b",
                        tensors=getattr(self, "lora_b_stacked", ()),
                    ),
                )
        return result

    @functools.wraps(original_forward)
    def forward(self: Any, input_: Any):
        trace_dir = _trace_dir()
        base_path = None
        if trace_dir is not None and _is_first_layer_attention_proj(self):
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            try:
                base_output = self.base_layer.quant_method.apply(
                    self.base_layer,
                    input_,
                    bias,
                )
                call_index = _next_index()
                base_path = _save_internal_tensor(
                    trace_dir,
                    stem=f"{_module_prefix(self)}.{call_index:06d}",
                    field="base_output",
                    tensor=base_output,
                )
            except Exception as exc:
                _event(
                    "lora_internal_base_output_failed",
                    module_name=_module_prefix(self),
                    error=repr(exc),
                )
        output = original_forward(self, input_)
        if trace_dir is not None and _is_first_layer_attention_proj(self):
            primary_output = _primary_tensor(output)
            call_index = _next_index()
            _event(
                "lora_internal_forward",
                call_index=call_index,
                module_name=_module_prefix(self),
                module_type=self.__class__.__name__,
                tp_rank=getattr(self, "tp_rank", None),
                tp_size=getattr(self, "tp_size", None),
                primary_input_shape=_shape(input_),
                primary_output_shape=_shape(primary_output),
                primary_input_path=_save_internal_tensor(
                    trace_dir,
                    stem=f"{_module_prefix(self)}.{call_index:06d}",
                    field="input",
                    tensor=input_,
                ),
                primary_output_path=_save_internal_tensor(
                    trace_dir,
                    stem=f"{_module_prefix(self)}.{call_index:06d}",
                    field="output",
                    tensor=primary_output,
                ),
                base_output_path=base_path,
            )
        return output

    MergedColumnParallelLinearWithLoRA.set_lora = set_lora
    ColumnParallelLinearWithLoRA.forward = forward
    _LORA_PATCHED = True
    _event("lora_internal_patch_active")


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
            if _trace_dir() is not None:
                _register_model_hooks(self)
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
    if name == "vllm.lora.layers.column_parallel_linear":
        _patch_lora_layers(module)
    if name == "vllm.model_executor.models.qwen3":
        _patch_causal_lm_class(module, "Qwen3ForCausalLM")
    elif name == "vllm.model_executor.models.qwen3_5":
        _patch_causal_lm_class(module, "Qwen3_5ForCausalLM")
        _patch_causal_lm_class(module, "Qwen3_5ForConditionalGeneration")
        _patch_causal_lm_class(module, "Qwen3_5MoeForCausalLM")
        _patch_causal_lm_class(module, "Qwen3_5MoeForConditionalGeneration")
    elif name == "vllm.model_executor.models.qwen3_moe":
        _patch_causal_lm_class(module, "Qwen3MoeForCausalLM")
    elif name == "vllm.model_executor.models.gpt_oss":
        _patch_causal_lm_class(module, "GptOssForCausalLM")


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
