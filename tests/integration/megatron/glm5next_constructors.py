"""Inspect installed constructors while building the GPU worker's model on meta.

No linear, spec, model, mHC, router, expert, or LoRA constructor is substituted.
CUDA availability and materialization are the only mocked hardware boundaries;
the real Gloo process groups exercise rank discovery. Stop before CUDA DDP.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
import importlib.metadata
import inspect
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator
from unittest.mock import patch

from megatron.bridge.models import model_provider
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear
import torch
import torch.distributed as dist

from art.megatron.lora import LoRA, LoraLayout
from art.megatron.runtime import bridge_runtime
from art.megatron.train import _is_bridge_hf_load_hook, build_training_runtime
from tests.integration.megatron.glm5next_qualification import (
    configure_qualification_provider,
)


@contextmanager
def audit_constructors() -> Iterator[dict[str, Any]]:
    """Bind actual call arguments to real installed Python constructor signatures.

    This covers direct calls, ModuleSpec/build_module, inherited constructors and
    factories such as TENorm.__new__. Keep every call site and its supplied keys.
    A missing required argument fails before the underlying constructor runs.
    """
    classes = {
        value
        for module in tuple(sys.modules.values())
        if module is not None
        and getattr(module, "__name__", "").startswith(
            ("megatron.", "transformer_engine.", "art.megatron.")
        )
        for value in tuple(vars(module).values())
        if inspect.isclass(value)
        and not issubclass(value, type)
        and isinstance(value.__module__, str)
        and value.__module__.startswith(
            ("megatron.core.", "transformer_engine.pytorch.")
        )
        and any(
            inspect.isfunction(raw.__func__ if isinstance(raw, staticmethod) else raw)
            for raw in (vars(value).get("__init__"), vars(value).get("__new__"))
        )
    }
    report: dict[str, Any] = {"class_signatures": {}, "signatures": {}, "calls": []}
    # Inspect the unmodified installed classes before wrapping any base class.
    class_signatures = {cls: str(inspect.signature(cls)) for cls in classes}
    with ExitStack() as stack:
        for cls in sorted(classes, key=lambda c: (c.__module__, c.__qualname__)):
            for method in ("__init__", "__new__"):
                raw = vars(cls).get(method)
                original = raw.__func__ if isinstance(raw, staticmethod) else raw
                if not inspect.isfunction(original):
                    continue
                signature = inspect.signature(original)
                name = f"{cls.__module__}.{cls.__qualname__}.{method}"
                report["class_signatures"][name] = class_signatures[cls]
                report["signatures"][name] = str(signature)

                def wrap(function: Any, sig: inspect.Signature, label: str) -> Any:
                    @wraps(function)
                    def checked(*args: Any, **kwargs: Any) -> Any:
                        bound = sig.bind(*args, **kwargs)
                        frame = inspect.currentframe()
                        assert frame is not None and frame.f_back is not None
                        caller = frame.f_back
                        report["calls"].append(
                            {
                                "constructor": label,
                                "file": caller.f_code.co_filename,
                                "line": caller.f_lineno,
                                "supplied": sorted(bound.arguments),
                                "is_expert": bound.arguments.get("is_expert"),
                            }
                        )
                        return function(*args, **kwargs)

                    return checked

                replacement = wrap(original, signature, name)
                stack.enter_context(
                    patch.object(
                        cls,
                        method,
                        staticmethod(replacement)
                        if method == "__new__"
                        else replacement,
                    )
                )
        yield report
    used = {call["constructor"] for call in report["calls"]}
    report["signatures"] = {
        key: value for key, value in report["signatures"].items() if key in used
    }
    report["class_signatures"] = {
        key: value for key, value in report["class_signatures"].items() if key in used
    }


class _BeforeCudaDDP(Exception):
    pass


def check_native_construction(config: Path) -> dict[str, Any]:
    """Run build_training_runtime through full TE construction and rank32 wrapping."""
    assert not torch.cuda.is_initialized()
    assert not dist.is_initialized(), (
        "run the CPU construction audit before GPU rank setup"
    )
    assert importlib.metadata.version("megatron-core") == "0.17.0"
    assert importlib.metadata.version("transformer-engine") in {"2.11.0", "2.14.1"}
    result: dict[str, Any] = {}

    def config_only(bundle: Any) -> None:
        # This is the pretrained/meta GPU path. Omit only checkpoint I/O; keep
        # native freeze/LoRA hooks registered by build_training_runtime itself.
        assert all(
            _is_bridge_hf_load_hook(hook) for hook in bundle.provider._pre_wrap_hooks
        )
        bundle.provider._pre_wrap_hooks = []

    def before_ddp(chunks: Any, *args: Any, **kwargs: Any) -> None:
        assert all(p.is_meta for chunk in chunks for p in chunk.parameters())
        modules = [module for chunk in chunks for module in chunk.modules()]
        classes = Counter(type(module).__name__ for module in modules)
        provider = chunks[0].config
        assert classes["Glm5NextTransformerLayer"] == provider.num_layers
        assert classes["KimiDeltaAttention"] == sum(provider.linear_attention_freq)
        assert classes["Glm5NextSelfAttention"] == provider.num_layers - sum(
            provider.linear_attention_freq
        )
        assert classes["TEGroupedMLP"] == sum(provider.moe_layer_freq)
        base = chunks[0]
        while hasattr(base, "module"):
            base = base.module
        assert isinstance(base.output_layer, ColumnParallelLinear)
        adapters = [module for module in modules if isinstance(module, LoRA)]
        assert adapters and all(
            module.A_T.shape[-1] == module.B_T.shape[-2] == 32 for module in adapters
        )
        result.update(
            layers=provider.num_layers,
            classes=dict(classes),
            parameter_elements=sum(
                p.numel() for chunk in chunks for p in chunk.parameters()
            ),
            trainable_elements=sum(
                p.numel()
                for chunk in chunks
                for p in chunk.parameters()
                if p.requires_grad
            ),
            lora_modules=len(adapters),
            parameter_device="meta",
        )
        raise _BeforeCudaDDP

    with tempfile.TemporaryDirectory(prefix="glm-native-construction-") as directory:
        dist.init_process_group(
            "gloo", init_method=f"file://{directory}/rendezvous", rank=0, world_size=1
        )
        try:
            parallel_state.initialize_model_parallel()
            with (
                audit_constructors() as audit,
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(
                    torch.cuda, "current_device", return_value=torch.device("meta")
                ),
                patch.object(
                    bridge_runtime,
                    "_materialization_device",
                    return_value=torch.device("meta"),
                ),
                patch.object(model_provider, "_ddp_wrap", side_effect=before_ddp),
            ):
                try:
                    build_training_runtime(
                        model_identifier=str(config.parent),
                        model_support_key="glm5next",
                        allow_unvalidated_arch=True,
                        build_optimizer=False,
                        lora_layout=LoraLayout(expert="shared_outer"),
                        provider_configure=configure_qualification_provider,
                        provider_bundle_configure=config_only,
                    )
                except _BeforeCudaDDP:
                    pass
                else:
                    raise AssertionError(
                        "native construction did not reach the CUDA DDP boundary"
                    )
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
    assert result and not torch.cuda.is_initialized()
    if result["layers"] == 45:
        assert result["trainable_elements"] == 77_923_584 * 32
    return {
        **result,
        "status": "pass",
        "path": "build_training_runtime/provide_distributed_model/GPTModel/native TE/meta materialization/freeze/rank32 LoRA/before CUDA DDP",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformer-engine", "megatron-core", "transformers")
        },
        "cuda_initialized": False,
        "weights_loaded": False,
        "cpu_topology": "TP1/PP1/CP1/EP1, real singleton Gloo groups",
        "cuda_mocks": [
            "torch.cuda.is_available=True for TE's unconditional constructor guard",
            "torch.cuda.current_device=meta for MCore's explicit CUDA allocations",
            "ART _materialization_device=meta instead of CUDA",
            "stop at Bridge _ddp_wrap before CUDA DDP allocation",
        ],
        "constructors": audit,
    }


def check_te_sm100() -> dict[str, Any]:
    """Read the real image's TE ELF targets on CPU, without loading a GPU."""
    command = shutil.which("cuobjdump")
    assert command, "CUDA13 stage0 requires cuobjdump to check installed TE targets"
    libraries = []
    for package in ("transformer-engine-cu13", "transformer-engine-torch"):
        distribution = importlib.metadata.distribution(package)
        for entry in distribution.files or ():
            if str(entry).endswith(".so"):
                path = Path(str(distribution.locate_file(entry))).resolve()
                process = subprocess.run(
                    [command, "--list-elf", str(path)],
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                output = process.stdout + process.stderr
                # TE's Python binding is host-only; kernels live in its common
                # CUDA library. cuobjdump returns 255 for that host-only ELF.
                if process.returncode and "does not contain device code" not in output:
                    process.check_returncode()
                libraries.append(
                    {
                        "path": str(path),
                        "targets": sorted(set(re.findall(r"sm_\d+[a-z]?", output))),
                    }
                )
    assert any(
        any(target.startswith("sm_100") for target in library["targets"])
        for library in libraries
    ), libraries
    assert not torch.cuda.is_initialized()
    return {"status": "pass", "required": "sm_100", "libraries": libraries}
