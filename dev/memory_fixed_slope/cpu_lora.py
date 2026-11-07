"""Real LoRA metadata methods, meta factors; no CUDA/TE model construction."""

import ast
import contextvars
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import types
import typing

import torch

root = Path(os.environ["ART_TREE"]) / "src/art"
m = types.ModuleType("art.megatron.lora")
m.__file__ = str(root / "megatron/lora.py")
e = m.__dict__
e.update(
    torch=torch,
    Any=typing.Any,
    cast=typing.cast,
    dataclass=dataclass,
    LoRASlot=torch.nn.Module,
    LoRAParallelSpec=lambda: None,
    LORA_ALPHA=32,
    _CURRENT_LORA_SLOT=contextvars.ContextVar("slot", default=None),
    _CURRENT_LORA_SEGMENTS=contextvars.ContextVar("seg", default=None),
    _CURRENT_EXPERT_ADAPTER_PLAN=contextvars.ContextVar("expert", default=None),
)
tree = ast.parse((root / "megatron/lora.py").read_text())
nodes = [
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name in {"LoRA", "LoRASlotRef"}
]
sys.modules[m.__name__] = m
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    *nodes,
                ],
                type_ignores=[],
            )
        ),
        str(root / "megatron/lora.py"),
        "exec",
    ),
    e,
)
m.SharedExpertsLinearFC1LoRA = type(
    "SharedExpertsLinearFC1LoRA", (torch.nn.Module,), {}
)
LoRA = e["LoRA"]
Ref = e["LoRASlotRef"]


def adapter(prefix, a, b, ref, rank=32):
    x = LoRA.__new__(LoRA)
    torch.nn.Module.__init__(x)
    x.adapter_model_prefix = prefix
    x.shared_factor = None
    x.lora_layout = types.SimpleNamespace(shared_outer_kernel=True)
    x.in_features = a
    x.out_features = b
    x.scale = 1.0
    x._num_local_experts = 1
    x.A_T = torch.nn.Parameter(
        torch.empty((a, rank), device="meta", dtype=torch.bfloat16), requires_grad=False
    )
    x.B_T = torch.nn.Parameter(
        torch.empty((rank, b), device="meta", dtype=torch.bfloat16), requires_grad=False
    )
    slot = torch.nn.Module()
    slot.A_T = torch.nn.Parameter(torch.empty_like(x.A_T))
    slot.B_T = torch.nn.Parameter(torch.empty_like(x.B_T))
    slot.scale = 1.0
    x._slot_keys = {ref: "selected"}
    x._slot_modules = torch.nn.ModuleDict({"selected": slot})
    # Stark precision.execution installs a per-call closure in the real model.
    x.forward = lambda *args, **kw: None
    return x
