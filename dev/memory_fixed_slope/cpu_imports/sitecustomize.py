"""CPU-only ART imports without public service/backends; no planner replacement."""

import ast
import os
import pathlib
import sys
import types

if os.environ.get("ART_CPU_HARNESS") == "1":
    root = pathlib.Path(os.environ["ART_TREE"]) / "src/art"
    for n in [
        "art",
        "art.megatron",
        "art.utils",
        "art.preprocessing",
        "art.dev",
        "art.megatron.model_support",
        "art.megatron.model_support.handlers",
        "art.megatron.gdn",
    ]:
        m = types.ModuleType(n)
        m.__path__ = [str(root / n.removeprefix("art").strip(".").replace(".", "/"))]
        sys.modules[n] = m
    sys.modules["art.dev"].TrainConfig = object
    m = types.ModuleType("art.preprocessing.pack")
    m.PackedTensors = dict
    sys.modules[m.__name__] = m
    # Runtime CP planning imports these provider helpers, not the trainer loop.
    m = types.ModuleType("art.megatron.training.microbatches")
    from typing import Any

    import torch

    from art.megatron.context_parallel.types import (
        ContextParallelConfig,
        estimate_owned_token_ms,
    )

    env = m.__dict__
    env.update(
        Any=Any,
        torch=torch,
        ContextParallelConfig=ContextParallelConfig,
        estimate_owned_token_ms=estimate_owned_token_ms,
        flash_sparse_block_size_for_head_dim=lambda **kw: (128, 64),
    )
    tree = ast.parse((root / "megatron/training/microbatches.py").read_text())
    wanted = {
        "_owned_token_ms_for_provider",
        "_context_parallel_config_for_provider",
        "_gdn_planner_config_for_provider",
    }
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
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
            str(root / "megatron/training/microbatches.py"),
            "exec",
        ),
        env,
    )
    sys.modules[m.__name__] = m
