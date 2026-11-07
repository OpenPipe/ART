"""CPU planner harness adapted from the host-overhead lane, with local outputs.

Meta LoRA factors and import shims inspect planning metadata only. No model
execution is represented; costs and width search are production ART methods.
"""

from collections import Counter
import cProfile
import io
import json
import logging
import os
from pathlib import Path
import pstats
import statistics
import time
from types import SimpleNamespace

from cpu_lora import Ref, adapter
from megatron.core.transformer.transformer_block import TransformerBlock
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.model_support.handlers.qwen3_5 import Qwen35DenseHandler
from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRank, _impl
from art.trainer_rank._impl import _MemoryProfile

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
logging.getLogger("art.trainer_rank._impl").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[2]


class CpuRank(TrainerRank):
    def _topology_key(self):
        return (1, 1, 8, 1)

    def _topology(self):
        return ParallelTopology(cp=8)

    def _dp_rank_and_size(self):
        return 0, 1


def make_rank(resident_slots=1, lora_rank=8):
    cfg = json.loads(
        (ROOT / "tests/unit/fixtures/memory_fixed_slope/dense-config.json").read_text()
    )
    provider = SimpleNamespace(**cfg)
    provider.num_layers = 64
    provider.ffn_hidden_size = 17408
    provider.kv_channels = 256
    provider.num_query_groups = 4
    for k, v in dict(
        context_parallel_size=8,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=1,
        params_dtype=torch.bfloat16,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        sequence_parallel=False,
        fp32_residual_connection=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
        padded_vocab_size=248320,
    ).items():
        setattr(provider, k, v)
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = provider
    layers = []
    ref = Ref(kind="checkpoint", name="run0")
    for i in range(64):
        layer = torch.nn.Module()
        layer.base = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        layer.base.requires_grad_(False)
        if i % 4 == 3:
            shapes = [
                ("q", 5120, 12288),
                ("k", 5120, 1024),
                ("v", 5120, 1024),
                ("o", 6144, 5120),
            ]
        else:
            shapes = [
                ("q", 5120, 2048),
                ("k", 5120, 2048),
                ("v", 5120, 6144),
                ("z", 5120, 6144),
                ("o", 6144, 5120),
            ]
        shapes += [("gate", 5120, 17408), ("up", 5120, 17408), ("down", 17408, 5120)]
        layer.adapters = torch.nn.ModuleList(
            [
                adapter(
                    f"base_model.model.model.layers.{i}.{n}", a, b, ref, rank=lora_rank
                )
                for n, a, b in shapes
            ]
        )
        layers.append(layer)
    block.layers = torch.nn.ModuleList(layers)
    block.num_layers_per_pipeline_rank = 64
    model = torch.nn.Module()
    model.decoder = block
    model.config = provider
    model._preprocess = lambda **kw: None
    for number in range(1, resident_slots):
        extra_ref = Ref(kind="checkpoint", name=f"run{number}")
        for layer in block.layers:
            for owner in layer.adapters:
                key = f"extra{number}"
                slot = torch.nn.Module()
                slot.A_T = torch.nn.Parameter(torch.empty_like(owner.A_T))
                slot.B_T = torch.nn.Parameter(torch.empty_like(owner.B_T))
                slot.scale = 1.0
                owner._slot_keys[extra_ref] = key
                owner._slot_modules[key] = slot
    rank = CpuRank(
        SimpleNamespace(
            model=[model],
            optimizer=None,
            provider=provider,
            model_support_handler=Qwen35DenseHandler(),
        )
    )
    rank._default_slot_ref = ref
    for number in range(resident_slots):
        slot_ref = Ref(kind="checkpoint", name=f"run{number}")
        rank._checkpoint_slots[slot_ref.name] = _impl._CheckpointSlot(
            params=tuple(
                p
                for layer in block.layers
                for owner in layer.adapters
                for p in owner.lora_slot_params(slot_ref)
            ),
            config={},
        )
    rank._gdn_layers = 48
    rank._mixer_top_gaps = (0, 1)
    rank._lora_modules_per_layer = 8
    rank._dense_fc1_adapted = True
    rank.rounds = Counter()
    rank.budget = 24 * 2**30
    rank._available_memory_bytes = lambda: rank.budget

    def agree(v):
        rank.rounds["all_ranks_true"] += 1
        return v

    rank._all_ranks_true = agree
    original = rank._memory_check_required

    def memory(required, **kw):
        rank.rounds["memory_checks"] += 1
        rank.rounds["memory_collective_rounds"] += 2
        return original(required, **kw)

    rank._memory_check_required = memory
    return rank


def requests(count=568, prompt=240, completion=213, group=16, variant=0):
    out = []
    generator = torch.Generator().manual_seed(932 + variant)
    for i in range(count):
        family = i // group
        head = torch.full((prompt,), 1000 + family, dtype=torch.long)
        tail = torch.randint(10000, 200000, (completion,), generator=generator)
        tokens = torch.cat((head, tail))
        out.append(
            ForwardInput(
                input_tokens=tokens, target_tokens=tokens.clone(), no_grad=False
            )
        )
    return out
