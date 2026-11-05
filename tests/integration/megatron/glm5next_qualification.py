"""M4 qualification workers. CPU rehearsal substitutes the engine, not the steps."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import torch

MODEL = "zai-org/GLM-5.3-Flash-BF16"
REVISION = "f9fed3bbc82291073373d67452e9f25317ef7577"
CONTEXTS = (32768, 65536, 131072, 262144)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire(record: dict, destination: Path) -> dict:
    path = destination / record["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{record['path']}"
        with (
            urllib.request.urlopen(url, timeout=180) as src,
            temporary.open("wb") as dst,
        ):
            while chunk := src.read(8 << 20):
                dst.write(chunk)
        temporary.rename(path)
    if path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
        raise ValueError(f"weight integrity failure: {path.name}")
    return {"path": path.name, "size": path.stat().st_size, "sha256": record["sha256"]}


def stage0(
    output: Path,
    expected_cuda: str | None = None,
    *,
    require_toolkit: bool = False,
    inputs: Path | None = None,
) -> None:
    from art.megatron.model_support.handlers.glm5next import GLM5NEXT_HANDLER
    from art.megatron.train import build_training_runtime
    from art.trainer_rank import TrainerRank

    expected_cuda = expected_cuda or os.environ.get("M4_EXPECTED_CUDA", "12.8")
    require_toolkit = require_toolkit or os.environ.get("M4_REQUIRE_TOOLKIT") == "1"
    toolkit = None
    if require_toolkit:
        import shutil
        import subprocess

        cuda_home = Path(os.environ.get("CUDA_HOME", "/missing"))
        assert (cuda_home / "bin/nvcc").is_file() and shutil.which("nvcc")
        toolkit = subprocess.check_output(
            [str(cuda_home / "bin/nvcc"), "--version"], text=True, timeout=30
        )
    assert torch.version.cuda == expected_cuda
    assert GLM5NEXT_HANDLER.single_root_packing
    assert callable(build_training_runtime)
    for name in ("forward_micro_batches", "backward", "optim_step", "export_lora"):
        assert callable(getattr(TrainerRank, name))
    assert not torch.cuda.is_initialized()
    real_configs = None
    te_architecture = None
    if inputs is not None:
        from tests.integration.megatron.glm5next_topology import validate_stage_plans

        validate_stage_plans(inputs, output.parent / "topology-plans.json")
        from tests.integration.megatron.glm5next_constructors import (
            check_native_construction,
            check_te_sm100,
        )
        from tests.integration.megatron.glm5next_real_config import check_real_configs

        real_configs = check_real_configs(inputs / "real-configs")
        for name in ("full", "F4", "F8"):
            real_configs[name]["native_construction"] = check_native_construction(
                inputs / "real-configs" / name / "config.json"
            )
        if expected_cuda == "13.0" and os.environ.get("M4_FAKE_TOOLKIT") != "1":
            te_architecture = check_te_sm100()
    write(
        output,
        {
            "status": "pass",
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "cuda_initialized": False,
            "real_configs": real_configs,
            "te_architecture": te_architecture,
            "toolkit": toolkit,
            "toolkit_simulated": os.environ.get("M4_FAKE_TOOLKIT") == "1",
            "art_source": str(Path(__import__("art").__file__).resolve()),
        },
    )
    print("M4_STAGE0_PASS", flush=True)


def hybrid_ep_stage(output: Path, *, tiny: bool) -> None:
    import subprocess

    from art.megatron import hybrid_ep_setup as setup

    arch = os.environ["TORCH_CUDA_ARCH_LIST"]
    assert setup._arch_list() == arch
    assert os.environ["HYBRID_EP_MULTINODE"] == "0" and os.environ["USE_NIXL"] == "0"
    assert callable(setup.setup_hybrid_ep) and callable(setup.validate_hybrid_ep)
    validation = "import json,sys,os,subprocess,torch,deep_ep,hybrid_ep_cpp; from importlib.metadata import version; from art.megatron.hybrid_ep_setup import validate_hybrid_ep; validate_hybrid_ep(); assert hybrid_ep_cpp.SM_ARCH == '10.0'; assert torch.version.cuda == '13.0'; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'art-deep-ep':version('art-deep-ep'),'SM_ARCH':hybrid_ep_cpp.SM_ARCH,'nvcc':subprocess.check_output([os.environ['CUDA_HOME']+'/bin/nvcc','--version'],text=True)}))"
    commands = [
        [sys.executable, "-m", "art.megatron.hybrid_ep_setup"],
        [sys.executable, "-c", validation],
    ]
    records = []
    started = time.monotonic()
    for argv in commands:
        if tiny:
            # Build/validation are external CUDA-toolchain operations. Exercise the
            # exact command schema; compilation remains an explicit GPU-window output.
            if "-c" in argv:
                compile(argv[-1], "hybridep-validation", "exec")
            actual = [
                sys.executable,
                "-c",
                "print('CPU HybridEP command-plan validation')",
            ]
        else:
            actual = argv
        subprocess.run(
            actual, check=True, timeout=max(1, 1800 - (time.monotonic() - started))
        )
        records.append({"argv": argv, "executed": actual, "simulation": tiny})
    write(
        output / "hybridep.json",
        {
            "status": "pass",
            "simulation": tiny,
            "source_sha256": setup._source_hash(),
            "arch": arch,
            "seconds": time.monotonic() - started,
            "commands": records,
        },
    )


def volumes(inputs: Path, output: Path) -> None:
    from art.megatron.glm5next.lora import LoraDimensions, lora_targets

    shapes = json.loads((inputs / "text-shapes.json").read_text())
    index = json.loads((inputs / "index.json").read_text())
    checkpoint = torch.empty(
        index["metadata"]["total_size"], dtype=torch.uint8, device="meta"
    )
    assert len(shapes) == 37534 and checkpoint.numel() == 642646653816
    dtype_map = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
        "U8": torch.uint8,
    }
    tensors = {
        name: torch.empty(shape, dtype=dtype_map[dtype], device="meta")
        for name, (dtype, shape) in shapes.items()
    }
    config = json.loads((inputs / "config.json").read_text())
    dimensions = LoraDimensions.from_config(config)
    adapter = [
        torch.empty(shape, dtype=torch.bfloat16, device="meta")
        for target in lora_targets(dimensions)
        for shape in target.shapes(32)
    ]
    assert sum(t.numel() for t in adapter) == 77_923_584 * 32
    contexts = []
    for length in CONTEXTS:
        # Full activation/logit shapes, including every mHC layer boundary.
        residual = torch.empty(
            (45, length, 4, 4096), dtype=torch.bfloat16, device="meta"
        )
        logits = torch.empty((length, 154880), dtype=torch.float32, device="meta")
        contexts.append(
            {
                "tokens": length,
                "mhc_bytes": residual.numel() * 2,
                "dense_logits_bytes": logits.numel() * 4,
                "selected_head": True,
                "cp": 8,
                "ep": 8,
            }
        )
    write(
        output,
        {
            "status": "pass",
            "base_tensors": len(tensors),
            "checkpoint_tensor_count": len(index["weight_map"]),
            "checkpoint_payload_bytes": checkpoint.numel(),
            "excluded_from_training": "MTP and vision; full download volume retained",
            "base_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "rank32_elements": sum(t.numel() for t in adapter),
            "contexts": contexts,
            "parity_diagnostics": diagnostic_volumes(inputs),
            "cuda_initialized": torch.cuda.is_initialized(),
            "execution": "meta",
        },
    )


def diagnostic_volumes(inputs: Path) -> list[dict]:
    """Represent every document/layer/position at real F4/F8 geometry on meta."""
    rows: list[dict] = []
    for layers in (4, 8):
        for document, ids in documents(inputs):
            positions = len(ids) - 1
            tensors = []
            for layer in range(layers):
                # attention, MLP, 2 collapsed mHC, 3 stream outputs,
                # 2 FP32 post/comb maps, and selected IDs/weights.
                tensors += [
                    torch.empty(
                        (1, positions, 4096), dtype=torch.bfloat16, device="meta"
                    )
                    for _ in range(4)
                ]
                tensors += [
                    torch.empty(
                        (1, positions, 4, 4096), dtype=torch.bfloat16, device="meta"
                    )
                    for _ in range(3)
                ]
                tensors += [
                    torch.empty(
                        (1, positions, width), dtype=torch.float32, device="meta"
                    )
                    for width in (4, 16, 4, 16)
                ]
                if layer >= 3:
                    tensors += [
                        torch.empty((positions, 8), dtype=dtype, device="meta")
                        for dtype in (torch.int32, torch.float32)
                    ]
                if layer % 4 == 3:
                    tensors.append(
                        torch.empty(
                            (1, positions, 2112), dtype=torch.int32, device="meta"
                        )
                    )
            rows.append(
                {
                    "fixture": f"F{layers}",
                    "document": document,
                    "positions": positions,
                    "layers": layers,
                    "capture_bytes": sum(t.numel() * t.element_size() for t in tensors),
                    "execution": "meta",
                }
            )
    return rows


def identity(model: Path, output: Path) -> None:
    from art.megatron.glm5next.lora import LoraDimensions, lora_targets
    from art.megatron.glm5next.lora_export import adapter_config, to_vllm_lora_tensors
    from art.megatron.model_support.lora_disk import save_vllm_lora_tensors

    dimensions = LoraDimensions.from_config(
        json.loads((model / "config.json").read_text())
    )
    torch.manual_seed(20261105)
    tensors = {}
    for target in lora_targets(dimensions):
        a, b = target.shapes(32)
        a_tensor = torch.randn(a, dtype=torch.bfloat16) * 0.01
        b_tensor = torch.zeros(b, dtype=torch.bfloat16)
        for expert in range(target.experts or 1):
            name = target.name.format(expert=expert)
            tensors[name + ".lora_A.weight"] = (
                a_tensor[0 if target.shared == "A" else expert]
                if target.experts
                else a_tensor
            )
            tensors[name + ".lora_B.weight"] = (
                b_tensor[0 if target.shared == "B" else expert]
                if target.experts
                else b_tensor
            )
    tensors, config = to_vllm_lora_tensors(
        tensors, adapter_config=adapter_config(32, dimensions=dimensions)
    )
    save_vllm_lora_tensors(output, tensors, config)


def configure_qualification_provider(provider: Any) -> None:
    provider._art_lora_rank = 32
    provider._art_lora_target_modules = list(
        provider._art_model_support_spec.default_target_modules
    )
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1


def capture_layer_outputs(chunks: Any, values: dict) -> list[Any]:
    """Capture outside the layer forward, through DDP/Float16 wrappers."""
    handles = []
    for wrapped in chunks:
        chunk = wrapped
        while hasattr(chunk, "module"):
            chunk = chunk.module
        for layer in chunk.decoder.layers:

            def capture(
                module: Any,
                args: Any,
                result: Any,
                key: str = f"layers.{layer.layer_number - 1}.streams",
            ) -> None:
                value = result[0] if isinstance(result, tuple) else result
                # All real positions, with HF's B,S,... orientation. Keep BF16
                # captures compact; the comparator promotes its arithmetic.
                if value.ndim >= 3:
                    value = value.transpose(0, 1)
                values[key] = value.detach().to(device="cpu", copy=True).contiguous()

            handles.append(layer.register_forward_hook(capture))
            if hasattr(layer, "hc_util"):
                from tests.integration.megatron.glm5next_parity_diagnostics import (
                    capture_sublayers,
                )

                handles.extend(capture_sublayers(layer, values))
    return handles


class TinyEngine:
    """Native ART LoRA storage on a tiny CPU projection; real token volumes."""

    def __init__(self, model: Path, output: Path):
        from types import SimpleNamespace

        from art.megatron.glm5next.lora_layers import _attach_projection

        provider = SimpleNamespace(
            _art_model_support_spec=SimpleNamespace(default_target_modules=["lm_head"])
        )
        configure_qualification_provider(provider)

        self.rank, self.device = 0, torch.device("cpu")
        torch.manual_seed(42)
        self.embedding = torch.nn.Embedding(257, 16)
        self.head: Any = torch.nn.Linear(16, 257, bias=False)
        self.head.requires_grad_(False)
        _attach_projection(
            self.head, "base_model.model.lm_head", provider._art_lora_rank, 32
        )
        self.params = [
            p for n, p in self.head.named_parameters() if n.endswith(("A_T", "B_T"))
        ]
        self.optimizer = torch.optim.AdamW(self.params, lr=1e-3, weight_decay=0)
        self.output = output
        self.layers: dict = {}
        from types import SimpleNamespace

        self.head.layer_number = 1
        wrapped = SimpleNamespace(
            module=SimpleNamespace(
                module=SimpleNamespace(decoder=SimpleNamespace(layers=[self.head]))
            )
        )
        self.handles = (
            capture_layer_outputs([wrapped], self.layers)
            if output.name in ("F4", "F8")
            else []
        )

    def forward(
        self, tokens: list[int], backward: bool = False, logits: bool = False
    ) -> dict:
        ids = torch.tensor(tokens, dtype=torch.long).remainder(257)
        if backward:
            self.optimizer.zero_grad()
        self.layers.clear()
        all_lp, sampled_logits, all_top, streams = [], [], [], []
        # Bound actual CPU memory while traversing all real-context tokens.
        for begin in range(0, len(tokens) - 1, 512):
            x = self.embedding(ids[begin : begin + 512]).detach()
            count = min(len(x), len(ids) - begin - 1)
            out = self.head(x[:count]).float()
            if self.layers:
                streams.append(self.layers["layers.0.streams"])
            lp = (
                out.log_softmax(-1)
                .gather(1, ids[begin + 1 : begin + 1 + count, None])
                .squeeze(1)
            )
            if backward:
                (-lp.sum() / (len(tokens) - 1)).backward()
            all_lp.append(lp.detach())
            all_top.append(out.detach().topk(2, dim=-1))
            if logits:
                sampled_logits.append(out.detach())
        if streams:
            self.layers["layers.0.streams"] = torch.cat(streams)
        return {
            "logprobs": torch.cat(all_lp),
            "logits": torch.cat(sampled_logits) if logits else None,
            "top1": torch.cat([top.indices[:, 0] for top in all_top]),
            "top2.values": torch.cat([top.values for top in all_top]),
        }

    def step(self) -> dict:
        assert all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in self.params
        )
        norm = float(
            torch.stack(
                [
                    p.grad.float().square().sum()
                    for p in self.params
                    if p.grad is not None
                ]
            )
            .sum()
            .sqrt()
        )
        self.optimizer.step()
        return {"finite": True, "grad_norm": norm}

    def export(self, path: Path) -> None:
        from safetensors.torch import load_file

        from art.megatron.glm5next.lora import LoraDimensions, LoraFlags
        from art.megatron.glm5next.lora_export import (
            adapter_config,
            from_vllm_lora_tensors,
            to_vllm_lora_tensors,
            validate_native_lora,
        )
        from art.megatron.model_support.lora_disk import save_vllm_lora_tensors

        dimensions = LoraDimensions(
            hidden=16,
            kda_width=16,
            q_rank=8,
            kv_rank=4,
            mla_width=16,
            dense_width=32,
            expert_width=8,
            experts=4,
            vocab=257,
            dense_layers=1,
            layers=1,
            mla_layers=(),
        )
        config = adapter_config(
            32, dimensions=dimensions, flags=LoraFlags(False, False, True)
        )
        native = {
            "base_model.model.lm_head.lora_A.weight": self.head.lora.A_T.detach().T.contiguous(),
            "base_model.model.lm_head.lora_B.weight": self.head.lora.B_T.detach().T.contiguous(),
        }
        tensors, serving_config = to_vllm_lora_tensors(native, adapter_config=config)
        save_vllm_lora_tensors(path, tensors, serving_config)
        restored = from_vllm_lora_tensors(
            load_file(str(path / "adapter_model.safetensors")),
            adapter_config=serving_config,
        )
        validate_native_lora(restored, adapter_config=config)
        for name in native:
            assert torch.equal(restored[name], native[name])

    def memory(self) -> dict:
        return {"allocated": 0, "reserved": 0, "execution": "tiny_cpu"}

    def sync(self) -> None:
        return None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class ArtEngine:
    def __init__(self, model: Path, output: Path):
        import torch.distributed as dist

        self.rank = int(os.environ["RANK"])
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(self.device)
        from art.megatron.lora import LoraLayout
        from art.megatron.train import build_training_runtime
        from art.trainer_rank import TrainerRank

        dist.init_process_group("nccl", timeout=timedelta(seconds=300))  # ty: ignore[possibly-missing-attribute]
        self.runtime = build_training_runtime(
            model_identifier=str(model),
            model_support_key="glm5next",
            allow_unvalidated_arch=True,
            build_optimizer=False,
            lora_layout=LoraLayout(expert="shared_outer"),
            provider_configure=configure_qualification_provider,
        )
        self.trainer = TrainerRank(self.runtime)
        self.trainer.load_checkpoint(str(output.parents[1] / "initial"))
        self.layers: dict = {}
        # Do not add D2H capture synchronization to performance/context measurements.
        self.handles = (
            capture_layer_outputs(self.runtime.model, self.layers)
            if output.name in ("F4", "F8")
            else []
        )
        self.output = output
        write(
            output / f"topology-rank{self.rank}.json",
            {
                "cp": self.runtime.provider.context_parallel_size,
                "ep": self.runtime.provider.expert_model_parallel_size,
                "recompute_granularity": self.runtime.provider.recompute_granularity,
                "recompute_method": self.runtime.provider.recompute_method,
                "recompute_num_layers": self.runtime.provider.recompute_num_layers,
            },
        )

    def forward(
        self, tokens: list[int], backward: bool = False, logits: bool = False
    ) -> dict:
        from art.trainer_rank import ForwardInput

        self.trainer.zero_grad()
        self.layers.clear()
        request = ForwardInput(
            input_tokens=torch.tensor(tokens[:-1]),
            target_tokens=torch.tensor(tokens[1:]),
            logits=logits,
            top_k=2,
        )
        from megatron.core import parallel_state as ps

        requests = [request] * ps.get_data_parallel_world_size()
        results = []
        for mb in self.trainer.forward_micro_batches(requests, no_grad=not backward):
            out = mb.outputs[0]
            assert out.target_logprobs is not None
            if backward:
                self.trainer.backward([-out.target_logprobs.mean()])
            results.append(
                {
                    "logprobs": out.target_logprobs.detach().float().cpu().reshape(-1),
                    "top1": out.top_k.tokens.detach()
                    .cpu()
                    .reshape(-1, 2)[:, 0]
                    .contiguous()
                    if out.top_k is not None
                    else None,
                    # A shared logsumexp cancels: top-2 logprob differences
                    # are logit margins, including on long documents.
                    "top2.values": out.top_k.logprobs.detach()
                    .float()
                    .cpu()
                    .reshape(-1, 2)
                    if out.top_k is not None
                    else None,
                    "logits": out.logits.detach().float().cpu()
                    if out.logits is not None
                    else None,
                }
            )
        assert len(results) == 1
        return results[0]

    def step(self) -> dict:
        from art.trainer_rank import AdamParams

        metrics = self.trainer.optim_step(
            params=AdamParams(learning_rate=1e-4, weight_decay=0, grad_clip_norm=0.1)
        )
        assert all(math.isfinite(v) for v in metrics.values())
        return {"finite": True, **metrics}

    def export(self, path: Path) -> None:
        self.trainer.export_lora(str(path))
        self.sync()
        if self.rank == 0:
            from safetensors.torch import load_file

            from art.megatron.glm5next.lora_export import (
                from_vllm_lora_tensors,
                validate_native_lora,
            )

            config = json.loads((path / "adapter_config.json").read_text())
            tensors = load_file(str(path / "adapter_model.safetensors"))
            native = from_vllm_lora_tensors(tensors, adapter_config=config)
            from art.megatron.glm5next.lora_export import META

            native_config = {**config, META: {**config[META], "format": "art"}}
            validate_native_lora(native, adapter_config=native_config)

    def memory(self) -> dict:
        return {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }

    def sync(self) -> None:
        torch.cuda.synchronize()

    def close(self) -> None:
        import torch.distributed as dist

        for handle in self.handles:
            handle.remove()
        dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def documents(inputs: Path) -> list[tuple[str, list[int]]]:
    result = []
    for pack in json.loads((inputs / "prompts.json").read_text())["packs"]:
        offsets = pack["cu_seqlens"]
        for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
            result.append((f"{pack['id']}-doc{i}", pack["input_ids"][start:end]))
    return result


def experiment(engine: Any, task: str, inputs: Path, output: Path) -> None:
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    docs = documents(inputs)
    if task in ("forward", "learn"):
        first_forward_repeat(engine, docs[0][1], output)
    if task in ("forward", "parity"):
        from tests.integration.megatron.glm5next_parity_diagnostics import trim_captures

        for name, ids in docs:
            # Dense logits only on short docs; every doc retains all component
            # positions and top-2 summaries, including the long prompts.
            result = engine.forward(ids, logits=task == "parity" and len(ids) <= 128)
            values = {
                "target_logprobs": result["logprobs"],
                **trim_captures(engine.layers, len(ids) - 1),
            }
            if task == "parity" and isinstance(engine, ArtEngine):
                from tests.integration.megatron.glm5next_parity_diagnostics import (
                    validate_captures,
                )

                validate_captures(
                    values, engine.runtime.provider.num_layers, len(ids) - 1
                )
            if result.get("top1") is not None:
                values["top1"] = result["top1"]
            if result.get("top2.values") is not None:
                values["top2.values"] = result["top2.values"].contiguous()
            if result["logits"] is not None:
                values["logits"] = result["logits"].reshape(len(ids) - 1, -1)
            nonfinite = [
                key
                for key, value in values.items()
                if value.is_floating_point() and not bool(torch.isfinite(value).all())
            ]
            save_file(values, str(output / f"{name}-rank{engine.rank}.safetensors"))
            if nonfinite:
                write(
                    output / f"failure-rank{engine.rank}.json",
                    {"status": "hard_bug", "document": name, "nonfinite": nonfinite},
                )
                raise ValueError(f"Non-finite output: {nonfinite}")
        write(
            output / f"result-rank{engine.rank}.json",
            {
                "status": "pass",
                "task": task,
                "documents": len(docs),
                "memory": engine.memory(),
            },
        )
    elif task == "learn":
        ids = docs[0][1]
        losses, steps = [], []
        for _ in range(3):
            engine.sync()
            started = time.monotonic()
            result = engine.forward(ids, backward=True)
            losses.append(float(-result["logprobs"].mean()))
            metrics = engine.step()
            engine.sync()
            seconds = time.monotonic() - started
            steps.append(
                {
                    "seconds": seconds,
                    "tokens_s_gpu": (len(ids) - 1)
                    / seconds
                    / int(os.environ.get("WORLD_SIZE", "1")),
                    "optimizer": metrics,
                    "memory": engine.memory(),
                }
            )
        engine.export(output / "adapter")
        report = {
            "status": "pass"
            if losses[-1] < losses[0] and all(math.isfinite(x) for x in losses)
            else "fail",
            "losses": losses,
            "steps": steps,
        }
        write(output / f"result-rank{engine.rank}.json", report)
        assert report["status"] == "pass", report
    elif task.startswith("context-"):
        length = int(task.split("-")[1])
        ids = (docs[0][1] * ((length + 1) // len(docs[0][1]) + 1))[: length + 1]
        engine.sync()
        started = time.monotonic()
        result = engine.forward(ids, backward=True)
        metrics = engine.step()
        engine.sync()
        seconds = time.monotonic() - started
        assert bool(torch.isfinite(result["logprobs"]).all())
        write(
            output / f"result-rank{engine.rank}.json",
            {
                "status": "pass",
                "tokens": length,
                "seconds": seconds,
                "tokens_s_gpu": length
                / seconds
                / int(os.environ.get("WORLD_SIZE", "1")),
                "memory": engine.memory(),
                "optimizer": metrics,
            },
        )
    else:
        raise ValueError(task)


def first_forward_repeat(engine: Any, ids: list[int], output: Path) -> dict:
    """First two operations on each fresh forward/learning engine, before training."""
    from safetensors.torch import save_file

    with torch.no_grad():
        arrays = [
            engine.forward(ids)["logprobs"].detach().float().cpu() for _ in range(2)
        ]
    save_file(
        {"first": arrays[0], "second": arrays[1]},
        str(output / f"first-forward-repeat-rank{engine.rank}.safetensors"),
    )
    finite = all(bool(torch.isfinite(value).all()) for value in arrays)
    delta = (arrays[0] - arrays[1]).abs().reshape(-1)
    metrics = {
        "mean": float(delta.mean()) if finite else None,
        "p99": float(torch.quantile(delta, 0.99)) if finite else None,
        "max": float(delta.max()) if finite else None,
    }
    hard_bug = not finite or float(delta.mean()) > 0.1
    report = {
        "status": "fail" if hard_bug else "pass",
        "finite": finite,
        "hard_bug": hard_bug,
        "threshold_mean": 0.1,
        "p99_max_policy": "report-only",
        "tokens_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        **metrics,
    }
    write(output / f"first-forward-repeat-rank{engine.rank}.json", report)
    if hard_bug:
        raise RuntimeError(f"first-forward repeat hard bug: {report}")
    return report


def oom_evidence(error: BaseException, task: str) -> dict | None:
    """TrainerRank wraps execution OOMs; admission refusals are distinct errors."""
    chain, seen = [], set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    if not any(isinstance(item, torch.cuda.OutOfMemoryError) for item in chain):
        return None
    return {
        "status": "oom",
        "task": task,
        "error_type": type(error).__name__,
        "error": str(error),
        "chain": [
            {"type": type(item).__name__, "message": str(item)} for item in chain
        ],
    }


def compare(
    candidate: Path,
    fixtures: Path,
    name: str,
    output: Path,
    tiny: bool = False,
    supplemental: Path | None = None,
) -> dict:
    from safetensors.torch import load_file

    from art.megatron.glm5next.hf_oracle import logprob_metrics
    from tests.integration.megatron.glm5next_parity_diagnostics import (
        sublayer_metrics,
        top1_diagnostics,
    )

    paths = sorted(candidate.glob("*-rank0.safetensors"))
    if not paths:
        raise ValueError("No native parity captures")
    rows: list[dict] = []
    for actual_path in paths:
        expected_path = (
            actual_path
            if tiny
            else fixtures / "dumps" / name / actual_path.name.replace("-rank0", "")
        )
        a, b = load_file(str(actual_path)), load_file(str(expected_path))
        lp = logprob_metrics(
            a["target_logprobs"].reshape(-1), b["target_logprobs"].reshape(-1)
        )
        top = top1_diagnostics(a, b)
        diagnostics_reference = dict(b)
        reference_source = "original M1 dump"
        if supplemental is not None and not tiny:
            extra_path = supplemental / "dumps" / name / expected_path.name
            extra = load_file(str(extra_path))
            diagnostics_reference.update(
                {key: value for key, value in extra.items() if key not in b}
            )
            reference_source = (
                "original M1 dump; missing F8 sublayers from supplemental M1 HF capture"
            )
        record = {
            "document": actual_path.name,
            **lp,
            "top1": top["agreements"] / top["positions"],
            "top1_diagnostics": top,
            "layers": sublayer_metrics(a, diagnostics_reference, top["positions"]),
            "diagnostics_reference": reference_source,
            "diagnostics_policy": "report-only",
        }
        if supplemental is not None and not tiny:
            record["supplemental_vs_original_logprobs"] = logprob_metrics(
                extra["target_logprobs"].reshape(-1), b["target_logprobs"].reshape(-1)
            )
            record["supplemental_vs_original_top1"] = float(
                (extra["top64.indices"][..., 0] == b["top64.indices"][..., 0])
                .float()
                .mean()
            )
        rows.append(record)
    passed = all(
        r["logprob_abs_mean"] <= 0.02
        and r["logprob_abs_p99"] <= 0.2
        and r["top1"] >= 0.99
        for r in rows
    )
    report = {
        "status": "pass" if passed else "fail",
        "hard_bug": any(r["logprob_abs_mean"] > 0.1 for r in rows),
        "rows": rows,
        "execution": "tiny_cpu_self_comparison" if tiny else "native_vs_original_m1_hf",
    }
    write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--supplemental", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.task.startswith("compare-"):
        compare(
            args.model,
            args.inputs,
            args.task.removeprefix("compare-"),
            args.output / "comparison.json",
            args.tiny,
            args.supplemental,
        )
    elif args.task == "hf-reference-F8":
        from tests.integration.megatron.glm5next_parity_diagnostics import (
            supplemental_reference,
        )

        supplemental_reference(args.model, args.output, tiny=args.tiny)
    elif args.task == "hf-stage0":
        from tests.integration.megatron.model_support.glm5next_oracle import versions

        runtime = versions()
        assert runtime["cuda"] == "12.8"
        assert runtime["transformers"] == "5.20.2"
        assert (
            runtime["modeling_sha256"]
            == "2e8a08614860e5117a410080df42e67e7947bfdd116e90c19e46229c2d7ea645"
        )
        assert not torch.cuda.is_initialized()
        write(
            args.output / "stage0.json",
            {"status": "pass", "cuda_initialized": False, **runtime},
        )
    elif args.task == "stage0":
        stage0(args.output / "stage0.json", inputs=args.inputs)
    elif args.task == "topology-plans":
        from tests.integration.megatron.glm5next_topology import validate_stage_plans

        validate_stage_plans(args.inputs, args.output / "plans.json")
    elif args.task == "volumes":
        volumes(args.inputs, args.output / "volumes.json")
    elif args.task == "acquire":
        records = json.loads((args.inputs / "weights.json").read_text())["files"]
        if args.tiny:
            records = [r for r in records if r["path"] == "config.json"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda r: acquire(r, args.model), records))
        write(args.output / "acquisition.json", results)
    elif args.task == "hybridep-setup":
        hybrid_ep_stage(args.output, tiny=args.tiny)
    elif args.task == "oracle-setup":
        from tests.integration.megatron.glm5next_qualification_launch import (
            prepare_worker_fixtures,
        )

        prepare_worker_fixtures(
            Path(__file__).resolve().parents[3], args.output.parents[1], tiny=args.tiny
        )
    elif args.task == "identity":
        # Identical CPU construction/codec/file IO in rehearsal and GPU execution.
        # This materializes the actual rank32 adapter volume, even for a tiny engine.
        identity(args.model, args.output.parents[1] / "initial")

    else:
        engine = None
        try:
            engine = (TinyEngine if args.tiny else ArtEngine)(args.model, args.output)
            experiment(engine, args.task, args.inputs, args.output)
        except Exception as exc:
            if (evidence := oom_evidence(exc, args.task)) is not None:
                write(
                    args.output / f"failure-rank{os.environ.get('RANK', '0')}.json",
                    evidence,
                )
            raise
        finally:
            if engine is not None:
                engine.close()


if __name__ == "__main__":
    main()
