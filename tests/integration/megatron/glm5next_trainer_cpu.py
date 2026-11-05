"""Real TrainerRank / native GLM architecture on CPU, with explicit kernel swaps.

Run with torchrun --standalone --nproc-per-node=8 --module ... --inputs DIR
--output DIR. This is a plumbing regression, not GPU numerical qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator
from unittest.mock import patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
import torch.distributed as dist
import torch.nn.functional as F

SUBSTITUTIONS = [
    "Native TE linear/grouped-linear CUDA GEMMs -> torch F.linear, preserving native modules and LoRA wrappers",
    "TE indexer LayerNorm CUDA kernel -> torch layer_norm",
    "GLM52 DSA LoRA A-projection Triton BF16 GEMM -> torch matmul; native LoRA forward/backward retained",
    "Shared-outer routed-expert LoRA CUDA GEMMs -> per-expert torch matmuls; compact storage, wrappers and gradient synchronization retained",
    "FLA KDA convolution/recurrence -> KimiDeltaAttention backend=reference; real Gloo autograd exchanges",
    "TileLang sparse MLA forward/backward -> masked torch attention/autograd; real CP stage fetch/dKV reductions",
    "HybridEP CUDA buffer/dispatcher -> MCore allgather dispatcher on real EP Gloo groups; topology guard retained",
    "MCore CUDA RNG checkpoint primitive -> torch checkpoint (full/uniform/one layer policy retained)",
    "CUDA constructor allocation -> meta then deterministic CPU materialization; tiny random weights, no BF16 download",
    "MCore current-device and global communication buffer CUDA allocation -> CPU allocation",
]
GPU_ONLY = [
    "Flash/TileLang sparse MLA kernels, BF16 rounding and FP32 dKV kernel partials",
    "FLA chunk KDA, CUDA convolution halo/CP precision kernels",
    "TE CUDA GEMMs/grouped GEMMs/norms, CUTLASS geometry, CUDA RNG and CUDA DDP",
    "Triton BF16 DSA LoRA projection and CUDA grouped/shared-outer LoRA kernels",
    "HybridEP sm_100 build/buffer/dispatch/combine/stream ordering and NCCL",
    "CUDA admission/memory telemetry, pinned-host asynchronous export and stream lifetime",
    "Actual 45-layer BF16 checkpoint loading, real-volume activations and 32K–262K GPU fit/throughput",
    "GPU repeated-forward numerics, learning/export gates and HF/serving parity",
]


def sparse_math(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, *, scale: float, **_: Any
) -> torch.Tensor:
    mask = (indices[..., None] == torch.arange(kv.shape[1])[None, None, None, :]).any(
        -2
    )
    scores = torch.einsum("bqhd,bkd->bhqk", q, kv) * scale
    # Empty padded queries must have zero gradients as well as zero outputs.
    masked = scores.masked_fill(~mask[:, None], -torch.inf)
    masked = torch.where(
        mask[:, None].any(-1, keepdim=True), masked, torch.zeros_like(masked)
    )
    probs = masked.softmax(-1) * mask[:, None]
    return torch.einsum("bhqk,bkd->bqhd", probs, kv[..., :512])


def sparse_forward(q: Any, kv: Any, indices: Any, *, scale: float) -> Any:
    return sparse_math(q, kv, indices, scale=scale), q.new_zeros(q.shape[:-1])


def sparse_backward(
    q: Any, kv: Any, indices: Any, out: Any, lse: Any, grad: Any, *, scale: float
) -> Any:
    del out, lse
    with torch.enable_grad():
        q1, kv1 = q.detach().requires_grad_(), kv.detach().requires_grad_()
        return torch.autograd.grad(
            sparse_math(q1, kv1, indices, scale=scale), (q1, kv1), grad
        )


def linear(self: Any, x: torch.Tensor, **_: Any) -> Any:
    bias = getattr(self, "bias", None)
    if bias is not None and not bias.numel():
        bias = None
    skip = getattr(self, "skip_bias_add", False)
    return F.linear(x, self.weight, None if skip else bias), bias if skip else None


def grouped_linear(self: Any, x: torch.Tensor, counts: Any) -> Any:
    counts = counts.tolist() if isinstance(counts, torch.Tensor) else counts
    parts = x.split(counts)
    return torch.cat(
        [F.linear(part, getattr(self, f"weight{i}")) for i, part in enumerate(parts)]
    ), None


def shared_outer(
    x: Any, a: Any, b: Any, counts: Any, *, scale: float, shared_factor: str
) -> Any:
    counts = counts.tolist() if isinstance(counts, torch.Tensor) else counts
    return torch.cat(
        [
            (part @ a[0 if shared_factor == "A" else i])
            @ b[0 if shared_factor == "B" else i]
            * scale
            for i, part in enumerate(x.split(counts))
        ]
    )


@contextmanager
def cpu_kernels() -> Iterator[None]:
    from megatron.core import tensor_parallel
    from megatron.core.extensions import transformer_engine as te
    from megatron.core.utils import GlobalMemoryBuffer
    from torch.utils.checkpoint import checkpoint
    import transformer_engine.pytorch as tep

    from art.megatron.glm52 import attention, cp_attention, lora_projection
    import art.megatron.kernels.shared_outer_lora as shared_kernel

    def cpu_checkpoint(function: Any, distribute: bool, *args: Any) -> Any:
        assert not distribute
        # The real handler's preprocess hook enables frozen-input gradients.
        # Reentrant replay completes the layer's EP/CP collectives on all ranks.
        return checkpoint(function, *args, use_reentrant=True)

    def cpu_buffer(
        self: Any, shape: Any, dtype: Any, name: Any, *args: Any, **kwargs: Any
    ) -> Any:
        del self, name, args, kwargs
        return torch.empty(shape, dtype=dtype, device="cpu")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(torch.cuda, "current_device", return_value=torch.device("cpu"))
        )
        for cls in (te.TELinear, te.TEColumnParallelLinear, te.TERowParallelLinear):
            stack.enter_context(patch.object(cls, "forward", linear))
        for cls in (te.TEColumnParallelGroupedLinear, te.TERowParallelGroupedLinear):
            stack.enter_context(patch.object(cls, "forward", grouped_linear))
        stack.enter_context(
            patch.object(
                tep.LayerNorm,
                "forward",
                lambda self, x: F.layer_norm(
                    x, (x.shape[-1],), self.weight, self.bias, self.eps
                ),
            )
        )
        stack.enter_context(patch.object(tensor_parallel, "checkpoint", cpu_checkpoint))
        stack.enter_context(patch.object(GlobalMemoryBuffer, "get_tensor", cpu_buffer))
        stack.enter_context(patch.object(attention, "sparse_mla", sparse_math))
        stack.enter_context(
            patch.object(lora_projection, "_forward", lambda x, a: x @ a)
        )
        stack.enter_context(
            patch.object(shared_kernel, "shared_outer_lora", shared_outer)
        )
        stack.enter_context(
            patch.object(cp_attention, "sparse_mla_forward", sparse_forward)
        )
        stack.enter_context(
            patch.object(cp_attention, "sparse_mla_backward", sparse_backward)
        )
        yield


def build_trainer(world: int, config_path: Path) -> Any:
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import OptimizerConfig

    from art.megatron.glm5next.kda import KimiDeltaAttention
    from art.megatron.lora import LoraLayout
    from art.megatron.provider import _build_provider_bundle
    from art.megatron.train import TrainingRuntime
    from art.trainer_rank import TrainerRank
    from tests.integration.megatron.glm5next_qualification import (
        configure_qualification_provider,
    )

    bundle = _build_provider_bundle(
        str(config_path.parent),
        torch_dtype=torch.float32,
        model_support_key="glm5next",
        allow_unvalidated_arch=True,
        load_weights=True,
    )
    provider = bundle.provider
    provider._art_model_support_spec = bundle.spec
    bundle.handler.patch_provider(provider, bundle.bridge)
    configure_qualification_provider(provider)
    provider._art_lora_layout = LoraLayout(expert="shared_outer")
    provider.tensor_model_parallel_size = provider.pipeline_model_parallel_size = (
        provider.expert_tensor_parallel_size
    ) = 1
    provider.context_parallel_size = provider.expert_model_parallel_size = world
    # Match native meta construction; CPU-init would copy a meta master into
    # explicitly CPU TE parameters before materialization.
    provider.use_cpu_initialization = False
    provider.init_model_with_meta_device = True
    provider.params_dtype = torch.float32
    provider.bf16 = provider.fp16 = False
    provider.moe_token_dispatcher_type = "allgather"
    provider.moe_permute_fusion = False
    provider.bias_dropout_fusion = False
    provider.gradient_accumulation_fusion = False
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.finalize()
    with (
        torch.device("meta"),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=torch.device("meta")),
    ):
        model = GPTModel(
            config=provider,
            transformer_layer_spec=provider.transformer_layer_spec(provider),
            vocab_size=provider.vocab_size,
            max_sequence_length=256,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )
    model.to_empty(device="cpu")
    torch.manual_seed(20261105)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "hc_" in name:
                param.zero_()
                if name.endswith("scale"):
                    param.fill_(1)
            elif param.ndim == 1:
                param.fill_(0 if name.endswith("bias") else 1)
            else:
                param.normal_(std=0.02)
        for module in model.modules():
            if isinstance(module, KimiDeltaAttention):
                module.backend = "reference"
            for name, buffer in module.named_buffers(recurse=False):
                if "expert" in name:
                    buffer.zero_()
    bundle.handler.apply_lora_adapters(
        [model],
        provider,
        target_modules=list(bundle.spec.default_target_modules),
        rank=32,
        alpha=32,
        lora_layout=LoraLayout(expert="shared_outer"),
    )
    bundle.handler.install_preprocess_patch([model])
    runtime = TrainingRuntime(
        provider_bundle=bundle,
        provider=provider,
        model=[model],
        optimizer=None,
        optimizer_config=OptimizerConfig(),
        rank=dist.get_rank(),
        world_size=world,
        lora_layout=LoraLayout(expert="shared_outer"),
    )
    trainer = TrainerRank(runtime)
    return trainer


def run(inputs: Path, output: Path) -> dict:
    from megatron.core import parallel_state as ps
    from safetensors.torch import load_file

    from art.megatron.glm5next.dsa import Glm5NextSelfAttention
    from art.megatron.glm5next.dsa_cp import Glm5NextDsaState
    from art.megatron.glm5next.lora_export import (
        META,
        from_vllm_lora_tensors,
        validate_native_lora,
    )
    from art.trainer_rank import AdamParams, ForwardInput

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    world = int(os.environ["WORLD_SIZE"])
    assert world >= 2
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    ps.initialize_model_parallel(
        context_parallel_size=world,
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
    )
    records: list[dict] = []
    reached: list[dict] = []
    try:
        with (
            tempfile.TemporaryDirectory(prefix="glm-trainer-cpu-") as directory,
            cpu_kernels(),
            ExitStack() as stack,
        ):
            cfg = json.loads(
                (
                    Path(__file__).parents[2]
                    / "unit/fixtures/glm5next_configs/R2/config.json"
                ).read_text()
            )
            cfg["n_routed_experts"] = max(4, world)
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(cfg))
            trainer = build_trainer(world, config)
            architecture = Counter(
                type(module).__name__
                for chunk in trainer.runtime.model
                for module in chunk.modules()
            )
            for name, count in {
                "Glm5NextTransformerLayer": 3,
                "KimiDeltaAttention": 2,
                "Glm5NextSelfAttention": 1,
                "Glm5NextRouter": 2,
                "TEGroupedMLP": 2,
            }.items():
                assert architecture[name] == count
            assert ps.get_context_parallel_world_size() == world
            assert ps.get_expert_model_parallel_world_size() == world
            from tests.integration.megatron.glm5next_qualification import identity

            if dist.get_rank() == 0:
                identity(config.parent, output / "initial")
            dist.barrier()
            trainer.load_checkpoint(str(output / "initial"))

            def reference_dispatch(batches: Any, *, topology: Any) -> None:
                del batches
                trainer._validate_hybridep_topology(topology)

            stack.enter_context(
                patch.object(trainer, "_configure_hybridep", reference_dispatch)
            )
            original_state = Glm5NextSelfAttention._attention_state

            def checked_state(self: Any, bias: Any) -> Any:
                state = original_state(self, bias)
                assert (
                    isinstance(state, Glm5NextDsaState)
                    and state.context_parallel_state is bias
                )
                reached.append(
                    {
                        "grad_enabled": torch.is_grad_enabled(),
                        "local_positions": state.position_ids.numel(),
                    }
                )
                return state

            stack.enter_context(
                patch.object(Glm5NextSelfAttention, "_attention_state", checked_state)
            )
            plans = json.loads((inputs / "stage-plans.json").read_text())
            for plan in plans:
                assert (plan["world_size"], plan["cp"], plan["ep"]) == (
                    world,
                    world,
                    world,
                )
                # Preserve every sealed topology/mode, scale only tokens/weights.
                ids = torch.arange(32 * world + 1) % cfg["vocab_size"]
                request = ForwardInput(
                    input_tokens=ids[:-1], target_tokens=ids[1:], top_k=2
                )
                trainer.zero_grad()
                parameters = [
                    param
                    for slot in trainer._checkpoint_slots.values()
                    for param in slot.params
                ]
                before = [param.detach().clone() for param in parameters]
                modes = []
                for no_grad in (True, False):
                    for mb in trainer.forward_micro_batches([request], no_grad=no_grad):
                        logprobs = mb.outputs[0].target_logprobs
                        assert logprobs is not None and torch.isfinite(logprobs).all()
                        assert logprobs.requires_grad is (not no_grad)
                        loss = -logprobs.mean()
                        if not no_grad:
                            trainer.backward([loss])
                        modes.append(
                            {
                                "no_grad": no_grad,
                                "loss": float(loss.detach()),
                                "tokens": logprobs.numel(),
                            }
                        )
                metrics = trainer.optim_step(
                    params=AdamParams(
                        learning_rate=1e-4, weight_decay=0, grad_clip_norm=0.1
                    )
                )
                assert all(math.isfinite(value) for value in metrics.values())
                changed = sum(
                    not torch.equal(previous, param)
                    for previous, param in zip(before, parameters, strict=True)
                )
                assert changed > 0, "optim_step did not change any adapter parameter"
                destination = output / plan["name"] / "adapter"
                trainer.export_lora(str(destination))
                dist.barrier()
                if dist.get_rank() == 0:
                    adapter_config = json.loads(
                        (destination / "adapter_config.json").read_text()
                    )
                    native = from_vllm_lora_tensors(
                        load_file(str(destination / "adapter_model.safetensors")),
                        adapter_config=adapter_config,
                    )
                    validate_native_lora(
                        native,
                        adapter_config={
                            **adapter_config,
                            META: {**adapter_config[META], "format": "art"},
                        },
                    )
                records.append(
                    {
                        "sealed_plan": plan,
                        "modes": modes,
                        "optimizer": metrics,
                        "updated_parameters": changed,
                        "export": str(destination),
                        "status": "pass",
                    }
                )
            assert reached and any(item["grad_enabled"] for item in reached)
            assert not torch.cuda.is_initialized()
            result = {
                "status": "pass",
                "rank": dist.get_rank(),
                "world_size": world,
                "cp": world,
                "ep": world,
                "dp": 1,
                "architecture": dict(architecture),
                "tiny_config": cfg,
                "scaled_context_tokens": 32 * world,
                "plans": records,
                "dsa_state_visits": reached,
                "substitutions": SUBSTITUTIONS,
                "gpu_only_unreached": GPU_ONLY,
                "cuda_initialized": False,
            }
            output.mkdir(parents=True, exist_ok=True)
            (output / f"rank-{dist.get_rank()}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            return result
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)


if __name__ == "__main__":
    main()
