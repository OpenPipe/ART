"""CPU CP1-versus-CPn probe of real TrainerRank, handlers, packing and plans.

CUDA kernels alone are replaced by torch math. Sparse attention interprets both
the partial and full block lists, so errors in block classification stay visible.
Run with torchrun --standalone --nproc-per-node=N --module
tests.integration.megatron.cp_trainer_cpu --output DIR.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import timedelta
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Generator
from unittest.mock import patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import torch
import torch.distributed as dist
import torch.nn.functional as F

MASK_AUDITS: list[dict[str, int]] = []


def block_mask_dense(mask: Any, qlen: int, klen: int) -> torch.Tensor:
    """Interpret flex's sparse metadata, including unmasked full blocks."""
    qb, kb = mask.BLOCK_SIZE
    partial = torch.zeros(qlen, klen, dtype=torch.bool)
    full = torch.zeros_like(partial)
    for counts, indices, target in (
        (mask.kv_num_blocks, mask.kv_indices, partial),
        (mask.full_kv_num_blocks, mask.full_kv_indices, full),
    ):
        if counts is None:
            continue
        for row, count in enumerate(counts[0, 0].tolist()):
            for col in indices[0, 0, row, :count].tolist():
                target[row * qb : (row + 1) * qb, col * kb : (col + 1) * kb] = True
    q = torch.arange(qlen)[:, None]
    k = torch.arange(klen)[None, :]
    exact = mask.mask_mod(torch.tensor(0), torch.tensor(0), q, k)
    result = full | (partial & exact)
    MASK_AUDITS.append(
        {"q": qlen, "k": klen, "mismatched_pairs": int((result != exact).sum())}
    )
    return result


def flex_math(
    q: Any, k: Any, v: Any, *, block_mask: Any, scale: float, **_: Any
) -> Any:
    mask = block_mask_dense(block_mask, q.shape[2], k.shape[2])
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    lse = scores.logsumexp(-1)
    probs = torch.where(
        mask.any(-1, keepdim=True), scores, torch.zeros_like(scores)
    ).softmax(-1)
    return (probs * mask) @ v, lse


def linear(self: Any, x: Any, **_: Any) -> Any:
    bias = getattr(self, "bias", None)
    if bias is not None and not bias.numel():
        bias = None
    skip = getattr(self, "skip_bias_add", False)
    if hasattr(self, "layer_norm_weight"):
        x = norm_math(
            self,
            x,
            weight=self.layer_norm_weight,
            bias=getattr(self, "layer_norm_bias", None),
        )
    output = F.linear(x, self.weight, None if skip else bias)
    if getattr(self, "return_layernorm_output", False):
        output = (output, x)
    return output, bias if skip else None


def norm_math(self: Any, x: Any, *, weight: Any = None, bias: Any = None) -> Any:
    weight = self.weight if weight is None else weight
    if getattr(self, "normalization", "RMSNorm") == "LayerNorm":
        return F.layer_norm(x, (x.shape[-1],), weight, bias, self.eps)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * weight


def conv_math(
    gdn: Any, qkv: Any, state: Any, bounds: Any, *, output_final_state: bool
) -> Any:
    outputs, finals = [], []
    width = state.shape[-1]
    for i, (start, end) in enumerate(zip(bounds.tolist(), bounds.tolist()[1:])):
        history = torch.cat((state[i], qkv[start:end].T), dim=-1)
        y = F.conv1d(
            history[None], gdn.conv1d.weight, gdn.conv1d.bias, groups=qkv.shape[-1]
        )
        outputs.append(gdn.act_fn(y[0].T))
        finals.append(history[:, -width:] if width else history[:, :0])
    return torch.cat(outputs), torch.stack(finals) if output_final_state else None


def scan_conv(parent: Any, tails: Any, lengths: Any, *, stop_rank: int) -> Any:
    rows = list(parent.unbind())
    for rank in range(stop_rank):
        rows = [
            torch.cat((row, tails[rank, i, :, :n]), dim=-1)[:, -parent.shape[-1] :]
            for i, (row, n) in enumerate(zip(rows, lengths[rank].tolist()))
        ]
    return torch.stack(rows)


def gather_streams(
    qkv: Any,
    beta: Any,
    g: Any,
    rows: Any,
    positions: Any,
    *,
    token_count: int,
    sequence_length: int,
) -> Any:
    indices = rows[:token_count] * sequence_length + positions[:token_count]
    return tuple(x[indices] for x in (qkv, beta, g))


def scatter_streams(
    output: Any, values: Any, rows: Any, positions: Any, mask: Any, **_: Any
) -> Any:
    result = output.clone()
    result[rows[mask], positions[mask]] = values.reshape(-1, *values.shape[-2:])[mask]
    return result


@contextmanager
def cpu_kernels() -> Generator[None]:
    if os.environ.get("CP_CPU_TRITON"):
        from transformer_engine.common.triton.cross_entropy import cross_entropy_kernel

        setattr(cross_entropy_kernel, "__art_fp32_target_grad__", True)
    # Install ART's runtime wrappers before replacing their CUDA leaf kernels.
    from megatron.core import tensor_parallel
    from megatron.core.extensions import transformer_engine as te
    from megatron.core.transformer.moe import router
    from megatron.core.utils import GlobalMemoryBuffer
    from torch.utils.checkpoint import checkpoint
    import transformer_engine.pytorch as tep

    from art.megatron.context_parallel import executor
    from art.megatron.flex_attn.attention import FlexAttentionWrapper
    from art.megatron.gdn import operator
    import art.megatron.kernels.shared_outer_lora as shared_kernel
    import art.megatron.lora as lora
    import art.megatron.provider  # noqa: F401
    from tests.integration.megatron.glm5next_trainer_cpu import (
        grouped_linear,
        shared_outer,
    )
    from tests.integration.megatron.model_support.gdn_fp32_reference import (
        _torch_chunk_gated_delta_rule_native_cp_reference,
        _torch_chunk_gated_delta_rule_reference,
    )

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch._dynamo.config, "disable", True))
        stack.enter_context(
            patch.object(torch.cuda, "current_device", return_value=torch.device("cpu"))
        )
        for cls in (
            te.TELinear,
            te.TEColumnParallelLinear,
            te.TERowParallelLinear,
            te.TELayerNormColumnParallelLinear,
        ):
            stack.enter_context(patch.object(cls, "forward", linear))
        for cls in (te.TEColumnParallelGroupedLinear, te.TERowParallelGroupedLinear):
            stack.enter_context(patch.object(cls, "forward", grouped_linear))
        stack.enter_context(patch.object(tep.RMSNorm, "forward", norm_math))
        stack.enter_context(
            patch.object(
                tep.LayerNorm,
                "forward",
                lambda self, x: F.layer_norm(
                    x, (x.shape[-1],), self.weight, self.bias, self.eps
                ),
            )
        )
        stack.enter_context(
            patch.object(
                GlobalMemoryBuffer,
                "get_tensor",
                lambda self, shape, dtype, *a, **kw: torch.empty(shape, dtype=dtype),
            )
        )
        stack.enter_context(
            patch.object(
                tensor_parallel,
                "checkpoint",
                lambda fn, distribute, *a: checkpoint(fn, *a, use_reentrant=True),
            )
        )
        stack.enter_context(
            patch.object(
                FlexAttentionWrapper,
                "forward",
                lambda self, *a, **kw: flex_math(*a, **kw)[0],
            )
        )
        stack.enter_context(
            patch.object(
                executor.FlexAttentionKernel,
                "run",
                lambda self, *a, **kw: flex_math(*a, **kw),
            )
        )
        if os.environ.get("CP_CPU_TRITON"):
            assert os.environ.get("TRITON_INTERPRET") == "1"
            from art.megatron.gdn import conv_gelu, segment_layout

            stack.enter_context(
                patch.object(conv_gelu, "_validate_packed_inputs", lambda *a: None)
            )
            stack.enter_context(
                patch.object(segment_layout, "_validate_cuda", lambda *a: None)
            )
        else:
            stack.enter_context(
                patch.object(
                    operator, "_causal_conv1d_packed_varlen_with_state", conv_math
                )
            )
            stack.enter_context(
                patch.object(operator, "_scan_conv_tail_batch", scan_conv)
            )
            stack.enter_context(
                patch.object(
                    operator, "_gather_bucket_streams_compact_fused", gather_streams
                )
            )
            stack.enter_context(
                patch.object(operator, "_scatter_bucket_output_fused", scatter_streams)
            )
        stack.enter_context(
            patch.object(
                operator,
                "_chunk_gated_delta_rule",
                _torch_chunk_gated_delta_rule_reference,
            )
        )
        stack.enter_context(
            patch.object(
                operator,
                "chunk_gated_delta_rule_native_cp",
                _torch_chunk_gated_delta_rule_native_cp_reference,
            )
        )
        stack.enter_context(
            patch.object(
                operator, "_l2norm", lambda x: F.normalize(x, p=2, dim=-1, eps=1e-6)
            )
        )
        if not os.environ.get("CP_CPU_TRITON"):
            stack.enter_context(
                patch.object(
                    operator,
                    "_prepare_packed_recurrent_inputs_fused",
                    lambda qkv, beta, g, **kw: operator._prepare_dense_recurrent_inputs(
                        qkv[None], beta[None], g[None], **kw
                    ),
                )
            )
        stack.enter_context(
            patch.object(shared_kernel, "shared_outer_lora", shared_outer)
        )
        stack.enter_context(
            patch.object(
                lora,
                "quack_grouped_lora",
                lambda x, a, b, counts, scale=1: shared_outer(
                    x, a, b, counts, scale=scale, shared_factor="none"
                ),
            )
        )
        stack.enter_context(
            patch.object(
                lora,
                "quack_grouped_lora_dual",
                lambda x, ga, gb, ua, ub, counts, scale_gate=1, scale_up=1: torch.cat(
                    (
                        shared_outer(
                            x, ga, gb, counts, scale=scale_gate, shared_factor="none"
                        ),
                        shared_outer(
                            x, ua, ub, counts, scale=scale_up, shared_factor="none"
                        ),
                    ),
                    dim=-1,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                router,
                "router_gating_linear",
                lambda x, w, b, dtype: F.linear(
                    x.to(dtype or x.dtype),
                    w.to(dtype or x.dtype),
                    None if b is None else b.to(dtype or x.dtype),
                ),
            )
        )
        yield


def qwen_config() -> dict:
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 8,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "full_attention_interval": 4,
            "vocab_size": 128,
            "max_position_embeddings": 32768,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
            "tie_word_embeddings": False,
            "attention_dropout": 0.0,
        },
    }


def build_qwen(world: int, config: Path) -> Any:
    from megatron.core.optimizer import OptimizerConfig

    from art.megatron.lora import LoraLayout
    from art.megatron.provider import _build_provider_bundle
    from art.megatron.train import TrainingRuntime
    from art.trainer_rank import TrainerRank

    bundle = _build_provider_bundle(
        str(config.parent),
        torch_dtype=torch.float32,
        model_support_key="qwen3_5_moe",
        allow_unvalidated_arch=True,
        load_weights=True,
    )
    p = bundle.provider
    p._art_model_support_spec = bundle.spec
    p._art_lora_rank = 32
    bundle.handler.configure_provider_for_runtime(p)
    bundle.handler.patch_provider(p, bundle.bridge)
    p.tensor_model_parallel_size = p.pipeline_model_parallel_size = (
        p.expert_tensor_parallel_size
    ) = p.expert_model_parallel_size = 1
    p.context_parallel_size = world
    p._art_lora_layout = LoraLayout()
    p.use_cpu_initialization = False
    p.init_model_with_meta_device = True
    p.params_dtype = torch.float32
    p.bf16 = p.fp16 = False
    p.moe_token_dispatcher_type = "allgather"
    p.moe_permute_fusion = p.bias_dropout_fusion = p.gradient_accumulation_fusion = (
        p.bias_activation_fusion
    ) = False
    p.recompute_granularity = "full"
    p.recompute_method = "uniform"
    p.recompute_num_layers = 1
    from art.megatron.provider import _finalize_provider_with_art_overrides

    _finalize_provider_with_art_overrides(p)
    with (
        torch.device("meta"),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=torch.device("meta")),
    ):
        model = p.provide_language_model(pre_process=True, post_process=True)
    model.to_empty(device="cpu")
    rotary = model.rotary_pos_emb
    rotary_dim = rotary.inv_freq.numel() * 2
    rotary.inv_freq = 1.0 / (
        p.rotary_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Name-based seeds make replicated parameters independent of topology.
            torch.manual_seed(
                int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            )
            if name.endswith("A_log"):
                param.zero_()
            elif param.ndim == 1:
                param.fill_(0 if name.endswith("bias") else 1)
            else:
                param.normal_(std=0.1)
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if "expert" in name:
                    buffer.zero_()
    bundle.handler.apply_lora_adapters(
        [model],
        p,
        target_modules=list(bundle.spec.default_target_modules),
        rank=32,
        alpha=32,
        lora_layout=LoraLayout(),
    )
    bundle.handler.install_preprocess_patch([model])
    trainer = TrainerRank(
        TrainingRuntime(
            provider_bundle=bundle,
            provider=p,
            model=[model],
            optimizer=None,
            optimizer_config=OptimizerConfig(),
            rank=dist.get_rank(),
            world_size=world,
            lora_layout=LoraLayout(),
        )
    )
    from art.megatron.identity_lora import create_identity_lora

    identity = config.parent.parent / "initial"
    if dist.get_rank() == 0:
        create_identity_lora(
            str(config.parent),
            str(identity),
            rank=32,
            lora_alpha=32,
            random_state=20261105,
            target_modules=list(bundle.spec.default_target_modules),
            handler=bundle.handler,
            allow_unvalidated_arch=True,
        )
        adapter_config = identity / "adapter_config.json"
        metadata = json.loads(adapter_config.read_text())
        metadata["base_model_name_or_path"] = bundle.spec.model_names[0]
        adapter_config.write_text(json.dumps(metadata))
    dist.barrier()
    trainer.load_checkpoint(str(identity))
    return trainer


def run(
    output: Path,
    lengths: list[int],
    *,
    model_name: str = "qwen",
    expert_parallel: int = 1,
) -> None:
    cache = output.parent / "cache"
    for key, value in {
        "ART_MEGATRON_CACHE_ROOT": cache,
        "XDG_CACHE_HOME": cache,
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
    }.items():
        os.environ[key] = str(value)
    from megatron.core import parallel_state as ps

    from art.trainer_rank import ForwardInput

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    MASK_AUDITS.clear()
    faulthandler.dump_traceback_later(
        int(os.environ.get("CP_CPU_TRACE_TIMEOUT", "180")), repeat=True
    )
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    ps.initialize_model_parallel(
        context_parallel_size=world, expert_model_parallel_size=expert_parallel
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (
            tempfile.TemporaryDirectory(dir=output) as directory,
            cpu_kernels(),
            ExitStack() as stack,
        ):
            config = Path(directory) / "config.json"
            if model_name == "qwen":
                config.write_text(json.dumps(qwen_config()))
                trainer = build_qwen(world, config)
                vocab = 128
            else:
                from tests.integration.megatron import glm5next_trainer_cpu as glm
                from tests.integration.megatron.glm5next_qualification import identity

                cfg = json.loads(
                    (
                        Path(__file__).parents[2]
                        / "unit/fixtures/glm5next_configs/R2/config.json"
                    ).read_text()
                )
                cfg["n_routed_experts"] = 8
                config.write_text(json.dumps(cfg))
                stack.enter_context(glm.cpu_kernels())
                trainer = glm.build_trainer(
                    world, config, expert_parallel=expert_parallel
                )
                if dist.get_rank() == 0:
                    identity(config.parent, output / "initial")
                dist.barrier()
                trainer.load_checkpoint(str(output / "initial"))
                stack.enter_context(
                    patch.object(
                        trainer,
                        "_configure_hybridep",
                        lambda batches, *, topology: (
                            trainer._validate_hybridep_topology(topology)
                        ),
                    )
                )
                vocab = cfg["vocab_size"]
            records: dict[str, Any] = {}
            metadata: list[Any] = []
            prepared: list[Any] = []
            original = trainer._decoder_hidden

            def decoder(batch: Any) -> Any:
                prepared.append(batch)
                if model_name == "glm":
                    state = batch.attention_state.model_state["glm5next_dsa"]
                    metadata.append(
                        [
                            {
                                "pools": plan.pool_starts.tolist(),
                                "queries": plan.query_ids.tolist(),
                                "tail": plan.tail_origin,
                            }
                            for plan in state.pool_plans
                        ]
                    )
                if os.environ.get("CP_CPU_TRACE"):
                    print(
                        f"TRACE rank={dist.get_rank()} decoder tokens={batch.tokens.shape}",
                        flush=True,
                    )
                try:
                    return original(batch)
                except BaseException:
                    traceback.print_exc()
                    raise

            stack.enter_context(patch.object(trainer, "_decoder_hidden", decoder))
            model = trainer.runtime.model[0]
            for index, layer in enumerate(model.decoder.layers):

                def capture(module: Any, args: Any, result: Any, *, name: str) -> None:
                    x = result[0] if isinstance(result, tuple) else result
                    records[name] = x.detach().clone()
                    from art.megatron.gdn import operator

                    state = prepared[-1].attention_state
                    plan = getattr(state, "gdn_execution_plan", None)
                    if plan is not None and world > 1:
                        layout = operator._gdn_runtime_state(state).hidden_layout
                        records[name + ".uids"] = operator._local_layout_token_uids(
                            plan, layout, hidden_states=x, gdn=None
                        )
                    else:
                        records[name + ".uids"] = prepared[-1].token_uids.reshape(-1)

                for suffix, module in (
                    ("layer", layer),
                    ("attention", layer.self_attention),
                    ("moe", layer.mlp),
                ):
                    if os.environ.get("CP_CPU_TRACE"):
                        stack.callback(
                            module.register_forward_pre_hook(
                                lambda m, a, name=f"{index}.{suffix}": print(
                                    f"TRACE rank={dist.get_rank()} enter={name}",
                                    flush=True,
                                )
                            ).remove
                        )
                    stack.callback(
                        module.register_forward_hook(
                            lambda m, a, o, name=f"{index}.{suffix}": capture(
                                m, a, o, name=name
                            )
                        ).remove
                    )
                core = getattr(layer.self_attention, "core_attention", None)
                if core is not None:
                    indexer = getattr(core, "indexer", None)
                    if indexer is not None:

                        def capture_indices(
                            module: Any,
                            args: Any,
                            result: Any,
                            *,
                            prefix: str = str(index),
                        ) -> None:
                            state = args[2]
                            ids = (
                                result
                                if isinstance(result, torch.Tensor)
                                else result.indices
                            )[0].long()
                            if state.context_parallel_state is not None:
                                route = state.combined_global_ids
                                ids = torch.where(
                                    ids < route.numel(),
                                    route[ids.clamp(max=max(route.numel() - 1, 0))],
                                    -1,
                                )
                            else:
                                ids = torch.where(
                                    ids < state.position_ids.shape[1], ids, -1
                                )
                            records[prefix + ".indices"] = ids[:, None].clone()
                            records[prefix + ".indices.uids"] = prepared[
                                -1
                            ].token_uids.reshape(-1)

                        stack.callback(
                            indexer.register_forward_hook(capture_indices).remove
                        )

                    def capture_qkv(
                        module: Any, args: Any, kwargs: Any, *, prefix: str = str(index)
                    ) -> None:
                        for i, name in enumerate(("query", "key", "value")):
                            value = args[i] if len(args) > i else kwargs[name]
                            records[prefix + "." + name] = value.detach().clone()

                    stack.callback(
                        core.register_forward_pre_hook(
                            capture_qkv, with_kwargs=True
                        ).remove
                    )
            requests = []
            for i, length in enumerate(lengths):
                ids = torch.randint(
                    0,
                    vocab,
                    (length + 1,),
                    generator=torch.Generator().manual_seed(1234 + i),
                )
                requests.append(
                    ForwardInput(input_tokens=ids[:-1], target_tokens=ids[1:])
                )
            print(f"START cp={world} lengths={lengths}", flush=True)
            values: list[Any] = [None] * len(requests)
            pieces: dict[str, list[Any]] = {}
            doc_offsets = [sum(lengths[:i]) for i in range(len(lengths))]
            for batch in trainer.forward_micro_batches(
                [requests] if model_name == "qwen" else requests, no_grad=False
            ):
                local = prepared[-1]
                canonical = torch.full((sum(lengths) + 1024,), -1, dtype=torch.long)
                indices = (
                    range(len(requests)) if model_name == "qwen" else batch.indices
                )
                outputs = batch.outputs[0] if model_name == "qwen" else batch.outputs
                for index, positions, sources, value in zip(
                    indices,
                    local.positions_by_item,
                    local.source_positions_by_item,
                    outputs,
                ):
                    values[index] = value.target_logprobs.detach()
                    canonical[local.token_uids.reshape(-1)[positions]] = (
                        sources + doc_offsets[index]
                    )
                if world > 1:
                    dist.all_reduce(canonical, op=dist.ReduceOp.MAX)
                records["token_uids"] = local.token_uids.reshape(-1)
                for name, value in records.items():
                    if name.endswith("uids"):
                        value = torch.where(
                            value >= 0, canonical[value.clamp_min(0)], value
                        )
                    pieces.setdefault(name, []).append(value)
                records.clear()
            records.update({name: torch.cat(parts) for name, parts in pieces.items()})
            records["logprobs"] = torch.cat(values)
            records["lengths"] = lengths
            records["mask_audits"] = MASK_AUDITS
            records["prepared_packs"] = [
                int(batch.tokens.numel()) for batch in prepared
            ]
            records["architecture"] = {
                name: type(module).__name__
                for name, module in model.named_modules()
                if name.endswith("self_attention")
            }
            records["dsa_metadata"] = metadata
            torch.save(records, output / f"rank{dist.get_rank()}.pt")
            assert not torch.cuda.is_initialized()
            print(
                f"DONE cp={world} rank={dist.get_rank()} loss={-records['logprobs'].mean().item()}",
                flush=True,
            )
    finally:
        faulthandler.cancel_dump_traceback_later()
        ps.destroy_model_parallel()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[503, 775, 1001, 1907, 4099]
    )
    parser.add_argument("--model", choices=("qwen", "glm"), default="qwen")
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.lengths, model_name=args.model, expert_parallel=args.ep)


if __name__ == "__main__":
    main()
