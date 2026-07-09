from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (  # noqa: E402
    Qwen3_5MoeVisionConfig,
    Qwen35VLMoEModelProvider,
)
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # noqa: E402
from megatron.core.tensor_parallel.random import (  # noqa: E402
    model_parallel_cuda_manual_seed,
)
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.megatron.lora import apply_lora_adapters  # noqa: E402
from art.megatron.model_support import QWEN3_5_MOE_SPEC  # noqa: E402
from art.megatron.model_support.handlers import QWEN3_5_MOE_HANDLER  # noqa: E402

from .cases import GdnPhase0Case, default_phase0_cases  # noqa: E402
from .distributed_init import file_init_method  # noqa: E402
from .metrics import GDN_CORRECTNESS_DTYPE, assert_real_gdn_metrics  # noqa: E402
from .packed_layout import build_phase0_packed_tensors  # noqa: E402
from .real_gdn_oracle import (  # noqa: E402
    attach_main_grads,
    compare_real_gdn_cp1_to_flattened,
    zero_parameter_grads,
)
from .test_real_gdn_cp1_packed_vs_flattened import (  # noqa: E402
    _single_rank_model_parallel,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN LoRA coverage.",
)
def test_real_qwen35_gdn_lora_gradients_match_flattened() -> None:
    case = next(
        case
        for case in default_phase0_cases(conv_width=2)
        if case.name == "ragged_family_mix"
    )
    with _single_rank_model_parallel():
        packed_gdn, flat_gdn = _make_matching_gdn_pair(tp_size=1, lora=True)
        tensors = build_phase0_packed_tensors(case)
        metrics = compare_real_gdn_cp1_to_flattened(
            packed_gdn=packed_gdn,
            flat_gdn=flat_gdn,
            hidden_states=_hidden(case),
            group_ids=tensors["group_ids"].cuda(),
            parent_ids=tensors["parent_ids"].cuda(),
            assistant_mask=tensors["assistant_mask"].cuda(),
        )
        assert_real_gdn_metrics(metrics, "lora")
        assert _gdn_lora_grad_names(packed_gdn)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="At least two CUDA devices are required for TP2 GDN coverage.",
)
def test_real_qwen35_gdn_tp2_gradients_match_flattened(tmp_path: Path) -> None:
    init_method = file_init_method(tmp_path, "real_gdn_tp2_lora")
    mp.spawn(
        _tp2_worker,
        args=(init_method, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        assert (tmp_path / f"rank_{rank}.ok").read_text() == "ok\n"


def _tp2_worker(rank: int, init_method: str, output_dir: str) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        case = next(
            case
            for case in default_phase0_cases(conv_width=2)
            if case.name == "multi_family_repeated"
        )
        packed_gdn, flat_gdn = _make_matching_gdn_pair(tp_size=2, lora=False)
        tensors = build_phase0_packed_tensors(case)
        metrics = compare_real_gdn_cp1_to_flattened(
            packed_gdn=packed_gdn,
            flat_gdn=flat_gdn,
            hidden_states=_hidden(case, seed=20410426 + rank),
            group_ids=tensors["group_ids"].cuda(),
            parent_ids=tensors["parent_ids"].cuda(),
            assistant_mask=tensors["assistant_mask"].cuda(),
        )
        assert_real_gdn_metrics(metrics, "tp2")
        Path(output_dir, f"rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _make_matching_gdn_pair(
    *, tp_size: int, lora: bool
) -> tuple[GatedDeltaNet, GatedDeltaNet]:
    model_parallel_cuda_manual_seed(1234)
    packed_model = _make_model(tp_size=tp_size)
    model_parallel_cuda_manual_seed(5678)
    flat_model = _make_model(tp_size=tp_size)
    if lora:
        apply_lora_adapters([packed_model], _make_provider(tp_size=tp_size))
        apply_lora_adapters([flat_model], _make_provider(tp_size=tp_size))
        _randomize_lora_parameters(packed_model)
    flat_model.load_state_dict(packed_model.state_dict())
    packed_gdn = _first_gdn(packed_model)
    flat_gdn = _first_gdn(flat_model)
    attach_main_grads(packed_gdn)
    attach_main_grads(flat_gdn)
    zero_parameter_grads(packed_gdn)
    zero_parameter_grads(flat_gdn)
    return packed_gdn, flat_gdn


def _make_model(*, tp_size: int) -> torch.nn.Module:
    return (
        _make_provider(tp_size=tp_size)
        .provide_language_model(pre_process=True, post_process=True)
        .cuda()
    )


def _make_provider(*, tp_size: int) -> Qwen35VLMoEModelProvider:
    assert Qwen3_5MoeVisionConfig is not None
    provider = Qwen35VLMoEModelProvider(
        num_layers=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=16,
        num_attention_heads=4,
        num_query_groups=tp_size,
        kv_channels=16,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_output_gate=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=2,
        vocab_size=128,
        seq_length=128,
        position_embedding_type="mrope",
        vision_config=Qwen3_5MoeVisionConfig(),
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=GDN_CORRECTNESS_DTYPE,
    )
    provider.finalize()
    setattr(provider, "_art_model_support_handler", QWEN3_5_MOE_HANDLER)
    setattr(provider, "_art_model_support_spec", QWEN3_5_MOE_SPEC)
    return provider


def _first_gdn(model: torch.nn.Module) -> GatedDeltaNet:
    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            return module
    raise AssertionError("expected Qwen3.5 provider to build a GDN layer")


def _hidden(case: GdnPhase0Case, seed: int = 20400426) -> torch.Tensor:
    return torch.randn(
        case.sequence_length,
        len(case.rows),
        64,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )


def _randomize_lora_parameters(model: torch.nn.Module) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20420426)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("A_T", "B_T")):
                parameter.copy_(
                    torch.randn(
                        parameter.shape, device=parameter.device, generator=generator
                    )
                    * 0.03
                )


def _gdn_lora_grad_names(gdn: torch.nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in gdn.named_parameters()
        if name.endswith(("A_T", "B_T"))
        and parameter.grad is not None
        and bool(parameter.grad.abs().max().item() > 0)
    )
