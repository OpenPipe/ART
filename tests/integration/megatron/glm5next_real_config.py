"""Weight-free rehearsal of the GPU worker's real native configuration path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_layer import MoELayer
import torch
import transformers

from art.megatron.glm5next.bridge import ArtGlm5NextBridge
from art.megatron.glm5next.dsa import Glm5NextSelfAttention
from art.megatron.glm5next.kda import KimiDeltaAttention
from art.megatron.glm5next.kpool_indexer import KPoolIndexer
from art.megatron.glm5next.layer import Glm5NextRouter, Glm5NextTransformerLayer
from art.megatron.provider import (
    _build_provider_bundle,
    finalize_provider_bundle,
    prepare_provider_bundle,
)
from tests.integration.megatron.glm5next_qualification import (
    configure_qualification_provider,
)


def check_real_config(path: Path) -> dict[str, Any]:
    assert transformers.__version__ == "5.12.1", transformers.__version__
    assert not torch.cuda.is_initialized()
    raw = path.read_bytes()
    config = json.loads(raw)
    cfg = config.get("text_config", config)
    expected_dsa = [
        i for i, kind in enumerate(cfg["layer_types"]) if kind != "linear_attention"
    ]
    if cfg["num_hidden_layers"] == 45:
        assert expected_dsa == list(range(3, 45, 4))
    # Only config.json exists: the pretrained path must remain lazy, with no
    # checkpoint, tokenizer, vision model, or HF inference implementation.
    with tempfile.TemporaryDirectory(prefix="m4-real-config-") as directory:
        local = Path(directory)
        (local / "config.json").write_bytes(raw)
        # R2 is M1's unaligned CPU oracle (H=32, expert=16). Exercise the
        # identical native config/Bridge builder without ART's GPU CUTLASS
        # admission, which intentionally requires 128-aligned dimensions.
        production_geometry = (
            cfg["hidden_size"] % 128 == 0 and cfg["moe_intermediate_size"] % 128 == 0
        )
        builder = (
            prepare_provider_bundle if production_geometry else _build_provider_bundle
        )
        bundle = builder(
            str(local),
            torch_dtype=torch.bfloat16,
            model_support_key="glm5next",
            allow_unvalidated_arch=True,
            load_weights=True,
        )
        assert isinstance(bundle.bridge._model_bridge, ArtGlm5NextBridge)
        provider = bundle.provider
        if not production_geometry:
            provider._art_model_support_spec = bundle.spec
            bundle.handler.patch_provider(provider, bundle.bridge)
        configure_qualification_provider(provider)
        if production_geometry:
            finalize_provider_bundle(bundle)
        else:
            provider.finalize()
        # Distributed rank discovery is the only unavailable CPU process state.
        # All native spec builders and module choices are unchanged.
        with (
            patch.object(
                parallel_state, "get_pipeline_model_parallel_rank", return_value=0
            ),
            torch.device("meta"),
        ):
            block = provider.transformer_layer_spec(provider)
            indexer = KPoolIndexer(
                hidden_size=provider.hidden_size,
                q_lora_rank=provider.q_lora_rank,
                num_heads=provider.dsa_indexer_n_heads,
                head_dim=provider.dsa_indexer_head_dim,
                topk=provider.dsa_indexer_topk,
                pool_size=provider.dsa_indexer_kpool,
            )
            streams = torch.empty(4, 1, provider.dsv4_hc_mult, provider.hidden_size)
        assert streams.is_meta and all(p.is_meta for p in indexer.parameters())
        assert tuple(indexer.index_kpool_compress_ape.shape) == (
            cfg["index_kpool"],
            cfg["index_head_dim"],
        )
        assert tuple(indexer.index_kpool_compress_gate.shape) == (
            cfg["index_head_dim"],
            cfg["hidden_size"],
        )
        layers = block.layer_specs
        assert (
            len(layers)
            == len({id(layer) for layer in layers})
            == cfg["num_hidden_layers"]
        )
        actual_dsa = []
        for i, layer in enumerate(layers):
            assert layer.module is Glm5NextTransformerLayer
            modules = layer.submodules
            attention = modules.self_attention.module
            assert attention is (
                Glm5NextSelfAttention if i in expected_dsa else KimiDeltaAttention
            )
            if attention is Glm5NextSelfAttention:
                actual_dsa.append(i)
            assert (modules.mlp.module is MoELayer) == (
                cfg["mlp_layer_types"][i] == "sparse"
            )
            if modules.mlp.module is MoELayer:
                assert modules.mlp.submodules.router is Glm5NextRouter
        families = bundle.handler.collect_layer_families(provider)
        assert [
            f.layer_index for f in families if f.key == "nope_dsa_attention"
        ] == expected_dsa[:1]  # Handler returns one representative per family.
        assert actual_dsa == expected_dsa
        original = provider.glm5next_hf_config
        assert original.art_layer_types == cfg["layer_types"]
        assert original.layer_types == [
            "full_attention" if kind == "indexed_attention" else kind
            for kind in cfg["layer_types"]
        ]
        for field in (
            "index_kpool",
            "index_kpool_compress",
            "index_kpool_always_select_tail",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "swiglu_limit",
            "moe_router_dtype",
        ):
            assert getattr(original, field) == cfg[field]
        assert provider.dsa_indexer_kpool == cfg["index_kpool"]
        assert provider.dsa_indexer_kpool_compress == cfg["index_kpool_compress"]
        assert (
            provider.dsa_indexer_kpool_always_select_tail
            == cfg["index_kpool_always_select_tail"]
        )
        assert provider.dsv4_hc_mult == cfg["hc_mult"]
        assert provider.dsv4_hc_sinkhorn_iters == cfg["hc_sinkhorn_iters"]
        assert provider.dsv4_hc_eps == cfg["hc_eps"]
        assert provider.activation_func_clamp_value == cfg["swiglu_limit"]
        assert provider.moe_router_dtype == "fp32"
        assert provider.kv_channels == cfg["v_head_dim"]  # head_dim=0 sentinel
        assert provider.mtp_num_layers is None
        assert provider.position_embedding_type == "none"
    assert path.read_bytes() == raw and not torch.cuda.is_initialized()
    return {
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "layers": len(layers),
        "dsa_layers": actual_dsa,
        "router_dtype": provider.moe_router_dtype,
        "swiglu_limit": provider.activation_func_clamp_value,
        "kpool": provider.dsa_indexer_kpool,
        "mhc": [
            provider.dsv4_hc_mult,
            provider.dsv4_hc_sinkhorn_iters,
            provider.dsv4_hc_eps,
        ],
        "spec_device": "meta",
        "path": "AutoBridge.from_hf_pretrained/config shim/handler/provider.finalize/native spec",
        "gpu_runtime_admission": production_geometry,
        "cuda_initialized": False,
        "weights_loaded": False,
        "status": "pass",
    }


def check_real_configs(root: Path) -> dict[str, Any]:
    return {
        name: check_real_config(root / name / "config.json")
        for name in ("full", "F4", "F8", "R2")
    }
