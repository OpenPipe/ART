from __future__ import annotations

from collections.abc import Mapping
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

FIXTURE_PATH_ENV = "ART_MODEL_SUPPORT_FIXTURE_PATH"
FIXTURE_CACHE_ENV = "ART_MODEL_SUPPORT_FIXTURE_CACHE"
FIXTURE_ROOT_ENV = "ART_MODEL_SUPPORT_FIXTURE_ROOT"
FIXTURE_VERSION = 18
_MODEL_FIXTURE_VERSION_OFFSETS = {"nemotron_h_moe": 7}
_CANONICAL_CACHE_VERSION = 16
_ROOT = Path("/tmp/art-models/main-merge-oracle")
_CACHE_ROOT = Path("/tmp/art-model-support-workflow/hf-cache")
_TOKENIZER_FIXTURE_ROOT = Path("/tmp/art-model-support-workflow/tokenizer-compatible")
_TOKENIZER_CACHE_ROOT = Path("/tmp/art-model-support-workflow/tokenizer-hf-cache")
_CANONICAL_CACHE_ROOT = Path("/tmp/art-model-support-workflow/canonical-hf-cache")
_FUNCTIONAL_FIXTURE_ROOT = Path("/tmp/art-model-support-workflow/functional")
_FUNCTIONAL_CACHE_ROOT = Path("/tmp/art-model-support-workflow/functional-hf-cache")
_GEMMA_CANONICAL_WEIGHT_STAGES = frozenset({"hf_parity", "packing_invariance"})
_PRETRAINED_WEIGHT_STAGES = frozenset({"length_trainability"})
_FUNCTIONAL_STAGES = frozenset(
    {
        "train_inf_mismatch",
    }
)
_RESIDENT_FUNCTIONAL_ENV = {
    "gemma4_dense": {"ART_MODEL_SUPPORT_LENGTH_MAX_MODEL_LEN": "2560"},
    "gemma4_moe": {"ART_MODEL_SUPPORT_LENGTH_MAX_MODEL_LEN": "2560"},
}
_REDUCED_TRAINABILITY_ENV: dict[str, dict[str, dict[str, str]]] = {
    "glm52": {
        "length_trainability": {
            "ART_MODEL_SUPPORT_LENGTH_ALLOWED_TOKEN_IDS": "154820,38069",
            "ART_MODEL_SUPPORT_LENGTH_MIN_TOKENS": "2",
            "ART_MODEL_SUPPORT_LENGTH_FREQUENCY_PENALTY": "0.5",
        }
    },
}
_TOKENIZER_FIXTURE_VERSION = 3
_FUNCTIONAL_FIXTURE_VERSION = 1
_FUNCTIONAL_FIXTURE_VERSION_OFFSETS = {"nemotron_h_moe": 5}
_FUNCTIONAL_REMOTE_CODE_FILES = {
    "nemotron_h_moe": (
        "configuration_nemotron_h.py",
        "modeling_nemotron_h.py",
    )
}
_REVISIONS = {
    "meta-llama/Llama-3.2-1B-Instruct": "9213176726f574b556790deb65791e0c5aa438b6",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen/Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "deepseek-ai/DeepSeek-V4-Flash": "60d8d70770c6776ff598c94bb586a859a38244f1",
    "zai-org/GLM-5.2": "b4734de4facf877f85769a911abafc5283eab3d9",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": (
        "2d59de1cbd51c0adf384eb906b766d1aee0e0517"
    ),
}
_MULTIMODAL = {"qwen3_5_dense", "qwen3_5_moe", "gemma4_dense", "gemma4_moe"}


class WorkflowFixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_model: str
    model_key: str
    source_revision: str
    path: str
    hf_home: str
    manifest: dict[str, object]
    tokenizer_compatible_path: str | None = None
    tokenizer_compatible_hf_home: str | None = None
    tokenizer_compatible_manifest: dict[str, object] | None = None
    functional_path: str | None = None
    functional_hf_home: str | None = None
    functional_manifest: dict[str, object] | None = None
    canonical_path: str | None = None
    canonical_hf_home: str | None = None

    def environment(self, stage_name: str | None = None) -> dict[str, str]:
        reduced_trainability = _REDUCED_TRAINABILITY_ENV.get(self.model_key, {}).get(
            stage_name
        )
        use_functional = stage_name in _FUNCTIONAL_STAGES
        use_canonical = (
            stage_name in _PRETRAINED_WEIGHT_STAGES and reduced_trainability is None
        ) or (
            self.model_key.startswith("gemma4_")
            and stage_name in _GEMMA_CANONICAL_WEIGHT_STAGES
        )
        use_tokenizer_compatible = (
            self.model_key.startswith("gemma4_") and reduced_trainability is not None
        )
        path = (
            self.functional_path
            if use_functional
            else self.canonical_path
            if use_canonical
            else self.tokenizer_compatible_path
            if use_tokenizer_compatible
            else self.path
        )
        hf_home = (
            self.functional_hf_home
            if use_functional
            else self.canonical_hf_home
            if use_canonical
            else self.tokenizer_compatible_hf_home
            if use_tokenizer_compatible
            else self.hf_home
        )
        if path is None or hf_home is None:
            contract = (
                "pretrained production-width functional weights"
                if use_functional
                else "canonical weights"
                if use_canonical
                else "canonical vocabulary"
            )
            raise RuntimeError(f"{self.model_key} {stage_name} requires {contract}")
        hub = str(Path(hf_home) / "hub")
        environment = {
            FIXTURE_PATH_ENV: path,
            FIXTURE_CACHE_ENV: hf_home,
            "ART_ORACLE_BASE_MODEL": path,
            "HF_HOME": hf_home,
            "HF_HUB_CACHE": hub,
            "HUGGINGFACE_HUB_CACHE": hub,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        if use_functional:
            assert self.functional_manifest is not None
            num_layers = self.functional_manifest.get("num_layers")
            if type(num_layers) is not int:
                raise RuntimeError(f"{self.model_key} functional depth is invalid")
            environment["ART_MODEL_SUPPORT_FUNCTIONAL_NUM_LAYERS"] = str(num_layers)
        if reduced_trainability is not None:
            environment.update(reduced_trainability)
        return environment

    def resident_functional_environment(self) -> dict[str, str]:
        environment = self.environment("length_trainability")
        environment.update(_RESIDENT_FUNCTIONAL_ENV.get(self.model_key, {}))
        if self.model_key == "glm52":
            environment.update(self.environment("train_inf_mismatch"))
        return environment


def _set(config: Any, **values: Any) -> Any:
    for name, value in values.items():
        setattr(config, name, value)
    return config


def _text(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _common(
    config: Any,
    *,
    layers: int,
    hidden: int,
    vocab_size: int,
    preserve_token_ids: bool,
) -> Any:
    text = _text(config)
    for name in ("layer_types", "mlp_layer_types", "indexer_types"):
        if (values := getattr(text, name, None)) is not None:
            setattr(text, name, list(values[:layers]))
    values = {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": vocab_size,
    }
    if not preserve_token_ids:
        values.update(pad_token_id=0, bos_token_id=2, eos_token_id=1)
    return _set(
        text,
        **values,
    )


# fmt: off
_DENSE_TEXT = {
    "intermediate_size": 512, "num_attention_heads": 8,
    "num_key_value_heads": 2, "head_dim": 32,
    "tie_word_embeddings": False,
}
_PLAIN_TEXT: dict[str, tuple[int, int, dict[str, Any]]] = {
    "llama3_dense": (4, 256, _DENSE_TEXT),
    "qwen3_dense": (4, 256, _DENSE_TEXT),
    "qwen3_moe": (
        4,
        256,
        {
            **_DENSE_TEXT, "moe_intermediate_size": 256,
            "num_experts": 4, "num_local_experts": 4,
            "num_experts_per_tok": 2, "quantization_config": None,
        },
    ),
    "glm52": (
        12,
        512,
        {
            "intermediate_size": 1024, "moe_intermediate_size": 256,
            "layer_types": ["deepseek_sparse_attention"] * 12,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 9,
            "indexer_types": ["full"] * 3
            + ["shared", "shared", "shared", "full"]
            + ["shared", "shared", "shared", "full", "shared"],
            "num_attention_heads": 64, "num_key_value_heads": 64,
            "q_lora_rank": 512, "qk_head_dim": 256,
            "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
            "v_head_dim": 256, "index_n_heads": 32, "index_topk": 128,
            "n_routed_experts": 4, "num_experts": 4, "num_local_experts": 4,
            "num_experts_per_tok": 2, "num_nextn_predict_layers": 0,
            "tie_word_embeddings": False, "quantization_config": None,
        },
    ),
    "gpt_oss_moe": (
        4,
        320,
        {
            "intermediate_size": 768,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "head_dim": 64, "num_attention_heads": 4, "num_key_value_heads": 1,
            "num_experts": 4, "num_local_experts": 4,
            "num_experts_per_tok": 2, "experts_per_token": 2,
            "initial_context_length": 2048, "sliding_window": 128,
            "tie_word_embeddings": False, "quantization_config": None,
        },
    ),
    "nemotron_h_moe": (
        6,
        256,
        {
            "intermediate_size": 512,
            "hybrid_override_pattern": "MEMEM*",
            "head_dim": 32,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "mamba_num_heads": 8,
            "mamba_head_dim": 64,
            "n_groups": 2,
            "ssm_state_size": 128,
            "moe_intermediate_size": 256,
            "moe_shared_expert_intermediate_size": 512,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": False,
        },
    ),
}
_QWEN35_TEXT = {
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
    "intermediate_size": 512, "head_dim": 256,
    "num_attention_heads": 4, "num_key_value_heads": 1,
    "full_attention_interval": 4, "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128, "linear_num_key_heads": 4,
    "linear_num_value_heads": 8, "linear_value_head_dim": 128,
    "tie_word_embeddings": False,
}
_QWEN35_VISION = {
    "depth": 1, "num_hidden_layers": 1,
    "hidden_size": 128, "intermediate_size": 256,
    "num_heads": 4, "num_attention_heads": 4,
    "num_position_embeddings": 16, "out_hidden_size": 1024,
    "deepstack_visual_indexes": [],
}
_GEMMA_TEXT = {
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 2,
    "intermediate_size": 512, "head_dim": 256, "global_head_dim": 512,
    "num_attention_heads": 4, "num_key_value_heads": 2,
    "num_global_key_value_heads": 1, "num_kv_shared_layers": 0,
    "sliding_window": 1024,
    "hidden_size_per_layer_input": 0,
    "tie_word_embeddings": True,
}
_GEMMA_VISION = {
    "depth": 1, "num_hidden_layers": 1,
    "hidden_size": 128, "intermediate_size": 256,
    "head_dim": 32, "global_head_dim": 32,
    "num_attention_heads": 4, "num_key_value_heads": 4,
    "patch_size": 16, "position_embedding_size": 64,
}
_MULTIMODAL_SHAPES = {
    "qwen3_5": (
        8,
        _QWEN35_TEXT,
        _QWEN35_VISION,
        {
            "moe_intermediate_size": 256, "shared_expert_intermediate_size": 256,
            "num_experts": 4, "num_local_experts": 4, "num_experts_per_tok": 2,
        },
        {
            "image_token_id": 2, "video_token_id": 3,
            "vision_start_token_id": 4, "vision_end_token_id": 5,
        },
    ),
    "gemma4": (
        12,
        _GEMMA_TEXT,
        _GEMMA_VISION,
        {
            "moe_intermediate_size": 256, "num_experts": 4,
            "num_local_experts": 4, "top_k_experts": 2, "num_experts_per_tok": 2,
        },
        {"image_token_id": 2, "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1},
    ),
}
# fmt: on

_FUNCTIONAL_LAYER_FIELDS = (
    "layer_types",
    "layers_block_type",
    "mlp_layer_types",
    "indexer_types",
)
_HYBRID_LAYER_SYMBOLS = {
    "mamba": "M",
    "moe": "E",
    "attention": "*",
    "mlp": "-",
}
_WIDTH_TERMS = ("hidden", "intermediate", "head", "expert", "lora_rank", "topk")


class _FunctionalPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    depth: int
    prefix: str
    config_key: str | None = None
    checkpoint: str = "model.safetensors.index.json"
    vision: tuple[str, str] | None = None
    auxiliary: tuple[str | None, str] | None = None


def _plan(depth: int, prefix: str, **values: Any) -> _FunctionalPlan:
    return _FunctionalPlan(depth=depth, prefix=prefix, **values)


# fmt: off
_QWEN35 = {"layer_types": (("linear_attention",) * 3 + ("full_attention",)) * 2}
_GEMMA4 = {"layer_types": (("sliding_attention",) * 5 + ("full_attention",)) * 2}
_FUNCTIONAL_PLANS = {
    "llama3_dense": _plan(2, "model.layers", checkpoint="model.safetensors"),
    "qwen3_dense": _plan(2, "model.layers"),
    "qwen3_moe": _plan(2, "model.layers"),
    "qwen3_5_dense": _plan(8, "model.language_model.layers", config_key="text_config", vision=("model.visual.blocks", "depth")),
    "qwen3_5_moe": _plan(8, "model.language_model.layers", config_key="text_config", vision=("model.visual.blocks", "depth")),
    "gemma4_dense": _plan(12, "model.language_model.layers", config_key="text_config", vision=("model.vision_tower.encoder.layers", "num_hidden_layers")),
    "gemma4_moe": _plan(12, "model.language_model.layers", config_key="text_config", vision=("model.vision_tower.encoder.layers", "num_hidden_layers")),
    "dsv4": _plan(6, "layers", auxiliary=("mtp", "num_nextn_predict_layers")),
    "glm52": _plan(10, "model.layers", auxiliary=(None, "num_nextn_predict_layers")),
    "gpt_oss_moe": _plan(4, "model.layers"),
    "nemotron_h_moe": _plan(
        6,
        "backbone.layers",
        auxiliary=(None, "num_nextn_predict_layers"),
    ),
}
_FUNCTIONAL_PATTERNS = {
    "qwen3_dense": {"layer_types": ("full_attention",) * 2},
    "qwen3_5_dense": _QWEN35, "qwen3_5_moe": _QWEN35,
    "gemma4_dense": _GEMMA4, "gemma4_moe": _GEMMA4,
    "dsv4": {
        "layer_types": ("sliding_attention", "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention", "compressed_sparse_attention", "heavily_compressed_attention"),
        "mlp_layer_types": ("hash_moe",) * 3 + ("moe",) * 3,
    },
    "glm52": {
        "mlp_layer_types": ("dense",) * 3 + ("sparse",) * 7,
        "indexer_types": ("full",) * 3 + ("shared",) * 3 + ("full",) + ("shared",) * 3,
    },
    "gpt_oss_moe": {"layer_types": ("sliding_attention", "full_attention") * 2},
    "nemotron_h_moe": {"hybrid_override_pattern": "MEMEM*"},
}
# fmt: on


def _configure(
    model_key: str,
    config: Any,
    *,
    source_vocab_size: int,
    tokenizer_compatible: bool,
) -> Any:
    common = {
        "vocab_size": source_vocab_size if tokenizer_compatible else 8192,
        "preserve_token_ids": tokenizer_compatible,
    }
    if model_key in _PLAIN_TEXT:
        layers, hidden, values = _PLAIN_TEXT[model_key]
        text = _set(_common(config, layers=layers, hidden=hidden, **common), **values)
        if model_key == "glm52":
            text.vocab_size = source_vocab_size
        return config
    family = model_key.rsplit("_", 1)[0]
    if family in _MULTIMODAL_SHAPES:
        moe = model_key.endswith("_moe")
        layers, text_shape, vision_shape, moe_shape, token_ids = _MULTIMODAL_SHAPES[
            family
        ]
        text = _set(_common(config, layers=layers, hidden=1024, **common), **text_shape)
        top_level = {"tie_word_embeddings": True} if family == "gemma4" else {}
        if family == "gemma4":
            _set(
                text,
                enable_moe_block=moe,
                vocab_size_per_layer_input=common["vocab_size"],
            )
        if moe:
            _set(text, **moe_shape)
        _set(config.vision_config, **vision_shape)
        if not tokenizer_compatible:
            top_level.update(token_ids)
        return _set(config, **top_level)
    if model_key == "dsv4":
        return _set(
            config,
            num_hidden_layers=4,
            compress_ratios=[0, 0, 4, 128],
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            mlp_layer_types=["moe"] * 4,
        )
    raise KeyError(f"No correctness fixture for {model_key}")


def _pack_qwen35_experts(path: Path, config: Any) -> None:
    from safetensors.torch import load_file, save_file
    import torch

    checkpoint = path / "model.safetensors"
    tensors = load_file(checkpoint)
    text = _text(config)
    for layer in range(text.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        gate_up, down = [], []
        for expert in range(text.num_experts):
            expert_prefix = f"{prefix}.{expert}"
            gate_up.append(
                torch.cat(
                    (
                        tensors.pop(f"{expert_prefix}.gate_proj.weight"),
                        tensors.pop(f"{expert_prefix}.up_proj.weight"),
                    )
                )
            )
            down.append(tensors.pop(f"{expert_prefix}.down_proj.weight"))
        tensors[f"{prefix}.gate_up_proj"] = torch.stack(gate_up)
        tensors[f"{prefix}.down_proj"] = torch.stack(down)
    save_file(tensors, checkpoint, metadata={"format": "pt"})


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    return _json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def _functional_plan(model_key: str) -> _FunctionalPlan:
    try:
        return _FUNCTIONAL_PLANS[model_key]
    except KeyError:
        raise RuntimeError(
            f"no pretrained functional fixture for {model_key}"
        ) from None


def _config_text(config: dict[str, Any], plan: _FunctionalPlan) -> dict[str, Any]:
    value = config if plan.config_key is None else config.get(plan.config_key)
    if not isinstance(value, dict):
        raise RuntimeError(f"functional fixture lacks {plan.config_key} config")
    return value


def _config_shape(config: dict[str, Any], *exclude: str) -> dict[str, object]:
    values = {key: value for key, value in config.items() if key not in exclude}
    return {
        "dimensions": {
            key: value
            for key, value in values.items()
            if type(value) is int and any(term in key for term in _WIDTH_TERMS)
        },
        "sha256": _json_sha256(values),
    }


def _functional_config(
    source: dict[str, Any], *, model_key: str
) -> tuple[dict[str, Any], dict[str, object]]:
    plan = _functional_plan(model_key)
    text = _config_text(source, plan)
    source_depth = text.get("num_hidden_layers")
    if type(source_depth) is not int or source_depth < plan.depth:
        raise RuntimeError(f"{model_key} has invalid production depth")
    reduced = json.loads(json.dumps(source))
    reduced_text = _config_text(reduced, plan)
    reduced_text["num_hidden_layers"] = plan.depth
    if plan.auxiliary:
        _, count_field = plan.auxiliary
        count = text.get(count_field)
        if type(count) is not int or count < 0:
            raise RuntimeError(f"{model_key} has invalid {count_field}")
        reduced_text[count_field] = 0
    patterns: dict[str, object] = {}
    for field in _FUNCTIONAL_LAYER_FIELDS:
        if (values := text.get(field)) is not None:
            if not isinstance(values, list) or len(values) != source_depth:
                raise RuntimeError(f"{model_key} production {field} is incomplete")
            patterns[field] = values[: plan.depth]
            reduced_text[field] = patterns[field]
    serialized_hybrid_pattern = text.get("hybrid_override_pattern")
    hybrid_pattern = serialized_hybrid_pattern
    if (layer_types := text.get("layers_block_type")) is not None:
        try:
            layer_types_pattern = "".join(
                _HYBRID_LAYER_SYMBOLS[layer_type] for layer_type in layer_types
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"{model_key} production layers_block_type is invalid"
            ) from exc
        if hybrid_pattern is not None and hybrid_pattern != layer_types_pattern:
            raise RuntimeError(f"{model_key} production hybrid patterns disagree")
        hybrid_pattern = layer_types_pattern
    if hybrid_pattern is not None:
        if not isinstance(hybrid_pattern, str) or len(hybrid_pattern) != source_depth:
            raise RuntimeError(f"{model_key} production hybrid pattern is invalid")
        patterns["hybrid_override_pattern"] = hybrid_pattern[: plan.depth]
        if serialized_hybrid_pattern is not None:
            reduced_text["hybrid_override_pattern"] = hybrid_pattern[: plan.depth]
    for field, expected in _FUNCTIONAL_PATTERNS.get(model_key, {}).items():
        actual = patterns.get(field)
        if (tuple(actual) if isinstance(actual, list) else actual) != expected:
            raise RuntimeError(f"{model_key} production {field} pattern changed")

    shape_exclusions = (
        "num_hidden_layers",
        *_FUNCTIONAL_LAYER_FIELDS,
        "hybrid_override_pattern",
        *((plan.auxiliary[1],) if plan.auxiliary else ()),
    )
    width = {"text": _config_shape(text, *shape_exclusions)}
    if width["text"] != _config_shape(reduced_text, *shape_exclusions):
        raise RuntimeError(f"{model_key} functional fixture changed text width")
    vision_policy: object = "not_applicable"
    if plan.vision:
        _, depth_field = plan.vision
        vision, reduced_vision = (
            source.get("vision_config"),
            reduced.get("vision_config"),
        )
        if not isinstance(vision, dict) or not isinstance(reduced_vision, dict):
            raise RuntimeError(f"{model_key} functional fixture lacks vision config")
        source_vision_depth = vision.get(depth_field)
        if type(source_vision_depth) is not int or source_vision_depth < 1:
            raise RuntimeError(f"{model_key} has invalid production vision depth")
        width["vision"] = _config_shape(vision, depth_field)
        reduced_vision[depth_field] = 1
        if width["vision"] != _config_shape(reduced_vision, depth_field):
            raise RuntimeError(f"{model_key} functional fixture changed vision width")
        vision_policy = {
            "mode": "one_pretrained_production_width_layer",
            "source_depth": source_vision_depth,
            "text_path_semantics": "unchanged",
        }
    dtype = text.get("dtype") or text.get("torch_dtype")
    dtype = dtype or source.get("dtype") or source.get("torch_dtype")
    # fmt: off
    return reduced, {
        "source_num_layers": source_depth,
        "selected_layer_sequence": list(range(plan.depth)),
        "selected_layer_patterns": patterns,
        "production_width": width,
        "config_vocab_size": int(text["vocab_size"]),
        "configured_dtype": str(dtype) if dtype else None,
        "inference_quantization": {
            "quantization_config": text.get("quantization_config") or source.get("quantization_config"),
            "expert_dtype": text.get("expert_dtype") or source.get("expert_dtype"),
        },
        "vision_policy": vision_policy,
    }
    # fmt: on


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_weight_map(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    if path.name == "model.safetensors":
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            values: object = dict.fromkeys(checkpoint.keys(), path.name)
    else:
        try:
            values = json.loads(path.read_text())["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid checkpoint index {path}") from exc
    if (
        not isinstance(values, dict)
        or not values
        or any(
            not isinstance(key, str)
            or not isinstance(shard, str)
            or Path(shard).name != shard
            for key, shard in values.items()
        )
    ):
        raise RuntimeError(f"invalid checkpoint weight map {path}")
    return cast(dict[str, str], values)


def _layer_index(name: str, prefix: str) -> int | None:
    if not name.startswith(f"{prefix}."):
        return None
    index, separator, suffix = name[len(prefix) + 1 :].partition(".")
    if not separator or not index.isdecimal() or not suffix:
        raise RuntimeError(f"malformed checkpoint layer key {name!r}")
    return int(index)


def _select_functional_weights(
    weights: Mapping[str, str], config: dict[str, Any], *, model_key: str
) -> dict[str, str]:
    plan = _functional_plan(model_key)
    text_config = _config_text(config, plan)
    source_depth = int(text_config["num_hidden_layers"])
    text_layers: set[int] = set()
    vision_layers: set[int] = set()
    auxiliary_layers: set[int] = set()
    selected: dict[str, str] = {}
    for name, shard in weights.items():
        text_layer = _layer_index(name, plan.prefix)
        vision_layer = (
            _layer_index(name, plan.vision[0])
            if text_layer is None and plan.vision
            else None
        )
        auxiliary_layer = (
            _layer_index(name, plan.auxiliary[0])
            if text_layer is None
            and vision_layer is None
            and plan.auxiliary
            and plan.auxiliary[0]
            else None
        )
        text_layers.update(() if text_layer is None else (text_layer,))
        vision_layers.update(() if vision_layer is None else (vision_layer,))
        auxiliary_layers.update(() if auxiliary_layer is None else (auxiliary_layer,))
        if (
            text_layer is None
            and vision_layer is None
            and auxiliary_layer is None
            or text_layer is not None
            and text_layer < plan.depth
            or vision_layer == 0
        ):
            selected[name] = shard
    auxiliary_prefix, auxiliary_count = plan.auxiliary or (None, None)
    count = text_config.get(auxiliary_count) if auxiliary_count else 0
    if type(count) is not int or count < 0:
        raise RuntimeError(f"{model_key} has invalid {auxiliary_count}")
    expected = set(
        range(source_depth + (count if plan.auxiliary and not auxiliary_prefix else 0))
    )
    if text_layers != expected:
        raise RuntimeError(f"{model_key} canonical text-layer coverage changed")
    if auxiliary_prefix:
        if auxiliary_layers != set(range(count)):
            raise RuntimeError(
                f"{model_key} canonical auxiliary-layer coverage changed"
            )
    if plan.vision:
        vision = config.get("vision_config")
        if not isinstance(vision, dict):
            raise RuntimeError(f"{model_key} canonical vision config is missing")
        if vision_layers != set(range(int(vision[plan.vision[1]]))):
            raise RuntimeError(f"{model_key} canonical vision-layer coverage changed")
    if not selected:
        raise RuntimeError(f"{model_key} functional checkpoint selection is empty")
    return selected


def _fixture_files(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != "fixture_manifest.json"
    }


def _fixture_file_sizes(path: Path) -> dict[str, int]:
    return {
        file.relative_to(path).as_posix(): file.stat().st_size
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != "fixture_manifest.json"
    }


def _checkpoint_is_complete(path: Path) -> bool:
    try:
        if any(
            not file.is_file() or file.stat().st_size == 0
            for file in (path / "config.json", path / "tokenizer_config.json")
        ):
            return False
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            checkpoint = path / "model.safetensors"
            return checkpoint.is_file() and checkpoint.stat().st_size > 0
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = set(weight_map.values())
        return bool(shards) and all(
            isinstance(name, str)
            and Path(name).name == name
            and (path / name).is_file()
            and (path / name).stat().st_size > 0
            for name in shards
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False


def _fixture_namespace(
    *,
    canonical_model: str,
    revision: str,
    model_key: str,
    version: int,
    tokenizer_compatible: bool,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "model": canonical_model,
                "revision": revision,
                "handler": model_key,
                "version": version,
                "tokenizer_compatible": tokenizer_compatible,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]


def _fixture_version(model_key: str) -> int:
    return FIXTURE_VERSION + _MODEL_FIXTURE_VERSION_OFFSETS.get(model_key, 0)


def _functional_fixture_version(model_key: str) -> int:
    return _FUNCTIONAL_FIXTURE_VERSION + _FUNCTIONAL_FIXTURE_VERSION_OFFSETS.get(
        model_key, 0
    )


def _is_current(
    path: Path,
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    tokenizer_compatible: bool,
    parent_manifest_sha256: str | None,
    version: int | None = None,
) -> bool:
    try:
        manifest = json.loads((path / "fixture_manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "version": version
        or (
            _TOKENIZER_FIXTURE_VERSION
            if tokenizer_compatible
            else _fixture_version(model_key)
        ),
        "source_model": canonical_model,
        "source_revision": revision,
        "handler": model_key,
        "seed": 0,
        "source_identity": {"model": canonical_model, "revision": revision},
        "parent_manifest_sha256": parent_manifest_sha256,
    }
    if tokenizer_compatible:
        expected["vocabulary_contract"] = "canonical"
    return (
        _checkpoint_is_complete(path)
        and all(manifest.get(key) == value for key, value in expected.items())
        and (
            manifest.get("file_sizes") == _fixture_file_sizes(path)
            if version is not None
            else manifest.get("files") == _fixture_files(path)
        )
        and (
            "manifest_sha256" not in manifest
            or manifest["manifest_sha256"] == _manifest_sha256(manifest)
        )
    )


def _publish(staging: Path, output: Path) -> None:
    previous = output.with_name(f".{output.name}.previous")
    if previous.exists():
        shutil.rmtree(previous)
    if output.exists():
        os.replace(output, previous)
    try:
        os.replace(staging, output)
    except BaseException:
        if previous.exists():
            os.replace(previous, output)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def _build(
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    output: Path,
    tokenizer_compatible: bool,
    source_fixture: Path | None = None,
    functional: bool = False,
) -> None:
    from safetensors.torch import load_file, save_file
    import torch
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    with tempfile.TemporaryDirectory(prefix=f".{model_key}-", dir=output.parent) as tmp:
        staging = Path(tmp) / model_key
        staging.mkdir()
        source_model = (
            source_fixture / "production_config"
            if source_fixture is not None
            else canonical_model
        )
        source_kwargs = (
            {"local_files_only": True}
            if source_fixture is not None
            else {"revision": revision}
        )
        source = AutoConfig.from_pretrained(
            source_model, trust_remote_code=True, **source_kwargs
        )
        source_config = cast(dict[str, Any], source.to_dict())
        source.save_pretrained(staging / "production_config")
        tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(
                source_fixture or canonical_model,
                trust_remote_code=True,
                **source_kwargs,
            ),
        )
        source_vocab_size = int(_text(source).vocab_size)
        tokenizer_max_id = max(map(int, tokenizer.get_vocab().values()))
        if tokenizer_max_id >= source_vocab_size:
            raise RuntimeError(
                f"{model_key} tokenizer ID {tokenizer_max_id} exceeds canonical "
                f"vocab_size={source_vocab_size}"
            )
        functional_contract: dict[str, object] | None = None
        if functional:
            reduced, functional_contract = _functional_config(
                source_config, model_key=model_key
            )
            config = source
        else:
            config = _configure(
                model_key,
                source,
                source_vocab_size=source_vocab_size,
                tokenizer_compatible=tokenizer_compatible,
            )
        config.save_pretrained(staging)
        if functional:
            (staging / "config.json").write_text(json.dumps(reduced, indent=2) + "\n")
            for name in _FUNCTIONAL_REMOTE_CODE_FILES.get(model_key, ()):
                if source_fixture is None:
                    raise RuntimeError(
                        f"{model_key} remote code requires a parent fixture"
                    )
                shutil.copy2(source_fixture / name, staging / name)
        tokenizer.save_pretrained(staging)
        if model_key in _MULTIMODAL:
            AutoProcessor.from_pretrained(
                source_fixture or canonical_model,
                trust_remote_code=True,
                **source_kwargs,
            ).save_pretrained(staging)
        if model_key.startswith("gemma4_"):
            AutoImageProcessor.from_pretrained(
                source_fixture or canonical_model,
                trust_remote_code=True,
                **source_kwargs,
            ).save_pretrained(staging)
        parameters = 0
        provenance: dict[str, object] | None = None
        if functional:
            provenance = _write_functional_weights(
                canonical_model=canonical_model,
                model_key=model_key,
                revision=revision,
                config=source_config,
                output=staging,
            )
        elif model_key == "dsv4":
            save_file(
                {"_art_fixture_dummy": torch.zeros(1)}, staging / "model.safetensors"
            )
        else:
            auto = (
                AutoModelForImageTextToText
                if model_key in _MULTIMODAL
                else AutoModelForCausalLM
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                configured_dtype = getattr(config, "dtype", None)
                model = auto.from_config(
                    config,
                    trust_remote_code=True,
                    **(
                        {"dtype": torch.float32}
                        if model_key == "nemotron_h_moe"
                        else {}
                    ),
                )
                if model_key == "nemotron_h_moe":
                    config.dtype = configured_dtype
                fp32 = {}
                if model_key == "nemotron_h_moe":
                    pattern = str(config.hybrid_override_pattern)
                    for index, symbol in enumerate(pattern):
                        mixer = model.backbone.layers[index].mixer
                        if symbol == "E":
                            torch.nn.init.normal_(
                                mixer.gate.weight, std=float(config.initializer_range)
                            )
                        if symbol == "M":
                            fp32[f"backbone.layers.{index}.mixer.A_log"] = (
                                mixer.A_log.detach().clone()
                            )
                            fp32[f"backbone.layers.{index}.mixer.D"] = (
                                mixer.D.detach().clone()
                            )
                        elif symbol == "E":
                            fp32[
                                f"backbone.layers.{index}.mixer.gate.e_score_correction_bias"
                            ] = mixer.gate.e_score_correction_bias.detach().clone()
                model = model.to(torch.bfloat16)
                tensors = dict(model.named_parameters()) | dict(model.named_buffers())
                for name, value in fp32.items():
                    tensors[name].data = value
            if model_key.startswith("gemma4_"):
                layers = model.model.language_model.layers
                residual_scale = (2 * len(layers)) ** -0.5
                with torch.no_grad():
                    for layer in layers:
                        layer.post_attention_layernorm.weight.fill_(residual_scale)
                        layer.post_feedforward_layernorm.weight.fill_(residual_scale)
            if model_key == "nemotron_h_moe":
                if model._tied_weights_keys != ["lm_head.weight"]:
                    raise RuntimeError("Nemotron-H HF tied-weight metadata changed")
                model._tied_weights_keys = {}
                model.register_for_auto_class("AutoModelForCausalLM")
            parameters = sum(parameter.numel() for parameter in model.parameters())
            model.save_pretrained(
                staging,
                safe_serialization=True,
                max_shard_size="2GB",
                **(
                    {"save_original_format": False}
                    if model_key == "nemotron_h_moe"
                    else {}
                ),
            )
            del model
            gc.collect()
            if model_key == "qwen3_5_moe":
                _pack_qwen35_experts(staging, config)
            if model_key.startswith("gemma4_"):
                checkpoint = staging / "model.safetensors"
                weight_map = dict.fromkeys(load_file(checkpoint), checkpoint.name)
                (staging / "model.safetensors.index.json").write_text(
                    json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
                    + "\n"
                )
        parent_manifest_sha256 = (
            _sha256(source_fixture / "fixture_manifest.json")
            if source_fixture is not None
            else None
        )
        manifest = {
            "version": (
                _TOKENIZER_FIXTURE_VERSION
                if tokenizer_compatible
                else _fixture_version(model_key)
            ),
            "source_model": canonical_model,
            "source_revision": revision,
            "source_identity": {"model": canonical_model, "revision": revision},
            "parent_manifest_sha256": parent_manifest_sha256,
            "handler": model_key,
            "parameters": parameters,
            "num_layers": int(_text(config).num_hidden_layers),
            "dtype": "bfloat16" if model_key != "dsv4" else None,
            "seed": 0,
            "vocabulary_contract": (
                "canonical" if tokenizer_compatible else "compact_8192"
            ),
            "config_vocab_size": int(_text(config).vocab_size),
            "tokenizer_size": len(tokenizer),
            "tokenizer_max_id": tokenizer_max_id,
        }
        if functional:
            if functional_contract is None or provenance is None:
                raise RuntimeError("functional fixture construction is incomplete")
            manifest.update(
                {
                    "version": _functional_fixture_version(model_key),
                    "fixture_kind": "functional_pretrained",
                    "pretrained": True,
                    "num_layers": _functional_plan(model_key).depth,
                    "dtype": {
                        "configured": functional_contract["configured_dtype"],
                        "checkpoint": provenance["checkpoint_dtypes"],
                    },
                    "weight_provenance": provenance,
                    "contract_sha256": _functional_contract_sha256(model_key),
                    **functional_contract,
                }
            )
        if tokenizer_compatible:
            _validate_tokenizer_compatible_fixture(staging, manifest)
        if functional and not _checkpoint_is_complete(staging):
            raise RuntimeError(f"{model_key} functional checkpoint is incomplete")
        if functional:
            manifest["file_sizes"] = _fixture_file_sizes(staging)
            manifest["manifest_sha256"] = _manifest_sha256(manifest)
        else:
            manifest["files"] = _fixture_files(staging)
        (staging / "fixture_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        _publish(staging, output)


def _cache_alias(
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    fixture: Path,
    root: Path,
    version: int,
    namespace: str,
) -> Path:
    hf_home = root / f"v{version}" / model_key / namespace
    repo = hf_home / "hub" / f"models--{canonical_model.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and not snapshot.is_symlink():
        raise RuntimeError(f"fixture cache alias is not a symlink: {snapshot}")
    if snapshot.is_symlink() and snapshot.resolve() != fixture.resolve():
        snapshot.unlink()
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.symlink_to(fixture, target_is_directory=True)
    if not snapshot.is_symlink() or snapshot.resolve() != fixture.resolve():
        raise RuntimeError(
            f"fixture cache alias does not identify {fixture}: {snapshot}"
        )
    (repo / "refs" / "main").write_text(revision)
    return hf_home


def _flatten_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _validate_tokenizer_compatible_fixture(
    fixture: Path, manifest: dict[str, object]
) -> None:
    from transformers import AutoTokenizer

    tokenizer = cast(Any, AutoTokenizer.from_pretrained(fixture, local_files_only=True))
    vocab_size_value = manifest["config_vocab_size"]
    if not isinstance(vocab_size_value, int):
        raise RuntimeError(
            f"fixture config_vocab_size is not an integer: {vocab_size_value!r}"
        )
    vocab_size = vocab_size_value
    registered_max_id = max(map(int, tokenizer.get_vocab().values()))
    if registered_max_id >= vocab_size:
        raise RuntimeError(
            f"registered tokenizer ID {registered_max_id} exceeds "
            f"vocab_size={vocab_size}"
        )
    samples = (
        "Return one token.",
        "Explain how distributed training preserves policy-version provenance.",
        "Unicode tokenizer check: cafe Tokyo resume.",
    )
    encoded: list[int] = []
    for sample in samples:
        encoded.extend(_flatten_token_ids(tokenizer(sample, add_special_tokens=True)))
    if getattr(tokenizer, "chat_template", None):
        for sample in samples:
            encoded.extend(
                _flatten_token_ids(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": sample}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            )
    max_encoded_id = max(encoded)
    if max_encoded_id >= vocab_size:
        raise RuntimeError(
            f"representative tokenizer ID {max_encoded_id} exceeds vocab_size={vocab_size}"
        )
    manifest["representative_max_token_id"] = max_encoded_id
    manifest["tokenizer_max_id"] = registered_max_id


def _functional_contract_sha256(model_key: str) -> str:
    return _json_sha256(
        (
            _functional_fixture_version(model_key),
            _functional_plan(model_key).model_dump(mode="json"),
            _FUNCTIONAL_PATTERNS.get(model_key),
        )
    )


def _write_functional_weights(
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    config: dict[str, Any],
    output: Path,
) -> dict[str, object]:
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    plan = _functional_plan(model_key)
    cache = _CANONICAL_CACHE_ROOT / f"v{_CANONICAL_CACHE_VERSION}" / model_key / "hub"
    checkpoint = Path(
        hf_hub_download(
            repo_id=canonical_model,
            filename=plan.checkpoint,
            revision=revision,
            cache_dir=cache,
        )
    )
    weights = _checkpoint_weight_map(checkpoint)
    selected = _select_functional_weights(weights, config, model_key=model_key)
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)
    source_root = Path(
        snapshot_download(
            repo_id=canonical_model,
            revision=revision,
            cache_dir=cache,
            allow_patterns=[plan.checkpoint, *by_shard],
        )
    )

    source_blobs: dict[str, str] = {}
    dtypes: dict[str, int] = {}
    total_size = 0
    for source_name, names in sorted(by_shard.items()):
        source_path = source_root / source_name
        blob = source_path.resolve(strict=True).name
        if len(blob) != 64 or any(c not in "0123456789abcdef" for c in blob):
            raise RuntimeError(
                f"canonical shard is not content-addressed: {source_path}"
            )
        with safe_open(source_path, framework="pt", device="cpu") as source:
            tensors = {name: source.get_tensor(name) for name in sorted(names)}
            for name in names:
                dtype = str(source.get_slice(name).get_dtype())
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
        save_file(tensors, output / source_name, metadata={"format": "pt"})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        source_blobs[source_name] = blob
        del tensors
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(selected.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    # fmt: off
    return {
        "method": "safetensors_safe_open_get_tensor_v1",
        "source_checkpoint": checkpoint.name,
        "source_index_sha256": _sha256(checkpoint) if checkpoint.name.endswith(".json") else None,
        "source_weight_map_sha256": _json_sha256(weights),
        "source_shards": source_blobs,
        "selected_key_count": len(selected),
        "selected_keys_sha256": _json_sha256(sorted(selected)),
        "checkpoint_dtypes": dtypes,
    }
    # fmt: on


def _canonical_snapshot(
    *, canonical_model: str, model_key: str, revision: str
) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    hf_home = _CANONICAL_CACHE_ROOT / f"v{_CANONICAL_CACHE_VERSION}" / model_key
    snapshot = snapshot_download(
        repo_id=canonical_model,
        revision=revision,
        cache_dir=hf_home / "hub",
    )
    repo = hf_home / "hub" / f"models--{canonical_model.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(revision)
    return Path(snapshot), hf_home


def _ensure_cached_fixture(
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    root: Path,
    cache_root: Path,
    version: int,
    tokenizer_compatible: bool,
    source_fixture: Path | None = None,
    functional: bool = False,
) -> tuple[Path, dict[str, object], Path]:
    namespace = _fixture_namespace(
        canonical_model=canonical_model,
        revision=revision,
        model_key=model_key,
        version=version,
        tokenizer_compatible=tokenizer_compatible,
    )
    model_root = root / model_key
    model_root.mkdir(parents=True, exist_ok=True)
    output = model_root / namespace
    parent_manifest_sha256 = (
        _sha256(source_fixture / "fixture_manifest.json")
        if source_fixture is not None
        else None
    )
    with (model_root / f".{namespace}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = _is_current(
            output,
            canonical_model=canonical_model,
            model_key=model_key,
            revision=revision,
            tokenizer_compatible=tokenizer_compatible,
            parent_manifest_sha256=parent_manifest_sha256,
            version=version if functional else None,
        )
        if functional and current:
            current = json.loads((output / "fixture_manifest.json").read_text()).get(
                "contract_sha256"
            ) == _functional_contract_sha256(model_key)
        if not current:
            if functional and source_fixture is None:
                raise RuntimeError("functional fixture requires compact metadata")
            _build(
                canonical_model=canonical_model,
                model_key=model_key,
                revision=revision,
                output=output,
                tokenizer_compatible=tokenizer_compatible,
                source_fixture=source_fixture,
                functional=functional,
            )
        manifest = cast(
            dict[str, object],
            json.loads((output / "fixture_manifest.json").read_text()),
        )
        if tokenizer_compatible:
            _validate_tokenizer_compatible_fixture(output, manifest)
        hf_home = _cache_alias(
            canonical_model=canonical_model,
            model_key=model_key,
            revision=revision,
            fixture=output,
            root=cache_root,
            version=version,
            namespace=namespace,
        )
    return output, manifest, hf_home


def ensure_workflow_fixture(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
    required_stages: set[str] | frozenset[str] = frozenset(),
) -> WorkflowFixture:
    from art.megatron.model_support.registry import get_model_support_spec

    model_key = get_model_support_spec(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    ).key
    try:
        revision = _REVISIONS[base_model]
    except KeyError:
        raise ValueError(
            "workflow fixtures require an exact pinned representative model; "
            f"unrecognized model {base_model!r} for handler {model_key!r}"
        ) from None
    root = Path(os.environ.get(FIXTURE_ROOT_ENV, str(_ROOT)))
    output, manifest, hf_home = _ensure_cached_fixture(
        canonical_model=base_model,
        model_key=model_key,
        revision=revision,
        root=root,
        cache_root=Path(os.environ.get(FIXTURE_CACHE_ENV, str(_CACHE_ROOT))),
        version=_fixture_version(model_key),
        tokenizer_compatible=False,
    )
    tokenizer_path: Path | None = None
    tokenizer_hf_home: Path | None = None
    tokenizer_manifest: dict[str, object] | None = None
    reduced_trainability_stages = _REDUCED_TRAINABILITY_ENV.get(model_key, {})
    tokenizer_required = model_key.startswith("gemma4_") and bool(
        required_stages & reduced_trainability_stages.keys()
    )
    if tokenizer_required:
        tokenizer_path, tokenizer_manifest, tokenizer_hf_home = _ensure_cached_fixture(
            canonical_model=base_model,
            model_key=model_key,
            revision=revision,
            root=_TOKENIZER_FIXTURE_ROOT / f"v{_TOKENIZER_FIXTURE_VERSION}",
            cache_root=_TOKENIZER_CACHE_ROOT,
            version=_TOKENIZER_FIXTURE_VERSION,
            tokenizer_compatible=True,
            source_fixture=output,
        )
    functional_path: Path | None = None
    functional_hf_home: Path | None = None
    functional_manifest: dict[str, object] | None = None
    if required_stages & _FUNCTIONAL_STAGES:
        functional_version = _functional_fixture_version(model_key)
        functional_path, functional_manifest, functional_hf_home = (
            _ensure_cached_fixture(
                canonical_model=base_model,
                model_key=model_key,
                revision=revision,
                root=_FUNCTIONAL_FIXTURE_ROOT / f"v{functional_version}",
                cache_root=_FUNCTIONAL_CACHE_ROOT,
                version=functional_version,
                tokenizer_compatible=True,
                source_fixture=output,
                functional=True,
            )
        )
    canonical_path: Path | None = None
    canonical_hf_home: Path | None = None
    canonical_required = any(
        stage in _PRETRAINED_WEIGHT_STAGES and stage not in reduced_trainability_stages
        for stage in required_stages
    ) or (
        model_key.startswith("gemma4_")
        and bool(required_stages & _GEMMA_CANONICAL_WEIGHT_STAGES)
    )
    if canonical_required:
        canonical_path, canonical_hf_home = _canonical_snapshot(
            canonical_model=base_model,
            model_key=model_key,
            revision=revision,
        )
    return WorkflowFixture(
        canonical_model=base_model,
        model_key=model_key,
        source_revision=revision,
        path=str(output),
        hf_home=str(hf_home),
        manifest=manifest,
        tokenizer_compatible_path=(
            str(tokenizer_path) if tokenizer_path is not None else None
        ),
        tokenizer_compatible_hf_home=(
            str(tokenizer_hf_home) if tokenizer_hf_home is not None else None
        ),
        tokenizer_compatible_manifest=tokenizer_manifest,
        functional_path=(str(functional_path) if functional_path is not None else None),
        functional_hf_home=(
            str(functional_hf_home) if functional_hf_home is not None else None
        ),
        functional_manifest=functional_manifest,
        canonical_path=str(canonical_path) if canonical_path is not None else None,
        canonical_hf_home=(
            str(canonical_hf_home) if canonical_hf_home is not None else None
        ),
    )
