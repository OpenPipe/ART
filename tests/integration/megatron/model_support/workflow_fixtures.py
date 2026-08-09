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
FIXTURE_VERSION = 17
_CANONICAL_CACHE_VERSION = 16
_ROOT = Path("/tmp/art-models/main-merge-oracle")
_CACHE_ROOT = Path("/tmp/art-model-support-workflow/hf-cache")
_TOKENIZER_FIXTURE_ROOT = Path("/tmp/art-model-support-workflow/tokenizer-compatible")
_TOKENIZER_CACHE_ROOT = Path("/tmp/art-model-support-workflow/tokenizer-hf-cache")
_CANONICAL_CACHE_ROOT = Path("/tmp/art-model-support-workflow/canonical-hf-cache")
_GEMMA_CANONICAL_WEIGHT_STAGES = frozenset({"hf_parity", "packing_invariance"})
_PRETRAINED_WEIGHT_STAGES = frozenset({"length_trainability", "yes_no_trainability"})
_GEMMA_YES_NO_ENV = {
    "ART_MODEL_SUPPORT_YES_NO_ALLOWED_TOKEN_IDS": "4443,951,7463",
    "ART_MODEL_SUPPORT_YES_NO_MAX_TOKENS": "1",
}
_REDUCED_TRAINABILITY_ENV: dict[str, dict[str, dict[str, str]]] = {
    "gemma4_dense": {"yes_no_trainability": _GEMMA_YES_NO_ENV},
    "gemma4_moe": {"yes_no_trainability": _GEMMA_YES_NO_ENV},
    "glm52": {
        "length_trainability": {
            "ART_MODEL_SUPPORT_LENGTH_ALLOWED_TOKEN_IDS": "154820,38069",
            "ART_MODEL_SUPPORT_LENGTH_MIN_TOKENS": "2",
            "ART_MODEL_SUPPORT_LENGTH_FREQUENCY_PENALTY": "0.5",
        },
        "yes_no_trainability": {
            "ART_MODEL_SUPPORT_YES_NO_ALLOWED_TOKEN_IDS": "9829,902,36569",
            "ART_MODEL_SUPPORT_YES_NO_MAX_STEPS": "8",
            "ART_MODEL_SUPPORT_YES_NO_MAX_TOKENS": "1",
        },
    },
}
_TOKENIZER_COMPATIBLE_STAGES = frozenset(
    {
        "train_inf_mismatch",
        "merged_vllm_serving",
        "native_vllm_lora",
    }
)
_TOKENIZER_FIXTURE_VERSION = 3
_REVISIONS = {
    "meta-llama/Llama-3.2-1B-Instruct": "9213176726f574b556790deb65791e0c5aa438b6",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "Qwen/Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "deepseek-ai/DeepSeek-V4-Flash": "60d8d70770c6776ff598c94bb586a859a38244f1",
    "zai-org/GLM-5.2": "b4734de4facf877f85769a911abafc5283eab3d9",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
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
    canonical_path: str | None = None
    canonical_hf_home: str | None = None

    def environment(self, stage_name: str | None = None) -> dict[str, str]:
        reduced_trainability = _REDUCED_TRAINABILITY_ENV.get(self.model_key, {}).get(
            stage_name
        )
        use_canonical = (
            (stage_name in _PRETRAINED_WEIGHT_STAGES and reduced_trainability is None)
            or (
                self.model_key.startswith("gemma4_")
                and stage_name in _GEMMA_CANONICAL_WEIGHT_STAGES
            )
            or (self.model_key == "gpt_oss_moe" and stage_name == "train_inf_mismatch")
        )
        use_tokenizer_compatible = stage_name in _TOKENIZER_COMPATIBLE_STAGES or (
            self.model_key.startswith("gemma4_") and reduced_trainability is not None
        )
        path = (
            self.canonical_path
            if use_canonical
            else self.tokenizer_compatible_path
            if use_tokenizer_compatible
            else self.path
        )
        hf_home = (
            self.canonical_hf_home
            if use_canonical
            else self.tokenizer_compatible_hf_home
            if use_tokenizer_compatible
            else self.hf_home
        )
        if path is None or hf_home is None:
            contract = "canonical weights" if use_canonical else "canonical vocabulary"
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
        if reduced_trainability is not None:
            environment.update(reduced_trainability)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_files(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): _sha256(file)
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


def _is_current(
    path: Path,
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    tokenizer_compatible: bool,
    parent_manifest_sha256: str | None,
) -> bool:
    try:
        manifest = json.loads((path / "fixture_manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "version": (
            _TOKENIZER_FIXTURE_VERSION if tokenizer_compatible else FIXTURE_VERSION
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
        and manifest.get("files") == _fixture_files(path)
    )


def _build(
    *,
    canonical_model: str,
    model_key: str,
    revision: str,
    output: Path,
    tokenizer_compatible: bool,
    source_fixture: Path | None = None,
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
        config = _configure(
            model_key,
            source,
            source_vocab_size=source_vocab_size,
            tokenizer_compatible=tokenizer_compatible,
        )
        config.save_pretrained(staging)
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
        if model_key == "dsv4":
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
                model = auto.from_config(config, trust_remote_code=True).to(
                    torch.bfloat16
                )
            if tokenizer_compatible and model_key.startswith("gemma4_"):
                layers = model.model.language_model.layers
                residual_scale = (2 * len(layers)) ** -0.5
                with torch.no_grad():
                    for layer in layers:
                        layer.post_attention_layernorm.weight.fill_(residual_scale)
                        layer.post_feedforward_layernorm.weight.fill_(residual_scale)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            model.save_pretrained(
                staging, safe_serialization=True, max_shard_size="2GB"
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
                _TOKENIZER_FIXTURE_VERSION if tokenizer_compatible else FIXTURE_VERSION
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
        if tokenizer_compatible:
            _validate_tokenizer_compatible_fixture(staging, manifest)
        manifest["files"] = _fixture_files(staging)
        (staging / "fixture_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
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
        if not _is_current(
            output,
            canonical_model=canonical_model,
            model_key=model_key,
            revision=revision,
            tokenizer_compatible=tokenizer_compatible,
            parent_manifest_sha256=parent_manifest_sha256,
        ):
            _build(
                canonical_model=canonical_model,
                model_key=model_key,
                revision=revision,
                output=output,
                tokenizer_compatible=tokenizer_compatible,
                source_fixture=source_fixture,
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
        version=FIXTURE_VERSION,
        tokenizer_compatible=False,
    )
    tokenizer_path: Path | None = None
    tokenizer_hf_home: Path | None = None
    tokenizer_manifest: dict[str, object] | None = None
    if required_stages & _TOKENIZER_COMPATIBLE_STAGES:
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
    canonical_path: Path | None = None
    canonical_hf_home: Path | None = None
    reduced_trainability_stages = _REDUCED_TRAINABILITY_ENV.get(model_key, {})
    canonical_required = (
        any(
            stage in _PRETRAINED_WEIGHT_STAGES
            and stage not in reduced_trainability_stages
            for stage in required_stages
        )
        or (
            model_key.startswith("gemma4_")
            and bool(required_stages & _GEMMA_CANONICAL_WEIGHT_STAGES)
        )
        or (model_key == "gpt_oss_moe" and "train_inf_mismatch" in required_stages)
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
        canonical_path=str(canonical_path) if canonical_path is not None else None,
        canonical_hf_home=(
            str(canonical_hf_home) if canonical_hf_home is not None else None
        ),
    )
