"""Build fixtures and capture native HF references (standalone, no ART runtime).

Examples (run with the pinned M1 interpreter):
  python glm5next_oracle.py download --hfmeta /path/to/glm-mvp/hfmeta
  python glm5next_oracle.py build R2 --hfmeta /path/to/glm-mvp/hfmeta
  python glm5next_oracle.py build F4 --hfmeta /path/to/glm-mvp/hfmeta
  python glm5next_oracle.py dump F4 --device cuda
  python glm5next_oracle.py vllm --base-url http://localhost:8000/v1 --model MODEL
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from typing import Any

# Must be set before importing huggingface_hub or transformers.
os.environ.setdefault("HF_HOME", "/tmp/stark-gpu/glm-hf")

from safetensors.torch import save_file
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ORACLE_PATH = (
    Path(__file__).resolve().parents[4] / "src/art/megatron/glm5next/hf_oracle.py"
)
_spec = importlib.util.spec_from_file_location("glm5next_hf_oracle", ORACLE_PATH)
assert _spec is not None and _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

ROOT = oracle.DEFAULT_FIXTURES
WORK = Path("/tmp/stark-gpu/glm-m1")
LENGTHS = (128, 512, 2048, 4096, 8192, 16384)
SEED = 20261105


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def versions() -> dict[str, Any]:
    spec = importlib.util.find_spec("transformers.models.glm5_next.modeling_glm5_next")
    assert spec is not None and spec.origin is not None
    packages = (
        "transformers",
        "torch",
        "flash-linear-attention",
        "fla-core",
        "triton",
        "safetensors",
        "huggingface-hub",
        "accelerate",
    )
    return {
        **{name: importlib.metadata.version(name) for name in packages},
        "cuda": torch.version.cuda,
        "modeling_sha256": oracle.sha256_file(Path(spec.origin)),
    }


def manifest(root: Path) -> dict[str, Any]:
    path = root / "MANIFEST.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "schema_version": 1,
        "source": {"repo": oracle.HF_REPO, "revision": oracle.HF_REVISION},
        "versions": versions(),
        "fixtures": {},
        "dumps": {},
        "seed": SEED,
        "packing": "independent HF forwards per document, concatenated in order with cu_seqlens; no padding",
        "selection": {
            "index_topk": 2048,
            "index_kpool": 4,
            "selected_pools": 512,
            "output_width": 2051,
            "invalid": -1,
            "first_pool_competition_query_zero_based": 2051,
        },
    }


def keep_key(key: str, layers: int) -> bool:
    if key in (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ):
        return True
    match = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
    return match is not None and int(match[1]) < layers


def fixture_config(hfmeta: Path, name: str) -> Any:
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    config = json.loads((hfmeta / "config.json").read_text())["text_config"]
    layers = {"F4": 4, "F8": 8, "R2": 3}[name]
    config.update(num_hidden_layers=layers, use_cache=False, num_nextn_predict_layers=0)
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        config[key] = config[key][:layers]
    config["layer_types"] = [
        "indexed_attention" if value == "deepseek_sparse_attention" else value
        for value in config["layer_types"]
    ]
    linear = config.pop("linear_attn_config")
    config.update(
        linear_head_dim=linear["head_dim"],
        linear_num_heads=linear["num_heads"],
        linear_conv_kernel_dim=linear["short_conv_kernel_size"],
        linear_lower_bound=linear["gate_lower_bound"],
    )
    if name == "R2":
        config.update(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            vocab_size=257,
            pad_token_id=256,
            eos_token_id=[256],
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_head_dim=8,
            v_head_dim=8,
            index_head_dim=8,
            index_n_heads=2,
            linear_head_dim=8,
            linear_num_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            index_topk=8,
            layer_types=["linear_attention", "indexed_attention", "linear_attention"],
            mlp_layer_types=["dense", "sparse", "sparse"],
        )
    # Do not turn an EOS token into an Embedding padding_idx: it is a real token.
    config["pad_token_id"] = None
    return Glm5NextTextConfig(**config)


def download(hfmeta: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    index = json.loads((hfmeta / "index.json").read_text())
    selected = sorted(
        {shard for key, shard in index["weight_map"].items() if keep_key(key, 8)}
    )
    info = HfApi().model_info(
        oracle.HF_REPO, revision=oracle.HF_REVISION, files_metadata=True
    )
    assert info.siblings is not None
    metadata = {item.rfilename: item for item in info.siblings}
    assert info.sha == oracle.HF_REVISION
    source = WORK / "source"
    source.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> dict[str, Any]:
        item = metadata[name]
        path = Path(hf_hub_download(oracle.HF_REPO, name, revision=oracle.HF_REVISION))
        if path.stat().st_size != item.size:
            raise ValueError(f"Wrong shard size: {name}")
        sha = oracle.sha256_file(path)
        if item.lfs is None or sha != item.lfs.sha256:
            raise ValueError(f"Wrong shard sha256: {name}")
        link = source / name
        if not link.exists():
            link.symlink_to(path)
        print(f"verified {name} {item.size} {sha}", flush=True)
        return {"name": name, "size": item.size, "sha256": sha, "cache_path": str(path)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, selected))
    # Keep a pinned copy of the original config/index alongside verified shards.
    for filename in ("config.json", "model.safetensors.index.json"):
        downloaded = Path(
            hf_hub_download(oracle.HF_REPO, filename, revision=oracle.HF_REVISION)
        )
        if filename.endswith("index.json"):
            if json.loads(downloaded.read_text()) != index:
                raise ValueError("Local hfmeta index differs from pinned BF16 index")
        (source / filename).write_bytes(downloaded.read_bytes())
    write_json(
        WORK / "SOURCE.json",
        {
            "repo": oracle.HF_REPO,
            "revision": oracle.HF_REVISION,
            "shards": records,
            "total_bytes": sum(r["size"] for r in records),
        },
    )


def build(name: str, hfmeta: Path, root: Path, *, meta: bool = False) -> dict[str, Any]:
    config = fixture_config(hfmeta, name)
    model_class = oracle.hf_model_class()
    if meta:
        with torch.device("meta"):
            model = model_class._from_config(config, dtype=torch.bfloat16)
        summary = {
            "name": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "layers": len(model.model.language_model.layers),
            "config": config.to_dict(),
        }
        print(json.dumps(summary), flush=True)
        return summary
    out = root / name
    if name == "R2":
        torch.manual_seed(SEED)
        model = model_class(config).eval()
        info = {}
    else:
        index = json.loads((hfmeta / "index.json").read_text())
        selected = {
            key: shard
            for key, shard in index["weight_map"].items()
            if keep_key(key, config.num_hidden_layers)
        }
        view = WORK / f"source-{name}"
        view.mkdir(parents=True, exist_ok=True)
        for shard in sorted(set(selected.values())):
            link = view / shard
            if not link.exists():
                link.symlink_to((WORK / "source" / shard).resolve(strict=True))
        write_json(
            view / "model.safetensors.index.json",
            {"metadata": {}, "weight_map": selected},
        )
        config.save_pretrained(view)
        offload = (
            {
                "device_map": {"model.language_model": "disk", "lm_head": "cpu"},
                "offload_folder": str(WORK / "offload-F8"),
            }
            if name == "F8"
            else {}
        )
        model, info = model_class.from_pretrained(
            view,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            output_loading_info=True,
            **offload,
        )
        # Entire selected shards may contain later layers; never tolerate missing converted weights.
        if any(
            info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError(f"Incomplete real-weight load: {info}")
        if name != "F8" and any(p.is_meta for p in model.parameters()):
            raise RuntimeError("Unmaterialized real-weight parameter")
    model.save_pretrained(
        out, safe_serialization=True, max_shard_size="4GB", save_original_format=False
    )
    entry = {
        "status": "ready",
        "random": name == "R2",
        "seed": SEED if name == "R2" else None,
        "config": config.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loading_info": {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in info.items()
        },
        "files": [
            oracle.file_record(p, root) for p in sorted(out.iterdir()) if p.is_file()
        ],
    }
    data = manifest(root)
    data["fixtures"][name] = entry
    write_json(root / "MANIFEST.json", data)
    print(f"built {name} parameters={entry['parameters']}", flush=True)
    return entry


def prompts(
    root: Path, vocab_size: int = 154880, lengths: tuple[int, ...] = LENGTHS
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    packs = []
    for length in lengths:
        sizes = [13, length - 42, 29]
        tokens = torch.randint(0, vocab_size, (length,), generator=generator).tolist()
        # Deliberate EOS-in-input regression case; do not mask it.
        tokens[17] = 154820 if vocab_size == 154880 else 256
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        packs.append(
            {"id": f"packed-{length}", "input_ids": tokens, "cu_seqlens": offsets}
        )
    data = {"schema_version": 1, "seed": SEED, "vocab_size": vocab_size, "packs": packs}
    write_json(root / "prompts.json", data)
    return data


def lora_family(name: str) -> str | None:
    if name == "lm_head.weight":
        return "unembed"
    if ".self_attn." in name:
        leaf = name.split(".")[-2]
        if leaf in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_a_proj",
            "kv_a_proj_with_mqa",
            "o_proj",
        ):
            return "attention/" + leaf
    if ".mlp." in name:
        if ".experts." in name:
            return "routed/" + name.split(".")[-1]
        if name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight")):
            kind = "shared" if ".shared_experts." in name else "dense"
            return kind + "/" + name.split(".")[-2]
    return None


def save_tensors(values: dict[str, torch.Tensor], path: Path) -> None:
    for key, value in values.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite reference tensor: {path.name}:{key}")
    save_file(values, path)


def dump(
    name: str,
    root: Path,
    device: str,
    *,
    lengths: tuple[int, ...] = LENGTHS,
    full_volume_rehearsal: bool = False,
) -> None:
    model = oracle.load_fixture(name, root=root, device=device, verify=True)
    if full_volume_rehearsal:
        if name != "R2" or device != "cpu":
            raise ValueError("Full-volume rehearsal requires R2 on CPU")
        model.config.index_topk = 2048
        for layer in model.model.language_model.layers:
            if getattr(layer.self_attn, "indexer", None) is not None:
                layer.self_attn.indexer.index_topk = 2048
    model.requires_grad_(False)
    context = (
        oracle.cpu_reference_kernels()
        if device == "cpu"
        else sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        )
    )
    prompt_path = root / ("prompts-r2" if name == "R2" else "")
    prompt_path.mkdir(parents=True, exist_ok=True)
    prompt_set = prompts(prompt_path, model.config.vocab_size, lengths)
    out = root / "dumps" / name
    out.mkdir(parents=True, exist_ok=True)
    files = []
    document_records = []
    with context:
        for pack in prompt_set["packs"]:
            for doc, (start, end) in enumerate(
                zip(pack["cu_seqlens"][:-1], pack["cu_seqlens"][1:], strict=True)
            ):
                ids = torch.tensor([pack["input_ids"][start:end]], device=device)
                values: dict[str, torch.Tensor] = {
                    "input_ids": ids.cpu(),
                    "position_ids": torch.arange(end - start).unsqueeze(0),
                    "pack_offset": torch.tensor([start, end]),
                }
                handles = []

                def input_hook(key: str):
                    def record(_module: Any, inputs: Any, kwargs: Any) -> None:
                        value = inputs[0] if inputs else kwargs["hidden_states"]
                        values[key] = (
                            value.detach().to(device="cpu", copy=True).contiguous()
                        )

                    return record

                def hook(key: str, kind: str):
                    def record(_module: Any, _inputs: Any, output: Any) -> None:
                        if kind == "layer":
                            values[key] = output[0].detach().cpu().contiguous()
                        elif kind == "router":
                            values[key + ".weights"] = (
                                output[1].detach().cpu().contiguous()
                            )
                            values[key + ".indices"] = (
                                output[2].detach().cpu().to(torch.int32).contiguous()
                            )
                        elif kind == "hc":
                            for field, item in zip(
                                ("post", "comb", "collapsed"), output, strict=True
                            ):
                                values[key + "." + field] = (
                                    item.detach().cpu().contiguous()
                                )
                        else:
                            values[key] = output.detach().cpu().contiguous()

                    return record

                for index, layer in enumerate(model.model.language_model.layers):
                    prefix = f"layers.{index}"
                    if name in ("F4", "R2"):
                        handles.append(
                            layer.register_forward_pre_hook(
                                input_hook(prefix + ".input_streams"), with_kwargs=True
                            )
                        )
                        handles.append(
                            layer.self_attn.register_forward_pre_hook(
                                input_hook(prefix + ".attention_input"),
                                with_kwargs=True,
                            )
                        )
                        handles.append(
                            layer.mlp.register_forward_pre_hook(
                                input_hook(prefix + ".mlp_input"), with_kwargs=True
                            )
                        )
                        handles.append(
                            layer.register_forward_hook(
                                hook(prefix + ".streams", "layer")
                            )
                        )
                        handles.append(
                            layer.attn_hc.register_forward_hook(
                                hook(prefix + ".attn_hc", "hc")
                            )
                        )
                        handles.append(
                            layer.ffn_hc.register_forward_hook(
                                hook(prefix + ".ffn_hc", "hc")
                            )
                        )
                        handles.append(
                            layer.self_attn.register_forward_hook(
                                hook(
                                    prefix + ".attention",
                                    "layer"
                                    if layer.block_type != "linear_attention"
                                    else "tensor",
                                )
                            )
                        )
                        handles.append(
                            layer.mlp.register_forward_hook(
                                hook(prefix + ".mlp", "tensor")
                            )
                        )
                    if getattr(layer.self_attn, "indexer", None) is not None:
                        handles.append(
                            layer.self_attn.indexer.register_forward_hook(
                                hook(prefix + ".indexer.indices", "tensor")
                            )
                        )
                    if hasattr(layer.mlp, "gate"):
                        handles.append(
                            layer.mlp.gate.register_forward_hook(
                                hook(prefix + ".router", "router")
                            )
                        )
                try:
                    with torch.inference_mode():
                        hidden = model.model.language_model(
                            input_ids=ids, use_cache=False
                        ).last_hidden_state
                        summaries: dict[str, list[torch.Tensor]] = {
                            k: []
                            for k in (
                                "logsumexp",
                                "top64.values",
                                "top64.indices",
                                "target_logprobs",
                            )
                        }
                        sample_positions = sorted(
                            {
                                0,
                                end - start - 1,
                                (end - start) // 2,
                                *[
                                    p
                                    for p in (2047, 2048, 2050, 2051, 8191, 8192)
                                    if p < end - start
                                ],
                            }
                        )
                        full = []
                        for begin in range(0, hidden.shape[1], 64):
                            logits = model.lm_head(
                                hidden[:, begin : begin + 64]
                            ).float()
                            lse = logits.logsumexp(-1)
                            top = logits.topk(min(64, logits.shape[-1]), dim=-1)
                            summaries["logsumexp"].append(lse.cpu())
                            summaries["top64.values"].append(top.values.cpu())
                            summaries["top64.indices"].append(
                                top.indices.to(torch.int32).cpu()
                            )
                            count = min(logits.shape[1], ids.shape[1] - begin - 1)
                            lp = (
                                logits[:, :count]
                                .gather(-1, ids[:, begin + 1 : begin + 1 + count, None])
                                .squeeze(-1)
                                - lse[:, :count]
                            )
                            summaries["target_logprobs"].append(lp.cpu())
                            full.extend(
                                logits[:, p - begin].cpu()
                                for p in sample_positions
                                if begin <= p < begin + logits.shape[1]
                            )
                        values.update(
                            {
                                key: torch.cat(parts, dim=1)
                                for key, parts in summaries.items()
                            }
                        )
                        values["full_logits.positions"] = torch.tensor(sample_positions)
                        values["full_logits"] = torch.stack(full, dim=1)
                finally:
                    for handle in handles:
                        handle.remove()
                path = out / f"{pack['id']}-doc{doc}.safetensors"
                for key, selection in values.items():
                    if not key.endswith("indexer.indices"):
                        continue
                    budget, pool = model.config.index_topk, model.config.index_kpool
                    if selection.shape[-1] != budget + pool - 1:
                        raise ValueError(f"Invalid KPool output width: {key}")
                    for position in (
                        0,
                        2,
                        3,
                        7,
                        2047,
                        2048,
                        2050,
                        2051,
                        end - start - 1,
                    ):
                        if position >= end - start:
                            continue
                        row = selection[0, position]
                        valid = row[row >= 0].tolist()
                        expected_count = (
                            min(budget, (position + 1) // pool * pool)
                            + (position + 1) % pool
                        )
                        if len(set(valid)) != expected_count or any(
                            i > position for i in valid
                        ):
                            raise ValueError(
                                f"Invalid causal KPool selection at {key}:{position}"
                            )
                save_tensors(values, path)
                files.append(oracle.file_record(path, root))
                document_records.append(
                    {
                        "pack": pack["id"],
                        "document": doc,
                        "start": start,
                        "end": end,
                        "file": str(path.relative_to(root)),
                        "keys": sorted(values),
                    }
                )
                print(f"dumped {path.name}", flush=True)
        if name in ("F4", "R2"):
            files.extend(gradient_dump(model, prompt_set, root, out, device))
    data = manifest(root)
    data["dumps"][name] = {
        "status": "ready",
        "versions": versions(),
        "files": files,
        "prompts": oracle.file_record(prompt_path / "prompts.json", root),
        "documents": document_records,
        "backend": "HF torch fallback"
        if device == "cpu"
        else "HF with FLA KDA; SDPA DSA (flash/efficient/cuDNN; math disabled)",
        "full_volume_cpu_rehearsal": full_volume_rehearsal,
        "index_topk": model.config.index_topk,
    }
    write_json(root / "MANIFEST.json", data)


def gradient_dump(
    model: Any, prompt_set: dict[str, Any], root: Path, out: Path, device: str
) -> list[dict[str, Any]]:
    """Exact CE gradients one family at a time, avoiding all-expert gradient residency."""
    pack = prompt_set["packs"][0]
    ids = torch.tensor([pack["input_ids"]], device=device)
    boundaries = pack["cu_seqlens"]
    target_count = sum(
        end - start - 1
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, parameter in model.named_parameters():
        family = lora_family(name)
        if family:
            groups.setdefault(family, []).append((name, parameter))
    files = []
    for family, parameters in groups.items():
        for _, parameter in parameters:
            parameter.requires_grad_(True)
        values: dict[str, torch.Tensor] = {}
        total_loss = torch.zeros(1)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            document = ids[:, start:end]
            logits = model(document, use_cache=False).logits.float()
            loss = (
                torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    document[:, 1:].reshape(-1),
                    reduction="sum",
                )
                / target_count
            )
            gradients = torch.autograd.grad(
                loss, [parameter for _, parameter in parameters]
            )
            for (key, _), gradient in zip(parameters, gradients, strict=True):
                # Accumulate in FP32 on the host; no cross-document target or state.
                cpu_gradient = gradient.detach().float().cpu()
                if key in values:
                    values[key].add_(cpu_gradient)
                else:
                    values[key] = cpu_gradient.contiguous()
            total_loss += loss.detach().float().cpu()
            del gradients, logits, loss
        path = out / ("gradients-" + family.replace("/", "-") + ".safetensors")
        save_tensors(
            {
                **values,
                "ce_loss": total_loss,
                "input_ids": ids.cpu(),
                "cu_seqlens": torch.tensor(boundaries),
                "target_count": torch.tensor([target_count]),
            },
            path,
        )
        files.append(oracle.file_record(path, root))
        for _, parameter in parameters:
            parameter.requires_grad_(False)
        del values
        print(f"gradient family {family}", flush=True)
    if not groups:
        raise RuntimeError("No LoRA-target weights found")
    return files


def vllm_reference(
    root: Path, base_url: str, model: str, output: Path, *, client: Any = None
) -> None:
    """The first prompt token has no logprob; no generated token is used."""
    import httpx

    prompt_set = json.loads((root / "prompts.json").read_text())
    headers = {}
    if token := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + token
    owned = client is None
    if owned:
        client = httpx.Client(timeout=1800, headers=headers)
    records = []
    try:
        for pack in prompt_set["packs"]:
            for doc, (start, end) in enumerate(
                zip(pack["cu_seqlens"][:-1], pack["cu_seqlens"][1:], strict=True)
            ):
                response = client.post(
                    base_url.rstrip("/") + "/completions",
                    json={
                        "model": model,
                        "prompt": pack["input_ids"][start:end],
                        "max_tokens": 1,
                        "temperature": 0,
                        "seed": SEED,
                        "echo": True,
                        "logprobs": 1,
                        "prompt_logprobs": 1,
                        "return_tokens_as_token_ids": True,
                    },
                )
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                logprobs = choice["logprobs"]
                probabilities = logprobs["token_logprobs"][: end - start]
                token_ids = logprobs["tokens"][: end - start]
                expected = pack["input_ids"][start:end]
                if len(probabilities) != len(expected) or token_ids != [
                    f"token_id:{i}" for i in expected
                ]:
                    raise ValueError(
                        "vLLM echo token alignment differs from fixed prompt IDs"
                    )
                if probabilities[0] is not None or any(
                    p is None or not torch.isfinite(torch.tensor(p))
                    for p in probabilities[1:]
                ):
                    raise ValueError("Missing/nonfinite prompt logprob")
                records.append(
                    {
                        "pack": pack["id"],
                        "document": doc,
                        "start": start,
                        "end": end,
                        "input_ids": expected,
                        "prompt_logprobs": probabilities,
                        "raw_response": body,
                    }
                )
    finally:
        if owned:
            client.close()
    write_json(
        output,
        {
            "schema_version": 1,
            "model": model,
            "prompts_sha256": oracle.sha256_file(root / "prompts.json"),
            "records": records,
        },
    )
    write_json(
        output.with_suffix(".manifest.json"), oracle.file_record(output, output.parent)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--hfmeta", type=Path, required=True)
    make = sub.add_parser("build")
    make.add_argument("name", choices=("R2", "F4", "F8"))
    make.add_argument("--hfmeta", type=Path, required=True)
    make.add_argument("--meta", action="store_true")
    capture = sub.add_parser("dump")
    capture.add_argument("name", choices=("R2", "F4", "F8"))
    capture.add_argument("--device", default="cpu")
    capture.add_argument("--full-volume-rehearsal", action="store_true")
    client = sub.add_parser("vllm")
    client.add_argument("--base-url", required=True)
    client.add_argument("--model", required=True)
    client.add_argument("--output", type=Path, required=True)
    sub.add_parser("prompts")
    args = parser.parse_args()
    started = time.monotonic()
    if args.command == "download":
        download(args.hfmeta)
    elif args.command == "build":
        build(args.name, args.hfmeta, args.root, meta=args.meta)
    elif args.command == "dump":
        dump(
            args.name,
            args.root,
            args.device,
            lengths=(128,)
            if args.name == "R2" and not args.full_volume_rehearsal
            else LENGTHS,
            full_volume_rehearsal=args.full_volume_rehearsal,
        )
    elif args.command == "prompts":
        prompts(args.root)
    else:
        vllm_reference(args.root, args.base_url, args.model, args.output)
    print(f"completed in {time.monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
