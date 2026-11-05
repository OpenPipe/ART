from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

# This oracle runs without importing ART's optional serving/training dependencies.
torch = pytest.importorskip("torch")
_spec = importlib.util.spec_from_file_location(
    "glm5next_helper", Path(__file__).with_name("glm5next_oracle.py")
)
assert _spec is not None and _spec.loader is not None
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
oracle = helper.oracle


def _native_runtime_available() -> bool:
    try:
        return importlib.metadata.version(
            "transformers"
        ) == oracle.TRANSFORMERS_VERSION and all(
            importlib.metadata.version(name)
            for name in (
                "flash-linear-attention",
                "fla-core",
                "triton",
                "safetensors",
                "huggingface-hub",
                "accelerate",
            )
        )
    except importlib.metadata.PackageNotFoundError:
        return False


requires_native_oracle = pytest.mark.skipif(
    not _native_runtime_available(),
    reason="requires the isolated Transformers 5.20.2 HF oracle runtime",
)


def tiny_config() -> Any:
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    return Glm5NextTextConfig(
        vocab_size=257,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_head_dim=8,
        index_n_heads=2,
        linear_head_dim=8,
        linear_num_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_topk=8,
        index_kpool=4,
        pad_token_id=None,
        eos_token_id=[256],
        use_cache=False,
        layer_types=["linear_attention", "indexed_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
    )


def test_metrics_detect_errors_and_ignore_selection_sentinels() -> None:
    expected = torch.tensor([1.0, 2.0])
    measured = oracle.logprob_metrics(torch.tensor([2.0, 2.0]), expected)
    assert measured["max_abs"] == 1.0
    assert measured["rel_l2"] == pytest.approx(1 / 5**0.5)
    assert measured["cosine"] == pytest.approx(6 / (8 * 5) ** 0.5)
    assert measured["logprob_abs_mean"] == 0.5
    assert measured["logprob_abs_p99"] == pytest.approx(0.99)
    assert (
        oracle.topk_set_agreement(
            torch.tensor([[2, 1, -1, 1]]), torch.tensor([[1, 2, -1]])
        )
        == 1.0
    )
    assert oracle.topk_set_agreement(
        torch.tensor([[1, 3]]), torch.tensor([[1, 2]])
    ) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="Non-finite"):
        oracle.tensor_metrics(torch.tensor([float("nan")]), torch.ones(1))
    with pytest.raises(ValueError, match="keys differ"):
        oracle.compare_dump({}, {"x": expected})


def test_manifest_integrity_rejects_corruption_and_path_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"reference")
    record = oracle.file_record(artifact, tmp_path)
    oracle.verify_records(tmp_path, [record])
    artifact.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        oracle.verify_records(tmp_path, [record])
    with pytest.raises(ValueError, match="escapes"):
        oracle.verify_records(tmp_path, [{**record, "path": "../escape"}])


@requires_native_oracle
def test_from_pretrained_converts_checkpoint_keys_and_preserves_every_weight(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(42)
    model_class = oracle.hf_model_class()
    config = tiny_config()
    model = model_class(config).eval()
    original = model.state_dict()
    checkpoint = {}
    for name, tensor in original.items():
        if name.endswith("self_attn.conv1d.weight"):
            for axis, part in zip(("q", "k", "v"), tensor.chunk(3, dim=0), strict=True):
                checkpoint[name.replace("conv1d", axis + "_conv1d")] = part.contiguous()
        elif ".experts.gate_up_proj" in name:
            for expert, packed in enumerate(tensor):
                for projection, part in zip(
                    ("gate", "up"), packed.chunk(2, dim=0), strict=True
                ):
                    checkpoint[
                        name.replace(
                            "experts.gate_up_proj",
                            f"experts.{expert}.{projection}_proj.weight",
                        )
                    ] = part.contiguous()
        elif ".experts.down_proj" in name:
            for expert, part in enumerate(tensor):
                checkpoint[
                    name.replace(
                        "experts.down_proj", f"experts.{expert}.down_proj.weight"
                    )
                ] = part.contiguous()
        else:
            key = name.replace(".forget_gate.", ".")
            for kind in ("attn", "ffn"):
                for field in ("fn", "base", "scale"):
                    key = key.replace(f".{kind}_hc.{field}", f".hc_{kind}_{field}")
            checkpoint[key] = tensor.contiguous()
    config.save_pretrained(tmp_path)
    save_file(checkpoint, tmp_path / "model.safetensors")
    loaded, info = model_class.from_pretrained(
        tmp_path, output_loading_info=True, dtype=torch.float32
    )
    assert not any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    assert original.keys() == loaded.state_dict().keys()
    for name, expected in original.items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)
    with oracle.cpu_reference_kernels(), torch.inference_mode():
        ids = torch.tensor([[4, 256, 5, 6, 7]])
        torch.testing.assert_close(
            loaded(ids).logits, model(ids).logits, rtol=0, atol=0
        )


@requires_native_oracle
def test_indexer_selects_complete_pools_plus_causal_tail() -> None:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    config = tiny_config()
    indexer = Glm5NextTextIndexer(config, 1)
    torch.nn.init.normal_(indexer.wq_b.weight)
    hidden = torch.randn(1, 17, config.hidden_size)
    query = torch.randn(1, 17, config.q_lora_rank)
    indices = indexer(hidden, query, torch.ones(1, 17, dtype=torch.bool), None)[0]
    assert indices.shape == (17, 11)
    for position, row in enumerate(indices.tolist()):
        selected = {i for i in row if i >= 0}
        visible = position + 1
        complete = visible // 4
        assert len(selected) == min(8, complete * 4) + visible % 4
        assert all(i <= position for i in selected)
        assert set(range(4 * complete, visible)) <= selected
        for pool in range(complete):
            intersection = selected & set(range(pool * 4, pool * 4 + 4))
            assert len(intersection) in (0, 4)


def test_prompt_packing_includes_eos_without_masking(tmp_path: Path) -> None:
    data = helper.prompts(tmp_path)
    assert [len(pack["input_ids"]) for pack in data["packs"]] == list(helper.LENGTHS)
    for pack in data["packs"]:
        assert pack["input_ids"][17] == 154820
        assert pack["cu_seqlens"][-1] == len(pack["input_ids"])
        assert len(set(pack["cu_seqlens"])) == 4
    assert data == helper.prompts(tmp_path)


def test_vllm_client_records_aligned_prompt_logprobs(tmp_path: Path) -> None:
    import httpx

    helper.prompts(tmp_path, lengths=(128,))
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        ids = body["prompt"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "logprobs": {
                            "tokens": [f"token_id:{i}" for i in ids] + ["token_id:4"],
                            "token_logprobs": [None] + [-1.0] * len(ids),
                        }
                    }
                ]
            },
        )

    output = tmp_path / "vllm.json"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        helper.vllm_reference(
            tmp_path, "http://oracle.test/v1", "glm", output, client=client
        )
    records = json.loads(output.read_text())["records"]
    assert len(requests) == len(records) == 3
    assert sum(len(r["prompt_logprobs"]) for r in records) == 128
    assert all(r["prompt_logprobs"][0] is None for r in records)
    oracle.verify_records(
        tmp_path, [json.loads(output.with_suffix(".manifest.json").read_text())]
    )


@requires_native_oracle
def test_packed_capture_and_ce_gradients_do_not_cross_document_boundaries(
    tmp_path: Path,
) -> None:
    from safetensors.torch import load_file

    torch.manual_seed(helper.SEED)
    model = oracle.hf_model_class()(tiny_config()).eval()
    fixture = tmp_path / "R2"
    model.save_pretrained(fixture, save_original_format=False)
    helper.write_json(
        tmp_path / "MANIFEST.json",
        {
            "fixtures": {
                "R2": {
                    "status": "ready",
                    "files": [
                        oracle.file_record(p, tmp_path) for p in fixture.iterdir()
                    ],
                }
            },
            "dumps": {},
        },
    )
    helper.dump("R2", tmp_path, "cpu", lengths=(128,))
    data = json.loads((tmp_path / "MANIFEST.json").read_text())
    oracle.verify_records(tmp_path, data["dumps"]["R2"]["files"])
    pack = json.loads((tmp_path / "prompts-r2/prompts.json").read_text())["packs"][0]
    loss = torch.tensor(0.0)
    model.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)
    with oracle.cpu_reference_kernels():
        for doc, (start, end) in enumerate(
            zip(pack["cu_seqlens"][:-1], pack["cu_seqlens"][1:], strict=True)
        ):
            ids = torch.tensor([pack["input_ids"][start:end]])
            logits = model(ids).logits.float()
            reference = load_file(
                tmp_path / f"dumps/R2/packed-128-doc{doc}.safetensors"
            )
            expected = (
                logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            )
            torch.testing.assert_close(
                reference["target_logprobs"], expected, rtol=1e-5, atol=1e-6
            )
            assert reference["layers.0.input_streams"].shape == (1, end - start, 4, 32)
            assert int(reference["layers.1.indexer.indices"].max()) < end - start
            loss = (
                loss
                + torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 257),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                / 125
            )
        (gradient,) = torch.autograd.grad(loss, [model.lm_head.weight])
    saved = load_file(tmp_path / "dumps/R2/gradients-unembed.safetensors")
    assert saved["target_count"].item() == 125
    assert saved["input_ids"][0, 17].item() == 256
    torch.testing.assert_close(
        saved["ce_loss"].squeeze(), loss.detach(), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(saved["lm_head.weight"], gradient, rtol=1e-5, atol=1e-7)


def test_module_comparison_cli_selects_dump_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.torch import save_file

    dump = tmp_path / "dump.safetensors"
    save_file(
        {"target_logprobs": torch.tensor([[-2.0, -3.0]]), "other": torch.zeros(1)}, dump
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def run_oracle(inputs):\n    return inputs\n")
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hf_oracle",
            str(dump),
            "--module",
            str(candidate),
            "--inputs",
            str(dump),
            "--key",
            "target_logprobs",
            "--output",
            str(output),
        ],
    )
    oracle.main()
    report = json.loads(output.read_text())
    assert report["target_logprobs"]["max_abs"] == 0.0
    assert report["target_logprobs"]["logprob_abs_p99"] == 0.0
    assert "other" not in report
