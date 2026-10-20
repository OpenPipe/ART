import torch

from art.megatron.weights.stark_peft import convert


def test_fused_gdn_and_packed_expert_updates_preserve_each_factor():
    torch.manual_seed(4)
    rank, hidden, intermediate, experts = 2, 7, 5, 3
    original = {}
    prefix = "base_model.model.model.layers.0"
    for part, size in zip(("q", "k", "v"), (4, 4, 6), strict=True):
        original[f"{prefix}.linear_attn.in_proj_{part}.lora_A.weight"] = torch.randn(
            rank, hidden
        )
        original[f"{prefix}.linear_attn.in_proj_{part}.lora_B.weight"] = torch.randn(
            size, rank
        )
    for expert in range(experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            incoming, outgoing = (
                (intermediate, hidden)
                if projection == "down_proj"
                else (hidden, intermediate)
            )
            original[f"{prefix}.mlp.experts.{expert}.{projection}.lora_A.weight"] = (
                torch.randn(rank, incoming)
            )
            original[f"{prefix}.mlp.experts.{expert}.{projection}.lora_B.weight"] = (
                torch.randn(outgoing, rank)
            )
    actual, config = convert(
        original,
        adapter_config={
            "r": rank,
            "lora_alpha": 32,
            "base_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
        },
    )
    public = prefix.replace(".model.layers.", ".model.language_model.layers.")
    a = actual[f"{public}.linear_attn.in_proj_qkv.lora_A.weight"]
    b = actual[f"{public}.linear_attn.in_proj_qkv.lora_B.weight"]
    expected = torch.cat(
        [
            original[f"{prefix}.linear_attn.in_proj_{p}.lora_B.weight"]
            @ original[f"{prefix}.linear_attn.in_proj_{p}.lora_A.weight"]
            for p in ("q", "k", "v")
        ]
    )
    torch.testing.assert_close(b @ a, expected)
    assert config["rank_pattern"]["in_proj_qkv"] == 3 * rank
    assert config["alpha_pattern"]["in_proj_qkv"] == 96
    assert (
        config["alpha_pattern"]["gate_up_proj"] / config["rank_pattern"]["gate_up_proj"]
        == 32 / rank
    )
    for nested, projections, rows in (
        (".base_layer", ("gate_proj", "up_proj"), intermediate * 2),
        ("", ("down_proj",), hidden),
    ):
        a = actual[f"{public}.mlp.experts{nested}.lora_A.weight"].reshape(
            experts, -1, hidden if nested else intermediate
        )
        b = actual[f"{public}.mlp.experts{nested}.lora_B.weight"].reshape(
            rows, -1, experts
        )
        delta = torch.einsum("ore,eri->eoi", b, a)
        expected = torch.stack(
            [
                torch.cat(
                    [
                        original[f"{prefix}.mlp.experts.{e}.{p}.lora_B.weight"]
                        @ original[f"{prefix}.mlp.experts.{e}.{p}.lora_A.weight"]
                        for p in projections
                    ]
                )
                for e in range(experts)
            ]
        )
        torch.testing.assert_close(delta, expected)
