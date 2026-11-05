"""CPU geometry, real PEFT merge, scaling and publication contract checks."""

from dataclasses import replace
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from art.megatron.glm5next.lora import (
    ART_PREFIX,
    LoraDimensions,
    LoraFlags,
    lora_targets,
    parameter_counts,
)
from art.megatron.glm5next.lora_export import (
    META,
    adapter_config,
    from_peft_lora_tensors,
    from_vllm_lora_tensors,
    gather_and_export_lora,
    to_peft_lora_tensors,
    to_vllm_lora_tensors,
)

FLAGS = [LoraFlags(*values) for values in itertools.product((False, True), repeat=3)]
TINY = LoraDimensions(
    hidden=8,
    kda_width=12,
    q_rank=4,
    kv_rank=2,
    mla_width=16,
    dense_width=10,
    expert_width=6,
    experts=4,
    vocab=17,
    dense_layers=3,
    layers=4,
    mla_layers=(3,),
)


def native_state(rank=8, flags=LoraFlags(), dimensions=TINY, dtype=torch.float32):
    rng = torch.Generator().manual_seed(412)
    state = {}
    for target in lora_targets(dimensions, flags):
        a, b = [
            torch.randn(s, generator=rng, dtype=dtype) * 0.04
            for s in target.shapes(rank)
        ]
        for expert in range(target.experts or 1):
            for factor, value in (("A", a), ("B", b)):
                if target.experts:
                    value = value[0 if target.shared == factor else expert]
                state[target.name.format(expert=expert) + f".lora_{factor}.weight"] = (
                    value
                )
    return state


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("rank", (1, 8, 32))
def test_exact_counts_derived_from_bf16_shape_dump(flags, rank):
    meta = json.loads(
        (Path(__file__).parent / "fixtures/glm5next_lora_hfmeta.json").read_text()
    )
    dimensions = LoraDimensions.from_hfmeta(meta["shapes"], meta["config"])
    assert dimensions == LoraDimensions()
    counts = parameter_counts(rank, dimensions=dimensions, flags=flags)
    assert counts == {
        "attn": 2_009_088 * rank * flags.train_attn,
        "mlp": 75_755_520 * rank * flags.train_mlp,
        "unembed": 158_976 * rank * flags.train_unembed,
    }
    if flags == LoraFlags():
        assert sum(counts.values()) == 77_923_584 * rank


def test_exact_target_set_and_shared_geometry():
    targets = lora_targets()
    attn = [t for t in targets if t.group == "attn"]
    assert len(attn) == 34 * 4 + 11 * 3
    assert {t.name.rsplit(".", 1)[-1] for t in attn} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
    }
    for t in targets:
        assert not any(
            x in t.name
            for x in (
                "indexer",
                "q_b_proj",
                "kv_b_proj",
                "f_a_proj",
                "g_a_proj",
                "b_proj",
            )
        )
        if t.experts:
            a, b = t.shapes(32)
            assert (a[0], b[0]) == ((288, 1) if t.shared == "B" else (1, 288))


@pytest.mark.parametrize("rank", (0, 33, True, 1.5))
def test_rank_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        adapter_config(rank)


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize(
    "codec,decode",
    (
        (to_vllm_lora_tensors, from_vllm_lora_tensors),
        (to_peft_lora_tensors, from_peft_lora_tensors),
    ),
)
def test_roundtrip_flags_and_random_factors(flags, codec, decode):
    state = native_state(flags=flags)
    config = adapter_config(8, alpha=7.5, dimensions=TINY, flags=flags)
    exported, published = codec(state, adapter_config=config)
    restored = decode(exported, adapter_config=published)
    assert restored.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(restored[name], state[name], rtol=0, atol=0)
    assert config[META]["format"] == "art"


def test_vllm_stack_names_shapes_and_alpha_over_rank():
    state = native_state()
    exported, config = to_vllm_lora_tensors(
        state, adapter_config=adapter_config(8, alpha=32, dimensions=TINY)
    )
    prefix = "base_model.model.language_model.model.layers.3.mlp.experts."
    assert {
        k.removeprefix(prefix): tuple(v.shape)
        for k, v in exported.items()
        if k.startswith(prefix)
    } == {
        "w1.lora_A.weight": (1, 8, 8),
        "w1.lora_B.weight": (4, 6, 8),
        "w2.lora_A.weight": (4, 8, 6),
        "w2.lora_B.weight": (1, 8, 8),
        "w3.lora_A.weight": (1, 8, 8),
        "w3.lora_B.weight": (4, 6, 8),
    }
    assert "base_model.model.language_model.lm_head.lora_A.weight" in exported
    assert config["r"] == 8 and config["lora_alpha"] == 32
    # Independent sampler math, including nonzero factors on both sides.
    for projection, slot in (
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
    ):
        a, b = (exported[prefix + slot + f".lora_{f}.weight"] for f in ("A", "B"))
        for e in range(TINY.experts):
            native = f"{ART_PREFIX}3.mlp.experts.{e}.{projection}"
            expected = (
                4 * state[native + ".lora_B.weight"] @ state[native + ".lora_A.weight"]
            )
            actual = (
                4
                * b[0 if projection == "down_proj" else e]
                @ a[e if projection == "down_proj" else 0]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vllm_inplace_scale_cannot_mutate_native_snapshot():
    state = native_state()
    original = {k: v.clone() for k, v in state.items()}
    exported, config = to_vllm_lora_tensors(
        state, adapter_config=adapter_config(8, dimensions=TINY)
    )
    for key, tensor in exported.items():
        if key.endswith(".lora_B.weight"):
            tensor.mul_(config["lora_alpha"] / config["r"])
    for key in state:
        torch.testing.assert_close(state[key], original[key], rtol=0, atol=0)
    # A reload starts from immutable artifact bytes, or a fresh export/copy.
    again, _ = to_vllm_lora_tensors(
        state, adapter_config=adapter_config(8, dimensions=TINY)
    )
    restored = from_vllm_lora_tensors(again, adapter_config=config)
    for key in state:
        torch.testing.assert_close(restored[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation", ("missing", "extra", "shape", "independent", "nan", "expert_rank")
)
def test_bad_native_artifact_fails_before_publication(mutation):
    state = native_state()
    name = f"{ART_PREFIX}3.mlp.experts.1.gate_proj.lora_A.weight"
    if mutation == "missing":
        del state[name]
    elif mutation == "extra":
        state[name.replace("gate_proj", "q_b_proj")] = state[name]
    elif mutation == "shape":
        state[name] = state[name].T
    elif mutation == "independent":
        state[name] = state[name] + 1
    elif mutation == "nan":
        state[name] = torch.full_like(state[name], float("nan"))
    else:
        state[name] = state[name][:1]
    with pytest.raises(ValueError):
        to_vllm_lora_tensors(state, adapter_config=adapter_config(8, dimensions=TINY))


@pytest.mark.parametrize(
    "option",
    ("rank_pattern", "alpha_pattern", "use_rslora", "use_dora", "lora_dropout"),
)
def test_reduced_expert_scale_and_variants_rejected(option):
    config = adapter_config(8, dimensions=TINY)
    config[option] = {"mlp.experts": 1} if "pattern" in option else True
    with pytest.raises(ValueError, match="uniform"):
        to_vllm_lora_tensors(native_state(), adapter_config=config)


def test_peft_padding_and_shared_ties_checked():
    exported, config = to_peft_lora_tensors(
        native_state(), adapter_config=adapter_config(8, dimensions=TINY)
    )
    key = "base_model.model.lm_head.lora_A.weight"
    exported[key][9, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        from_peft_lora_tensors(exported, adapter_config=config)


@pytest.mark.parametrize("format", ("vllm", "peft"))
def test_ep_gather_uses_art_merge_matches_single_rank(format):
    pytest.importorskip("megatron.core")
    state = native_state()
    config = adapter_config(8, dimensions=TINY)
    entries: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    # Fake EP4: each logical expert is owned by a different rank. Ordinary
    # replicas have already been selected by ART's canonical owner policy.
    fake_ranks: list[dict[str, list[tuple[dict[str, Any], Any]]]] = [{}, {}, {}, {}]
    for key, tensor in state.items():
        manifest = {"sharded": False, "shard_world_size": 1, "shard_rank": 0}
        owner = (
            int(key.split(".experts.")[1].split(".")[0]) if ".experts." in key else 0
        )
        fake_ranks[owner][key] = [(manifest, tensor)]
    for local in reversed(fake_ranks):
        for key, values in local.items():
            entries.setdefault(key, []).extend(values)
    # Exercise ART's actual shard ordering too, using the head's input axis.
    key = "base_model.model.lm_head.lora_A.weight"
    entries[key] = [
        (
            {
                "sharded": True,
                "shard_world_size": 2,
                "shard_rank": i,
                "export_shard_strategy": "uniform",
                "export_shard_dim": 1,
            },
            t,
        )
        for i, t in reversed(list(enumerate(state[key].chunk(2, dim=1))))
    ]
    gathered, gc = gather_and_export_lora(entries, adapter_config=config, format=format)
    codec = to_vllm_lora_tensors if format == "vllm" else to_peft_lora_tensors
    single, sc = codec(state, adapter_config=config)
    assert gc == sc
    assert gathered.keys() == single.keys()
    for key in single:
        torch.testing.assert_close(gathered[key], single[key], rtol=0, atol=0)
    missing_rank = {k: v for k, v in entries.items() if k not in fake_ranks[2]}
    with pytest.raises(ValueError, match="Missing"):
        gather_and_export_lora(missing_rank, adapter_config=config, format=format)
    duplicate = "base_model.model.lm_head.lora_B.weight"
    entries[duplicate].append(entries[duplicate][0])
    with pytest.raises(RuntimeError, match="expected 1 shard|Duplicate shard"):
        gather_and_export_lora(entries, adapter_config=config, format=format)


def test_peft_020_actual_load_and_merge(tmp_path):
    peft = pytest.importorskip("peft", minversion="0.20.0")
    from safetensors.torch import save_file

    # Same fused [E,out,in] parameter orientation as HF GLM5NextTextExperts;
    # load through real PEFT instead of reimplementing its packing convention.
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(4, 12, 8))
            self.down_proj = torch.nn.Parameter(torch.zeros(4, 8, 6))

        def forward(self, x):
            gate, up = torch.nn.functional.linear(x, self.gate_up_proj[2]).chunk(2, -1)
            return torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, self.down_proj[2]
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.language_model.layers[0].mlp = torch.nn.Module()
            self.model.language_model.layers[0].mlp.experts = Experts()
            self.model.language_model.layers[0].mlp.shared_experts = torch.nn.Module()
            for name, inp, out in (
                ("gate_proj", 8, 6),
                ("up_proj", 8, 6),
                ("down_proj", 6, 8),
            ):
                setattr(
                    self.model.language_model.layers[0].mlp.shared_experts,
                    name,
                    torch.nn.Linear(inp, out, bias=False),
                )

        def forward(self, x):
            return self.model.language_model.layers[0].mlp.experts(x)

    dims = replace(TINY, layers=1, dense_layers=0, mla_layers=())
    flags = LoraFlags(False, True, False)
    state = native_state(dimensions=dims, flags=flags)
    config = adapter_config(8, alpha=7.5, dimensions=dims, flags=flags)
    tensors, published = to_peft_lora_tensors(state, adapter_config=config)
    # Generic PeftModel allows a minimal fixture without generation methods.
    published["task_type"] = None
    (tmp_path / "adapter_config.json").write_text(json.dumps(published))
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    with pytest.warns(UserWarning, match="Unexpected keyword arguments"):
        adapted = peft.PeftModel.from_pretrained(Model(), tmp_path)
    experts = adapted.base_model.model.model.language_model.layers[0].mlp.experts
    base = experts.get_base_layer()
    for name in ("gate_up_proj", "down_proj"):
        actual = getattr(base, name).clone()
        assert torch.count_nonzero(actual) == 0
    merged = adapted.merge_and_unload()
    weights = merged.model.language_model.layers[0].mlp.experts
    prefix = ART_PREFIX + "0.mlp.experts."
    for e in range(4):
        delta = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            p = prefix + str(e) + "." + name
            delta[name] = (
                (7.5 / 8) * state[p + ".lora_B.weight"] @ state[p + ".lora_A.weight"]
            )
        torch.testing.assert_close(
            weights.gate_up_proj[e], torch.cat((delta["gate_proj"], delta["up_proj"]))
        )
        torch.testing.assert_close(weights.down_proj[e], delta["down_proj"])


def test_peft_empty_flags_load_as_identity(tmp_path):
    peft = pytest.importorskip("peft", minversion="0.20.0")
    from safetensors.torch import save_file

    config = adapter_config(8, dimensions=TINY, flags=LoraFlags(False, False, False))
    tensors, config = to_peft_lora_tensors({}, adapter_config=config)
    config["task_type"] = None
    assert config["target_modules"] == "dummy-target-modules"
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    base = torch.nn.Linear(3, 4, bias=False)
    x = torch.randn(2, 3)
    expected = base(x).clone()
    with pytest.warns(UserWarning, match="Unexpected keyword arguments"):
        model = peft.PeftModel.from_pretrained(base, tmp_path)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
