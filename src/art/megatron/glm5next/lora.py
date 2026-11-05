"""GLM-5.3 Flash's Tinker target selection and unique-parameter geometry.

This module is intentionally importable without Megatron/CUDA. Runtime wrappers
are imported only by ``apply_glm5next_lora``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

ART_PREFIX = "base_model.model.model.layers."
HEAD_PREFIX = "base_model.model.lm_head"
MAX_RANK = 32


@dataclass(frozen=True)
class LoraFlags:
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = True

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in vars(self).values()):
            raise TypeError("GLM LoRA flags must be bools")


@dataclass(frozen=True)
class LoraDimensions:
    hidden: int = 4096
    kda_width: int = 8192
    q_rank: int = 1536
    kv_rank: int = 512
    mla_width: int = 16384
    dense_width: int = 12288
    expert_width: int = 2048
    experts: int = 288
    vocab: int = 154880
    dense_layers: int = 3
    layers: int = 45
    mla_layers: tuple[int, ...] = tuple(range(3, 45, 4))

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name not in {"mla_layers", "dense_layers"}:
                if type(value) is not int or value < 1:
                    raise ValueError(f"Invalid GLM dimension {name}={value}")
        if not 0 <= self.dense_layers <= self.layers:
            raise ValueError("Invalid dense layer count")
        if len(set(self.mla_layers)) != len(self.mla_layers) or any(
            not 0 <= layer < self.layers for layer in self.mla_layers
        ):
            raise ValueError("Invalid MLA layer indices")

    @classmethod
    def from_config(cls, config: Any) -> LoraDimensions:
        def get(obj: Any, key: str, default: Any = None) -> Any:
            return (
                obj.get(key, default)
                if isinstance(obj, Mapping)
                else getattr(obj, key, default)
            )

        c = get(config, "text_config", config)
        if (
            int(get(c, "n_shared_experts", 1)) != 1
            or int(get(c, "qk_rope_head_dim", 0)) != 0
        ):
            raise ValueError("Flash LoRA requires one shared expert and NoPE MLA")
        layers = int(get(c, "num_hidden_layers"))
        kinds = get(c, "layer_types")
        mlp_kinds = get(c, "mlp_layer_types")
        dense_layers = int(get(c, "first_k_dense_replace", min(3, layers)))
        if mlp_kinds:
            dense_layers = next(
                (i for i, kind in enumerate(mlp_kinds) if kind != "dense"), layers
            )
            if any(kind == "dense" for kind in mlp_kinds[dense_layers:]):
                raise ValueError(
                    "GLM LoRA requires leading dense layers followed by MoE"
                )
        linear = get(c, "linear_attn_config", {})
        return cls(
            hidden=int(get(c, "hidden_size")),
            kda_width=int(get(c, "linear_num_heads", get(linear, "num_heads", 64)))
            * int(get(c, "linear_head_dim", get(linear, "head_dim", 128))),
            q_rank=int(get(c, "q_lora_rank")),
            kv_rank=int(get(c, "kv_lora_rank")),
            mla_width=int(get(c, "num_attention_heads")) * int(get(c, "v_head_dim")),
            dense_width=int(get(c, "intermediate_size")),
            expert_width=int(get(c, "moe_intermediate_size")),
            experts=int(get(c, "n_routed_experts", get(c, "num_local_experts"))),
            vocab=int(get(c, "vocab_size")),
            dense_layers=dense_layers,
            layers=layers,
            mla_layers=tuple(
                i
                for i in range(layers)
                if (kinds[i] != "linear_attention" if kinds else i % 4 == 3)
            ),
        )

    @classmethod
    def from_hfmeta(cls, shapes: Mapping[str, Any], config: Any) -> LoraDimensions:
        """Derive target widths from the supplied BF16 shape dump, not counts."""
        from dataclasses import replace

        dims = cls.from_config(config)

        def shape(name: str) -> tuple[int, ...]:
            return tuple(shapes[name][1])

        p = "model.language_model.layers."
        q = shape(p + "0.self_attn.q_proj.weight")
        qa = shape(p + "3.self_attn.q_a_proj.weight")
        kva = shape(p + "3.self_attn.kv_a_proj_with_mqa.weight")
        derived = replace(
            dims,
            hidden=q[1],
            kda_width=q[0],
            q_rank=qa[0],
            kv_rank=kva[0],
            mla_width=shape(p + "3.self_attn.o_proj.weight")[1],
            dense_width=shape(p + "0.mlp.gate_proj.weight")[0],
            expert_width=shape(p + "4.mlp.experts.0.gate_proj.weight")[0],
            vocab=shape("lm_head.weight")[0],
        )
        # Check every supplied target shape, including the reverse projections;
        # a width borrowed from q/gate must not conceal inconsistent k/up/down.
        templates = {
            "0.self_attn.q_proj": (derived.kda_width, derived.hidden),
            "0.self_attn.k_proj": (derived.kda_width, derived.hidden),
            "0.self_attn.v_proj": (derived.kda_width, derived.hidden),
            "0.self_attn.o_proj": (derived.hidden, derived.kda_width),
            "3.self_attn.q_a_proj": (derived.q_rank, derived.hidden),
            "3.self_attn.kv_a_proj_with_mqa": (derived.kv_rank, derived.hidden),
            "3.self_attn.o_proj": (derived.hidden, derived.mla_width),
            "0.mlp.gate_proj": (derived.dense_width, derived.hidden),
            "0.mlp.up_proj": (derived.dense_width, derived.hidden),
            "0.mlp.down_proj": (derived.hidden, derived.dense_width),
            "4.mlp.experts.0.gate_proj": (derived.expert_width, derived.hidden),
            "4.mlp.experts.0.up_proj": (derived.expert_width, derived.hidden),
            "4.mlp.experts.0.down_proj": (derived.hidden, derived.expert_width),
        }
        for module, expected in templates.items():
            if shape(p + module + ".weight") != expected:
                raise ValueError(f"Inconsistent GLM target weight shape: {module}")
        if shape("lm_head.weight") != (derived.vocab, derived.hidden):
            raise ValueError("Inconsistent GLM lm_head shape")
        return derived


@dataclass(frozen=True)
class LoraTarget:
    name: str
    group: Literal["attn", "mlp", "unembed"]
    in_features: int
    out_features: int
    experts: int = 0
    shared: Literal["A", "B"] | None = None

    def shapes(self, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        validate_rank(rank)
        if not self.experts:
            return (rank, self.in_features), (self.out_features, rank)
        return (
            (1 if self.shared == "A" else self.experts, rank, self.in_features),
            (1 if self.shared == "B" else self.experts, self.out_features, rank),
        )

    def count(self, rank: int) -> int:
        return sum(math.prod(s) for s in self.shapes(rank))


def validate_rank(rank: int) -> None:
    if type(rank) is not int or not 1 <= rank <= MAX_RANK:
        raise ValueError(f"GLM LoRA rank must be an integer in 1..{MAX_RANK}")


def lora_targets(
    dimensions: LoraDimensions = LoraDimensions(), flags: LoraFlags = LoraFlags()
) -> tuple[LoraTarget, ...]:
    d = dimensions
    result: list[LoraTarget] = []
    for layer in range(d.layers):
        p = f"{ART_PREFIX}{layer}"
        if flags.train_attn:
            projections = (
                (("q_a_proj", d.q_rank), ("kv_a_proj_with_mqa", d.kv_rank))
                if layer in d.mla_layers
                else tuple((name + "_proj", d.kda_width) for name in ("q", "k", "v"))
            )
            result.extend(
                LoraTarget(f"{p}.self_attn.{name}", "attn", d.hidden, out)
                for name, out in projections
            )
            result.append(
                LoraTarget(
                    f"{p}.self_attn.o_proj",
                    "attn",
                    d.mla_width if layer in d.mla_layers else d.kda_width,
                    d.hidden,
                )
            )
        if flags.train_mlp:
            width = d.dense_width if layer < d.dense_layers else d.expert_width
            prefix = f"{p}.mlp" + ("" if layer < d.dense_layers else ".shared_experts")
            for name in ("gate_proj", "up_proj", "down_proj"):
                down = name == "down_proj"
                result.append(
                    LoraTarget(
                        f"{prefix}.{name}",
                        "mlp",
                        width if down else d.hidden,
                        d.hidden if down else width,
                    )
                )
                if layer >= d.dense_layers:
                    result.append(
                        LoraTarget(
                            f"{p}.mlp.experts.{{expert}}.{name}",
                            "mlp",
                            d.expert_width if down else d.hidden,
                            d.hidden if down else d.expert_width,
                            d.experts,
                            "B" if down else "A",
                        )
                    )
    if flags.train_unembed:
        result.append(LoraTarget(HEAD_PREFIX, "unembed", d.hidden, d.vocab))
    return tuple(result)


def parameter_counts(
    rank: int,
    *,
    dimensions: LoraDimensions = LoraDimensions(),
    flags: LoraFlags = LoraFlags(),
) -> dict[str, int]:
    validate_rank(rank)
    result = dict.fromkeys(("attn", "mlp", "unembed"), 0)
    for target in lora_targets(dimensions, flags):
        result[target.group] += target.count(rank)
    return result


def apply_glm5next_lora(
    model_chunks: Sequence[Any],
    provider: Any,
    *,
    rank: int,
    alpha: int = 32,
    flags: LoraFlags = LoraFlags(),
) -> None:
    """M4 installation entrypoint; TP1, ART shared-outer expert wrappers.

    Attention bindings are installed by ``lora_layers`` after the corresponding
    lane has posted its projection interface. No generic target suffix matching.
    """
    validate_rank(rank)
    from art.megatron.glm5next.lora_layers import install_glm5next_lora

    install_glm5next_lora(model_chunks, provider, rank=rank, alpha=alpha, flags=flags)
