from types import SimpleNamespace

import torch

from art.megatron.portable_optimizer_archive import (
    LoadedPortableOptimizerArchive,
    PortableOptimizerArchiveMetadata,
    portable_optimizer_logical_keys_for_sites,
    reconstruct_portable_optimizer_components,
    reconstruct_trainer_rank_optimizer_state,
)
from art.trainer_rank._impl import _optimizer_parameter_key_owners


class _DestinationModule:
    def __init__(self, a_key: str, b_key: str) -> None:
        self._keys = {"lora_A": (a_key,), "lora_B": (b_key,)}

    def _expected_weight_keys_for_param(
        self, suffix: str, _parameter: torch.Tensor
    ) -> tuple[str, ...]:
        return self._keys[suffix]

    def _adapter_weight(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        suffix: str,
        moe_parameterization: object,
    ) -> torch.Tensor:
        del moe_parameterization
        (key,) = self._keys[suffix]
        return tensors[key]

    @staticmethod
    def _localized_weight(weight: torch.Tensor, *, into: torch.Tensor) -> torch.Tensor:
        assert weight.shape == into.shape
        return weight


class _ExpertDestinationModule(_DestinationModule):
    def __init__(self) -> None:
        self._keys = {
            "lora_A": ("layer.shared.lora_A.weight",),
            "lora_B": (
                "layer.0.lora_B.weight",
                "layer.1.lora_B.weight",
            ),
        }

    def _adapter_weight(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        suffix: str,
        moe_parameterization: object,
    ) -> torch.Tensor:
        assert moe_parameterization == "shared_outer"
        keys = self._keys[suffix]
        if len(keys) == 1:
            return tensors[keys[0]]
        return torch.stack([tensors[key] for key in keys])


class _ShardedDestinationModule(_DestinationModule):
    def __init__(self, start: int) -> None:
        super().__init__("layer.lora_A.weight", "layer.lora_B.weight")
        self._start = start

    def _localized_weight(
        self, weight: torch.Tensor, *, into: torch.Tensor
    ) -> torch.Tensor:
        return weight.narrow(1, self._start, into.shape[1]).contiguous()


class _PaddedExpertDestinationModule:
    def __init__(self, expert_ids: tuple[int | None, ...]) -> None:
        self.expert_ids = expert_ids

    def _expected_weight_keys_for_param(
        self, suffix: str, _parameter: torch.Tensor
    ) -> tuple[str, ...]:
        return tuple(
            f"layer.{expert}.{suffix}.weight"
            for expert in self.expert_ids
            if expert is not None
        )

    def _adapter_weight(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        suffix: str,
        moe_parameterization: object,
    ) -> torch.Tensor:
        del moe_parameterization
        real = iter(
            tensors[key]
            for key in self._expected_weight_keys_for_param(
                suffix, torch.empty(0)
            )
        )
        first = next(iter(tensors.values()))
        return torch.stack(
            [torch.zeros_like(first) if expert is None else next(real) for expert in self.expert_ids]
        )

    @staticmethod
    def _localized_weight(weight: torch.Tensor, *, into: torch.Tensor) -> torch.Tensor:
        assert weight.shape == into.shape
        return weight


def _archive(
    *,
    source_rank: int,
    values: dict[str, float],
    shape: tuple[int, ...] = (2, 3),
    source_world_size: int = 2,
) -> LoadedPortableOptimizerArchive:
    keys = tuple(sorted(values))
    tensors = {
        f"{component}/{key}": torch.full(shape, value + offset)
        for component, offset in (
            ("master", 0.0),
            ("exp_avg", 10.0),
            ("exp_avg_sq", 20.0),
        )
        for key, value in values.items()
    }
    return LoadedPortableOptimizerArchive(
        metadata=PortableOptimizerArchiveMetadata(
            source_rank=source_rank,
            source_world_size=source_world_size,
            logical_keys=keys,
            steps=dict.fromkeys(keys, 31.0),
            param_group={"lr": 3e-5, "betas": [0.9, 0.95], "step": 31},
        ),
        loaded_logical_keys=keys,
        tensors=tensors,
    )


def test_logical_optimizer_repartitions_across_source_rank_ownership() -> None:
    # The destination site intentionally combines one tensor from each source
    # rank. A topology-strict checkpoint could not reconstruct this placement.
    module = _DestinationModule("layer.lora_A.weight", "layer.lora_B.weight")
    slot = SimpleNamespace(
        A_T=torch.nn.Parameter(torch.empty(2, 3)),
        B_T=torch.nn.Parameter(torch.empty(2, 3)),
    )
    sites = ((module, slot),)
    keys = portable_optimizer_logical_keys_for_sites(sites)
    assert keys == ("layer.lora_A.weight", "layer.lora_B.weight")

    components = reconstruct_portable_optimizer_components(
        (
            _archive(source_rank=0, values={"layer.lora_B.weight": 2.0}),
            _archive(source_rank=1, values={"layer.lora_A.weight": 1.0}),
        )
    )
    layout = {
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("A",), (2, 3), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (2, 3), "torch.float32", "cpu", True, None, "", ()),
        ),
    }
    state = reconstruct_trainer_rank_optimizer_state(components, sites, layout)

    torch.testing.assert_close(state["master_params"][0], torch.full((2, 3), 1.0))
    torch.testing.assert_close(state["master_params"][1], torch.full((2, 3), 2.0))
    optimizer = state["optimizer"]
    assert optimizer["param_groups"] == [
        {"lr": 3e-5, "betas": [0.9, 0.95], "step": 31, "params": [0, 1]}
    ]
    optimizer_state = optimizer["state"]
    torch.testing.assert_close(optimizer_state[0]["exp_avg"], torch.full((2, 3), 11.0))
    torch.testing.assert_close(
        optimizer_state[1]["exp_avg_sq"], torch.full((2, 3), 22.0)
    )
    assert "step" not in optimizer_state[0]
    assert "step" not in optimizer_state[1]


def test_shared_outer_reconstruction_combines_shared_and_expert_keys() -> None:
    module = _ExpertDestinationModule()
    a = torch.nn.Parameter(torch.empty(2, 3))
    b = torch.nn.Parameter(torch.empty(2, 2, 3))
    setattr(a, "lora_moe_parameterization", "shared_outer")
    setattr(b, "lora_moe_parameterization", "shared_outer")
    sites = ((module, SimpleNamespace(A_T=a, B_T=b)),)
    assert portable_optimizer_logical_keys_for_sites(sites) == (
        "layer.0.lora_B.weight",
        "layer.1.lora_B.weight",
        "layer.shared.lora_A.weight",
    )
    components = reconstruct_portable_optimizer_components(
        (
            _archive(
                source_rank=0,
                values={
                    "layer.0.lora_B.weight": 2.0,
                    "layer.shared.lora_A.weight": 1.0,
                },
            ),
            _archive(source_rank=1, values={"layer.1.lora_B.weight": 2.0}),
        )
    )
    layout = {
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("A",), (2, 3), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (2, 2, 3), "torch.float32", "cpu", True, None, "", ()),
        ),
    }
    state = reconstruct_trainer_rank_optimizer_state(components, sites, layout)
    torch.testing.assert_close(state["master_params"][0], torch.full((2, 3), 1.0))
    torch.testing.assert_close(
        state["master_params"][1], torch.full((2, 2, 3), 2.0)
    )
    assert state["optimizer"]["param_groups"][0]["step"] == 31


def test_cp2_ep2_source_semantics_reconstruct_into_cp2_ep1_destination() -> None:
    """Compare logical Adam state, not source/destination layout bytes.

    Four archives model CP2/EP2 source ownership, including the CP ranks that
    contribute no additional logical keys. One destination parameter per LoRA
    side models EP1 on each CP rank. Packed length is intentionally absent from
    reconstruction because it does not change logical optimizer state.
    """
    source = reconstruct_portable_optimizer_components(
        (
            _archive(
                source_rank=0,
                source_world_size=4,
                values={
                    "layer.0.lora_A.weight": 1.0,
                    "layer.0.lora_B.weight": 2.0,
                },
            ),
            _archive(
                source_rank=1,
                source_world_size=4,
                values={
                    "layer.1.lora_A.weight": 1.0,
                    "layer.1.lora_B.weight": 2.0,
                },
            ),
            # The other CP replica contributes no additional logical state.
            # Its empty archives still prove that zero-key source ranks are a
            # valid part of the topology-neutral generation.
            _archive(source_rank=2, source_world_size=4, values={}),
            _archive(source_rank=3, source_world_size=4, values={}),
        )
    )
    module = _PaddedExpertDestinationModule((0, 1))
    a = torch.nn.Parameter(torch.empty(2, 2, 3))
    b = torch.nn.Parameter(torch.empty(2, 2, 3))
    sites = ((module, SimpleNamespace(A_T=a, B_T=b)),)
    destination_layout = {
        # Physical placement differs from the source archives and is not part
        # of the semantic comparison.
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("A",), (2, 2, 3), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (2, 2, 3), "torch.float32", "cpu", True, None, "", ()),
        ),
    }

    # Both destination CP ranks reconstruct the same topology-neutral semantic
    # state despite different source ownership and a changed physical layout.
    restored_by_cp_rank = tuple(
        reconstruct_trainer_rank_optimizer_state(source, sites, destination_layout)
        for _destination_cp_rank in range(2)
    )

    for restored in restored_by_cp_rank:
        for parameter_index, suffix in enumerate(("lora_A", "lora_B")):
            state = restored["optimizer"]["state"][parameter_index]
            for expert_index in range(2):
                key = f"layer.{expert_index}.{suffix}.weight"
                torch.testing.assert_close(
                    restored["master_params"][parameter_index][expert_index],
                    source.master[key],
                )
                torch.testing.assert_close(
                    state["exp_avg"][expert_index], source.exp_avg[key]
                )
                torch.testing.assert_close(
                    state["exp_avg_sq"][expert_index], source.exp_avg_sq[key]
                )
                assert "step" not in state
                assert restored["optimizer"]["param_groups"][0]["step"] == 31

    for parameter_index in range(2):
        torch.testing.assert_close(
            restored_by_cp_rank[0]["master_params"][parameter_index],
            restored_by_cp_rank[1]["master_params"][parameter_index],
        )


def test_logical_optimizer_localizes_into_new_destination_shards() -> None:
    components = reconstruct_portable_optimizer_components(
        (
            _archive(
                source_rank=0,
                values={"layer.lora_A.weight": 1.0},
                shape=(2, 4),
            ),
            _archive(
                source_rank=1,
                values={"layer.lora_B.weight": 2.0},
                shape=(2, 4),
            ),
        )
    )
    layout = {
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("A",), (2, 2), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (2, 2), "torch.float32", "cpu", True, None, "", ()),
        ),
    }
    for start in (0, 2):
        module = _ShardedDestinationModule(start)
        sites = (
            (
                module,
                SimpleNamespace(
                    A_T=torch.nn.Parameter(torch.empty(2, 2)),
                    B_T=torch.nn.Parameter(torch.empty(2, 2)),
                ),
            ),
        )
        state = reconstruct_trainer_rank_optimizer_state(components, sites, layout)
        torch.testing.assert_close(
            state["master_params"][0], torch.full((2, 2), 1.0)
        )
        torch.testing.assert_close(
            state["optimizer"]["state"][1]["exp_avg_sq"],
            torch.full((2, 2), 22.0),
        )


def test_padding_only_destination_uses_zero_optimizer_coordinates() -> None:
    source = _archive(source_rank=0, values={"layer.0.lora_A.weight": 1.0})
    metadata_only = LoadedPortableOptimizerArchive(
        metadata=source.metadata,
        loaded_logical_keys=(),
        tensors={},
    )
    components = reconstruct_portable_optimizer_components((metadata_only,))
    module = _PaddedExpertDestinationModule((None, None))
    parameter = torch.nn.Parameter(torch.empty(2, 2, 3))
    sites = ((module, SimpleNamespace(A_T=parameter, B_T=parameter.clone())),)
    layout = {
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("A",), (2, 2, 3), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (2, 2, 3), "torch.float32", "cpu", True, None, "", ()),
        ),
    }

    state = reconstruct_trainer_rank_optimizer_state(components, sites, layout)

    assert state["optimizer"]["param_groups"][0]["lr"] == 3e-5
    for index, master in enumerate(state["master_params"]):
        torch.testing.assert_close(master, torch.zeros_like(master))
        optimizer_state = state["optimizer"]["state"][index]
        torch.testing.assert_close(
            optimizer_state["exp_avg"], torch.zeros_like(master)
        )
        torch.testing.assert_close(
            optimizer_state["exp_avg_sq"], torch.zeros_like(master)
        )
        assert "step" not in optimizer_state
        assert state["optimizer"]["param_groups"][0]["step"] == 31


def test_mixed_expert_destination_maps_keys_around_physical_padding() -> None:
    module = _PaddedExpertDestinationModule((4, None, 7))
    parameter = torch.nn.Parameter(torch.empty(3, 2, 3))
    keys = module._expected_weight_keys_for_param("lora_A", parameter)

    owners, padding = _optimizer_parameter_key_owners(module, parameter, keys)

    assert owners == ((0, keys[0]), (2, keys[1]))
    assert padding == (1,)
    source = _archive(
        source_rank=0,
        values={
            keys[0]: 1.0,
            keys[1]: 2.0,
            "layer.4.lora_B.weight": 3.0,
            "layer.7.lora_B.weight": 4.0,
        },
    )
    source = source.model_copy(
        update={
            "metadata": source.metadata.model_copy(
                update={"steps": dict.fromkeys(source.metadata.logical_keys, 31.0)}
            )
        }
    )
    components = reconstruct_portable_optimizer_components((source,))
    sites = ((module, SimpleNamespace(A_T=parameter, B_T=parameter.clone())),)
    layout = {
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("A",), (3, 2, 3), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (3, 2, 3), "torch.float32", "cpu", True, None, "", ()),
        ),
    }

    state = reconstruct_trainer_rank_optimizer_state(components, sites, layout)

    master = state["master_params"][0]
    torch.testing.assert_close(master[0], torch.full((2, 3), 1.0))
    torch.testing.assert_close(master[1], torch.zeros((2, 3)))
    torch.testing.assert_close(master[2], torch.full((2, 3), 2.0))
