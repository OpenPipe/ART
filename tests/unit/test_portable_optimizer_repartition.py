from types import SimpleNamespace

import torch

from art.megatron.portable_optimizer_archive import (
    LoadedPortableOptimizerArchive,
    PortableOptimizerArchiveMetadata,
    portable_optimizer_logical_keys_for_sites,
    reconstruct_portable_optimizer_components,
    reconstruct_trainer_rank_optimizer_state,
)


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


def _archive(
    *, source_rank: int, values: dict[str, float], shape: tuple[int, ...] = (2, 3)
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
            source_world_size=2,
            logical_keys=keys,
            steps={key: value + 30.0 for key, value in values.items()},
            param_group={"lr": 3e-5, "betas": [0.9, 0.95]},
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
        {"lr": 3e-5, "betas": [0.9, 0.95], "params": [0, 1]}
    ]
    optimizer_state = optimizer["state"]
    torch.testing.assert_close(optimizer_state[0]["exp_avg"], torch.full((2, 3), 11.0))
    torch.testing.assert_close(
        optimizer_state[1]["exp_avg_sq"], torch.full((2, 3), 22.0)
    )
    assert optimizer_state[0]["step"].item() == 31.0
    assert optimizer_state[1]["step"].item() == 32.0


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
    assert state["optimizer"]["state"][1]["step"].item() == 32.0


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
