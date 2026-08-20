from __future__ import annotations

from collections.abc import Callable
from functools import cache

from .adapter import LinearRecurrentFamilyAdapter
from .contract import LinearRecurrentContract, PartitionKind

AdapterKey = tuple[str, PartitionKind]
AdapterFactory = Callable[[], LinearRecurrentFamilyAdapter]


def _gdn_adapter() -> LinearRecurrentFamilyAdapter:
    from .gdn_adapter import GdnRecurrentFamilyAdapter

    return GdnRecurrentFamilyAdapter()


def _mamba_2_adapter() -> LinearRecurrentFamilyAdapter:
    from art.megatron.mamba.adapter import Mamba2RecurrentFamilyAdapter

    return Mamba2RecurrentFamilyAdapter()


_ADAPTER_FACTORIES: dict[AdapterKey, AdapterFactory] = {
    ("gated_delta_net", "token_sharded_chain"): _gdn_adapter,
    ("mamba_2", "head_sharded_full_tree"): _mamba_2_adapter,
}


def get_recurrent_family_adapter(
    contract: LinearRecurrentContract,
) -> LinearRecurrentFamilyAdapter:
    """Resolve one statically registered family and partition implementation."""

    return _get_recurrent_family_adapter((contract.family_key, contract.partition_kind))


@cache
def _get_recurrent_family_adapter(key: AdapterKey) -> LinearRecurrentFamilyAdapter:
    factory = _ADAPTER_FACTORIES.get(key)
    if factory is None:
        raise NotImplementedError(
            "unregistered linear-recurrent adapter "
            f"family={key[0]!r} partition={key[1]!r}"
        )
    adapter = factory()
    if (adapter.family_key, adapter.partition_kind) != key:
        raise RuntimeError(
            f"linear-recurrent adapter registered under the wrong key: {key}"
        )
    return adapter
