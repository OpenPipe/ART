from types import SimpleNamespace

import pytest

from art.megatron.backend import _PackingConfig


@pytest.mark.parametrize(("cp", "expected"), [(1, 64), (2, 256)])
def test_megatron_packing_prunes_short_cross_rank_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    cp: int,
    expected: int,
) -> None:
    monkeypatch.setattr(
        "art.megatron.backend.get_megatron_runtime_config",
        lambda: SimpleNamespace(topology=SimpleNamespace(cp=cp)),
    )

    config = _PackingConfig.from_dev_config(
        {"packed_sequence_length": 1024},
        include_moe_routing=False,
        collect_packing_shapes=False,
    )

    assert config.min_prefix_tree_shared_segment_length == expected
