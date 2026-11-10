"""Rollback of containers replaced by production checkpoint staging."""

import gc
import weakref

import pytest
from test_trainer_rank_checkpoint_transactions import _run, checkpoint_rank

from art.trainer_rank import _checkpoint


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", ["stage", "commit"])
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("mode", ["current", "outside"])
def test_production_compaction_rollback(tmp_path, existing, fault, world, mode):
    _run(tmp_path, mode, fault, existing, world, world - 1)


@pytest.mark.parametrize("gap", [False, True])
def test_rollback_restores_the_live_module_map(gap):
    rank, owner, _config, adapter = checkpoint_rank(gap=gap)
    container, keys, registry = (
        owner._slot_modules,
        owner._slot_keys,
        rank._checkpoint_slots,
    )
    entries, indices, registered = (
        tuple(container.items()),
        tuple(keys.items()),
        tuple(registry.items()),
    )
    prior = {ref: owner._slot(ref) for ref in keys}
    snapshot = _checkpoint._slot_snapshot(rank)
    rollback = _checkpoint._load_rollback_changes(rank, snapshot)
    assert (
        rank._load_checkpoint_slot(
            "__art_loading_probe", adapter, alpha=1.0, _prepared=True
        )
        == 1
    )
    assert owner._slot_modules is not container  # production compaction ran
    staged = owner._slot(rank._slot_ref("__art_loading_probe"))
    references = [weakref.ref(staged), *(weakref.ref(p) for p in staged.parameters())]
    del staged
    rollback.apply()
    gc.collect()
    assert owner._slot_modules is container
    assert owner._slot_keys is keys
    assert rank._checkpoint_slots is registry
    assert tuple(owner._slot_modules.items()) == entries
    assert tuple(owner._slot_keys.items()) == indices
    assert tuple(rank._checkpoint_slots.items()) == registered
    assert all(owner._slot(ref) is slot for ref, slot in prior.items())
    assert not any(ref() is not None for ref in references)
