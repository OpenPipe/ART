from types import SimpleNamespace

import pytest

from art.megatron.runtime.executor import MCoreRunSlotExecutor


def _executor() -> tuple[MCoreRunSlotExecutor, SimpleNamespace]:
    executor = object.__new__(MCoreRunSlotExecutor)
    state = SimpleNamespace(
        unregistering=False,
        kl_reference_counts={"checkpoint": 2},
        kl_reference_acquisitions={
            "older": "checkpoint",
            "current": "checkpoint",
        },
    )
    executor._kl_reference_preparations = {}
    executor._run_cleanups = {}
    executor._require_run = lambda _run_id, **_kwargs: state  # type: ignore[method-assign]
    executor._trim_kl_references = lambda _state: None  # type: ignore[method-assign]
    return executor, state


def test_release_is_token_specific_and_idempotent() -> None:
    executor, state = _executor()

    executor.release_kl_reference("run", "checkpoint", "current")
    executor.release_kl_reference("run", "checkpoint", "current")

    assert state.kl_reference_counts == {"checkpoint": 1}
    assert state.kl_reference_acquisitions == {"older": "checkpoint"}


def test_release_identity_mismatch_preserves_the_acquisition() -> None:
    executor, state = _executor()

    with pytest.raises(RuntimeError, match="changed identity"):
        executor.release_kl_reference("run", "other", "current")

    assert state.kl_reference_counts == {"checkpoint": 2}
    assert state.kl_reference_acquisitions["current"] == "checkpoint"


def test_unregister_rejects_active_reference_acquisitions() -> None:
    executor, state = _executor()

    with pytest.raises(RuntimeError, match="active KL reference"):
        executor.start_unregister_run("run")

    assert not state.unregistering
