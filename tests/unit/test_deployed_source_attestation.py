from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.megatron import artifacts


def test_source_only_view_requires_complete_matching_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts, "_native_worktree", lambda: False)
    monkeypatch.setenv(artifacts.DEPLOYED_SOURCE_ROOT_ENV, str(artifacts.REPO_ROOT))
    monkeypatch.setenv(artifacts.DEPLOYED_SOURCE_COMMIT_ENV, "a" * 40)
    monkeypatch.setenv(artifacts.DEPLOYED_SOURCE_TREE_ENV, "b" * 40)

    state = artifacts.pinned_git_state("suite")

    assert state.path == str(artifacts.REPO_ROOT.resolve())
    assert state.commit == "a" * 40
    assert state.tree == "b" * 40
    assert not state.dirty


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (artifacts.DEPLOYED_SOURCE_ROOT_ENV, None),
        (artifacts.DEPLOYED_SOURCE_COMMIT_ENV, "short"),
        (artifacts.DEPLOYED_SOURCE_TREE_ENV, "short"),
    ],
)
def test_source_only_view_rejects_missing_or_malformed_attestation(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str | None
) -> None:
    monkeypatch.setattr(artifacts, "_native_worktree", lambda: False)
    monkeypatch.setenv(
        artifacts.DEPLOYED_SOURCE_ROOT_ENV, str(Path(artifacts.REPO_ROOT))
    )
    monkeypatch.setenv(artifacts.DEPLOYED_SOURCE_COMMIT_ENV, "a" * 40)
    monkeypatch.setenv(artifacts.DEPLOYED_SOURCE_TREE_ENV, "b" * 40)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match="attestation|object IDs"):
        artifacts.pinned_git_state("suite")
