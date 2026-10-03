"""Exercise the pre-offset_keys safe_open API without changing installed packages."""

import pytest
import safetensors
from test_checkpoint_selective_read import _dependencies, _files, _prepared, _trainer

from art.trainer_rank import _checkpoint


@pytest.mark.parametrize("optimizer", [False, True])
def test_legacy_keys_api_preserves_checkpoint_bytes_and_selective_reads(
    monkeypatch, tmp_path, optimizer
):
    _dependencies(monkeypatch)
    monkeypatch.setattr(_checkpoint, "_ensure_finalize_group", lambda trainer: None)
    real = safetensors.safe_open
    reads = []
    indexes = []

    class LegacyKeys:
        def __init__(self, path, **kwargs):
            self.reader = real(path, **kwargs)
            self.is_snapshot = path.parent.name == "snapshot"

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            if self.is_snapshot:
                indexes.append(True)
            return self.reader.keys()

        def get_tensor(self, key):
            if self.is_snapshot:
                reads.append(key)
            return self.reader.get_tensor(key)

    # safetensors 0.4.3/0.5.3 expose keys/get_tensor but no offset_keys.
    # Their torch.load_file iterates keys, which is the eager reference here.
    def eager_keys(prepared, relative, prefix, keys=None, *, snapshot=None):
        with LegacyKeys(
            prepared.snapshot / relative, framework="pt", device="cpu"
        ) as f:
            payload = {key: f.get_tensor(key) for key in f.keys()}
        if keys is None:
            return {
                key.removeprefix(prefix + "/"): value
                for key, value in payload.items()
                if key.startswith(prefix + "/")
            }
        return {key: payload[f"{prefix}/{key}"] for key in keys}

    monkeypatch.setattr(safetensors, "safe_open", LegacyKeys)
    actual = _prepared(tmp_path / "actual", optimizer=optimizer)
    expected = _prepared(tmp_path / "expected", optimizer=optimizer)
    _checkpoint.finish_checkpoint_save(_trainer(actual), str(actual.destination))
    assert len(indexes) == 2
    assert len(reads) == (20 if optimizer else 4)
    monkeypatch.setattr(_checkpoint, "_read_snapshot", eager_keys)
    _checkpoint.finish_checkpoint_save(_trainer(expected), str(expected.destination))
    assert _files(actual.destination) == _files(expected.destination)
    assert not actual.snapshot.exists() and not actual.reservation.exists()
